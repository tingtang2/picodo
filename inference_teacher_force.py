import hydra
import jax
import jax.numpy as jnp
import os
import math
import optax
import orbax.checkpoint as ocp
from etils import epath
from orbax.checkpoint._src.path import gcs_utils, step as step_lib
from omegaconf import OmegaConf, DictConfig
from flax import nnx
import tiktoken  # For GPT-2 tokenization
from functools import partial

from configs import resolver_setup 
import model as model_lib
import utils 
import data

class _StandardNameFormatHNS(step_lib._StandardNameFormat):
    """Fixes GCS HNS listing when step_prefix is None."""

    def _glob_step_paths(self, base_path: epath.PathLike) -> list[epath.Path]:
        base_path = epath.Path(base_path)
        if gcs_utils.is_hierarchical_namespace_enabled(base_path):
            bucket_name, path_prefix = gcs_utils.parse_gcs_path(base_path)
            bucket = gcs_utils.get_bucket(bucket_name)
            result = bucket.list_blobs(
                prefix=path_prefix,
                delimiter='/',
                include_folders_as_prefixes=True,
            )
            for _ in result.pages:
                pass
            step_prefix = self.step_prefix or ''
            return [
                epath.Path(f'gs://{bucket_name}/{folder}')
                for folder in result.prefixes
                if folder.startswith(os.path.join(path_prefix, step_prefix))
            ]
        return list(
            epath.Path(base_path).glob(
                f'{step_lib.step_prefix_with_underscore(self.step_prefix)}*'
            )
        )

@partial(jax.jit, static_argnames=('model_graphdef',))
def get_next_token_logits(model_state, model_graphdef, x, attention_mask):
    """
    JIT-compiled function to get the logits for the next token.
    x is expected to be [B, T]
    attention_mask is expected to be [B, T] (1 for real, 0 for pad)
    """
    model = nnx.merge(model_graphdef, model_state)
    # Get logits from the model, passing the mask
    logits = model(x, attention_mask=attention_mask) # [B, T, V]
    # Return logits for the very last token in the sequence
    return logits[:, -1, :].astype(jnp.float32) # [B, V]


def teacher_force_sequence(model_state, model_graphdef, tokenizer, sequence, top_k, context_size, diagnostics_dir, step_to_load):
    """
    Teacher-force a sequence through the model and print the top-K logits for
    the next token at each decoding step.
    """
    pad_token_id = tokenizer.eot_token
    seq = jnp.array(sequence, dtype=jnp.int32)[20:]
    # seq_len = seq.shape[0]
    seq_len = 20
    print(f"Teacher forcing first validation sequence of length {seq_len} with top_k={top_k}.")

    printable_prefix = tokenizer.decode(seq[:min(100, seq_len)].tolist())
    print(f"Prefix preview: {printable_prefix!r}")

    for step in range(seq_len - 1):
        # Use the gold prefix up to and including position `step`
        prefix = seq[: step + 1][None, :]  # [1, t]

        # Left-pad to the model context
        if prefix.shape[1] > context_size:
            idx_cond = prefix[:, -context_size:]
            attention_mask = jnp.ones_like(idx_cond, dtype=jnp.int32)
        else:
            pad_len = context_size - prefix.shape[1]
            padding = jnp.full((1, pad_len), pad_token_id, dtype=prefix.dtype)
            idx_cond = jnp.concatenate([padding, prefix], axis=1)

            mask_padding = jnp.zeros((1, pad_len), dtype=jnp.int32)
            mask_real = jnp.ones_like(prefix, dtype=jnp.int32)
            attention_mask = jnp.concatenate([mask_padding, mask_real], axis=1)

        logits = get_next_token_logits(model_state, model_graphdef, idx_cond, attention_mask)  # [1, V]
        utils.save_to_numpy(save_dir=diagnostics_dir, name=f'logits_step_{step}_teacher_force_checkpoint_{step_to_load}.npy', data=logits)
        capped_top_k = max(1, min(int(top_k), logits.shape[-1]))
        values, indices = jax.lax.top_k(logits[0], k=capped_top_k)

        target_token_id = int(seq[step + 1])
        target_token_str = tokenizer.decode([target_token_id])

        values_np = jax.device_get(values)
        indices_np = jax.device_get(indices)

        print(f"\nStep {step + 1}/{seq_len - 1} -> target id={target_token_id} token={target_token_str!r}")
        for rank, (token_id, logit_val) in enumerate(zip(indices_np.tolist(), values_np.tolist()), start=1):
            decoded_token = tokenizer.decode([token_id])
            decoded_token = decoded_token.replace("\n", "\\n")
            print(f"  {rank}. id={token_id:<6} logit={logit_val:.4f} token={decoded_token!r}")

@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c) # Resolve interpolations
    run_name = c.run_name if c.run_name else 'picodo_run'
    
    # For inference, force disable flash attention to use the
    # manual causal + padding mask in jax.nn.dot_product_attention.
    # The flash attention kernel is hard-coded for causal-only.
    print("Forcing 'c.model.use_flash_attn = False' for inference to support padding masks.")
    c.model.use_flash_attn = False
    
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(c))
    print("---------------------")

    # --- Setup JAX and Mesh ---
    jax.distributed.initialize()
    # For inference, we can just run on a single device.
    # We set up a minimal mesh.
    num_fsdp_devices = 1
    num_tp_devices = jax.device_count() // num_fsdp_devices
    if c.num_tp_devices > num_tp_devices:
        print(f"Warning: Requested {c.num_tp_devices} TP devices, but only {num_tp_devices} are available.")
        c.num_tp_devices = num_tp_devices
        
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    print(f"Using sharding mesh: {mesh.shape} (data, model)")
    
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3) 

    # --- Round Vocab Size ---
    # This MUST match the training config
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    print(f"Model vocab size rounded to: {c.model.V}")
    
    # --- Initialize Model Structure ---
    print('Initializing abstract model structure...')
    model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(model)
    
    # get num. model parameters
    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')
    
    
    print("Loading datasets...")
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.batch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size
    print(f"Loaded {len(ds_train)} training batches.")
    print(f"Loaded {len(ds_valid)} validation batches.")


    # --- Create Abstract Optimizer State ---
    print("Initializing abstract optimizer structure...")
    # We don't need the real optimizer, just the structure
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        c.opt.b2,
        eps=c.opt.eps,
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )
    if c.opt.clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(c.opt.clip_by_global_norm), tx)
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)
    
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)
    # --- Load Checkpoint ---
    gcp_bucket = getattr(c.checkpoint, 'gcp_bucket', None)
    if gcp_bucket:
        if not gcp_bucket.startswith('gs://'):
            gcp_bucket = f'gs://{gcp_bucket}'
        ckpt_dir = f'{gcp_bucket.rstrip("/")}/{run_name}'
        print(f'Loading checkpoint from GCS bucket: {ckpt_dir}')
    else:
        ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)

    step_prefix = getattr(c.checkpoint, 'step_prefix', None)
    if gcp_bucket:
        name_format = _StandardNameFormatHNS(step_prefix=step_prefix)
        mngr_options = ocp.CheckpointManagerOptions(
            create=False,
            step_name_format=name_format,
        )
    else:
        mngr_options = ocp.CheckpointManagerOptions(create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=mngr_options)
    
    step_to_load = c.checkpoint.start_step if c.checkpoint.start_step is not None else ckpt_mngr.latest_step()
    
    if step_to_load is None:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        return
        
    print(f"Restoring checkpoint from step {step_to_load} in {ckpt_dir}...")
    restored_data = ckpt_mngr.restore(step_to_load, args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_opt_state),
    training_metadata=ocp.args.JsonRestore(),))
    opt_state = restored_data['state']
    print("Checkpoint restored successfully.")
    ckpt_mngr.close()
    
    diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)                     
    # Extract the model state from the optimizer state
    model_state = opt_state.model

    # --- Initialize Tokenizer ---
    print("Loading GPT-2 tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Teacher forcing on first validation example ---
    print("Fetching first validation batch and sequence...")
    first_val_batch = jax.device_get(ds_valid[1]) # [batch_size, T]
    first_sequence = first_val_batch[0] # [T]

    teacher_force_sequence(
        model_state=model_state,
        model_graphdef=model_graphdef,
        tokenizer=tokenizer,
        sequence=first_sequence,
        top_k=c.teacher_force_top_k,
        context_size=c.model.T,
        diagnostics_dir=diagnostics_dir,
        step_to_load=step_to_load
    )


if __name__ == '__main__':
    main()
