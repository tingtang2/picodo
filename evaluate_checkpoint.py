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

from configs import resolver_setup 
import data
import model as model_lib
import utils
import train 
from functools import partial
import json

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

@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn_light(model_state, model_graphdef, x):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    losses = losses.at[:, -1].set(0)
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    manual_losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    manual_losses = manual_losses.at[:, -1].set(0)
    
    label_logits = jnp.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
    manual_optax_like = jax.nn.logsumexp(logits, axis=-1) - label_logits
    manual_optax_like = manual_optax_like.at[:, -1].set(0)
    return losses.mean(), losses, manual_losses, manual_losses - losses, manual_optax_like, manual_optax_like - losses

@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c) # Resolve interpolations (like model.F, model.N)
    
    run_name = c.run_name if c.run_name else 'picodo_run'
    
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(c))
    print("---------------------")

    # Init JAX and Mesh 
    jax.distributed.initialize()
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    print(f"Sharding mesh: {mesh.shape} (data, model)")
    
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # Round Vocab Size
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    print(f"Model vocab size rounded to: {c.model.V}")
    
    # model
    print('initializing model...')
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count()) # round V up to enable sharding
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

    print("Initializing abstract model structure...")
    model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(model)
    
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        c.opt.b2,
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )
    if c.opt.clip_by_global_norm:
         tx = optax.chain(optax.clip_by_global_norm(c.opt.clip_by_global_norm), tx)
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)
    
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    # Load Checkpoint
    gcp_bucket = getattr(c.checkpoint, 'gcp_bucket', None)
    if gcp_bucket:
        if not gcp_bucket.startswith('gs://'):
            gcp_bucket = f'gs://{gcp_bucket}'
        ckpt_dir = f'{gcp_bucket.rstrip("/")}/{run_name}'
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
        mngr_options = ocp.CheckpointManagerOptions(create=False) # Don't create
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=mngr_options)
    
    # Create a directory to store evaluation metrics
    metrics_dir = os.path.join(c.checkpoint.workdir, 'evaluation_metrics', run_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    step_to_load = c.checkpoint.start_step if c.checkpoint.start_step is not None else ckpt_mngr.latest_step()
    
    if step_to_load is None:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        return
        
    print(f"Restoring checkpoint from step {step_to_load} in {ckpt_dir}...")
    restored_data = ckpt_mngr.restore(step_to_load, args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_opt_state),
    training_metadata=ocp.args.JsonRestore(),))
    opt_state = restored_data['state']
    start_step = restored_data['training_metadata']['next_step']
    print("Checkpoint restored successfully.")
    print(f'start_step: {start_step}')
    ckpt_mngr.close()
    
    print('sim training step')
    metrics = {}
    
    train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
    # opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train.train_step(opt_state, opt_graphdef, model_graphdef, ds_train[start_step])
    batch_loss, train_raw_loss, manual_raw_loss, manual_diff, manual_optax_like, manual_optax_like_diff = loss_fn_light(
        opt_state.model, model_graphdef, ds_train[start_step]
    )
    train_raw_loss_flat = jnp.ravel(train_raw_loss)
    train_raw_loss_sorted = jnp.sort(train_raw_loss_flat)
    print("train_raw_loss top5 low:", train_raw_loss_sorted[:5])
    print("train_raw_loss top5 high:", train_raw_loss_sorted[-5:][::-1])
    print("train module:", train.__file__)
    neg_mask = train_raw_loss_flat < 0
    neg_count = jnp.sum(neg_mask)
    print("train_raw_loss min:", train_raw_loss_flat.min(), "neg_count:", neg_count, "dtype:", train_raw_loss_flat.dtype)
    manual_flat = jnp.ravel(manual_raw_loss)
    print(
        "manual_raw_loss min:",
        manual_flat.min(),
        "neg_count:",
        jnp.sum(manual_flat < 0),
        "diff_min:",
        manual_diff.min(),
        "diff_max:",
        manual_diff.max(),
        "diff_abs_max:",
        jnp.max(jnp.abs(manual_diff)),
    )
    manual_optax_like_flat = jnp.ravel(manual_optax_like)
    print(
        "manual_optax_like min:",
        manual_optax_like_flat.min(),
        "neg_count:",
        jnp.sum(manual_optax_like_flat < 0),
        "diff_min:",
        manual_optax_like_diff.min(),
        "diff_max:",
        manual_optax_like_diff.max(),
        "diff_abs_max:",
        jnp.max(jnp.abs(manual_optax_like_diff)),
    )
    if neg_count > 0:
        neg_idx = jnp.argmax(neg_mask)
        b = int(jax.device_get(neg_idx // train_raw_loss.shape[1]))
        t = int(jax.device_get(neg_idx % train_raw_loss.shape[1]))
        print("first negative idx:", b, t, "value:", train_raw_loss[b, t])
        batch = ds_train[start_step]
        y_full = jnp.roll(batch, -1, axis=1)
        max_token = jnp.max(y_full)
        oob_mask = y_full >= c.model.V
        oob_count = jnp.sum(oob_mask)
        print("max_token:", max_token, "vocab_size:", c.model.V, "oob_count:", oob_count)
        batch_ex = batch[b:b + 4]
        model = nnx.merge(model_graphdef, opt_state.model)
        logits_ex = model(batch_ex)
        y_ex = jnp.roll(batch_ex, -1, axis=1)
        log_probs_ex = jax.nn.log_softmax(logits_ex.astype(jnp.float32), axis=-1)
        manual = -log_probs_ex[0, t, y_ex[0, t]]
        print("manual ce:", manual, "is_last_token:", t == batch.shape[1] - 1)
        print("logits nan:", jnp.any(jnp.isnan(logits_ex)), "inf:", jnp.any(jnp.isinf(logits_ex)))
        try:
            import torch
            logits_ex_np = jax.device_get(logits_ex.astype(jnp.float32))
            y_ex_np = jax.device_get(y_ex)
            logits_t = torch.tensor(logits_ex_np)
            y_t = torch.tensor(y_ex_np, dtype=torch.long)
            ce_flat = torch.nn.functional.cross_entropy(
                logits_t.view(-1, logits_t.shape[-1]),
                y_t.view(-1),
                reduction="none",
            )
            ce_t = ce_flat.view(y_t.shape)
            print("torch ce:", float(ce_t[0, t]), "is_last_token:", t == y_t.shape[1] - 1)
        except Exception as e:
            print("torch ce skipped:", e)
    
    diagnostics_dir = os.path.join(c.checkpoint.workdir, 'top_loss_diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    # save diagnostic data
    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'train_raw_losses_step_{start_step}.npy', data=manual_raw_loss)
    # utils.save_to_numpy(diagnostics_dir, f'train_data_{start_step}', ds_train[start_step])
    mean_output_logit, output_logit_norm = train.get_mean_and_norm_output_logit(opt_state.model, model_graphdef, ds_train[start_step])
    print(mean_output_logit)
    
    # logging
    train_loss_sum += batch_loss
    train_med_loss_sum += jnp.median(train_raw_loss)
    train_lower_90th_mean_loss_sum += utils.compute_lower_90th_percentile_mean(train_raw_loss)
    train_loss_num += 1
    
    
    metrics['train_loss'] = train_loss_sum / train_loss_num
    metrics['train_med_loss'] = train_med_loss_sum / train_loss_num
    metrics['train_lower_90th_mean_loss'] = train_lower_90th_mean_loss_sum / train_loss_num
    metrics['train_mean_output_logit'] = mean_output_logit
    print(start_step)
    metrics['lr'] = lr_schedule(start_step)
    
    # Run Evaluation
    print("Running evaluation on validation set...")
    eval_loss, eval_raw_loss, eval_logits, mean_eval_output_logit = train.eval_step(c, opt_state.model, model_graphdef, ds_valid)
    eval_raw_loss_flat = jnp.concatenate(eval_raw_loss, axis=0)
    

    metrics.update({
        "step": step_to_load,
        "eval_loss": eval_loss,
        "eval_med_loss": jnp.median(eval_raw_loss_flat),
        "eval_lower_90th_mean_loss": utils.compute_lower_90th_percentile_mean(eval_raw_loss_flat),
        'eval_output_logit_mean': mean_eval_output_logit
    })
    

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("---------------")
    
    # # Save metrics to a JSON file
    # print(f"Saving metrics to {metrics_dir}...")
    # # Convert JAX/Numpy arrays to standard Python types for JSON serialization
    # serializable_metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
    # metrics_filename = os.path.join(metrics_dir, f'metrics_step_{step_to_load}.json')
    
    # try:
    #     with open(metrics_filename, 'w') as f:
    #         json.dump(serializable_metrics, f, indent=4)
    #     print(f"Successfully saved metrics to {metrics_filename}")
    # except Exception as e:
    #     print(f"Error saving metrics to {metrics_filename}: {e}")
    # # --- END ADDED ---


if __name__ == '__main__':
    main()
