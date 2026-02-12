import hydra
import jax
import jax.numpy as jnp
import os
import math
import optax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf, DictConfig
from flax import nnx
import tiktoken  # For GPT-2 tokenization
from functools import partial

from configs import resolver_setup 
import model as model_lib
import utils 
import data

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
    return logits[:, -1, :] # [B, V]


def generate(model_state, model_graphdef, tokenizer, prompt, max_new_tokens, context_size):
    """
    Generate text using greedy decoding, creating and passing an attention mask.
    """
    
    # The GPT-2 tokenizer uses <|endoftext|> (50256) as its EOT token
    # and also for padding.
    pad_token_id = tokenizer.eot_token 
    eot = tokenizer._special_tokens['<|endoftext|>'] 
    print(f"Using pad token ID: {pad_token_id}")

    # Encode the prompt string into token IDs
    prompt_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    
    # Add batch dimension
    idx = jnp.array([prompt_tokens], dtype=jnp.int32) # [1, T_prompt]
    
    print("Generating response...")
    print(f"Prompt: \"{prompt}\"")
    
    # Generation loop
    for _ in range(max_new_tokens):
        # --- Context Handling ---
        # If the sequence is longer than the model's context size (T),
        # crop it to the last T tokens.
        if idx.shape[1] > context_size:
            idx_cond = idx[:, -context_size:]
            # Create a mask that is all 1s (no padding)
            attention_mask = jnp.ones_like(idx_cond, dtype=jnp.int32) # [1, T]
        else:
            # --- Padding ---
            # If the sequence is shorter than T, pad it on the left.
            idx_cond_unpadded = idx
            pad_len = context_size - idx_cond_unpadded.shape[1]
            
            # Pad sequence on the left with the pad token
            padding = jnp.full((1, pad_len), pad_token_id, dtype=idx_cond_unpadded.dtype)
            idx_cond = jnp.concatenate([padding, idx_cond_unpadded], axis=1)
            
            # --- Create Attention Mask ---
            # Create mask: 0 for padding, 1 for real tokens
            mask_padding = jnp.zeros((1, pad_len), dtype=jnp.int32)
            mask_real = jnp.ones_like(idx_cond_unpadded, dtype=jnp.int32)
            attention_mask = jnp.concatenate([mask_padding, mask_real], axis=1) # [1, T]

        
        # Ensure the input is exactly [1, T]
        assert idx_cond.shape == (1, context_size)
        assert attention_mask.shape == (1, context_size)
            
        # Get logits for the next token, passing the mask
        logits = get_next_token_logits(
            model_state, 
            model_graphdef, 
            idx_cond, 
            attention_mask
        ) # [1, V]
        
        # Greedy decoding: pick the token with the highest probability
        next_token = jnp.argmax(logits, axis=-1) # [1]
        
        # Add the new token to our sequence
        idx = jnp.concatenate([idx, next_token[:, None]], axis=1) # [1, T_prompt + 1]

        # Check if we generated the end-of-text token
        if next_token.item() == tokenizer.eot_token:
            print(" (End-of-text token generated)")
            break
            
    # --- Decoding ---
    # Get all generated tokens (excluding the prompt)
    generated_tokens = idx[0, len(prompt_tokens):].tolist()
    
    # Decode the tokens back into a string
    try:
        generated_text = tokenizer.decode(generated_tokens)
    except Exception as e:
        print(f"Warning: Error during decoding: {e}")
        generated_text = f"[Decoding Error: {generated_tokens}]"
        
    return generated_text

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
    # We need this to tell Orbax what structure to expect when loading
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
    ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)
    mngr_options = ocp.CheckpointManagerOptions(create=False) # Don't create
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
    
    # Extract the model state from the optimizer state
    model_state = opt_state.model

    # --- Initialize Tokenizer ---
    print("Loading GPT-2 tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Generate ---
    # !-- Customize your prompt here --!
    prompt = "The capital of Spain is"
    max_new_tokens = 30
    
    generated_text = generate(
        model_state, 
        model_graphdef, 
        tokenizer, 
        prompt, 
        max_new_tokens,
        c.model.T # Pass the model's context size
    )
    
    print("\n--- Generated Text ---")
    print(generated_text)
    print("------------------------")


if __name__ == '__main__':
    main()
