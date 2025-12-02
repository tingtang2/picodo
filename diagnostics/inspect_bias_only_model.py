import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf, DictConfig
import hydra
import numpy as np
import tiktoken
import math

import matplotlib.pyplot as plt

# Import project modules
import utils
import model as model_lib
from configs import resolver_setup

# --- Define BiasOnlyModel (Must match training definition) ---
class BiasOnlyModel(nnx.Module):
    """
    Wraps a base model and adds a learnable bias to the output logits.
    """
    def __init__(self, base_graphdef, base_state, V):
        self.base_graphdef = base_graphdef
        # Store base_state as a Variable (not Param)
        self.base_state = nnx.Variable(base_state)
        # Initialize bias to zeros
        self.bias = nnx.Param(jnp.zeros(V))

    def __call__(self, x, attention_mask=None):
        frozen_state = jax.lax.stop_gradient(self.base_state.value)
        base_model = nnx.merge(self.base_graphdef, frozen_state)
        logits = base_model(x, attention_mask=attention_mask)
        return logits + self.bias

@hydra.main(version_base=None, config_path='../configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c)
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(c))
    
    # Init JAX
    jax.distributed.initialize()
    
    # Setup Mesh (Required for sharded model loading)
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    
    key = jax.random.key(c.seed)
    
    # Round vocab size (important for sharding)
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    print(f"Model V rounded to: {c.model.V}")

    # --- 1. Initialize Base Model Structure ---
    print('Initializing base model structure...')
    # We create a dummy base model just to get the structure/shapes right
    # The actual weights will be overwritten by the checkpoint
    base_model = model_lib.create_sharded_model(c.model, key)
    base_graphdef = nnx.graphdef(base_model)
    _, base_state = nnx.split(base_model)

    # --- 2. Initialize BiasOnlyModel Structure ---
    print('Initializing BiasOnlyModel structure...')
    model = BiasOnlyModel(base_graphdef, base_state, c.model.V)
    
    # --- 3. Setup Optimizer Structure (Needed for Orbax restoration) ---
    # We need to replicate the exact structure used in saving
    # The training script wrapped the model in ModelAndOptimizer
    print('Setting up optimizer structure...')
    num_opt_steps=10000
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    tx = optax.inject_hyperparams(optax.adamw)(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    
    clip_by_global_norm = c.opt.clip_by_global_norm
    if clip_by_global_norm:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_by_global_norm), tx)
    
    optimizer = nnx.ModelAndOptimizer(model, tx)
    _, opt_state = nnx.split(optimizer)
    
    # Create abstract state for restoration
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    # --- 4. Load Checkpoint ---
    # Construct the path to the bias-only run directory
    run_name = c.run_name if c.run_name else 'picodo_run'
    bias_run_name = f"{run_name}_bias_only_final"
    ckpt_dir = os.path.join(c.checkpoint.workdir, bias_run_name)
    
    print(f"Looking for checkpoints in: {ckpt_dir}")
    mngr_options = ocp.CheckpointManagerOptions(create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=mngr_options)
    
    step_to_load = ckpt_mngr.latest_step()
    if step_to_load is None:
        print("No checkpoint found.")
        return

    print(f"Restoring from step {step_to_load}...")
    restored_data = ckpt_mngr.restore(
        step_to_load, 
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_opt_state),
            training_metadata=ocp.args.JsonRestore()
        )
    )
    
    # Extract model state
    restored_opt_state = restored_data['state']
    # Restored model state contains {base_state: ..., bias: ...}
    bias_weights = restored_opt_state.model.bias.value
    
    print("\n=== Bias Weights Inspection ===")
    
    # --- 5. Inspect Weights ---
    # Move to numpy for easier printing
    bias_np = np.array(bias_weights)
    
    print(f"Shape: {bias_np.shape}")
    print(f"Mean:   {bias_np.mean():.6f}")
    print(f"Std:    {bias_np.std():.6f}")
    print(f"Min:    {bias_np.min():.6f}")
    print(f"Max:    {bias_np.max():.6f}")
    print(f"L2 Norm: {np.linalg.norm(bias_np):.6f}")

    # --- 6. Top/Bottom Tokens ---
    try:
        enc = tiktoken.get_encoding("gpt2")
        
        # Get indices of top 20 and bottom 20 biases
        top_k = 40
        sorted_indices = np.argsort(bias_np)
        bottom_indices = sorted_indices[:top_k]
        top_indices = sorted_indices[-top_k:][::-1] # Reverse to get highest first

        print(f"\n--- Top {top_k} Tokens (Highest Bias / Most Encouraged) ---")
        for idx in top_indices:
            try:
                token_str = enc.decode([idx])
                # Escape newlines/tabs for cleaner printing
                token_repr = repr(token_str)
            except:
                token_repr = "[Un-decodable]"
            print(f"Token ID {idx:<5} | Bias: {bias_np[idx]:.4f} | {token_repr}")

        print(f"\n--- Bottom {top_k} Tokens (Lowest Bias / Most Suppressed) ---")
        for idx in bottom_indices:
            try:
                token_str = enc.decode([idx])
                token_repr = repr(token_str)
            except:
                token_repr = "[Un-decodable]"
            print(f"Token ID {idx:<5} | Bias: {bias_np[idx]:.4f} | {token_repr}")
            
    except Exception as e:
        print(f"\nCould not decode tokens (requires tiktoken): {e}")

    # --- 7. Plot Histogram ---
    print("\nPlotting histogram...")
    try:
        plt.figure(figsize=(10, 6))
        # Use a reasonable number of bins (e.g., 100) to see distribution shape
        plt.hist(bias_np, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Histogram of Learned Bias Weights\nRun: {bias_run_name}")
        plt.xlabel("Bias Weight Value")
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.5)
        
        # Save to the checkpoint directory
        output_path = os.path.join(ckpt_dir, 'bias_histogram.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Histogram saved to: {output_path}")
    except Exception as e:
        print(f"Error plotting histogram: {e}")

if __name__ == '__main__':
    main()