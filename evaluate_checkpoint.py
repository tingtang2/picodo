import hydra
import jax
import jax.numpy as jnp
import os
import math
import optax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf, DictConfig
from flax import nnx

from configs import resolver_setup 
import data
import model as model_lib
import utils
import train 

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
    tx = optax.inject_hyperparams(optax.adamw)(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    if c.opt.clip_by_global_norm:
         tx = optax.chain(optax.clip_by_global_norm(c.opt.clip_by_global_norm), tx)
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)
    
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    # Load Checkpoint
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
    start_step = restored_data['training_metadata']['step']
    print("Checkpoint restored successfully.")
    ckpt_mngr.close()
    
    print('sim training step')
    metrics = {}
    train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
    opt_state, batch_loss, train_raw_loss, grad_norm = train.train_step(opt_state, opt_graphdef, model_graphdef, ds_train[start_step])
    
    # logging
    train_loss_sum += batch_loss
    train_med_loss_sum += jnp.median(train_raw_loss)
    train_lower_90th_mean_loss_sum += utils.compute_lower_90th_percentile_mean(train_raw_loss)
    train_loss_num += 1
    
    
    metrics['train_loss'] = train_loss_sum / train_loss_num
    metrics['train_med_loss'] = train_med_loss_sum / train_loss_num
    metrics['train_lower_90th_mean_loss'] = train_lower_90th_mean_loss_sum / train_loss_num
    print(start_step)
    metrics['lr'] = lr_schedule(start_step)
    metrics['global_grad_norm'] = grad_norm
    
    # Run Evaluation
    print("Running evaluation on validation set...")
    eval_loss, eval_raw_loss = train.eval_step(opt_state.model, model_graphdef, ds_valid)
    eval_raw_loss_flat = jnp.concatenate(eval_raw_loss, axis=0)
    

    metrics.update({
        "step": step_to_load,
        "eval_loss": eval_loss,
        "eval_med_loss": jnp.median(eval_raw_loss_flat),
        "eval_lower_90th_mean_loss": utils.compute_lower_90th_percentile_mean(eval_raw_loss_flat),
    })
    

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("---------------")


if __name__ == '__main__':
    main()