import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import eval_step, get_mean_output_logit, loss_fn
import utils
import data
import model as model_lib
from configs import resolver_setup

import jax
import math
from flax import nnx
import optax
from omegaconf import OmegaConf, DictConfig
import hydra
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy 
import jax.numpy as jnp
from tqdm.auto import tqdm
import wandb
import sys
from functools import partial


@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn_light(model_state, model_graphdef, x):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y)
    losses = losses.at[:, -1].set(0)
    return losses.mean(), losses


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state',))
def custom_train_step(opt_state, opt_graphdef, model_graphdef, batch):
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_light, has_aux=True)(opt_state.model, model_graphdef, batch)
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, grads


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c)
    print(OmegaConf.to_yaml(c))

    # init distributed env if using multiple vms
    jax.distributed.initialize()
    
    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # sharding
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    print('sharding mesh:', ', '.join(f'{k}={v}' for k, v in mesh.shape.items()))

    # --- 1. Initialize Base Model ---
    print('initializing base model...')
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count()) # round V up to enable sharding
    base_model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(base_model)
    
    # get num. model parameters
    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(base_model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')
    
    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.batch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    print("Setting up structure to load base model checkpoint...")
    
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    tx = optax.inject_hyperparams(optax.adamw)(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    
    clip_by_global_norm = c.opt.clip_by_global_norm
    if clip_by_global_norm:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_by_global_norm), tx)
    
    optimizer = nnx.ModelAndOptimizer(base_model, tx)
    _ , opt_state = nnx.split(optimizer)

    # set up checkpointing
    start_step = 0
    ckpt_mngr = None
    base_abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    run_name = c.run_name if c.run_name else 'picodo_run'
    ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)
    mngr_options = ocp.CheckpointManagerOptions(create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=mngr_options)
    
    step_to_load = c.checkpoint.start_step if c.checkpoint.start_step is not None else ckpt_mngr.latest_step()
    
    if step_to_load is None:
        raise ValueError(f"No checkpoint found in {ckpt_dir} to load base model from.")
    start_step = step_to_load
        
    print(f"Restoring base model from step {step_to_load} in {ckpt_dir}...")
    restored_data = ckpt_mngr.restore(step_to_load, args=ocp.args.Composite(state=ocp.args.StandardRestore(base_abstract_opt_state),
            training_metadata=ocp.args.JsonRestore(),))
    opt_state = restored_data['state']
    start_step = restored_data['training_metadata']['step']
    print(f'Successfully restored checkpoint. Resuming from step {start_step}.')
    
    model = nnx.merge(model_graphdef, opt_state.model)
    print("Setting up optimizer for continued training...")
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    
    # We define the global schedule but wrap it to shift the input 'count' by 'step_to_load'
    global_lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    
    def shifted_lr_schedule(count):
        # The optimizer starts counting at 0, so we add the checkpoint step to get the absolute step
        return global_lr_schedule(count + step_to_load)

    tx = optax.inject_hyperparams(optax.adamw)(shifted_lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    
    clip_by_global_norm = c.opt.clip_by_global_norm
    if clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(clip_by_global_norm), tx)
    
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, _ = nnx.split(optimizer)  # keep restored opt_state to stay consistent with checkpoint

    new_ckpt_mngr = None
    if c.checkpoint.turn_on:
        # Use a suffix or different run name for the bias tuning run
        bias_run_name = f"{run_name}_reverse_bad_checkpoints_only_{c.opt.peak_lr}"
        new_ckpt_dir = os.path.join(c.checkpoint.workdir, bias_run_name)
        
        mngr_options = ocp.CheckpointManagerOptions(
            create=True,
            preservation_policy=preservation_policy.LatestN(c.checkpoint.max_to_keep)
        )
        
        new_ckpt_mngr = ocp.CheckpointManager(new_ckpt_dir, options=mngr_options)
        print(f"New checkpoints will be saved to: {new_ckpt_dir}")

    # Initialize model with optimizer state
    model = nnx.merge(model_graphdef, opt_state.model)
    
    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=f"{run_name}_reverse_checkpoint_{c.opt.peak_lr}")
        wandb.summary.update(n_params) # Logs original params, maybe should log bias count too
    
    if c.diagnostics.end_step:
        num_opt_steps = c.diagnostics.end_step

    pbar = range(start_step, num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar, initial=start_step, total=num_opt_steps)
    last_opt_state = None
    for step in pbar:
        # training step
        last_opt_state = jax.tree_util.tree_map(jax.device_get, opt_state)  # keep snapshot on host to save device memory
        opt_state, batch_loss, train_raw_loss, grads = custom_train_step(opt_state, opt_graphdef, model_graphdef, ds_train[step])

        # logging
        metrics = {}
        metrics['train_loss'] = batch_loss
        metrics['train_med_loss'] = jnp.median(train_raw_loss)
        metrics['train_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(train_raw_loss)
        metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
        metrics['train_output_logit_mean'] = get_mean_output_logit(opt_state.model, model_graphdef, ds_train[step])
        metrics['lr'] = shifted_lr_schedule(step - start_step)
        # metrics.update(utils.get_layer_grad_norms(grads))
        # metrics.update(utils.get_layer_weight_norms(opt_state.model))
        # metrics.update(utils.get_layer_moment_norms(opt_state))

        if jax.process_index() == 0:
            wandb.log(metrics, step)
            pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
    
        # eval and checkpointing
        if step % c.eval_every_steps == 0:
            eval_loss, eval_raw_loss, eval_logits, mean_eval_output_logit = eval_step(c, opt_state.model, model_graphdef, ds_valid)
            flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
            metrics = {}
            metrics['eval_loss'] = eval_loss
            metrics['eval_output_logit_mean'] = mean_eval_output_logit
            metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
            metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            if jax.process_index() == 0:
                wandb.log(metrics, step)

            if float(eval_loss) > 10.0 and last_opt_state is not None:
                opt_state = jax.tree_util.tree_map(jax.device_put, last_opt_state)
                if jax.process_index() == 0:
                    print(f"Reversed optimizer update at step {step} due to high eval loss: {float(eval_loss):.2f}")
            
        if c.checkpoint.turn_on and step % c.checkpoint.checkpoint_every_steps == 0:
            new_ckpt_mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(opt_state), training_metadata=ocp.args.JsonSave({
                'step': step + 1})), force=True)
    
    if num_opt_steps != len(ds_train):
        print('exiting early')
        wandb.finish()
        if new_ckpt_mngr: new_ckpt_mngr.close()
        sys.exit(1)


if __name__ == '__main__':
    main()
