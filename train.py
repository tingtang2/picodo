import math
import jax
import jax.numpy as jnp
import optax
import wandb
from functools import partial
from flax import nnx
import optax
from tqdm.auto import tqdm
from omegaconf.dictconfig import DictConfig
import data, utils
import model as model_lib
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy 
import os
import sys



@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y) # [B, T]
    losses = losses.at[:, -1].set(0)
    
    return losses.mean(), losses

@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn_z_loss(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y) # [B, T]
    losses = losses.at[:, -1].set(0)

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)  

    z_loss = (z**2).mean()
    lam = 1e-4
    
    return losses.mean() + lam * z_loss, losses

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state'))
def train_step(opt_state, opt_graphdef, model_graphdef, batch):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(opt_state.model, model_graphdef, batch)
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, utils.get_layer_grad_norms(grads)

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state'))
def train_step_z_loss(opt_state, opt_graphdef, model_graphdef, batch):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_z_loss, has_aux=True)(opt_state.model, model_graphdef, batch)
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, utils.get_layer_grad_norms(grads)

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logits_by_lm_head(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    return logits.reshape(-1, logits.shape[-1]).astype(jnp.float32).mean(axis=0) # [B * T, V] -> [V]

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_mean_output_logit(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    return logits.astype(jnp.float32).mean() 


def eval_step(c, model_state, model_graphdef, dataset):
    loss_sum = jnp.zeros([], dtype=jnp.float32)
    raw_losses = []
    total_logits = []
    logit_mean_sum = jnp.zeros([], dtype=jnp.float32)

    
    for i in range(len(dataset)):
        batch = dataset[i]
        if c.opt.use_z_loss:
            batch_loss, raw_loss = loss_fn_z_loss(model_state, model_graphdef, batch)
        else:
            batch_loss, raw_loss = loss_fn(model_state, model_graphdef, batch)
        loss_sum += batch_loss
        raw_losses.append(raw_loss)
        logit_mean_sum += get_mean_output_logit(model_state, model_graphdef, batch)
        if c.diagnostics.save_raw_losses:
            total_logits.append(get_logits_by_lm_head(model_state, model_graphdef, batch).astype(jnp.float32))

    mean_loss = loss_sum / len(dataset)
    mean_output_logit = logit_mean_sum / len(dataset)
    
    return mean_loss, raw_losses, total_logits, mean_output_logit



def train_and_evaluate(c: DictConfig):
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
    
    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.batch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    tx = optax.inject_hyperparams(optax.adamw)(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    
    clip_by_global_norm = c.opt.clip_by_global_norm
    if clip_by_global_norm:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_by_global_norm), tx)
    
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # set up checkpointing
    start_step = 0
    ckpt_mngr = None
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)
    if c.checkpoint.turn_on:
        run_name = c.run_name if c.run_name else 'picodo_run'
        ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)
        
        mngr_options = ocp.CheckpointManagerOptions(
            create=True,
            preservation_policy=preservation_policy.LatestN(c.checkpoint.max_to_keep)
        )
        
        ckpt_mngr = ocp.CheckpointManager(
            ckpt_dir,
            options=mngr_options
        )
        
        print(f'Checking for existing checkpoints in: {ckpt_dir}')
        latest_step = c.checkpoint.start_step if c.checkpoint.start_step != None else ckpt_mngr.latest_step()

        if latest_step is not None:
            print(f'Restoring checkpoint from step {latest_step} in {ckpt_dir}...')
            
            restored_data = ckpt_mngr.restore(latest_step, args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_opt_state),
            training_metadata=ocp.args.JsonRestore(),))
            opt_state = restored_data['state']
            start_step = restored_data['training_metadata']['step']
            print(f'Successfully restored checkpoint. Resuming from step {start_step}.')
        else:
            print('No checkpoint found. Starting from scratch.')

    model = nnx.merge(model_graphdef, opt_state.model)

    if c.diagnostics.save_raw_losses:
        if ckpt_dir:
            diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
            os.makedirs(diagnostics_dir, exist_ok=True)
            utils.save_to_numpy(save_dir=diagnostics_dir, name='val_dataset', data=ds_valid)
            utils.save_to_numpy(save_dir=diagnostics_dir, name='train_dataset', data=ds_train[:c.diagnostics.end_step])

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0

    if c.diagnostics.end_step:
        num_opt_steps = c.diagnostics.end_step

    pbar = range(start_step, num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar, initial=start_step, total=num_opt_steps)
    for step in pbar:
        # training step
        if c.opt.use_z_loss:
            opt_state, batch_loss, train_raw_loss, grad_norms = train_step_z_loss(opt_state, opt_graphdef, model_graphdef, ds_train[step])
        else:
            opt_state, batch_loss, train_raw_loss, grad_norms = train_step(opt_state, opt_graphdef, model_graphdef, ds_train[step])
        
        if c.diagnostics.save_raw_losses:
            train_logits = get_logits_by_lm_head(opt_state.model, model_graphdef, ds_train[step])

        # logging
        train_loss_sum += batch_loss
        train_med_loss_sum += jnp.median(train_raw_loss)
        train_lower_90th_mean_loss_sum += utils.compute_lower_90th_percentile_mean(train_raw_loss)
        train_loss_num += 1
        if train_loss_num * tokens_per_opt_step >= c.log_every_tokens:
            metrics = {}
            metrics['train_loss'] = train_loss_sum / train_loss_num
            metrics['train_med_loss'] = train_med_loss_sum / train_loss_num
            metrics['train_lower_90th_mean_loss'] = train_lower_90th_mean_loss_sum / train_loss_num
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            metrics['train_output_logit_mean'] = get_mean_output_logit(opt_state.model, model_graphdef, ds_train[step])
            metrics['lr'] = lr_schedule(step)
            metrics.update(grad_norms)
            metrics.update(utils.get_layer_weight_norms(opt_state.model))
            metrics.update(utils.get_layer_moment_norms(opt_state))

            if jax.process_index() == 0:
                wandb.log(metrics, step)
                pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
            train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
        
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
            
            # diagnostics
            if c.diagnostics.save_raw_losses:
                if ckpt_dir:
                    diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
                    os.makedirs(diagnostics_dir, exist_ok=True)
                    
                    # save diagnostic data
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'train_raw_losses_step_{step}.npy', data=train_raw_loss)
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{step}.npy', data=eval_raw_loss)
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'train_logits_step_{step}.npy', data=train_logits)
                    utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_logits_step_{step}.npy', data=eval_logits)

        if c.checkpoint.turn_on and step % c.checkpoint.checkpoint_every_steps == 0:
            ckpt_mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(opt_state), training_metadata=ocp.args.JsonSave({
                'step': step + 1})), force=True)
    
    if num_opt_steps != len(ds_train):
        print('exiting early')
        wandb.finish()
        ckpt_mngr.close()
        sys.exit(1)

    # eval at end of training
    eval_loss, eval_raw_loss, eval_logits, mean_eval_output_logit = eval_step(c, opt_state.model, model_graphdef, ds_valid)
    metrics = {}
    flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
    metrics['eval_loss'] = eval_loss
    metrics['eval_output_logit_mean'] = mean_eval_output_logit
    metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
    metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)
    if jax.process_index() == 0:
        wandb.log(metrics)
        wandb.finish()
        if c.diagnostics.save_raw_losses:
            if ckpt_dir:
                diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
                os.makedirs(diagnostics_dir, exist_ok=True)
                
                # save diagnostic data
                utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{num_opt_steps}.npy', data=eval_raw_loss)
            
    # final checkpoint
    if c.checkpoint.turn_on and not c.diagnostics.save_raw_losses:
        ckpt_mngr.save(num_opt_steps, args=ocp.args.Composite(state=ocp.args.StandardSave(opt_state), training_metadata=ocp.args.JsonSave({
            'step': num_opt_steps + 1})))
        
        ckpt_mngr.wait_until_finished() 
        if jax.process_index() == 0:
            print(f'Saved final checkpoint at step {num_opt_steps} to {ckpt_mngr.directory}')
        ckpt_mngr.close()
