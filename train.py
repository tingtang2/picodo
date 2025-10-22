import math
import jax
import jax.numpy as jnp
import optax
import wandb
from functools import partial
from flax import nnx
from optax import tree_utils as otu
import optax
from tqdm.auto import tqdm
from omegaconf.dictconfig import DictConfig
import data, utils
import model as model_lib
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy 
import os
import heapq
import numpy as np
import sys



@partial(jax.jit, static_argnames=('model_graphdef'))
def loss_fn(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    loss_mask = jnp.ones(x.shape, dtype=bool).at[:, -1].set(False)
    logits = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y) # [B, T]
    losses = losses.at[:, -1].set(0) #
    
    per_sequence_loss = losses.mean(axis=1)
    mean_loss = per_sequence_loss.mean()
    
    return mean_loss, per_sequence_loss

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'), donate_argnames=('opt_state'))
def train_step(opt_state, opt_graphdef, model_graphdef, batch):
    # Use has_aux=True to get the per_sequence_loss
    (loss, per_sequence_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(opt_state.model, model_graphdef, batch)
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    
    # Return the per_sequence_loss as well
    return opt_state, loss, per_sequence_loss


def eval_step(model_state, model_graphdef, dataset, k_top_batches=0):
    loss_sum = jnp.zeros([], dtype=jnp.float32)
    top_k_heap = []
    all_losses = []
    
    for batch in dataset:
        batch_loss, per_sequence_loss = loss_fn(model_state, model_graphdef, batch)
        all_losses.append(batch_loss)
        loss_sum += batch_loss

        if k_top_batches > 0:
            if len(top_k_heap) < k_top_batches:
                heapq.heappush(top_k_heap, (batch_loss, batch))
            else:
                heapq.heappushpop(top_k_heap, (batch_loss, batch))

    mean_loss = loss_sum / len(dataset)
    
    if k_top_batches > 0:
        top_k_with_loss = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
        return mean_loss, all_losses, top_k_with_loss
    else:
        return mean_loss, all_losses, None


def train_and_evaluate(c: DictConfig):

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

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_loss_num = jnp.zeros([]), 0

    if c.diagnostics.end_step:
        num_opt_steps = c.diagnostics.end_step

    pbar = range(start_step, num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar, initial=start_step, total=num_opt_steps)
    for step in pbar:

        # training step
        opt_state, batch_loss, per_sequence_loss = train_step(opt_state, opt_graphdef, model_graphdef, ds_train[step])

        # logging
        train_loss_sum += batch_loss
        train_loss_num += 1
        if train_loss_num * tokens_per_opt_step >= c.log_every_tokens:
            metrics = {}
            metrics['train_loss'] = train_loss_sum / train_loss_num
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            metrics['lr'] = lr_schedule(step)
            if jax.process_index() == 0:
                wandb.log(metrics, step)
                pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
            train_loss_sum, train_loss_num = jnp.zeros([]), 0
        
        # eval and checkpointing
        if step % c.eval_every_steps == 0:
            k_top = 3 if c.diagnostics.save_top_token_ids else 0
            eval_loss, all_losses, top_eval_data = eval_step(opt_state.model, model_graphdef, ds_valid, k_top_batches=k_top)
            
            metrics = {}
            metrics['eval_loss'] = eval_loss
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            if jax.process_index() == 0:
                metrics['eval_loss_histogram'] = wandb.Histogram(all_losses)
                wandb.log(metrics, step)
            
            # diagnostics
            conditions_met = (
                c.diagnostics.save_top_token_ids and
                eval_loss > 7 and
                step > 300
            )
            
            if conditions_met and jax.process_index() == 0:
                print(f'Step {step}: eval_loss {eval_loss:.4f} > 7, saving top 3 eval batches to {ckpt_dir}...')
                
                if ckpt_dir:
                    diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
                    os.makedirs(diagnostics_dir, exist_ok=True)
                    
                    losses = []
                    batches_data = []
                    
                    for loss, batch in top_eval_data:
                        losses.append(float(loss))
                        batches_data.append(np.array(batch))
                        
                    current_batch = ds_train[step]
                    loss_path = os.path.join(diagnostics_dir, f'top_3_eval_losses_step_{step}.npy')
                    batches_path = os.path.join(diagnostics_dir, f'top_3_eval_batches_step_{step}.npy')
                    hist_path = os.path.join(diagnostics_dir, f'all_eval_losses_step_{step}.npy')
                    train_seq_loss_path = os.path.join(diagnostics_dir, f'per_seq_losses_{step}.npy')
                    train_batch_path = os.path.join(diagnostics_dir, f'train_batch_seq_{step}.npy')
                    
                    # Save the files
                    try:
                        np.save(loss_path, np.array(losses))
                        np.save(batches_path, np.stack(batches_data))
                        np.save(hist_path, np.array(all_losses))
                        np.save(train_seq_loss_path, np.array(per_sequence_loss))
                        np.save(train_batch_path, np.array(current_batch))
                        print(f'Saved diagnostic files to {diagnostics_dir} for step {step}')
                        
                    except Exception as e:
                        print(f'Error saving diagnostic files for step {step}: {e}')

        if c.checkpoint.turn_on and step % c.checkpoint.checkpoint_every_steps == 0:
            ckpt_mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(opt_state), training_metadata=ocp.args.JsonSave({
                'step': step + 1})))
    
    if num_opt_steps != len(ds_train):
        print('exiting early')
        wandb.finish()
        ckpt_mngr.close()
        sys.exit(1)

    # eval at end of training
    eval_loss, all_losses, top_eval_data = eval_step(opt_state.model, model_graphdef, ds_valid)
    if jax.process_index() == 0:
        wandb.log({'eval_loss': eval_loss, 'eval_loss_histogram': wandb.Histogram(all_losses)})
        wandb.finish()
        if ckpt_dir:
            diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
            os.makedirs(diagnostics_dir, exist_ok=True)
            
            hist_path = os.path.join(diagnostics_dir, f'all_eval_losses_step_{num_opt_steps}.npy')
            try:
                np.save(hist_path, np.array(all_losses))
                print(f'Saved diagnostic files to {diagnostics_dir} for step {num_opt_steps}')
                
            except Exception as e:
                print(f'Error saving diagnostic files for step {num_opt_steps}: {e}')

    # final checkpoint
    if c.checkpoint.turn_on:
        ckpt_mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(opt_state), training_metadata=ocp.args.JsonSave({
            'step': step + 1})))
        
        ckpt_mngr.wait_until_finished() 
        if jax.process_index() == 0:
            print(f'Saved final checkpoint at step {num_opt_steps} to {ckpt_mngr.directory}')
        ckpt_mngr.close()
