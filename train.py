import jax
import jax.numpy as jnp
import optax
import wandb
import data, utils
import model as model_lib
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig


@nnx.jit
def loss_fn(model, batch):
  x, y, weights = data.get_in_out(batch)
  logits = model(x)
  losses = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
  mean_loss = jnp.sum(losses * weights) / weights.sum()
  return mean_loss


@nnx.jit
def train_step(optimizer, batch):
  loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model, batch)
  optimizer.update(grads) # in-place update
  return {'train_loss': loss}


def eval_step(model, dataset):
  loss = 0
  for batch in dataset:
    loss += loss_fn(model, batch)
  mean_loss = loss / len(dataset)
  return {'eval_loss': mean_loss}


def train_and_evaluate(c: DictConfig):

  # datastes
  get_batch_train, ds_train_size = data.make_ds_loader(c.ds_path_train, c.model.L, c.batch_size_train)
  get_batch_valid, ds_valid_size = data.make_ds_loader(c.ds_path_valid, c.model.L, c.batch_size_valid)

  # get number of training/validation steps
  c.num_tokens_train = c.num_tokens_train or ds_train_size
  c.num_tokens_valid = c.num_tokens_valid or ds_valid_size
  tokens_per_train_step = c.batch_size_train * c.model.L
  tokens_per_valid_step = c.batch_size_valid * c.model.L
  num_train_steps = c.num_tokens_train // tokens_per_train_step
  num_valid_steps = c.num_tokens_valid // tokens_per_valid_step

  # model
  # all devices are aligned across a single mesh axis called 'data'
  # we use FSDP to shard data, model, and optimzier parameters across this axis
  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',))
  model = model_lib.create_sharded_model(c.model, mesh, c.seed)
  data_sharding = NamedSharding(mesh, P('data')) # data parallelism
  ds_valid = jnp.stack([get_batch_valid(i) for i in range(num_valid_steps)])
  with mesh: ds_valid = jax.device_put(ds_valid, NamedSharding(mesh, P(None, 'data')))

  # optimizer
  lr_schedule = optax.schedules.warmup_cosine_decay_schedule(c.opt.init_lr, c.opt.peak_lr, c.opt.warmup_steps, num_train_steps)
  lr_schedule_cpu = jax.jit(lr_schedule, backend='cpu') # for logging only
  tx = optax.adamw(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
  optimizer = nnx.Optimizer(model, tx)

  # start wandb
  if c.wandb_project is not None:
    wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode)

  # training loop
  # note: metrics for each steps are processed only after asynchronously dispatching the next step
  pending_train_metrics = None
  pending_eval_metrics = None
  pbar = tqdm(range(num_train_steps))
  with mesh:
    for step in pbar:

      # training step
      batch = jax.device_put(get_batch_train(step), data_sharding)
      train_metrics = train_step(optimizer, batch)

      # async logging
      if pending_train_metrics is not None:
        pending_train_metrics |= dict(learning_rate=lr_schedule_cpu(step), train_tokens_seen=step*tokens_per_train_step)
        pbar.set_postfix_str(f'loss={pending_train_metrics["train_loss"]:.2f}')
        wandb.log(pending_train_metrics, step-1)
      pending_train_metrics = train_metrics
      if pending_eval_metrics is not None:
        wandb.log(pending_eval_metrics, step-1)
        pending_eval_metrics = None

      # eval step
      if ((step+1) % c.eval_every_steps == 0) or ((step+1) == num_train_steps):
        pending_eval_metrics = eval_step(optimizer.model, ds_valid)

    wandb.log(pending_train_metrics, step)
    wandb.log(pending_eval_metrics, step)
