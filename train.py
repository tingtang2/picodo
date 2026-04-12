import math
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from functools import partial
from flax import nnx
from jax.experimental import multihost_utils
from tqdm.auto import tqdm
from omegaconf.dictconfig import DictConfig
import data, utils
import model as model_lib
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy 
from etils import epath
from orbax.checkpoint._src.path import gcs_utils, step as step_lib
import os
import sys

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



@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn(model_state, model_graphdef, x, collect_qkv_stats: bool = True): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True) # [B, T, V]
    else:
        logits = model(x, return_qkv=False) # [B, T, V]
        qkv_dict = None

    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}
    
    return losses.mean(), (losses, qkv_stats)

@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_z_loss(model_state, model_graphdef, x, collect_qkv_stats: bool = True): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True) # [B, T, V]
    else:
        logits = model(x, return_qkv=False) # [B, T, V]
        qkv_dict = None
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)  

    z_loss = (z**2).mean()
    lam = 1e-4

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}
    
    return losses.mean() + lam * z_loss, (losses, qkv_stats)


def _build_loss_skip_weights(
    losses,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
):
    use_log_loss = jnp.asarray(use_log_loss, dtype=jnp.bool_)
    stats_losses = jnp.where(
        use_log_loss,
        jnp.log1p(jnp.maximum(losses, 0.0)),
        losses,
    )
    scale = jnp.maximum(1.4826 * mad, eps)
    z = (stats_losses - center) / scale
    weights = jnp.ones_like(losses, dtype=jnp.float32)
    soft_band = jnp.logical_and(z > z_soft, z <= z_hard)
    weights = jnp.where(soft_band, jnp.asarray(soft_weight, dtype=jnp.float32), weights)
    hard_mask = jnp.logical_or(z > z_hard, stats_losses > abs_hard)
    weights = jnp.where(hard_mask, 0.0, weights)
    gate = jnp.asarray(apply_gate, dtype=weights.dtype)
    weights = gate * weights + (1.0 - gate) * jnp.ones_like(weights)
    return weights


def _to_host_numpy(x, dtype=np.float32, flatten: bool = False):
    """Converts possibly non-addressable JAX arrays to host NumPy arrays."""
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        host_value = multihost_utils.process_allgather(x)
    else:
        host_value = jax.device_get(x)
    arr = np.asarray(host_value, dtype=dtype)
    return arr.reshape(-1) if flatten else arr


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_with_skip(
    model_state,
    model_graphdef,
    x,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    weights = _build_loss_skip_weights(
        losses, center, mad, z_soft, z_hard, abs_hard, soft_weight, eps, apply_gate, use_log_loss
    )
    denom = jnp.maximum(weights.sum(), 1.0)
    masked_loss = (losses * weights).sum() / denom
    unmasked_loss = losses.mean()

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    skip_stats = {
        'loss_skip_keep_frac': jnp.mean(weights > 0),
        'loss_skip_weight_mean': jnp.mean(weights),
        'loss_skip_masked_loss': masked_loss,
        'loss_skip_unmasked_loss': unmasked_loss,
    }

    return masked_loss, (losses, qkv_stats, skip_stats)


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_z_loss_with_skip(
    model_state,
    model_graphdef,
    x,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    weights = _build_loss_skip_weights(
        losses, center, mad, z_soft, z_hard, abs_hard, soft_weight, eps, apply_gate, use_log_loss
    )
    denom = jnp.maximum(weights.sum(), 1.0)
    masked_ce = (losses * weights).sum() / denom
    unmasked_ce = losses.mean()

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)
    z_loss = (z**2).mean()
    lam = 1e-4

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    skip_stats = {
        'loss_skip_keep_frac': jnp.mean(weights > 0),
        'loss_skip_weight_mean': jnp.mean(weights),
        'loss_skip_masked_loss': masked_ce,
        'loss_skip_unmasked_loss': unmasked_ce,
    }

    return masked_ce + lam * z_loss, (losses, qkv_stats, skip_stats)


def _apply_tanh_soft_cap(losses, soft_cap):
    soft_cap = jnp.maximum(jnp.asarray(soft_cap, dtype=losses.dtype), jnp.asarray(1e-6, dtype=losses.dtype))
    return soft_cap * jnp.tanh(losses / soft_cap)


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_with_soft_cap(
    model_state,
    model_graphdef,
    x,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    capped_losses = _apply_tanh_soft_cap(losses, soft_cap)
    apply_cap_f = jnp.asarray(apply_cap, dtype=capped_losses.dtype)
    capped_losses = apply_cap_f * capped_losses + (1.0 - apply_cap_f) * losses
    capped_loss = capped_losses.mean()
    uncapped_loss = losses.mean()

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    cap_stats = {
        'loss_cap_applied': jnp.asarray(apply_cap, dtype=jnp.float32),
        'loss_cap_capped_loss': capped_loss,
        'loss_cap_uncapped_loss': uncapped_loss,
    }

    return capped_loss, (losses, qkv_stats, cap_stats)


@partial(jax.jit, static_argnames=('model_graphdef', 'collect_qkv_stats'))
def loss_fn_z_loss_with_soft_cap(
    model_state,
    model_graphdef,
    x,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_qkv_stats:
        logits, qkv_dict = model(x, return_qkv=True)
    else:
        logits = model(x, return_qkv=False)
        qkv_dict = None

    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    losses = losses[:, :-1]

    capped_losses = _apply_tanh_soft_cap(losses, soft_cap)
    apply_cap_f = jnp.asarray(apply_cap, dtype=capped_losses.dtype)
    capped_losses = apply_cap_f * capped_losses + (1.0 - apply_cap_f) * losses
    capped_ce = capped_losses.mean()
    uncapped_ce = losses.mean()

    z = jax.nn.logsumexp(logits[:, :-1].astype(jnp.float32), axis=-1)
    z_loss = (z**2).mean()
    lam = 1e-4

    if collect_qkv_stats:
        qkv_dict_detached = jax.tree.map(jax.lax.stop_gradient, qkv_dict)
        qkv_stats = utils.compute_qkv_stats(qkv_dict_detached)
    else:
        qkv_stats = {}

    cap_stats = {
        'loss_cap_applied': jnp.asarray(apply_cap, dtype=jnp.float32),
        'loss_cap_capped_loss': capped_ce,
        'loss_cap_uncapped_loss': uncapped_ce,
    }

    return capped_ce + lam * z_loss, (losses, qkv_stats, cap_stats)


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step(opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats: bool = True):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, grads

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_z_loss(opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats: bool = True):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_z_loss, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, grads

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_centered(opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats: bool = True):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, grads

@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_z_loss_centered(opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats: bool = True):
    # Use has_aux=True to get the raw losses
    (loss, raw_loss), grads = jax.value_and_grad(loss_fn_z_loss, has_aux=True)(
        opt_state.model, model_graphdef, batch, collect_qkv_stats
    )
    
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    
    return opt_state, loss, raw_loss, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_with_skip(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_z_loss_with_skip(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_with_skip_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_z_loss_with_skip_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    center,
    mad,
    z_soft,
    z_hard,
    abs_hard,
    soft_weight,
    eps,
    apply_gate,
    use_log_loss,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_skip, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        center,
        mad,
        z_soft,
        z_hard,
        abs_hard,
        soft_weight,
        eps,
        apply_gate,
        use_log_loss,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_with_soft_cap(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_z_loss_with_soft_cap(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_with_soft_cap_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'collect_qkv_stats'), donate_argnames=('opt_state'))
def train_step_z_loss_with_soft_cap_centered(
    opt_state,
    opt_graphdef,
    model_graphdef,
    batch,
    soft_cap,
    apply_cap,
    collect_qkv_stats: bool = True,
):
    (loss, aux), grads = jax.value_and_grad(loss_fn_z_loss_with_soft_cap, has_aux=True)(
        opt_state.model,
        model_graphdef,
        batch,
        soft_cap,
        apply_cap,
        collect_qkv_stats,
    )
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    model_lib.center_output_embeddings(optimizer.model)
    opt_state = nnx.state(optimizer)
    return opt_state, loss, aux, grads

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logits_by_lm_head(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    return logits.reshape(-1, logits.shape[-1]).astype(jnp.float32).mean(axis=0) # [B * T, V] -> [V]

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_gaps_by_lm_head(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x).astype(jnp.float32)[:, :-1, :] # [B, T, V]
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    gaps = logits - max_logits

    mean_gaps = jnp.mean(gaps, axis=(0, 1)) # [V]
    y = jnp.roll(x, -1, axis=1)[:, :-1]
    target_logits = jnp.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
    target_gaps = target_logits - max_logits.squeeze(-1) # [B, T]

    return mean_gaps, target_gaps

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_mean_and_norm_output_logit(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    logits = model(x) # [B, T, V]
    logits = logits[:, :-1].astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
    return logits.mean(), utils.get_l2_norm(logits), logits.std(), entropy

@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_grad_sum_stats(model_state, model_graphdef, x): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32) # [B, T, V]

    def loss_from_logits(l):
        log_probs = jax.nn.log_softmax(l, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        losses = losses[:, :-1]
        return losses.mean()

    grad_logits = jax.grad(loss_from_logits)(logits) # dL/dz
    grad_sum = grad_logits.sum(axis=-1)[:, :-1] # [B, T-1]

    return {
        'logit_grad_sum_mean_abs': jnp.mean(jnp.abs(grad_sum)),
        'logit_grad_sum_max_abs': jnp.max(jnp.abs(grad_sum)),
        'logit_grad_sum_mean': jnp.mean(grad_sum),
    }


@partial(jax.jit, static_argnames=('model_graphdef'))
def get_logit_grad_scaling_stats(model_state, model_graphdef, x): # [B, T]
    """Compares autodiff CE logits grads vs closed-form grads to detect scaling issues."""
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32) # [B, T, V]

    mask = jnp.ones_like(y, dtype=logits.dtype)
    mask = mask.at[:, -1].set(0)
    denom = jnp.maximum(mask.sum(), 1.0)

    def loss_from_logits(l):
        log_probs = jax.nn.log_softmax(l, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        return (losses * mask).sum() / denom

    grad_auto = jax.grad(loss_from_logits)(logits)

    probs = jax.nn.softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(y, logits.shape[-1], dtype=logits.dtype)
    grad_closed_form = (probs - one_hot) * mask[..., None] / denom

    grad_diff = grad_auto - grad_closed_form
    auto_norm = utils.get_l2_norm(grad_auto)
    diff_norm = utils.get_l2_norm(grad_diff)

    grad_sum = grad_auto.sum(axis=-1)[:, :-1]
    closed_form_sum = grad_closed_form.sum(axis=-1)[:, :-1]

    return {
        'logit_grad_rel_l2_err': diff_norm / jnp.maximum(auto_norm, 1e-30),
        'logit_grad_max_abs_err': jnp.max(jnp.abs(grad_diff[:, :-1])),
        'logit_grad_mean_abs_err': jnp.mean(jnp.abs(grad_diff[:, :-1])),
        'logit_grad_closed_form_sum_mean_abs': jnp.mean(jnp.abs(closed_form_sum)),
        'logit_grad_closed_form_sum_max_abs': jnp.max(jnp.abs(closed_form_sum)),
        'logit_grad_auto_sum_mean_abs': jnp.mean(jnp.abs(grad_sum)),
        'logit_grad_auto_sum_max_abs': jnp.max(jnp.abs(grad_sum)),
    }


def eval_step(c, model_state, model_graphdef, dataset, collect_qkv_stats: bool = True):
    loss_sum = jnp.zeros([], dtype=jnp.float32)
    raw_losses = []
    total_logits = []
    logit_mean_sum = jnp.zeros([], dtype=jnp.float32)
    logit_std_sum = jnp.zeros([], dtype=jnp.float32)
    logit_entropy_sum = jnp.zeros([], dtype=jnp.float32)

    
    for i in range(len(dataset)):
        batch = dataset[i]
        if c.opt.use_z_loss:
            batch_loss, (raw_loss, _) = loss_fn_z_loss(
                model_state, model_graphdef, batch, collect_qkv_stats
            )
        else:
            batch_loss, (raw_loss, _) = loss_fn(
                model_state, model_graphdef, batch, collect_qkv_stats
            )
        loss_sum += batch_loss
        raw_losses.append(raw_loss)
        output_logit_mean, _, output_logit_std, output_logit_entropy = get_mean_and_norm_output_logit(
            model_state, model_graphdef, batch
        )
        logit_mean_sum += output_logit_mean
        logit_std_sum += output_logit_std
        logit_entropy_sum += output_logit_entropy
        if c.diagnostics.save_raw_losses:
            total_logits.append(get_logits_by_lm_head(model_state, model_graphdef, batch).astype(jnp.float32))

    mean_loss = loss_sum / len(dataset)
    mean_output_logit = logit_mean_sum / len(dataset)
    mean_output_logit_std = logit_std_sum / len(dataset)
    mean_output_logit_entropy = logit_entropy_sum / len(dataset)
    
    return (
        mean_loss,
        raw_losses,
        total_logits,
        mean_output_logit,
        mean_output_logit_std,
        mean_output_logit_entropy,
    )


def _build_train_metrics(
    step,
    tokens_per_opt_step,
    train_loss,
    train_med_loss,
    train_lower_90th_mean_loss,
    output_logit_mean,
    output_logit_norm,
    output_logit_std,
    output_logit_entropy,
    opt_state,
    grads,
    lr_schedule,
    qkv_stats,
    logit_grad_stats,
    logit_grad_scaling_stats,
    loss_skip_stats=None,
):
    metrics = {}
    metrics['train_loss'] = train_loss
    metrics['train_med_loss'] = train_med_loss
    metrics['train_lower_90th_mean_loss'] = train_lower_90th_mean_loss
    metrics['train_tokens_seen'] = (step + 1) * tokens_per_opt_step
    metrics['train_output_logit_mean'] = output_logit_mean
    metrics['train_output_logit_norm'] = output_logit_norm
    metrics['train_output_logit_std'] = output_logit_std
    metrics['train_output_logit_entropy'] = output_logit_entropy
    metrics['lr'] = lr_schedule(step)
    metrics.update(utils.get_layer_grad_norms_split(grads))
    metrics.update(utils.get_layer_weight_norms_split(opt_state.model))
    metrics.update(utils.get_layer_moment_norms(opt_state))
    metrics.update(utils.get_layer_second_moment_rms_metrics(grads, opt_state))
    if logit_grad_stats is not None:
        metrics.update(logit_grad_stats)
    if logit_grad_scaling_stats is not None:
        metrics.update(logit_grad_scaling_stats)
    metrics.update(qkv_stats)
    if loss_skip_stats is not None:
        metrics.update(loss_skip_stats)
    return metrics


def _log_metrics_if_primary(metrics, step, pbar):
    if jax.process_index() == 0:
        wandb.log(metrics, step)
        pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')


def _resolve_diagnostics_dir(ckpt_dir):
    if not ckpt_dir:
        return None
    diagnostics_dir = os.path.join(ckpt_dir, 'top_loss_diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    return diagnostics_dir


def _save_eval_diagnostics(
    diagnostics_dir,
    step,
    train_raw_loss,
    eval_raw_loss,
    train_logit_gaps,
    train_target_gaps,
    eval_logits,
):
    if not diagnostics_dir:
        return
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'train_raw_losses_step_{step}.npy', data=train_raw_loss
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{step}.npy', data=eval_raw_loss
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'train_mean_logit_gaps_step_{step}.npy', data=train_logit_gaps
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'train_target_logit_gaps_step_{step}.npy', data=train_target_gaps
    )
    utils.save_to_numpy(
        save_dir=diagnostics_dir, name=f'eval_logits_step_{step}.npy', data=eval_logits
    )


def _save_checkpoint(ckpt_mngr, step, opt_state):
    ckpt_mngr.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(opt_state),
            training_metadata=ocp.args.JsonSave({
                'step': step,
                'next_step': step + 1,
            }),
        ),
        force=True,
    )


def _should_checkpoint(c, step):
    if not c.checkpoint.turn_on:
        return False
    checkpoint_steps = getattr(c.checkpoint, 'checkpoint_steps', None)
    if checkpoint_steps is not None:
        return step in checkpoint_steps
    return step % c.checkpoint.checkpoint_every_steps == 0



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
    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        c.opt.b2,
        eps=c.opt.eps,
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )
    
    clip_by_global_norm = c.opt.clip_by_global_norm
    if clip_by_global_norm:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_by_global_norm), tx)
    
    optimizer = nnx.ModelAndOptimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)


    # set up checkpointing
    start_step = 0
    ckpt_dir = None
    ckpt_mngr = None
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)
    if c.checkpoint.turn_on:
        run_name = c.run_name if c.run_name else 'picodo_run'

        # Use GCP bucket if specified, otherwise use local workdir
        gcp_bucket = getattr(c.checkpoint, 'gcp_bucket', None)
        if gcp_bucket:
            # Format: gs://bucket-name/path
            if not gcp_bucket.startswith('gs://'):
                gcp_bucket = f'gs://{gcp_bucket}'
            ckpt_dir = os.path.join(gcp_bucket, run_name)
            if jax.process_index() == 0:
                print(f'Checkpoints will be saved to GCS bucket: {ckpt_dir}')
        else:
            ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)

        step_prefix = getattr(c.checkpoint, 'step_prefix', None)

        # Base checkpoint manager options
        mngr_options_kwargs = {
            'create': True,
            'preservation_policy': preservation_policy.LatestN(c.checkpoint.max_to_keep)
        }

        if gcp_bucket:
            mngr_options_kwargs['step_name_format'] = _StandardNameFormatHNS(
                step_prefix=step_prefix
            )

        # Add multihost settings if running on multiple hosts
        is_multihost = jax.process_count() > 1
        if is_multihost:
            mngr_options_kwargs['enable_async_checkpointing'] = True
            # Orbax API changed across versions:
            # newer versions expose options under ocp.options.*,
            # older versions use ocp.multiprocessing.*.
            mp_options_cls = None
            if hasattr(ocp, 'options') and hasattr(ocp.options, 'MultiprocessingOptions'):
                mp_options_cls = ocp.options.MultiprocessingOptions
            elif hasattr(ocp, 'multiprocessing') and hasattr(ocp.multiprocessing, 'MultiprocessingOptions'):
                mp_options_cls = ocp.multiprocessing.MultiprocessingOptions

            if mp_options_cls is not None:
                mngr_options_kwargs['multiprocessing_options'] = mp_options_cls(
                    primary_host=0,
                    active_processes=set(range(jax.process_count()))
                )
                # Newer Orbax does not allow create=True when active_processes is set.
                mngr_options_kwargs['create'] = False
            if jax.process_index() == 0:
                print(f'Multihost checkpointing enabled with {jax.process_count()} processes')

        # If create=False (required for some multihost Orbax versions),
        # ensure root directory exists before CheckpointManager construction.
        if not mngr_options_kwargs.get('create', True):
            if jax.process_index() == 0:
                epath.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            if is_multihost:
                multihost_utils.sync_global_devices('ckpt_root_dir_ready')

        mngr_options = ocp.CheckpointManagerOptions(**mngr_options_kwargs)

        ckpt_mngr = ocp.CheckpointManager(
            ckpt_dir,
            options=mngr_options
        )
        
        print(f'Checking for existing checkpoints in: {ckpt_dir}')
        latest_step = c.checkpoint.start_step if c.checkpoint.start_step != None else ckpt_mngr.latest_step()

        if latest_step is not None:
            print(f'Restoring checkpoint from step {latest_step} in {ckpt_dir}...')
            
            restored_data = ckpt_mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_opt_state),
                    training_metadata=ocp.args.JsonRestore(),
                ),
            )
            opt_state = restored_data['state']
            training_metadata = restored_data.get('training_metadata', {})
            meta_step = training_metadata.get('step')
            meta_next_step = training_metadata.get('next_step')
            if meta_next_step is not None:
                start_step = meta_next_step
            elif meta_step is not None:
                if meta_step == latest_step + 1:
                    start_step = meta_step
                elif meta_step == latest_step:
                    start_step = meta_step + 1
                else:
                    print(
                        'Warning: checkpoint metadata step does not match checkpoint id '
                        f'(metadata step={meta_step}, checkpoint id={latest_step}). '
                        'Falling back to resume from checkpoint id + 1.'
                    )
                    start_step = latest_step + 1
            else:
                print(
                    'Warning: checkpoint metadata missing step/next_step; '
                    'falling back to resume from checkpoint id + 1.'
                )
                start_step = latest_step + 1
            print(f'Successfully restored checkpoint. Resuming from step {start_step}.')
        else:
            print('No checkpoint found. Starting from scratch.')

    if c.diagnostics.save_raw_losses:
        diagnostics_dir = _resolve_diagnostics_dir(ckpt_dir)
        if diagnostics_dir:
            utils.save_to_numpy(save_dir=diagnostics_dir, name='val_dataset', data=ds_valid)
            utils.save_to_numpy(save_dir=diagnostics_dir, name='train_dataset', data=ds_train[:c.diagnostics.end_step])

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
    log_metrics_per_step = bool(getattr(c, "log_metrics_per_step", False))
    log_logit_grad_stats = bool(getattr(c, "log_logit_grad_stats", False))
    log_logit_grad_scaling_stats = bool(getattr(c, "log_logit_grad_scaling_stats", False))
    collect_qkv_stats = bool(getattr(c.diagnostics, "collect_qkv_stats", True))
    loss_skip_cfg = getattr(c.opt, "loss_skip", None)
    loss_skip_enabled = bool(getattr(loss_skip_cfg, "enabled", False))
    loss_skip_warmup_steps = int(getattr(loss_skip_cfg, "warmup_steps", 1000))
    loss_skip_window_size = int(getattr(loss_skip_cfg, "window_size", 1000))
    loss_skip_min_history = int(getattr(loss_skip_cfg, "min_history", 256))
    loss_skip_z_soft = float(getattr(loss_skip_cfg, "z_soft", 5.0))
    loss_skip_z_hard = float(getattr(loss_skip_cfg, "z_hard", 8.0))
    loss_skip_soft_weight = float(getattr(loss_skip_cfg, "soft_weight", 1.0))
    loss_skip_eps = float(getattr(loss_skip_cfg, "eps", 1e-6))
    loss_skip_use_log_loss = bool(getattr(loss_skip_cfg, "use_log_loss", False))
    loss_skip_abs_hard_cfg = getattr(loss_skip_cfg, "abs_hard", None)
    loss_skip_abs_hard_fixed = None if loss_skip_abs_hard_cfg is None else float(loss_skip_abs_hard_cfg)
    loss_skip_history = deque(maxlen=loss_skip_window_size)
    loss_skip_cutoff_logged = False
    loss_cap_cfg = getattr(c.opt, "loss_cap", None)
    loss_cap_enabled = bool(getattr(loss_cap_cfg, "enabled", False))
    loss_cap_soft_cap = float(getattr(loss_cap_cfg, "soft_cap", 30.0))
    loss_cap_warmup_steps = int(getattr(loss_cap_cfg, "warmup_steps", 0))
    loss_rewrite_cfg = getattr(c.opt, "loss_rewrite", None)
    loss_rewrite_enabled = bool(getattr(loss_rewrite_cfg, "enabled", False))
    loss_rewrite_replacement_token_id = int(getattr(loss_rewrite_cfg, "replacement_token_id", 11))
    loss_rewrite_warmup_steps = int(getattr(loss_rewrite_cfg, "warmup_steps", 1000))
    loss_rewrite_window_size = int(getattr(loss_rewrite_cfg, "window_size", 1000))
    loss_rewrite_min_history = int(getattr(loss_rewrite_cfg, "min_history", 256))
    loss_rewrite_z_hard = float(getattr(loss_rewrite_cfg, "z_hard", 8.0))
    loss_rewrite_eps = float(getattr(loss_rewrite_cfg, "eps", 1e-6))
    loss_rewrite_use_log_loss = bool(getattr(loss_rewrite_cfg, "use_log_loss", False))
    loss_rewrite_history = deque(maxlen=loss_rewrite_window_size)
    loss_rewrite_cutoff_logged = False
    if loss_skip_enabled and loss_cap_enabled:
        raise ValueError("opt.loss_skip.enabled and opt.loss_cap.enabled cannot both be True.")
    if (loss_skip_enabled and loss_rewrite_enabled) or (loss_cap_enabled and loss_rewrite_enabled):
        raise ValueError(
            "opt.loss_rewrite.enabled cannot be combined with opt.loss_skip.enabled or opt.loss_cap.enabled."
        )
    if loss_skip_enabled and jax.process_index() == 0:
        print(
            "loss-skip enabled: "
            f"warmup_steps={loss_skip_warmup_steps}, window_size={loss_skip_window_size}, "
            f"min_history={loss_skip_min_history}, z_soft={loss_skip_z_soft}, z_hard={loss_skip_z_hard}, "
            f"soft_weight={loss_skip_soft_weight}, use_log_loss={loss_skip_use_log_loss}, abs_hard="
            f"{loss_skip_abs_hard_fixed if loss_skip_abs_hard_fixed is not None else 'inf'}"
        )
    if loss_cap_enabled and jax.process_index() == 0:
        print(f"loss-cap enabled: soft_cap={loss_cap_soft_cap}, warmup_steps={loss_cap_warmup_steps}")
    if loss_rewrite_enabled and jax.process_index() == 0:
        print(
            "loss-rewrite enabled: "
            f"replacement_token_id={loss_rewrite_replacement_token_id}, "
            f"warmup_steps={loss_rewrite_warmup_steps}, window_size={loss_rewrite_window_size}, "
            f"min_history={loss_rewrite_min_history}, z_hard={loss_rewrite_z_hard}, "
            f"use_log_loss={loss_rewrite_use_log_loss}"
        )

    if c.diagnostics.end_step:
        num_opt_steps = c.diagnostics.end_step
    
    mucentering = bool(getattr(c.opt, "mucentering", False))

    pbar = range(start_step, num_opt_steps)
    if jax.process_index() == 0: pbar = tqdm(pbar, initial=start_step, total=num_opt_steps)
    for step in pbar:
        batch = ds_train[step]
        will_log_train_metrics = log_metrics_per_step or (
            (train_loss_num + 1) * tokens_per_opt_step >= c.log_every_tokens
        )
        pre_output_logit_mean = None
        pre_output_logit_norm = None
        pre_output_logit_std = None
        pre_output_logit_entropy = None
        if will_log_train_metrics:
            (
                pre_output_logit_mean,
                pre_output_logit_norm,
                pre_output_logit_std,
                pre_output_logit_entropy,
            ) = get_mean_and_norm_output_logit(opt_state.model, model_graphdef, batch)
        logit_grad_stats = None
        logit_grad_scaling_stats = None
        if log_logit_grad_stats:
            logit_grad_stats = get_logit_grad_sum_stats(opt_state.model, model_graphdef, batch)
        if log_logit_grad_scaling_stats:
            logit_grad_scaling_stats = get_logit_grad_scaling_stats(opt_state.model, model_graphdef, batch)
        loss_skip_stats = None
        gate_apply = False
        gate_center = 0.0
        gate_mad = 1.0
        gate_abs_hard = float("inf")
        rewrite_gate_apply = False
        rewrite_gate_center = 0.0
        rewrite_gate_mad = 1.0
        rewrite_hard_mask_np = None
        rewrite_probe_raw_np = None
        rewrite_probe_loss = None
        if loss_skip_enabled and step >= loss_skip_warmup_steps and len(loss_skip_history) >= loss_skip_min_history:
            hist = np.asarray(loss_skip_history, dtype=np.float32)
            gate_center = float(np.median(hist))
            gate_mad = float(np.median(np.abs(hist - gate_center)))
            if loss_skip_abs_hard_fixed is not None:
                gate_abs_hard = float(loss_skip_abs_hard_fixed)
            else:
                gate_abs_hard = float("inf")
            gate_apply = True
            if (not loss_skip_cutoff_logged) and jax.process_index() == 0:
                scale = max(1.4826 * gate_mad, loss_skip_eps)
                z_hard_cutoff = gate_center + loss_skip_z_hard * scale
                domain = "log1p(loss)" if loss_skip_use_log_loss else "loss"
                print(
                    "loss-skip z_hard cutoff active: "
                    f"domain={domain}, z_hard={loss_skip_z_hard}, "
                    f"cutoff={z_hard_cutoff:.6f}, abs_hard={gate_abs_hard}"
                )
                loss_skip_cutoff_logged = True
        if (
            loss_rewrite_enabled
            and step >= loss_rewrite_warmup_steps
            and len(loss_rewrite_history) >= loss_rewrite_min_history
        ):
            hist = np.asarray(loss_rewrite_history, dtype=np.float32)
            rewrite_gate_center = float(np.median(hist))
            rewrite_gate_mad = float(np.median(np.abs(hist - rewrite_gate_center)))
            rewrite_gate_apply = True
            if (not loss_rewrite_cutoff_logged) and jax.process_index() == 0:
                scale = max(1.4826 * rewrite_gate_mad, loss_rewrite_eps)
                z_hard_cutoff = rewrite_gate_center + loss_rewrite_z_hard * scale
                domain = "log1p(loss)" if loss_rewrite_use_log_loss else "loss"
                print(
                    "loss-rewrite z_hard cutoff active: "
                    f"domain={domain}, z_hard={loss_rewrite_z_hard}, "
                    f"cutoff={z_hard_cutoff:.6f}"
                )
                loss_rewrite_cutoff_logged = True
        # training step
        if loss_skip_enabled:
            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads = train_step_z_loss_with_skip_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads = train_step_z_loss_with_skip(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads = train_step_with_skip_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, skip_stats_device), grads = train_step_with_skip(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        gate_center,
                        gate_mad,
                        loss_skip_z_soft,
                        loss_skip_z_hard,
                        gate_abs_hard,
                        loss_skip_soft_weight,
                        loss_skip_eps,
                        gate_apply,
                        loss_skip_use_log_loss,
                        collect_qkv_stats,
                    )
            skip_stats_host = {k: float(v) for k, v in jax.device_get(skip_stats_device).items()}
            skip_stats_host['loss_skip_gate_applied'] = float(gate_apply)
            skip_stats_host['loss_skip_center'] = float(gate_center)
            skip_stats_host['loss_skip_mad'] = float(gate_mad)
            skip_stats_host['loss_skip_abs_hard'] = float(gate_abs_hard)
            skip_stats_host['loss_skip_use_log_loss'] = float(loss_skip_use_log_loss)
            skip_stats_host['loss_skip_z_hard_cutoff'] = float(
                gate_center + loss_skip_z_hard * max(1.4826 * gate_mad, loss_skip_eps)
            )
            loss_skip_stats = skip_stats_host
        elif loss_cap_enabled:
            cap_apply = step >= loss_cap_warmup_steps
            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads = train_step_z_loss_with_soft_cap_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads = train_step_z_loss_with_soft_cap(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads = train_step_with_soft_cap_centered(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats, cap_stats_device), grads = train_step_with_soft_cap(
                        opt_state,
                        opt_graphdef,
                        model_graphdef,
                        batch,
                        loss_cap_soft_cap,
                        cap_apply,
                        collect_qkv_stats,
                    )
            loss_skip_stats = {k: float(v) for k, v in jax.device_get(cap_stats_device).items()}
        elif loss_rewrite_enabled:
            if c.opt.use_z_loss:
                rewrite_probe_loss, (rewrite_probe_raw_loss, _) = loss_fn_z_loss(
                    opt_state.model, model_graphdef, batch, False
                )
            else:
                rewrite_probe_loss, (rewrite_probe_raw_loss, _) = loss_fn(
                    opt_state.model, model_graphdef, batch, False
                )
            rewrite_probe_raw_np = _to_host_numpy(rewrite_probe_raw_loss, dtype=np.float32)
            rewrite_stats_np = (
                np.log1p(np.maximum(rewrite_probe_raw_np, 0.0))
                if loss_rewrite_use_log_loss
                else rewrite_probe_raw_np
            )
            if rewrite_gate_apply:
                scale = max(1.4826 * rewrite_gate_mad, loss_rewrite_eps)
                rewrite_z = (rewrite_stats_np - rewrite_gate_center) / scale
                rewrite_hard_mask_np = rewrite_z > loss_rewrite_z_hard
            else:
                rewrite_hard_mask_np = np.zeros_like(rewrite_stats_np, dtype=bool)

            rewrite_hard_mask = jnp.asarray(rewrite_hard_mask_np, dtype=jnp.bool_)
            replacement_token = jnp.asarray(loss_rewrite_replacement_token_id, dtype=batch.dtype)
            rewritten_inputs = jnp.where(rewrite_hard_mask, replacement_token, batch[:, :-1])
            rewritten_batch = batch.at[:, :-1].set(rewritten_inputs)

            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_z_loss_centered(
                        opt_state, opt_graphdef, model_graphdef, rewritten_batch, collect_qkv_stats
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_z_loss(
                        opt_state, opt_graphdef, model_graphdef, rewritten_batch, collect_qkv_stats
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_centered(
                        opt_state, opt_graphdef, model_graphdef, rewritten_batch, collect_qkv_stats
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step(
                        opt_state, opt_graphdef, model_graphdef, rewritten_batch, collect_qkv_stats
                    )
            num_replaced = int(np.count_nonzero(rewrite_hard_mask_np))
            total_rewrite_tokens = int(rewrite_hard_mask_np.size)
            loss_skip_stats = {
                'loss_rewrite_gate_applied': float(rewrite_gate_apply),
                'loss_rewrite_center': float(rewrite_gate_center),
                'loss_rewrite_mad': float(rewrite_gate_mad),
                'loss_rewrite_z_hard': float(loss_rewrite_z_hard),
                'loss_rewrite_z_hard_cutoff': float(
                    rewrite_gate_center + loss_rewrite_z_hard * max(1.4826 * rewrite_gate_mad, loss_rewrite_eps)
                ),
                'loss_rewrite_num_replaced': float(num_replaced),
                'loss_rewrite_frac_replaced': float(num_replaced / max(total_rewrite_tokens, 1)),
                'loss_rewrite_probe_loss': float(jax.device_get(rewrite_probe_loss)),
                'loss_rewrite_use_log_loss': float(loss_rewrite_use_log_loss),
            }
        else:
            if c.opt.use_z_loss:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_z_loss_centered(
                        opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_z_loss(
                        opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats
                    )
            else:
                if mucentering:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step_centered(
                        opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats
                    )
                else:
                    opt_state, batch_loss, (train_raw_loss, qkv_stats), grads = train_step(
                        opt_state, opt_graphdef, model_graphdef, batch, collect_qkv_stats
                    )
        if loss_skip_enabled:
            raw_np = _to_host_numpy(train_raw_loss, dtype=np.float32, flatten=True)
            stats_np = np.log1p(np.maximum(raw_np, 0.0)) if loss_skip_use_log_loss else raw_np
            if gate_apply:
                scale = max(1.4826 * gate_mad, loss_skip_eps)
                z = (stats_np - gate_center) / scale
                hard = np.logical_or(z > loss_skip_z_hard, stats_np > gate_abs_hard)
                keep = np.logical_not(hard)
            else:
                keep = np.ones_like(stats_np, dtype=bool)
            if keep.any():
                loss_skip_history.extend(stats_np[keep].tolist())
        if loss_rewrite_enabled and rewrite_probe_raw_np is not None:
            rewrite_stats_np = (
                np.log1p(np.maximum(rewrite_probe_raw_np, 0.0))
                if loss_rewrite_use_log_loss
                else rewrite_probe_raw_np
            ).reshape(-1)
            rewrite_keep = np.logical_not(rewrite_hard_mask_np.reshape(-1))
            if rewrite_gate_apply:
                if rewrite_keep.any():
                    loss_rewrite_history.extend(rewrite_stats_np[rewrite_keep].tolist())
            else:
                loss_rewrite_history.extend(rewrite_stats_np.tolist())
        batch_loss_to_log = (
            jnp.asarray(loss_skip_stats['loss_skip_unmasked_loss'], dtype=jnp.float32)
            if (loss_skip_enabled and loss_skip_stats is not None and 'loss_skip_unmasked_loss' in loss_skip_stats)
            else jnp.asarray(loss_skip_stats['loss_cap_uncapped_loss'], dtype=jnp.float32)
            if (loss_cap_enabled and loss_skip_stats is not None and 'loss_cap_uncapped_loss' in loss_skip_stats)
            else jnp.asarray(loss_skip_stats['loss_rewrite_probe_loss'], dtype=jnp.float32)
            if (loss_rewrite_enabled and loss_skip_stats is not None and 'loss_rewrite_probe_loss' in loss_skip_stats)
            else batch_loss
        )
        # if jax.process_index() == 0:
        #     min_train_loss = float(jax.device_get(jnp.min(train_raw_loss)))
        #     assert min_train_loss >= 0.0, f"negative train loss: {min_train_loss}"
        
        if c.diagnostics.save_raw_losses:
            train_logit_gaps, train_target_gaps = get_logit_gaps_by_lm_head(opt_state.model, model_graphdef, batch)

        # logging
        if log_metrics_per_step:
            metrics = _build_train_metrics(
                step,
                tokens_per_opt_step,
                batch_loss_to_log,
                jnp.median(train_raw_loss),
                utils.compute_lower_90th_percentile_mean(train_raw_loss),
                pre_output_logit_mean,
                pre_output_logit_norm,
                pre_output_logit_std,
                pre_output_logit_entropy,
                opt_state,
                grads,
                lr_schedule,
                qkv_stats,
                logit_grad_stats,
                logit_grad_scaling_stats,
                loss_skip_stats,
            )
            _log_metrics_if_primary(metrics, step, pbar)
        else:
            train_loss_sum += batch_loss_to_log
            train_med_loss_sum += jnp.median(train_raw_loss)
            train_lower_90th_mean_loss_sum += utils.compute_lower_90th_percentile_mean(train_raw_loss)
            train_loss_num += 1
            if train_loss_num * tokens_per_opt_step >= c.log_every_tokens:
                metrics = _build_train_metrics(
                    step,
                    tokens_per_opt_step,
                    train_loss_sum / train_loss_num,
                    train_med_loss_sum / train_loss_num,
                    train_lower_90th_mean_loss_sum / train_loss_num,
                    pre_output_logit_mean,
                    pre_output_logit_norm,
                    pre_output_logit_std,
                    pre_output_logit_entropy,
                    opt_state,
                    grads,
                    lr_schedule,
                    qkv_stats,
                    logit_grad_stats,
                    logit_grad_scaling_stats,
                    loss_skip_stats,
                )
                _log_metrics_if_primary(metrics, step, pbar)
                train_loss_sum, train_med_loss_sum, train_lower_90th_mean_loss_sum, train_loss_num = jnp.zeros([]), jnp.zeros([]), jnp.zeros([]), 0
        
        # eval and checkpointing
        if step % c.eval_every_steps == 0:
            (
                eval_loss,
                eval_raw_loss,
                eval_logits,
                mean_eval_output_logit,
                mean_eval_output_logit_std,
                mean_eval_output_logit_entropy,
            ) = eval_step(c, opt_state.model, model_graphdef, ds_valid, collect_qkv_stats)
            flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
            metrics = {}
            metrics['eval_loss'] = eval_loss
            metrics['eval_output_logit_mean'] = mean_eval_output_logit
            metrics['eval_output_logit_std'] = mean_eval_output_logit_std
            metrics['eval_output_logit_entropy'] = mean_eval_output_logit_entropy
            metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
            metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)
            metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
            if jax.process_index() == 0:
                wandb.log(metrics, step)
            
            # diagnostics
            if c.diagnostics.save_raw_losses:
                diagnostics_dir = _resolve_diagnostics_dir(ckpt_dir)
                _save_eval_diagnostics(
                    diagnostics_dir,
                    step,
                    train_raw_loss,
                    eval_raw_loss,
                    train_logit_gaps,
                    train_target_gaps,
                    eval_logits,
                )

        if _should_checkpoint(c, step):
            _save_checkpoint(ckpt_mngr, step, opt_state)
            # Wait for async checkpoint to complete in multihost setting
            if jax.process_count() > 1:
                ckpt_mngr.wait_until_finished()
    
    if num_opt_steps != len(ds_train):
        print('exiting early')
        wandb.finish()
        ckpt_mngr.close()
        sys.exit(1)

    # eval at end of training
    (
        eval_loss,
        eval_raw_loss,
        eval_logits,
        mean_eval_output_logit,
        mean_eval_output_logit_std,
        mean_eval_output_logit_entropy,
    ) = eval_step(c, opt_state.model, model_graphdef, ds_valid, collect_qkv_stats)
    metrics = {}
    flattened_eval_raw_loss = jnp.concatenate(eval_raw_loss, axis=0)
    metrics['eval_loss'] = eval_loss
    metrics['eval_output_logit_mean'] = mean_eval_output_logit
    metrics['eval_output_logit_std'] = mean_eval_output_logit_std
    metrics['eval_output_logit_entropy'] = mean_eval_output_logit_entropy
    metrics['eval_med_loss'] = jnp.median(flattened_eval_raw_loss)
    metrics['eval_lower_90th_mean_loss'] = utils.compute_lower_90th_percentile_mean(flattened_eval_raw_loss)
    if jax.process_index() == 0:
        wandb.log(metrics)
        wandb.finish()
        if c.diagnostics.save_raw_losses:
            diagnostics_dir = _resolve_diagnostics_dir(ckpt_dir)
            if diagnostics_dir:
                utils.save_to_numpy(save_dir=diagnostics_dir, name=f'eval_raw_losses_step_{num_opt_steps}.npy', data=eval_raw_loss)
            
    # final checkpoint
    if c.checkpoint.turn_on and not c.diagnostics.save_raw_losses:
        final_step = max(num_opt_steps - 1, 0)
        _save_checkpoint(ckpt_mngr, final_step, opt_state)

        ckpt_mngr.wait_until_finished()
        if jax.process_index() == 0:
            print(f'Saved final checkpoint at step {final_step} to {ckpt_mngr.directory}')
        ckpt_mngr.close()
