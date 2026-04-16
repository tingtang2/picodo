"""Per-layer attention heatmap diagnostics.

Runs a separate inference pass (no gradients) with return_qkv=True,
applies QK-norm + RoPE to the raw q, k, then computes explicit
softmax(QK^T / sqrt(H)) attention weights. Averages over batch and
heads, yielding one [T, T] heatmap per layer, logged to W&B as images.

Flash attention (used during training) never materialises the T x T
matrix. This diagnostic bypasses it by computing attention weights
from the same q, k that flash attention would use.
"""

from __future__ import annotations
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from rope import apply_rope

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@partial(jax.jit, static_argnames=('model_graphdef',))
def _forward_qkv(model_state, model_graphdef, batch):
    """Run the full forward pass with return_qkv=True. Returns the raw
    (pre-norm, pre-RoPE) qkv dict: {layer_idx: (q, k, v)}."""
    model = nnx.merge(model_graphdef, model_state)
    _, qkv_dict = model(batch, return_qkv=True)
    return qkv_dict


@partial(jax.jit, static_argnames=('has_qk_norm',))
def _layer_attn_map(q_raw, k_raw, has_qk_norm):
    """From raw (pre-norm, pre-RoPE) q and k for one layer, compute
    the causal attention weight matrix averaged over batch and heads.

    Returns [T, T] in fp32.
    """
    T = q_raw.shape[1]

    # QK-norm (RMSNorm without learnable scale, eps=1e-6).
    if has_qk_norm:
        q = q_raw.astype(jnp.float32)
        k = k_raw.astype(jnp.float32)
        q = q * jax.lax.rsqrt(jnp.mean(q ** 2, axis=-1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt(jnp.mean(k ** 2, axis=-1, keepdims=True) + 1e-6)
        q = q.astype(q_raw.dtype)
        k = k.astype(k_raw.dtype)
    else:
        q, k = q_raw, k_raw

    # RoPE.
    position = jnp.arange(T)
    q = apply_rope(q, position[None])
    k = apply_rope(k, position[None])

    # Explicit attention weights in fp32.
    scale = jnp.sqrt(jnp.float32(q.shape[-1]))
    # q, k: [B, T, N, H]  ->  logits: [B, N, T, T]
    logits = jnp.einsum(
        'BTNH,BSNH->BNTS',
        q.astype(jnp.float32),
        k.astype(jnp.float32),
    ) / scale
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    logits = jnp.where(causal[None, None], logits, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1)  # [B, N, T, T]

    return weights.mean(axis=0)  # [N, T, T]  (averaged over batch, per head)


def log_attention_heatmaps(
    model_state,
    model_graphdef,
    batch,
    step: int,
    qk_norm_flags: List[bool],
):
    """Compute per-layer attention heatmaps and log to W&B.

    Args:
        model_state: nnx model state (opt_state.model).
        model_graphdef: static graph definition.
        batch: input token IDs [B, T].
        step: current training step.
        qk_norm_flags: per-layer boolean, True if that layer uses QK-norm.
    """
    import wandb

    qkv_dict = _forward_qkv(model_state, model_graphdef, batch)

    images = {}
    num_layers = len(qk_norm_flags)
    for i in range(num_layers):
        q_raw, k_raw, _ = qkv_dict[i]
        attn_maps = _layer_attn_map(q_raw, k_raw, qk_norm_flags[i])  # [N, T, T]
        attn_np = np.asarray(jax.device_get(attn_maps))
        num_heads = attn_np.shape[0]

        for h in range(num_heads):
            fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
            im = ax.imshow(attn_np[h], aspect='auto', cmap='viridis',
                           interpolation='nearest')
            ax.set_xlabel('Key position')
            ax.set_ylabel('Query position')
            ax.set_title(f'Layer {i} Head {h} (avg over batch)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            images[f'attn_heatmap/layer_{i}_head_{h}'] = wandb.Image(fig)
            plt.close(fig)

    if jax.process_index() == 0:
        wandb.log(images, step)
