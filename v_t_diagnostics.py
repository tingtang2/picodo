"""Tier-1 evidence for the rare-token / v_t hypothesis.

For the output embedding (W_U = token_embed_out.embedding) and the input
embedding (W_E = token_embed_in.embedding) we log, per logged step:

1. Per-row sqrt(v_hat) quantiles across the V-token vocabulary:
   min, p1, p10, p50, p90, p99, max. On a log-y axis these show the
   dispersion of the second moment across tokens. Hypothesis predicts
   several orders of magnitude spread, with p1/p10 near machine zero.

2. Fraction of cells (and rows) with sqrt(v_hat) below the Adam epsilon.
   Directly measures what fraction of the matrix is in the eps-dominated
   ("unprotected") regime.

3. Frequency-bucketed mean sqrt(v_hat): tokens sorted by training-corpus
   frequency and sliced into `num_buckets` deciles (default 10). Per-step
   mean for each bucket. If the hypothesis holds, buckets separate on a
   log axis with the rarest at the bottom, near or below eps.

4. Spearman correlation between log(freq) and log(row_rms). One scalar
   that summarises whether row second moment is monotonically ranked by
   token frequency. Hypothesis predicts stably positive (> 0.7) past
   warmup.

All quantities are computed in fp32. Values are bias-corrected:
v_hat = v / (1 - b2^t). Under b2 annealing, optax's exact correction
uses per-step b2 accumulation; we approximate with the current step's
b2, which is within ~1% past warmup and preserves time-series shape.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Union
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Finding ScaleByAdamState and extracting nu for a specific embedding matrix.
# ---------------------------------------------------------------------------

def _find_scale_by_adam_state(tree: Any) -> Optional[Any]:
    """Recursively locate the first node that carries both 'mu' and 'nu'
    (i.e. the optax ScaleByAdamState). Handles Mapping, NamedTuple, and
    plain tuple/list containers. Returns None if not found."""
    if isinstance(tree, Mapping):
        if ('mu' in tree) and ('nu' in tree):
            return tree
        for v in tree.values():
            found = _find_scale_by_adam_state(v)
            if found is not None:
                return found
        return None
    if hasattr(tree, 'mu') and hasattr(tree, 'nu'):
        return tree
    if isinstance(tree, (list, tuple)):
        for v in tree:
            found = _find_scale_by_adam_state(v)
            if found is not None:
                return found
    return None


def _get_nu(adam_state: Any) -> Any:
    if hasattr(adam_state, 'nu'):
        return adam_state.nu
    return adam_state['nu']


def _extract_embedding_nu(nu_tree: Any, layer_name: str) -> Optional[jnp.ndarray]:
    """Walk the nu pytree (which mirrors model-param structure) to reach
    <layer_name>.embedding. Returns the fp32 array, or None if absent."""
    node = nu_tree
    for key in (layer_name, 'embedding'):
        found = False
        if hasattr(node, 'items'):
            for k, v in node.items():
                if str(k) == key:
                    node = v
                    found = True
                    break
        if not found and hasattr(node, key):
            node = getattr(node, key)
            found = True
        if not found:
            return None
    arr = node.value if hasattr(node, 'value') else node
    return arr.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Core per-matrix computation.
# ---------------------------------------------------------------------------

def _make_metrics_fn(num_buckets: int):
    """Build the jit-compiled core. num_buckets is baked in so
    segment_sum's num_segments is static from jit's perspective."""

    @jax.jit
    def _fn(nu, freq, bucket_ids, eps, bias_correction):
        # v_hat per cell, sqrt-taken. Clamp negatives from numerical noise.
        v_hat = nu / bias_correction
        sqrt_v_hat = jnp.sqrt(jnp.maximum(v_hat, 0.0))  # [V, D]

        # Cell-level fraction below eps (most sensitive measurement).
        frac_below_eps_cell = jnp.mean(sqrt_v_hat < eps)

        # Per-row reduction: mean sqrt(v_hat) over the D dim.
        row_rms = jnp.mean(sqrt_v_hat, axis=-1)  # [V]
        frac_below_eps_row = jnp.mean(row_rms < eps)

        # Quantiles across the vocabulary axis.
        q_levels = jnp.array([0.01, 0.10, 0.50, 0.90, 0.99], dtype=jnp.float32)
        quantiles = jnp.quantile(row_rms, q_levels)
        row_rms_min = jnp.min(row_rms)
        row_rms_max = jnp.max(row_rms)

        # Bucket means via segment_sum.
        ones = jnp.ones_like(row_rms)
        bucket_sums = jax.ops.segment_sum(row_rms, bucket_ids, num_segments=num_buckets)
        bucket_counts = jax.ops.segment_sum(ones, bucket_ids, num_segments=num_buckets)
        bucket_means = bucket_sums / jnp.maximum(bucket_counts, 1.0)  # [num_buckets]

        # Spearman rho(log f, log row_rms). Rank via argsort-of-argsort.
        log_floor = jnp.float32(1e-30)
        log_freq = jnp.log(freq + log_floor)
        log_row = jnp.log(row_rms + log_floor)
        rank_freq = jnp.argsort(jnp.argsort(log_freq)).astype(jnp.float32)
        rank_row = jnp.argsort(jnp.argsort(log_row)).astype(jnp.float32)
        rf = rank_freq - jnp.mean(rank_freq)
        rr = rank_row - jnp.mean(rank_row)
        denom = jnp.sqrt(jnp.sum(rf * rf) * jnp.sum(rr * rr)) + jnp.float32(1e-30)
        spearman = jnp.sum(rf * rr) / denom

        return {
            'row_rms_min': row_rms_min,
            'row_rms_p1': quantiles[0],
            'row_rms_p10': quantiles[1],
            'row_rms_p50': quantiles[2],
            'row_rms_p90': quantiles[3],
            'row_rms_p99': quantiles[4],
            'row_rms_max': row_rms_max,
            'frac_below_eps_cell': frac_below_eps_cell,
            'frac_below_eps_row': frac_below_eps_row,
            'spearman_logf_logv': spearman,
            'bucket_means': bucket_means,
        }

    return _fn


# ---------------------------------------------------------------------------
# Tracker.
# ---------------------------------------------------------------------------

class VtDiagnostics:
    """Construct once, call .step(opt_state, step, eps) per logged step.

    Args:
        freq_path: path to .npy from compute_token_frequencies.py, shape
            [V_tokenizer], normalised so sum == 1.
        vocab_size: the model's (possibly rounded-up) vocab size c.model.V.
        b2: either a scalar (Adam default b2) or a callable(step) returning
            a scalar (b2 annealing schedule). Used for bias correction.
        num_buckets: number of frequency deciles (default 10).
        layers: list of (layer_name, wandb_prefix) pairs. Default tracks
            both the unembedding (W_U) and the input embedding (W_E).
    """

    def __init__(
        self,
        freq_path: str,
        vocab_size: int,
        b2: Union[float, Callable[[Any], Any]],
        num_buckets: int = 10,
        layers: Optional[list] = None,
    ) -> None:
        freq_path = os.path.expanduser(str(freq_path))
        freq_raw = np.load(freq_path).astype(np.float32)  # [V_tokenizer]
        if len(freq_raw) < vocab_size:
            freq_raw = np.pad(freq_raw, (0, vocab_size - len(freq_raw)))
        freq_raw = freq_raw[:vocab_size]

        # Bucket 0 = rarest, bucket num_buckets - 1 = most common. Tokens
        # with freq == 0 (padding) land in bucket 0 deterministically.
        order = np.argsort(freq_raw, kind='stable')  # ascending
        ranks = np.argsort(order, kind='stable')
        bucket_ids = (ranks * num_buckets // vocab_size).astype(np.int32)
        bucket_ids = np.clip(bucket_ids, 0, num_buckets - 1)

        self.freq = jnp.asarray(freq_raw, dtype=jnp.float32)
        self.bucket_ids = jnp.asarray(bucket_ids, dtype=jnp.int32)
        self.num_buckets = num_buckets
        self.b2 = b2
        self._fn = _make_metrics_fn(num_buckets)
        self.layers = layers or [
            ('token_embed_out', 'W_U'),
            ('token_embed_in', 'W_E'),
        ]

    def _current_b2(self, step_t: int) -> float:
        if callable(self.b2):
            return float(self.b2(step_t))
        return float(self.b2)

    def step(self, opt_state: Any, step_t: int, eps: float) -> Dict[str, float]:
        adam_state = _find_scale_by_adam_state(opt_state)
        if adam_state is None:
            return {}
        nu_tree = _get_nu(adam_state)

        # Bias correction uses optax's post-increment count: after step_t
        # training steps, optax's internal count is step_t + 1.
        t = max(int(step_t) + 1, 1)
        b2_now = self._current_b2(step_t)
        bias_correction = max(1.0 - b2_now ** t, 1e-30)

        eps_arr = jnp.asarray(eps, dtype=jnp.float32)
        bc_arr = jnp.asarray(bias_correction, dtype=jnp.float32)

        out: Dict[str, float] = {}
        for layer_name, prefix in self.layers:
            nu = _extract_embedding_nu(nu_tree, layer_name)
            if nu is None:
                continue
            m = self._fn(nu, self.freq, self.bucket_ids, eps_arr, bc_arr)
            for k, v in m.items():
                if k == 'bucket_means':
                    for i in range(self.num_buckets):
                        out[f'v_t/{prefix}/bucket_{i}_mean'] = float(v[i])
                else:
                    out[f'v_t/{prefix}/{k}'] = float(v)
        return out
