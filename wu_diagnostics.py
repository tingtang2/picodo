"""Per-step diagnostics for the output embedding (LM head / unembedding) W_U.

Two families of metrics, all cheap enough to run every step:

1. **mu_W tracking** — tests the flat-direction hypothesis.
   mu_W = mean row of W_U.  The loss gradient along mu_W is structurally
   zero (softmax is shift-invariant), so Adam can random-walk mu_W with no
   restoring force.  We track its norm, per-step displacement, directional
   cosine, and an "amplification ratio" that measures whether Adam is
   moving mu_W faster than the average row.

2. **Spectral health of W_U** — tests the spectral-collapse hypothesis.
   We compute the singular values of W_U via the D x D Gram matrix
   (W_U^T W_U), then report the condition number, effective rank, top-5
   singular values, and the alignment between the dominant singular
   direction and the flat (mu_W) direction.

All quantities are computed in fp32 regardless of the model's activation
dtype.  The previous-step snapshot of W_U is held as a JIT-owned fp32
buffer (same donate_argnames-safe pattern as param_tracking.py).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Extracting W_U from the model state pytree.
# ---------------------------------------------------------------------------

def _extract_wu(model_state) -> jnp.ndarray:
    """Walk the model state and return token_embed_out.embedding as fp32.

    Uses the same walker pattern as utils.get_layer_weight_norms_split,
    matching keys by str(k) to handle non-string nnx.State keys safely.
    Returns the raw array (not the nnx.Param wrapper).
    """
    node = model_state
    for key in ("token_embed_out", "embedding"):
        found = False
        if hasattr(node, "items"):
            for k, v in node.items():
                if str(k) == key:
                    node = v
                    found = True
                    break
        if not found and hasattr(node, key):
            node = getattr(node, key)
            found = True
        if not found:
            raise KeyError(f"Cannot find '{key}' in model state")
    # Unwrap nnx.Param if present.
    arr = node.value if hasattr(node, "value") else node
    return arr.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Snapshot helper (donate_argnames-safe, mirrors param_tracking.py).
# ---------------------------------------------------------------------------

@jax.jit
def _snapshot_wu(wu_f32: jnp.ndarray) -> jnp.ndarray:
    """Return a fresh fp32 copy that XLA cannot alias to the input."""
    return jnp.zeros_like(wu_f32) + wu_f32


# ---------------------------------------------------------------------------
# Spectral quantities via the Gram matrix.
# ---------------------------------------------------------------------------

def _compute_spectral_metrics(wu_f32: jnp.ndarray, mu_w: jnp.ndarray) -> Dict[str, float]:
    """Compute condition number, effective rank, top-5 singular values, and
    flat-direction alignment from W_U [V, D] in fp32.

    Uses the D x D Gram matrix  G = W_U^T W_U  whose eigenvalues are the
    squared singular values.  On TPU this is a single 768 x 768 eigh call.
    """
    # Gram matrix [D, D]
    G = wu_f32.T @ wu_f32
    # Single eigh call — eigenvalues in ascending order, eigenvectors in
    # columns. All eigenvalues are non-negative for a PSD matrix.
    eigvals, eigvecs = jnp.linalg.eigh(G)
    # Clamp tiny negatives from numerical noise, then sqrt -> singular values.
    eigvals = jnp.maximum(eigvals, 0.0)
    sigmas = jnp.sqrt(eigvals)
    # Sort descending (eigh returns ascending).
    sigmas = sigmas[::-1]

    sigma_1 = float(sigmas[0])
    sigma_D = float(sigmas[-1])
    condition_number = sigma_1 / (sigma_D + 1e-30)

    # Effective rank = exp(entropy of normalised singular values).
    normed = sigmas / (jnp.sum(sigmas) + 1e-30)
    # Avoid log(0) by clamping.
    log_normed = jnp.log(jnp.maximum(normed, 1e-30))
    effective_rank = float(jnp.exp(-jnp.sum(normed * log_normed)))

    # Top right singular vector = eigenvector for the largest eigenvalue.
    # eigh returns ascending order, so last column is the top eigenvector.
    top_right_sv = eigvecs[:, -1]  # [D]

    # Alignment of the flat direction (mu_W) with the top singular direction.
    # Cosine between unit-mu_W and the top right singular vector.
    # Values near 1 = spectral collapse IS the flat-direction growth.
    mu_w_normed = mu_w / (jnp.linalg.norm(mu_w) + 1e-30)
    flat_alignment = float(jnp.abs(jnp.dot(mu_w_normed, top_right_sv)))

    metrics: Dict[str, float] = {}
    metrics["W_U/sigma_1"] = sigma_1
    metrics["W_U/sigma_D"] = sigma_D
    metrics["W_U/condition_number"] = condition_number
    metrics["W_U/effective_rank"] = effective_rank
    metrics["W_U/flat_alignment"] = flat_alignment
    # Top-5 singular values for spectrum shape.
    for i in range(min(5, len(sigmas))):
        metrics[f"W_U/sigma_{i+1}"] = float(sigmas[i])
    return metrics


# ---------------------------------------------------------------------------
# The tracker.
# ---------------------------------------------------------------------------

class WUDiagnostics:
    """Per-step mu_W + spectral diagnostics for the output embedding W_U.

    Construct once, call `.step(model_state)` after each optimizer update.
    Returns a flat dict of metrics ready for wandb.log.
    """

    def __init__(self) -> None:
        self._prev_wu: Optional[jnp.ndarray] = None  # fp32 [V, D]
        self._prev_mu: Optional[jnp.ndarray] = None   # fp32 [D]

    def step(self, model_state) -> Dict[str, float]:
        """Compute all W_U diagnostics for this step.

        On the first call, returns mu_W/norm and spectral metrics only (no
        delta/cosine/amplification since there is no previous snapshot yet).
        """
        wu_f32 = _extract_wu(model_state)  # [V, D] fp32
        # Detach from train_step's donate_argnames.
        wu_f32 = _snapshot_wu(wu_f32)
        jax.block_until_ready(wu_f32)

        mu_w = jnp.mean(wu_f32, axis=0)    # [D]
        mu_w_norm = float(jnp.linalg.norm(mu_w))

        metrics: Dict[str, float] = {}

        # ---- mu_W metrics (always) ----
        metrics["mu_W/norm"] = mu_w_norm

        if self._prev_wu is not None and self._prev_mu is not None:
            # Per-step displacement of mu_W.
            delta_mu = mu_w - self._prev_mu
            delta_mu_norm = float(jnp.linalg.norm(delta_mu))
            metrics["mu_W/step_norm"] = delta_mu_norm

            # Cosine between consecutive mu_W vectors.
            prev_mu_norm = float(jnp.linalg.norm(self._prev_mu))
            cos_mu = float(
                jnp.dot(mu_w, self._prev_mu) / (mu_w_norm * prev_mu_norm + 1e-30)
            )
            metrics["mu_W/cosine"] = cos_mu

            # Amplification ratio: is mu_W moving faster than the average row?
            delta_wu = wu_f32 - self._prev_wu        # [V, D]
            per_row_norms = jnp.linalg.norm(delta_wu, axis=1)  # [V]
            mean_row_step = float(jnp.mean(per_row_norms))
            metrics["mu_W/amplification_ratio"] = delta_mu_norm / (mean_row_step + 1e-30)

        # ---- Spectral metrics (always) ----
        spectral = _compute_spectral_metrics(wu_f32, mu_w)
        metrics.update(spectral)

        # ---- Update snapshots ----
        self._prev_wu = wu_f32
        self._prev_mu = mu_w

        return metrics
