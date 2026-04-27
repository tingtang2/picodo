"""Per-step dispersion diagnostics for the unembedding matrix W_U.

Tests the hypothesis that Adam's second moment is uniform within a row
(across feature dim D) but heavy-tailed across rows (across vocab dim V).
If true, vocab-dim operations (column centering / mu centering) are
ill-conditioned because their averages are dominated by outlier rows,
while feature-dim operations (row centering) are well-conditioned because
they average within-row over similar-magnitude quantities.

Three layers of metrics, all under the W&B prefix `unembed_dispersion/`:

  Layer 1 — second-moment dispersion (the core hypothesis):
    - within-row CoV of sqrt(v_hat) summarized over V
    - across-row CoV of sqrt(v_hat) summarized over D
    - the ratio of across-row median to within-row median (key prediction)

  Layer 2 — what each centering operation would subtract:
    - per-row mean of update (rowcenter target)
    - per-column mean of update (mucenter / colcenter target), L2 + median
    - mean/median ratio of |per-col-mean| (outlier indicator: >> 1 means
      a few rows dominate the average)
    - fraction of total update L2 contributed by the top-1% of rows

  Layer 3 — parameter-level dispersion of W_U:
    - per-row variance / per-row L2 norm (do some rows grow huge?)
    - per-column variance (do features develop heavy tails?)

Cost is one O(V*D) reduction per call. Module is self-contained; walks
the optimizer state via the same pattern as utils._find_moment_state.
"""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Union
import functools

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Tree walkers (no nnx dependency).
# ---------------------------------------------------------------------------

def _find_moment_state(opt_state: Any) -> Optional[Any]:
    if isinstance(opt_state, Mapping):
        if "mu" in opt_state and "nu" in opt_state:
            return opt_state
        for v in opt_state.values():
            f = _find_moment_state(v)
            if f is not None:
                return f
        return None
    if hasattr(opt_state, "mu") and hasattr(opt_state, "nu"):
        return opt_state
    if isinstance(opt_state, (list, tuple)):
        for v in opt_state:
            f = _find_moment_state(v)
            if f is not None:
                return f
    return None


def _get_field(state: Any, key: str) -> Any:
    if isinstance(state, Mapping):
        return state[key]
    return getattr(state, key)


def _extract_embedding(tree: Any, layer_name: str) -> Optional[jnp.ndarray]:
    node = tree
    for key in (layer_name, "embedding"):
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
            return None
    arr = node.value if hasattr(node, "value") else node
    return arr


# ---------------------------------------------------------------------------
# Core jit-compiled metric computation. V and D are static via shape.
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("top_k_count",))
def _compute_dispersion_metrics(W_U, mu, nu, eps, bias_correction, top_k_count):
    W = W_U.astype(jnp.float32)
    M = mu.astype(jnp.float32)
    N = nu.astype(jnp.float32)
    nu_hat = N / bias_correction
    sqrt_nu_hat = jnp.sqrt(jnp.maximum(nu_hat, 0.0))  # [V, D]

    # ---- Layer 1: nu dispersion ----
    eps_f = jnp.float32(1e-30)

    # Within-row CoV: for each token v, std/mean across features d.
    within_row_mean = jnp.mean(sqrt_nu_hat, axis=1)          # [V]
    within_row_std = jnp.std(sqrt_nu_hat, axis=1)            # [V]
    within_row_cov = within_row_std / (within_row_mean + eps_f)  # [V]

    # Across-row CoV: for each feature d, std/mean across vocab v.
    across_row_mean = jnp.mean(sqrt_nu_hat, axis=0)          # [D]
    across_row_std = jnp.std(sqrt_nu_hat, axis=0)            # [D]
    across_row_cov = across_row_std / (across_row_mean + eps_f)  # [D]

    within_row_cov_med = jnp.median(within_row_cov)
    within_row_cov_p99 = jnp.quantile(within_row_cov, 0.99)
    within_row_cov_max = jnp.max(within_row_cov)
    across_row_cov_med = jnp.median(across_row_cov)
    across_row_cov_p99 = jnp.quantile(across_row_cov, 0.99)
    across_row_cov_max = jnp.max(across_row_cov)
    cov_ratio = across_row_cov_med / (within_row_cov_med + eps_f)

    # ---- Layer 2: what each centering would subtract ----
    update = M / (sqrt_nu_hat + eps)  # [V, D]

    per_row_mean = jnp.mean(update, axis=1)  # [V] — what rowcenter subtracts
    per_col_mean = jnp.mean(update, axis=0)  # [D] — what mucenter / colcenter subtracts

    abs_prm = jnp.abs(per_row_mean)
    abs_pcm = jnp.abs(per_col_mean)

    prm_median = jnp.median(abs_prm)
    prm_max = jnp.max(abs_prm)
    pcm_median = jnp.median(abs_pcm)
    pcm_max = jnp.max(abs_pcm)
    pcm_l2 = jnp.linalg.norm(per_col_mean)

    # mean/median ratio: > 1 indicates outlier-driven mean.
    pcm_meanmedian_ratio = jnp.mean(abs_pcm) / (pcm_median + eps_f)
    prm_meanmedian_ratio = jnp.mean(abs_prm) / (prm_median + eps_f)

    # Outlier attribution: fraction of total update^2 contributed by top
    # k = top_k_count rows (k = max(V//100, 1) typically, set by caller).
    per_row_norm_sq = jnp.sum(update * update, axis=1)  # [V]
    sorted_desc = jnp.sort(per_row_norm_sq)[::-1]
    top_k_sum = jnp.sum(sorted_desc[:top_k_count])
    total_sum = jnp.sum(per_row_norm_sq)
    top_1pct_fraction = top_k_sum / (total_sum + eps_f)

    # ---- Layer 3: W_U parameter dispersion ----
    W_per_row_var = jnp.var(W, axis=1)        # [V]
    W_per_col_var = jnp.var(W, axis=0)        # [D]
    W_per_row_norm = jnp.linalg.norm(W, axis=1)  # [V]

    return {
        # Layer 1
        "nu/within_row_cov_median": within_row_cov_med,
        "nu/within_row_cov_p99":    within_row_cov_p99,
        "nu/within_row_cov_max":    within_row_cov_max,
        "nu/across_row_cov_median": across_row_cov_med,
        "nu/across_row_cov_p99":    across_row_cov_p99,
        "nu/across_row_cov_max":    across_row_cov_max,
        "nu/cov_ratio_across_to_within_median": cov_ratio,
        # Layer 2
        "update/per_row_mean_abs_median": prm_median,
        "update/per_row_mean_abs_max":    prm_max,
        "update/per_col_mean_l2":         pcm_l2,
        "update/per_col_mean_abs_median": pcm_median,
        "update/per_col_mean_abs_max":    pcm_max,
        "update/per_col_mean_meanmedian_ratio": pcm_meanmedian_ratio,
        "update/per_row_mean_meanmedian_ratio": prm_meanmedian_ratio,
        "update/top_1pct_row_fraction_of_total": top_1pct_fraction,
        # Layer 3
        "W_U/per_row_var_median": jnp.median(W_per_row_var),
        "W_U/per_row_var_p99":    jnp.quantile(W_per_row_var, 0.99),
        "W_U/per_row_var_max":    jnp.max(W_per_row_var),
        "W_U/per_col_var_median": jnp.median(W_per_col_var),
        "W_U/per_col_var_p99":    jnp.quantile(W_per_col_var, 0.99),
        "W_U/per_col_var_max":    jnp.max(W_per_col_var),
        "W_U/per_row_norm_median": jnp.median(W_per_row_norm),
        "W_U/per_row_norm_p99":    jnp.quantile(W_per_row_norm, 0.99),
        "W_U/per_row_norm_max":    jnp.max(W_per_row_norm),
    }


# ---------------------------------------------------------------------------
# Tracker.
# ---------------------------------------------------------------------------

class UnembedDispersionDiagnostic:
    """Construct once with `b2` and `eps` from the optimizer config.
    Call `.step(opt_state, step)` per logged step.
    """

    def __init__(
        self,
        b2: Union[float, Callable[[Any], Any]],
        eps: float,
        layer_name: str = "token_embed_out",
    ) -> None:
        self.b2 = b2
        self.eps = float(eps)
        self.layer_name = layer_name

    def _b2_at(self, step_t: int) -> float:
        if callable(self.b2):
            return float(self.b2(step_t))
        return float(self.b2)

    def step(self, opt_state: Any, step_t: int) -> Dict[str, float]:
        adam_state = _find_moment_state(opt_state)
        if adam_state is None:
            return {}
        mu_tree = _get_field(adam_state, "mu")
        nu_tree = _get_field(adam_state, "nu")

        # opt_state under nnx wraps both model params and optax state.
        # The model params are reachable via opt_state.model (an nnx.State).
        if hasattr(opt_state, "model"):
            model_state = opt_state.model
        else:
            # Fallback: treat opt_state itself as the model tree.
            model_state = opt_state

        W_U = _extract_embedding(model_state, self.layer_name)
        mu = _extract_embedding(mu_tree, self.layer_name)
        nu = _extract_embedding(nu_tree, self.layer_name)
        if W_U is None or mu is None or nu is None:
            return {}

        # Bias correction (approximation under b2 annealing; CoV metrics
        # are invariant under uniform scaling so the approximation only
        # affects absolute magnitudes in Layer 2 / Layer 3).
        t = max(int(step_t) + 1, 1)
        b2_now = self._b2_at(step_t)
        bias_correction = max(1.0 - b2_now ** t, 1e-30)

        V = W_U.shape[0]
        top_k_count = max(V // 100, 1)

        m = _compute_dispersion_metrics(
            W_U, mu, nu,
            jnp.float32(self.eps),
            jnp.float32(bias_correction),
            top_k_count,
        )

        return {f"unembed_dispersion/{k}": float(v) for k, v in m.items()}
