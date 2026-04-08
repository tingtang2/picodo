"""Per-step per-tensor cosine similarity between consecutive parameter values.

For every parameter tensor in the model we log

    cos(W_t, W_{t-1}) = <W_t, W_{t-1}> / (||W_t|| * ||W_{t-1}||)

Both vectors are upcast to fp32 before the inner product. The model lives in
bf16, and bf16 dot products of nearly-identical large vectors lose essentially
all precision -- the resulting cosine would be measuring round-off rather than
actual parameter movement.

The tracker holds a single fp32 snapshot of the previous step's parameters on
device (memory cost = one extra fp32 copy of the model parameters). On the
first call no cosines are returned (there is no previous snapshot yet), and the
snapshot is initialized.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


def _per_leaf_cosine(prev: jnp.ndarray, cur: jnp.ndarray) -> jnp.ndarray:
    p = prev.astype(jnp.float32)
    c = cur.astype(jnp.float32)
    num = jnp.vdot(p, c)
    den = jnp.linalg.norm(p) * jnp.linalg.norm(c) + jnp.float32(1e-30)
    return num / den


@jax.jit
def _compute_cosines_pytree(prev_state, cur_state):
    """Per-leaf cosine, pytree-in pytree-out. JIT'd; the model's pytree
    structure is stable across training steps so this compiles once."""
    return jax.tree.map(_per_leaf_cosine, prev_state, cur_state)


def _flatten_cosine_tree(tree, prefix: str = "cosine") -> Dict[str, float]:
    """Walk an nnx.State / nested-dict of scalar cosines into a flat
    {'<prefix>/<dotted.path>': float} dict suitable for wandb.log.

    Mirrors the walker style used in utils.get_layer_weight_norms_split so
    parameter paths are reported consistently across the codebase.
    """
    out: Dict[str, float] = {}

    def visit(path: str, node: Any) -> None:
        if hasattr(node, "value"):  # nnx.Param-like wrapper
            out[f"{prefix}/{path}"] = float(node.value)
            return
        if hasattr(node, "items"):  # nnx.State / dict
            for key, child in node.items():
                key_s = str(key)
                new_path = key_s if path == "" else f"{path}.{key_s}"
                visit(new_path, child)
            return
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            out[f"{prefix}/{path}"] = float(node)
            return

    visit("", tree)
    return out


class CosineTracker:
    """Tracks the previous training step's parameter snapshot and computes
    per-tensor cosine similarities against the current parameters."""

    def __init__(self) -> None:
        self._prev: Optional[Any] = None

    def step(self, cur_state) -> Dict[str, float]:
        """Compute per-tensor cosines against the previous snapshot, then
        replace the snapshot with the current parameters.

        Returns an empty dict on the very first call (no previous snapshot).
        """
        cur_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), cur_state)
        if self._prev is None:
            self._prev = cur_f32
            return {}
        cos_tree = _compute_cosines_pytree(self._prev, cur_f32)
        metrics = _flatten_cosine_tree(cos_tree)
        self._prev = cur_f32
        return metrics
