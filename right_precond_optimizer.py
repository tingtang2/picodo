"""Right-Preconditioned Adam (RPA) for the unembedding matrix.

Symmetry-preserving Adam variant for a 2D weight g of shape [V, D].

State per (masked-in) leaf:
    m_t = b1 * m_{t-1} + (1 - b1) * g_t                        # [V, D]
    R_t = b2 * R_{t-1} + (1 - b2) * g_t.T @ g_t                # [D, D]

Update:
    m_hat = m_t / (1 - b1^t)
    R_hat = R_t / (1 - b2^t)
    u_t   = m_hat @ (R_hat + eps * I_D)^(-1/2)                 # [V, D]
    W    -= lr_t * u_t

The two gradient conservation laws on W_U are preserved structurally:

  softmax (sum over v):
      sum_v u[v,d] = sum_{d'} (sum_v m_hat[v,d']) * M[d',d] = 0
      because m inherits sum_v m[v,:] = 0 from the gradient (linear
      momentum) and the right-preconditioner only acts on the D axis.

  LayerNorm gauge (sum over d):
      Under LN, sum_d g[v,d] = 0 for every row v. Hence
          (g.T @ g) @ 1_D = sum_v g[v,:] * (sum_d g[v,d]) = 0,
      so R @ 1_D = 0, and 1_D is an eigenvector of (R + eps*I)^{-1/2}
      with eigenvalue eps^{-1/2}. Therefore
          sum_d u[v,d] = eps^{-1/2} * (sum_d m_hat[v,d]) = 0.

Wrap with optax.masked to confine to W_U: every masked-in leaf must be
rank-2.

Cost: D x D state per leaf, one symmetric eigendecomposition of an
(D x D) matrix per step. For D=1024 this is ~10^9 flops/step, small
relative to the model forward+backward at gpt2m scale.
"""

from __future__ import annotations
from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
import optax


class RightPrecondAdamState(NamedTuple):
    count: jax.Array
    m: Any  # tree-of-[V,D]-arrays (or MaskedNode at masked-out leaves)
    R: Any  # tree-of-[D,D]-arrays (or MaskedNode at masked-out leaves)


def _is_masked(x: Any) -> bool:
    return isinstance(x, optax.MaskedNode)


def _arr(p: Any):
    return getattr(p, "value", p)


def right_precond_adam(
    learning_rate: Union[float, Callable[[Any], Any]],
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Right-Preconditioned Adam.

    Args:
        learning_rate: scalar or schedule(step) -> float.
        b1: first-moment decay (matches Adam's b1).
        b2: second-moment (R) decay (matches Adam's b2).
        eps: ridge added to R before its inverse square root. Note this
            is NOT dimensionally the same as Adam's elementwise eps;
            here eps regularizes the D x D matrix, with eps^{-1/2} setting
            the maximum gain along the LN-gauge direction.
    """
    lr_fn = learning_rate if callable(learning_rate) else (lambda _: learning_rate)
    eps_f32 = jnp.float32(eps)
    b1_f32 = jnp.float32(b1)
    b2_f32 = jnp.float32(b2)

    def init_fn(params):
        def init_m(p):
            if _is_masked(p):
                return p
            return jnp.zeros_like(_arr(p))

        def init_R(p):
            if _is_masked(p):
                return p
            v = _arr(p)
            if v.ndim != 2:
                raise ValueError(
                    "right_precond_adam expects rank-2 leaves under the mask, "
                    f"got shape={v.shape}"
                )
            D = v.shape[1]
            return jnp.zeros((D, D), dtype=jnp.float32)

        return RightPrecondAdamState(
            count=jnp.zeros([], jnp.int32),
            m=jax.tree.map(init_m, params, is_leaf=_is_masked),
            R=jax.tree.map(init_R, params, is_leaf=_is_masked),
        )

    def update_fn(updates, state, params=None):
        del params
        new_count = state.count + 1
        t = new_count.astype(jnp.float32)
        bc1 = 1.0 - jnp.power(b1_f32, t)
        bc2 = 1.0 - jnp.power(b2_f32, t)
        lr = jnp.float32(lr_fn(state.count))

        def step_one(g, m_prev, R_prev):
            if _is_masked(g):
                return g, m_prev, R_prev
            g_f = g.astype(jnp.float32)
            m_new = b1_f32 * m_prev + (1.0 - b1_f32) * g_f
            R_new = b2_f32 * R_prev + (1.0 - b2_f32) * (g_f.T @ g_f)
            m_hat = m_new / bc1
            R_hat = R_new / bc2
            D = R_hat.shape[0]
            R_sym = 0.5 * (R_hat + R_hat.T) + eps_f32 * jnp.eye(D, dtype=jnp.float32)
            w, U = jnp.linalg.eigh(R_sym)
            inv_sqrt_w = jnp.reciprocal(jnp.sqrt(jnp.maximum(w, eps_f32)))
            inv_sqrt = (U * inv_sqrt_w) @ U.T
            u = m_hat @ inv_sqrt
            update_out = (-lr * u).astype(g.dtype)
            return update_out, m_new, R_new

        triples = jax.tree.map(
            step_one, updates, state.m, state.R, is_leaf=_is_masked,
        )
        is_triple = (
            lambda x: isinstance(x, tuple) and len(x) == 3 and not isinstance(x, optax.MaskedNode)
        )
        new_updates = jax.tree.map(lambda t: t[0], triples, is_leaf=is_triple)
        new_m = jax.tree.map(lambda t: t[1], triples, is_leaf=is_triple)
        new_R = jax.tree.map(lambda t: t[2], triples, is_leaf=is_triple)
        return new_updates, RightPrecondAdamState(count=new_count, m=new_m, R=new_R)

    return optax.GradientTransformation(init_fn, update_fn)
