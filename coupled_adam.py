"""Coupled Adam optimizer (Stollenwerk & Stollenwerk, ACL 2025, arxiv 2502.08441).

The paper replaces Adam's per-parameter second moment v_hat for the embedding
matrix with its average across the vocabulary axis, leaving a single shared
per-feature preconditioner. This module implements that, generalized to allow
coupling along the vocab axis (paper behavior), the feature axis (extension),
or both axes simultaneously (scalar preconditioner).

Coupling axes selection:
    couple_vocab_axis=True,  couple_feature_axis=False  -> paper behavior
    couple_vocab_axis=False, couple_feature_axis=True   -> per-token shared
    couple_vocab_axis=True,  couple_feature_axis=True   -> scalar
    couple_vocab_axis=False, couple_feature_axis=False  -> equivalent to AdamW
"""

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax


class ScaleByCoupledAdamState(NamedTuple):
    count: jax.Array
    mu: Any
    nu: Any


def scale_by_coupled_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    *,
    couple_vocab_axis: bool = True,
    couple_feature_axis: bool = False,
    vocab_axis: int = 0,
    feature_axis: int = 1,
) -> optax.GradientTransformation:
    """Adam preconditioner with the second moment averaged over chosen axes."""

    axes = []
    if couple_vocab_axis:
        axes.append(vocab_axis)
    if couple_feature_axis:
        axes.append(feature_axis)
    couple_axes = tuple(sorted(set(axes)))

    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)
        nu = jax.tree.map(jnp.zeros_like, params)
        return ScaleByCoupledAdamState(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu,
        )

    def precondition(m_hat, v_hat):
        if couple_axes:
            for axis in couple_axes:
                if axis >= v_hat.ndim:
                    raise ValueError(
                        f"coupled_adam: axis {axis} out of bounds for shape {v_hat.shape}."
                    )
            v_hat = jnp.mean(v_hat, axis=couple_axes, keepdims=True)
        return m_hat / (jnp.sqrt(v_hat) + eps)

    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree.map(
            lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, updates,
        )
        nu = jax.tree.map(
            lambda v, g: b2 * v + (1.0 - b2) * jnp.square(g), state.nu, updates,
        )
        count = optax.safe_int32_increment(state.count)
        bc1 = 1.0 - jnp.asarray(b1, jnp.float32) ** count
        bc2 = 1.0 - jnp.asarray(b2, jnp.float32) ** count
        mu_hat = jax.tree.map(lambda m: m / bc1, mu)
        nu_hat = jax.tree.map(lambda v: v / bc2, nu)
        new_updates = jax.tree.map(precondition, mu_hat, nu_hat)
        return new_updates, ScaleByCoupledAdamState(count=count, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def coupled_adamw(
    learning_rate,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    couple_vocab_axis: bool = True,
    couple_feature_axis: bool = False,
    vocab_axis: int = 0,
    feature_axis: int = 1,
    mask: Optional[Any] = None,
) -> optax.GradientTransformation:
    """AdamW with the second moment coupled along selected axes."""
    return optax.chain(
        scale_by_coupled_adam(
            b1=b1, b2=b2, eps=eps,
            couple_vocab_axis=couple_vocab_axis,
            couple_feature_axis=couple_feature_axis,
            vocab_axis=vocab_axis, feature_axis=feature_axis,
        ),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale_by_learning_rate(learning_rate),
    )
