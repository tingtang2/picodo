"""Tests for coupled_adam.{scale_by_coupled_adam, coupled_adamw}.

Run:
    python -m tests.test_coupled_adam

Validates:
  T1. Both axes uncoupled is numerically identical to optax.adamw step-for-step.
  T2. Vocab-axis coupling collapses v_hat to shape (1, D) before rsqrt.
  T3. Feature-axis coupling collapses v_hat to shape (V, 1) before rsqrt.
  T4. Both axes coupled produce a scalar v_hat.
  T5. coupled_adamw with non-zero weight_decay actually decays params (AdamW-style).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
import optax

import coupled_adam


def _make_inputs(V=64, D=16, seed=0):
    key = jax.random.PRNGKey(seed)
    k_p, k_g = jax.random.split(key)
    params = jax.random.normal(k_p, (V, D), dtype=jnp.float32)
    grads = jax.random.normal(k_g, (V, D), dtype=jnp.float32)
    return params, grads


def _hand_coupled_step(params, grads, b1, b2, eps, lr, wd, axes):
    """Single-step hand calculation matching coupled_adamw's inner math."""
    mu_prev = jnp.zeros_like(params)
    nu_prev = jnp.zeros_like(params)
    mu = b1 * mu_prev + (1 - b1) * grads
    nu = b2 * nu_prev + (1 - b2) * jnp.square(grads)
    mu_hat = mu / (1 - b1)
    nu_hat = nu / (1 - b2)
    if axes:
        nu_hat = jnp.mean(nu_hat, axis=axes, keepdims=True)
    pre = mu_hat / (jnp.sqrt(nu_hat) + eps)
    pre = pre + wd * params  # AdamW decoupled WD acts on params
    return params - lr * pre


class TestCoupledAdam(unittest.TestCase):

    def test_t1_no_coupling_matches_adamw(self):
        params, grads = _make_inputs()
        b1, b2, eps, lr, wd = 0.9, 0.999, 1e-8, 1e-3, 0.01

        ref = optax.adamw(lr, b1, b2, eps=eps, weight_decay=wd)
        ref_state = ref.init(params)
        ref_upd, _ = ref.update(grads, ref_state, params)
        ref_params = optax.apply_updates(params, ref_upd)

        ours = coupled_adam.coupled_adamw(
            lr, b1, b2, eps=eps, weight_decay=wd,
            couple_vocab_axis=False, couple_feature_axis=False,
        )
        ours_state = ours.init(params)
        ours_upd, _ = ours.update(grads, ours_state, params)
        ours_params = optax.apply_updates(params, ours_upd)

        np.testing.assert_allclose(np.asarray(ours_params), np.asarray(ref_params), atol=1e-6, rtol=1e-6)

    def test_t2_vocab_axis_matches_hand_calc(self):
        params, grads = _make_inputs()
        b1, b2, eps, lr, wd = 0.9, 0.999, 1e-8, 1e-3, 0.0

        ours = coupled_adam.coupled_adamw(
            lr, b1, b2, eps=eps, weight_decay=wd,
            couple_vocab_axis=True, couple_feature_axis=False,
        )
        state = ours.init(params)
        upd, _ = ours.update(grads, state, params)
        new_params = optax.apply_updates(params, upd)

        expected = _hand_coupled_step(params, grads, b1, b2, eps, lr, wd, axes=(0,))
        np.testing.assert_allclose(np.asarray(new_params), np.asarray(expected), atol=1e-6, rtol=1e-6)

    def test_t3_feature_axis_matches_hand_calc(self):
        params, grads = _make_inputs()
        b1, b2, eps, lr, wd = 0.9, 0.999, 1e-8, 1e-3, 0.0

        ours = coupled_adam.coupled_adamw(
            lr, b1, b2, eps=eps, weight_decay=wd,
            couple_vocab_axis=False, couple_feature_axis=True,
        )
        state = ours.init(params)
        upd, _ = ours.update(grads, state, params)
        new_params = optax.apply_updates(params, upd)

        expected = _hand_coupled_step(params, grads, b1, b2, eps, lr, wd, axes=(1,))
        np.testing.assert_allclose(np.asarray(new_params), np.asarray(expected), atol=1e-6, rtol=1e-6)

    def test_t4_both_axes_yield_scalar_preconditioner(self):
        params, grads = _make_inputs()
        b1, b2, eps, lr, wd = 0.9, 0.999, 1e-8, 1e-3, 0.0

        ours = coupled_adam.coupled_adamw(
            lr, b1, b2, eps=eps, weight_decay=wd,
            couple_vocab_axis=True, couple_feature_axis=True,
        )
        state = ours.init(params)
        upd, _ = ours.update(grads, state, params)
        new_params = optax.apply_updates(params, upd)

        expected = _hand_coupled_step(params, grads, b1, b2, eps, lr, wd, axes=(0, 1))
        np.testing.assert_allclose(np.asarray(new_params), np.asarray(expected), atol=1e-6, rtol=1e-6)

        # The preconditioner is a single scalar — verify the update direction
        # is exactly proportional to the bias-corrected first moment.
        m_hat = grads  # since mu_prev=0, mu = (1-b1)*g, mu_hat = g
        scalar_v = float(jnp.mean(jnp.square(grads)))
        expected_dir = m_hat / (np.sqrt(scalar_v) + eps)
        actual_dir = -np.asarray(upd) / lr
        np.testing.assert_allclose(actual_dir, expected_dir, atol=1e-5, rtol=1e-5)

    def test_t5_weight_decay_applied(self):
        params, _ = _make_inputs()
        zero_grad = jnp.zeros_like(params)
        b1, b2, eps, lr, wd = 0.9, 0.999, 1e-8, 1e-3, 0.1

        ours = coupled_adam.coupled_adamw(
            lr, b1, b2, eps=eps, weight_decay=wd,
            couple_vocab_axis=True, couple_feature_axis=False,
        )
        state = ours.init(params)
        upd, _ = ours.update(zero_grad, state, params)
        new_params = optax.apply_updates(params, upd)

        # With zero gradient, only WD acts: params <- params - lr * wd * params
        expected = params * (1.0 - lr * wd)
        np.testing.assert_allclose(np.asarray(new_params), np.asarray(expected), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
