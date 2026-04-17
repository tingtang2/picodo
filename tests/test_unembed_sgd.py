"""Tests for the unembed_sgd optimizer path.

Run on the TPU (or any machine with the env-loss-spikes venv active):

    cd ~/loss-spikes-project/picodo
    source ../env-loss-spikes/bin/activate
    python -m tests.test_unembed_sgd

Validates:

  T1. utils.build_unembed_label_mask labels ONLY token_embed_out.embedding
      as 'unembed' and every other leaf as 'rest'.

  T2. Mutual-exclusion check: opt.unembed_sgd=true + opt.unembed_eps set
      raises ValueError (replicates train.py's pre-check exactly).

  T3. Conservation law under SGD (the core theorem, made empirical):
      given a synthetic gradient on a W-shape matrix whose row-sum is
      exactly zero (the softmax-CE gradient's defining property), plain
      SGD preserves the row-sum in the updated W.

  T4. Adam breaks the conservation law: same synthetic gradient, same
      W, same lr -> Adam produces a NONZERO change in the row-sum.
      This makes the theoretical claim empirical.

  T5. optax.multi_transform with SGD + adamw composes cleanly: tx.init
      and tx.update succeed on a simple dict-structured pytree that
      mirrors train.py's optimizer routing.

Tests T3-T5 operate on raw JAX arrays rather than nnx.Param-wrapped
pytrees so they don't depend on nnx's tree-traversal semantics; the
mathematical content is the same.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from omegaconf import OmegaConf

import utils


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

class _ToyModel(nnx.Module):
    """Minimal model matching picodo's attribute names. Only used for T1."""

    def __init__(self, V: int, D: int, rngs: nnx.Rngs):
        self.token_embed_in = nnx.Embed(num_embeddings=V, features=D, rngs=rngs)
        self.token_embed_out = nnx.Embed(num_embeddings=V, features=D, rngs=rngs)
        self.linear1 = nnx.Linear(D, D, rngs=rngs)
        self.linear2 = nnx.Linear(D, D, rngs=rngs)


def _make_zero_rowsum_grad(shape, key):
    """Synthetic gradient with exactly zero row-sum (matches softmax-CE)."""
    g = jax.random.normal(key, shape, dtype=jnp.float32)
    return g - jnp.mean(g, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# T1. Label mask correctness.
# ---------------------------------------------------------------------------

class TestLabelMask(unittest.TestCase):

    def test_only_token_embed_out_labeled_unembed(self):
        model = _ToyModel(V=128, D=32, rngs=nnx.Rngs(0))
        label_tree = utils.build_unembed_label_mask(model)

        unembed_paths = []
        rest_paths = []

        def visit(path, leaf):
            key = jax.tree_util.keystr(path, simple=True, separator='/')
            if leaf == 'unembed':
                unembed_paths.append(key)
            elif leaf == 'rest':
                rest_paths.append(key)
            else:
                self.fail(f"unexpected label {leaf!r} at {key}")

        jax.tree_util.tree_map_with_path(
            visit, label_tree,
            is_leaf=lambda x: isinstance(x, str),
        )

        self.assertGreaterEqual(len(unembed_paths), 1, "no 'unembed' leaves")
        self.assertGreater(len(rest_paths), 1, "no 'rest' leaves")

        for p in unembed_paths:
            self.assertIn('token_embed_out', p, f"'unembed' label on wrong path: {p}")
            self.assertIn('embedding', p, f"'unembed' label on wrong path: {p}")

        for p in rest_paths:
            self.assertFalse(
                'token_embed_out' in p and 'embedding' in p,
                f"'rest' label leaked onto W_U path: {p}",
            )


# ---------------------------------------------------------------------------
# T2. Mutual-exclusion check.
# ---------------------------------------------------------------------------

class TestMutualExclusion(unittest.TestCase):

    def _apply_precheck(self, c):
        """Replicates the exact precheck from train.py."""
        unembed_sgd = bool(getattr(c.opt, "unembed_sgd", False))
        unembed_eps = getattr(c.opt, "unembed_eps", None)
        if unembed_sgd and unembed_eps is not None:
            raise ValueError(
                "opt.unembed_sgd=true is incompatible with opt.unembed_eps being "
                f"set (got unembed_eps={unembed_eps}). SGD has no epsilon; pick one."
            )

    def test_conflict_raises(self):
        c = OmegaConf.create({"opt": {"unembed_sgd": True, "unembed_eps": 1e-4}})
        with self.assertRaises(ValueError):
            self._apply_precheck(c)

    def test_only_sgd_ok(self):
        c = OmegaConf.create({"opt": {"unembed_sgd": True, "unembed_eps": None}})
        self._apply_precheck(c)  # should not raise

    def test_only_eps_ok(self):
        c = OmegaConf.create({"opt": {"unembed_sgd": False, "unembed_eps": 1e-4}})
        self._apply_precheck(c)  # should not raise

    def test_both_off_ok(self):
        c = OmegaConf.create({"opt": {"unembed_sgd": False, "unembed_eps": None}})
        self._apply_precheck(c)  # should not raise


# ---------------------------------------------------------------------------
# T3 + T4. Conservation law: SGD preserves row-sum, Adam breaks it.
# ---------------------------------------------------------------------------

class TestConservationLaw(unittest.TestCase):

    V, D = 256, 64
    LR = 0.1

    def test_sgd_preserves_row_sum(self):
        key = jax.random.PRNGKey(0)
        k_w, k_g = jax.random.split(key)
        W = jax.random.normal(k_w, (self.V, self.D), dtype=jnp.float32)
        g = _make_zero_rowsum_grad(W.shape, k_g)

        self.assertLess(
            float(jnp.linalg.norm(jnp.sum(g, axis=0))), 1e-5,
            "test setup: synthetic gradient should have zero row-sum",
        )

        tx = optax.sgd(self.LR)
        state = tx.init(W)
        updates, _ = tx.update(g, state, W)
        W_new = W + updates

        pre_sum = jnp.sum(W, axis=0)
        post_sum = jnp.sum(W_new, axis=0)
        delta_norm = float(jnp.linalg.norm(post_sum - pre_sum))

        self.assertLess(
            delta_norm, 1e-4,
            f"SGD must preserve row-sum to roundoff; got ||Δ row-sum||={delta_norm}",
        )

    def test_adam_breaks_row_sum(self):
        key = jax.random.PRNGKey(0)
        k_w, k_g = jax.random.split(key)
        W = jax.random.normal(k_w, (self.V, self.D), dtype=jnp.float32)
        g = _make_zero_rowsum_grad(W.shape, k_g)

        tx = optax.adam(self.LR, eps=1e-8)
        state = tx.init(W)
        updates, _ = tx.update(g, state, W)
        W_new = W + updates

        pre_sum = jnp.sum(W, axis=0)
        post_sum = jnp.sum(W_new, axis=0)
        delta_norm = float(jnp.linalg.norm(post_sum - pre_sum))

        self.assertGreater(
            delta_norm, 1e-3,
            f"Adam must break row-sum conservation on zero-sum grad; "
            f"got ||Δ row-sum||={delta_norm}",
        )

    def test_adam_breaking_grows_with_row_dispersion(self):
        """Adam's row-sum violation should be larger when the per-row
        gradient-squared magnitudes are more dispersed across rows.
        Constructed to mirror the LM's rare/common token asymmetry."""
        key = jax.random.PRNGKey(1)
        W = jax.random.normal(key, (self.V, self.D), dtype=jnp.float32)

        # Gradient with strong per-row variance: row i scaled by 10**(i/V).
        k_base = jax.random.PRNGKey(2)
        base = jax.random.normal(k_base, (self.V, self.D), dtype=jnp.float32)
        scale = (10.0 ** (jnp.arange(self.V) / self.V))[:, None]
        g_dispersed = base * scale
        g_dispersed = g_dispersed - jnp.mean(g_dispersed, axis=0, keepdims=True)

        # Gradient with uniform per-row magnitude.
        g_uniform = base - jnp.mean(base, axis=0, keepdims=True)

        def adam_delta(g):
            tx = optax.adam(self.LR, eps=1e-8)
            state = tx.init(W)
            # Run multiple steps so nu differs substantially across rows.
            W_t = W
            for _ in range(10):
                updates, state = tx.update(g, state, W_t)
                W_t = W_t + updates
            return float(jnp.linalg.norm(jnp.sum(W_t, axis=0) - jnp.sum(W, axis=0)))

        delta_dispersed = adam_delta(g_dispersed)
        delta_uniform = adam_delta(g_uniform)

        # Dispersed grads -> more non-uniform denominators -> larger
        # conservation-law breakage.
        self.assertGreater(
            delta_dispersed, delta_uniform,
            f"dispersed-per-row gradients should break row-sum more than "
            f"uniform ones. Got dispersed={delta_dispersed}, "
            f"uniform={delta_uniform}",
        )


# ---------------------------------------------------------------------------
# T5. multi_transform composition with SGD + adamw (the path train.py builds).
# ---------------------------------------------------------------------------

class TestMultiTransformComposition(unittest.TestCase):

    def test_sgd_plus_adamw_via_multi_transform(self):
        # Param pytree matching the shape of the model's state.
        V, D = 128, 32
        params = {
            'token_embed_out': {
                'embedding': jnp.ones((V, D), dtype=jnp.float32) * 0.1,
            },
            'token_embed_in': {
                'embedding': jnp.ones((V, D), dtype=jnp.float32) * 0.2,
            },
            'other_layer': jnp.ones((D, D), dtype=jnp.float32) * 0.3,
        }

        def label_leaf(path, _):
            key = jax.tree_util.keystr(path, simple=True, separator='/')
            if 'token_embed_out' in key and 'embedding' in key:
                return 'unembed'
            return 'rest'

        labels = jax.tree_util.tree_map_with_path(label_leaf, params)

        lr = 0.01
        unembed_tx = optax.sgd(lr)
        rest_tx = optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.0)
        tx = optax.multi_transform(
            {'unembed': unembed_tx, 'rest': rest_tx},
            labels,
        )

        state = tx.init(params)
        self.assertTrue(hasattr(state, 'inner_states'),
                        "expected multi_transform state with inner_states")

        # A zero-row-sum grad on W_U, random grads elsewhere.
        key = jax.random.PRNGKey(42)
        k_u, k_i, k_o = jax.random.split(key, 3)
        g_unembed = _make_zero_rowsum_grad((V, D), k_u)
        grads = {
            'token_embed_out': {'embedding': g_unembed},
            'token_embed_in':  {'embedding': jax.random.normal(k_i, (V, D)) * 0.01},
            'other_layer':     jax.random.normal(k_o, (D, D)) * 0.01,
        }

        updates, new_state = tx.update(grads, state, params)

        new_params = jax.tree.map(lambda p, u: p + u, params, updates)

        # W_U row-sum must be preserved (SGD path).
        pre = jnp.sum(params['token_embed_out']['embedding'], axis=0)
        post = jnp.sum(new_params['token_embed_out']['embedding'], axis=0)
        self.assertLess(
            float(jnp.linalg.norm(post - pre)), 1e-4,
            "W_U row-sum should be preserved under SGD path of multi_transform",
        )

        # Other params must have been meaningfully updated (adamw path).
        change = float(jnp.linalg.norm(
            new_params['other_layer'] - params['other_layer']
        ))
        self.assertGreater(
            change, 1e-6,
            "'rest' group should have been updated by adamw",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
