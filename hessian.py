"""Per-step top-K Hessian eigenvalue tracking via pytree subspace iteration.

We approximate the top-K eigenvalues of the loss Hessian H = d^2 L / d theta^2
along the training trajectory and, for each top-K eigenvector, the fraction of
its squared L2-mass that lives on each parameter tensor (so you can see *where*
the sharpest curvature directions live).

Algorithm
---------
The tracker maintains a basis V of K orthonormal pytree-vectors that have the
same structure (and therefore the same FSDP sharding) as the model parameters.
Per training step:

  1. Apply the Hessian to each basis vector via a forward-over-reverse HVP
     (`jax.jvp` of `jax.grad(loss)`). Cost: K HVPs per step.
  2. Form the K x K Rayleigh-Ritz matrix M[i, j] = <V[i], H V[j]> and
     diagonalize it. The eigenvalues are the top-K Ritz approximations.
  3. Rotate V into the computed Ritz directions and re-orthonormalize via
     modified Gram-Schmidt. The rotated basis is carried into the next step
     as a warm start, so over consecutive steps it converges toward the
     dominant K-dimensional invariant subspace of H.

This is mathematically the standard Rayleigh-Ritz / subspace iteration scheme,
specialized to one outer iteration per training step. Because consecutive
training steps barely change the Hessian, one iteration per step is enough to
keep V tracking the top eigenspace -- this is the same warm-start trick that
makes per-step LOBPCG affordable, but expressed natively on pytrees so the
existing FSDP sharding is preserved without flattening parameters into one
huge vector.

Per-tensor mass
---------------
For each top-K eigenvector v_i (a pytree with the same structure as the model
params), we report ||v_i restricted to leaf p||^2 as a fraction of ||v_i||^2,
for every leaf p. The fractions for a single eigenvector sum to 1.0; the leaf
with the largest fraction is the parameter tensor along which that
sharpness-direction predominantly lives.
"""

from __future__ import annotations
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# ---------------------------------------------------------------------------
# Pytree linear-algebra helpers. All inner products are computed in fp32 to
# avoid the precision issues we hit with bf16 dot products on large vectors.
# ---------------------------------------------------------------------------

def tree_dot(a, b) -> jnp.ndarray:
    """Sum of element-wise products across all leaves, in fp32."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    acc = jnp.float32(0.0)
    for x, y in zip(leaves_a, leaves_b):
        acc = acc + jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32))
    return acc


def tree_norm(a) -> jnp.ndarray:
    return jnp.sqrt(tree_dot(a, a))


def gram_schmidt(vs: List[Any]) -> List[Any]:
    """Modified Gram-Schmidt orthonormalization of a list of pytree-vectors.

    Each output vector preserves the dtype of the corresponding input leaf.
    The arithmetic is done in fp32 and cast back at the end.
    """
    out: List[Any] = []
    for v in vs:
        for u in out:
            coef = tree_dot(u, v)
            v = jax.tree.map(
                lambda vi, ui: (
                    vi.astype(jnp.float32) - coef * ui.astype(jnp.float32)
                ).astype(vi.dtype),
                v, u,
            )
        n = tree_norm(v)
        v = jax.tree.map(
            lambda vi: (vi.astype(jnp.float32) / (n + jnp.float32(1e-30))).astype(vi.dtype),
            v,
        )
        out.append(v)
    return out


def _path_to_dotted(path) -> str:
    """Convert a jax tree path (sequence of key entries) into the dotted form
    used by utils.get_layer_weight_norms_split (e.g. ``blocks.3.attn.qkv_proj.kernel``)."""
    parts: List[str] = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return ".".join(parts)


def build_param_mask(template, param_filter: Optional[str]) -> Any:
    """Build a pytree of bool scalars matching ``template``'s structure.

    A leaf is True iff its dotted path matches ``param_filter`` (regex
    ``re.search``). When ``param_filter`` is None, every leaf is True.
    """
    if param_filter is None:
        return jax.tree_util.tree_map(lambda _: True, template)
    pattern = re.compile(param_filter)

    def per_leaf(path, _leaf):
        return bool(pattern.search(_path_to_dotted(path)))

    return jax.tree_util.tree_map_with_path(per_leaf, template)


def list_filter_matches(template, param_filter: Optional[str]) -> List[str]:
    """Return the list of dotted param paths matched by ``param_filter``."""
    matches: List[str] = []

    def visit(path, _leaf):
        dotted = _path_to_dotted(path)
        if param_filter is None or re.search(param_filter, dotted):
            matches.append(dotted)
        return _leaf

    jax.tree_util.tree_map_with_path(visit, template)
    return matches


def apply_stop_gradient_mask(params, mask):
    """Wrap each non-matching leaf in jax.lax.stop_gradient.

    Matching leaves pass through unchanged. The forward pass uses the same
    values either way, but the backward (and hence the JVP-of-grad used for
    HVPs) sees zero on the stop_gradient'd leaves, so XLA dead-code
    elimination drops the corresponding intermediate activations from the
    backward graph.
    """
    return jax.tree_util.tree_map(
        lambda leaf, m: leaf if m else jax.lax.stop_gradient(leaf),
        params, mask,
    )


def zero_out_unmasked(tree, mask):
    """Set every non-matching leaf of ``tree`` to zero (preserving dtype/sharding)."""
    return jax.tree_util.tree_map(
        lambda leaf, m: leaf if m else jnp.zeros_like(leaf),
        tree, mask,
    )


def random_pytree_like(key: jax.Array, template: Any) -> Any:
    """Sample a random pytree with the same structure (and sharding) as
    `template`. Each leaf is iid standard normal in the leaf's own dtype.

    To inherit `template`'s sharding we add `0 * template_leaf` to the random
    sample -- XLA propagates the sharding constraint of the (sharded) template
    leaf onto the result of the addition.
    """
    leaves = jax.tree_util.tree_leaves(template)
    keys = jax.random.split(key, len(leaves))
    new_leaves = []
    for k, leaf in zip(keys, leaves):
        x = jax.random.normal(k, leaf.shape, leaf.dtype)
        x = x + jnp.zeros_like(leaf)  # inherit leaf's sharding
        new_leaves.append(x)
    treedef = jax.tree_util.tree_structure(template)
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


# ---------------------------------------------------------------------------
# HVP construction. The loss is the standard next-token cross-entropy in fp32,
# matching `loss_fn` in train.py (without the qkv-stats side channel which is
# irrelevant to the Hessian).
# ---------------------------------------------------------------------------

def make_hvp(model_graphdef, mask=None):
    """Build a JIT-compiled Hessian-vector product closure.

    Returns a function `hvp(params, batch, v) -> H @ v` where `params`, `v`,
    and the returned tangent are all pytrees with the same structure as
    `nnx.state(model)`. `model_graphdef` is captured in the closure so the
    compiled HVP is keyed on it -- as long as we reuse the same graphdef
    object across training (we do; it is created once in train.py), the JIT
    cache hits on every call after the first.

    If ``mask`` is provided (a pytree of bool with the same structure as
    ``params``), every non-True leaf is wrapped in ``jax.lax.stop_gradient``
    inside the loss. The HVP then sees zero gradient through those leaves,
    so the result is the diagonal block of H restricted to the True leaves.
    XLA's dead-code elimination drops the backward-pass activations belonging
    to the masked-out parameters, which is what makes the per-subgroup HVP
    cheap enough to fit in HBM when the full HVP would not.
    """

    def loss_of_params(params, batch):
        if mask is not None:
            params = apply_stop_gradient_mask(params, mask)
        model = nnx.merge(model_graphdef, params)
        # TPU Splash flash attention's pallas_call has no jvp rule, so a
        # forward-over-reverse HVP through it raises NotImplementedError.
        # Force the HVP-time forward to take the model's non-flash attention
        # path. The merged model is fresh each call, so this mutation is local.
        for block in model.blocks:
            block.attn.use_flash_attn = False
            block.attn.attention = partial(jax.nn.dot_product_attention, is_causal=False)
        y = jnp.roll(batch, -1, axis=1)
        logits = model(batch, return_qkv=False)
        logits = logits.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        return losses[:, :-1].mean()

    @jax.jit
    def hvp(params, batch, v):
        # forward-over-reverse: (1) take the gradient of the loss as a
        # function of params; (2) take the JVP of that gradient in direction v.
        # The JVP tangent of grad(loss) is exactly the Hessian-vector product.
        _, hv = jax.jvp(lambda p: jax.grad(loss_of_params)(p, batch), (params,), (v,))
        return hv

    return hvp


# ---------------------------------------------------------------------------
# Eigenvector mass over parameter tensors.
# ---------------------------------------------------------------------------

def eigvec_mass_per_tensor(v) -> Dict[str, float]:
    """Walk an eigenvector pytree and return {param_path: ||v_path||^2 / ||v||^2}.

    The walker mirrors `utils.get_layer_weight_norms_split` so parameter paths
    are consistent with the rest of the codebase's logging.
    """
    sq_norms: Dict[str, float] = {}

    def visit(path: str, node: Any) -> None:
        if hasattr(node, "value"):
            arr = node.value
            sq_norms[path] = float(jnp.sum(arr.astype(jnp.float32) ** 2))
            return
        if hasattr(node, "items"):
            for key, child in node.items():
                key_s = str(key)
                new_path = key_s if path == "" else f"{path}.{key_s}"
                visit(new_path, child)
            return
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            sq_norms[path] = float(jnp.sum(node.astype(jnp.float32) ** 2))
            return

    visit("", v)
    total = sum(sq_norms.values()) + 1e-30
    return {p: s / total for p, s in sq_norms.items()}


# ---------------------------------------------------------------------------
# The tracker.
# ---------------------------------------------------------------------------

class HessianTracker:
    """Per-step top-K Hessian eigenvalue + eigenvector-mass tracker.

    Construct once per training run and call `.step(params, batch)` after each
    optimizer update. Returns a flat dict of metrics ready for wandb.log:

        hessian/eigval_0 ... hessian/eigval_{K-1}              (K floats)
        hessian/mass/eigvec_0/<param_path> ... hessian/mass/eigvec_{K-1}/<param_path>
            (K * num_params floats; per-eigenvector mass distributions sum to 1)
    """

    def __init__(
        self,
        model_graphdef,
        k: int,
        seed: int = 0,
        param_filter: Optional[str] = None,
    ) -> None:
        self.k = int(k)
        self.param_filter = param_filter
        self._model_graphdef = model_graphdef
        self._rng = jax.random.key(int(seed))
        self.V: List[Any] = []  # lazily initialized on first .step() call
        # Mask is built lazily once we see the actual params pytree, then the
        # HVP closure is constructed against that mask.
        self._mask: Any = None
        self.hvp = None  # type: ignore[assignment]

    def _init_basis(self, params_template) -> None:
        # Build the mask and the masked HVP closure now that we have a real
        # params template (we need its pytree structure).
        self._mask = build_param_mask(params_template, self.param_filter)
        self.hvp = make_hvp(self._model_graphdef, mask=self._mask)
        keys = jax.random.split(self._rng, self.k + 1)
        self._rng = keys[0]
        # Initial random basis, then zero out leaves outside the masked
        # subspace so the basis lives entirely within the subgroup.
        vs = [random_pytree_like(keys[i + 1], params_template) for i in range(self.k)]
        vs = [zero_out_unmasked(v, self._mask) for v in vs]
        self.V = gram_schmidt(vs)

    def step(self, params, batch) -> Dict[str, float]:
        """Run one subspace-iteration step against the current params and
        batch and return a flat metrics dict for wandb.log."""
        if not self.V:
            self._init_basis(params)

        K = self.k

        # 1. Hessian-vector products for each basis vector.
        HV: List[Any] = [self.hvp(params, batch, v) for v in self.V]

        # 2. Rayleigh-Ritz matrix M[i, j] = <V[i], H V[j]> and its
        #    eigendecomposition. M is symmetric in exact arithmetic;
        #    symmetrize to suppress numerical asymmetry.
        M_rows = []
        for i in range(K):
            row = jnp.stack([tree_dot(self.V[i], HV[j]) for j in range(K)])
            M_rows.append(row)
        M = jnp.stack(M_rows)
        M = 0.5 * (M + M.T)
        eigvals_dev, C_dev = jnp.linalg.eigh(M)

        # Sort descending so eigval_0 is the largest.
        order_dev = jnp.argsort(-eigvals_dev)
        eigvals_dev = eigvals_dev[order_dev]
        C_dev = C_dev[:, order_dev]

        # Move the small (K and K x K) results to host once.
        eigvals = np.asarray(eigvals_dev)
        C = np.asarray(C_dev)

        # 3. Rotate V into the Ritz basis: V_new[i] = sum_j C[j, i] * HV[j].
        #    Then re-orthonormalize for numerical health.
        V_new: List[Any] = []
        for i in range(K):
            weights = [float(C[j, i]) for j in range(K)]

            def combine(*leaves, _w=weights):
                acc = _w[0] * leaves[0].astype(jnp.float32)
                for j in range(1, len(leaves)):
                    acc = acc + _w[j] * leaves[j].astype(jnp.float32)
                return acc.astype(leaves[0].dtype)

            v_new = jax.tree.map(combine, *HV)
            V_new.append(v_new)
        self.V = gram_schmidt(V_new)

        # 4. Build metrics dict.
        metrics: Dict[str, float] = {}
        for i in range(K):
            metrics[f"hessian/eigval_{i}"] = float(eigvals[i])
            mass = eigvec_mass_per_tensor(self.V[i])
            for path, frac in mass.items():
                metrics[f"hessian/mass/eigvec_{i}/{path}"] = frac
        return metrics
