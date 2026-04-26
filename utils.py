import jax
import jax.numpy as jnp
import optax
from flax import nnx
from collections.abc import Mapping
from functools import partial
import os
import numpy as np


def flatten_dict(d, prefix=None, sep='.'):
    if isinstance(d, Mapping):
        out = {}
        for k, v in d.items():
            nested_prefix = k if prefix is None else f'{prefix}{sep}{k}'
            out |= flatten_dict(v, nested_prefix, sep)
        return out
    else:
        return {prefix: d}


def get_num_model_params(model: nnx.Module):
    graphdef, params = nnx.split(model, nnx.Param)
    n_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), params, 0)
    return n_params


def _is_output_embedding_path(path) -> bool:
    key = jax.tree_util.keystr(path, simple=True, separator='/')
    return key == 'token_embed_out/embedding'


def build_weight_decay_mask(model: nnx.Module, exclude_input_embedding: bool):
    if not exclude_input_embedding:
        return None
    _, params = nnx.split(model, nnx.Param)

    def mask_leaf(path, _):
        key = jax.tree_util.keystr(path, simple=True, separator='/')
        return 'token_embed_in' not in key

    return jax.tree_util.tree_map_with_path(
        mask_leaf,
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def build_output_embedding_mask(model: nnx.Module):
    _, params = nnx.split(model, nnx.Param)

    def mask_leaf(path, leaf):
        if not _is_output_embedding_path(path):
            return False
        value = getattr(leaf, 'value', leaf)
        if value.ndim != 2:
            raise ValueError(
                "Expected `token_embed_out.embedding` to be a rank-2 matrix for row-wise centering, "
                f"got shape={value.shape}."
            )
        return True

    return jax.tree_util.tree_map_with_path(
        mask_leaf,
        params,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )


def row_wise_mean_centering() -> optax.GradientTransformation:
    """Subtracts the per-row mean from masked 2D updates."""

    def init_fn(_):
        return optax.EmptyState()

    def center_update(update):
        if isinstance(update, optax.MaskedNode):
            return update
        update_f32 = update.astype(jnp.float32)
        centered = update_f32 - jnp.mean(update_f32, axis=1, keepdims=True)
        return centered.astype(update.dtype)

    def update_fn(updates, state, params=None):
        del params
        centered_updates = jax.tree_util.tree_map(
            center_update,
            updates,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return centered_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def magnitude_weighted_column_wise_centering(eps: float = 1e-8) -> optax.GradientTransformation:
    """Zero-centers per-column drift, weighted by squared row norms of the params."""

    def init_fn(_):
        return optax.EmptyState()

    def center_update(update, param):
        if isinstance(update, optax.MaskedNode) or isinstance(param, optax.MaskedNode):
            return update

        update_f32 = update.astype(jnp.float32)
        param_f32 = param.astype(jnp.float32)

        total_drift = jnp.sum(update_f32, axis=0, keepdims=True)
        sq_norms = jnp.sum(jnp.square(param_f32), axis=1, keepdims=True)
        sq_norm_sum = jnp.sum(sq_norms)
        uniform = jnp.full_like(sq_norms, 1.0 / sq_norms.shape[0])
        proportions = jnp.where(sq_norm_sum > eps, sq_norms / sq_norm_sum, uniform)

        centered = update_f32 - proportions * total_drift
        return centered.astype(update.dtype)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "magnitude_weighted_column_wise_centering requires `params` to compute row norms."
            )
        centered_updates = jax.tree_util.tree_map(
            center_update,
            updates,
            params,
            is_leaf=lambda x: isinstance(x, optax.MaskedNode),
        )
        return centered_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def save_to_numpy(save_dir: str, name: str, data):
    path = os.path.join(save_dir, name)
    np.save(path, np.array(data))
    

def compute_lower_90th_percentile_mean(x):
    k = int(0.9 * x.size)
    top_90th = jnp.partition(x.flatten(), k)[:k]
    return top_90th.mean()


def get_l2_norm(data):
    sq_sum = jax.tree_util.tree_reduce(lambda acc, g: acc + jnp.sum(g * g), data, initializer=0.)
    return jnp.sqrt(sq_sum)

def get_layer_grad_norms(grads):
    norms = {}
    norms['global_grad_norm'] = get_l2_norm(grads)
    
    for key, value in grads.items():
        if key == 'blocks':
            for i, block_grads in value.items():
                norms[f'grad_norm/blocks.{i}'] = get_l2_norm(block_grads)
        else:
            norms[f'grad_norm/{key}'] = get_l2_norm(value)
            
    return norms


def get_layer_grad_norms_split(grads):
    norms = {}
    norms['grad_norm/global'] = float(get_l2_norm(grads))
    def visit(path, node):
        # Case 1: Param leaf
        if hasattr(node, "value"):      # nnx.Param
            norms[f"grad_norm/{path}"] = float(get_l2_norm(node.value))
            return
        # Case 2: State or dict-like object
        if hasattr(node, "items"):      # nnx.State, nested dicts
            for key, value in node.items():
                key = str(key)
                new_path = key if path == "" else f"{path}.{key}"
                visit(new_path, value)
            return
        # Case 3: raw array leaf
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            norms[f"grad_norm/{path}"] = float(get_l2_norm(node))
            return
    visit("", grads)
    return norms

def get_layer_weight_norms(params):
    norms = {}
    norms['weight_norm/global'] = get_l2_norm(params)
    
    for key, value in params.items():
        if key == 'blocks':
            for i, block_params in value.items():
                norms[f'weight_norm/blocks.{i}'] = get_l2_norm(block_params)
        else:
            norms[f'weight_norm/{key}'] = get_l2_norm(value)
            
    return norms

def get_layer_weight_norms_split(params):
    norms = {}
    norms['weight_norm/global'] = float(get_l2_norm(params))
    def visit(path, node):
        # Case 1: Param leaf
        if hasattr(node, "value"):      # nnx.Param
            norms[f"weight_norm/{path}"] = float(get_l2_norm(node.value))
            return
        # Case 2: State or dict-like object
        if hasattr(node, "items"):      # nnx.State, nested dicts
            for key, value in node.items():
                key = str(key)
                new_path = key if path == "" else f"{path}.{key}"
                visit(new_path, value)
            return
        # Case 3: raw array leaf
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            norms[f"weight_norm/{path}"] = float(get_l2_norm(node))
            return
    visit("", params)
    return norms


def _find_moment_state(opt_state):
    def visit(tree):
        if isinstance(tree, Mapping):
            if "mu" in tree and "nu" in tree:
                return tree
            for value in tree.values():
                found = visit(value)
                if found is not None:
                    return found
        elif hasattr(tree, "mu") and hasattr(tree, "nu"):
            return tree
        elif isinstance(tree, (list, tuple)):
            for value in tree:
                found = visit(value)
                if found is not None:
                    return found
        return None

    return visit(opt_state)


def _get_state_component(state, key):
    if isinstance(state, Mapping):
        return state[key]
    return getattr(state, key)


def _get_nested_state_item(tree, path):
    node = tree
    for key in path:
        if isinstance(node, Mapping) or hasattr(node, "items"):
            node = node[key]
        else:
            node = getattr(node, key)
    return node


def _state_leaf_to_array(node, dtype=jnp.float32):
    value = node.value if hasattr(node, "value") else node
    return jnp.asarray(value, dtype=dtype)


def _get_first_nested_state_item(tree, candidate_paths):
    for path in candidate_paths:
        try:
            return _get_nested_state_item(tree, path)
        except (AttributeError, KeyError, TypeError):
            continue
    raise KeyError(f"Could not resolve any of the candidate paths: {candidate_paths}")


def _sanitize_metric_name(name: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or "unnamed"


def get_layer_moment_norms(opt_state):
    adam_state = _find_moment_state(opt_state)
    if adam_state is None:
        return {}

    metrics = {}
    # Helper to traverse the tree with a specific prefix
    def log_tree(tree, prefix):
        def visit(path, node):
            # Case 1: Param leaf (nnx.Param or similar wrapper with .value)
            if hasattr(node, "value"):      
                metrics[f"{prefix}/{path}"] = float(get_l2_norm(node.value))
                return
            # Case 2: State or dict-like object (recurse)
            if hasattr(node, "items"):      
                for key, value in node.items():
                    key = str(key)
                    new_path = key if path == "" else f"{path}.{key}"
                    visit(new_path, value)
                return
            # Case 3: raw array leaf
            if isinstance(node, (jnp.ndarray, np.ndarray)):
                metrics[f"{prefix}/{path}"] = float(get_l2_norm(node))
                return
        
        visit("", tree)
    
    # Process First Moment (mu)
    mu = _get_state_component(adam_state, "mu")
    metrics['moment1_norm/global'] = float(get_l2_norm(mu))
    log_tree(mu, 'moment1_norm')

    # Process Second Moment (nu)
    nu = _get_state_component(adam_state, "nu")
    metrics['moment2_norm/global'] = float(get_l2_norm(nu))
    log_tree(nu, 'moment2_norm')

    return metrics


def _collect_array_leaves(tree):
    leaves = {}

    def visit(path, node):
        if hasattr(node, "value"):
            leaves[path] = jnp.asarray(node.value, dtype=jnp.float32)
            return
        if hasattr(node, "items"):
            for key, value in node.items():
                key = str(key)
                child_path = key if path == "" else f"{path}.{key}"
                visit(child_path, value)
            return
        if isinstance(node, (jnp.ndarray, np.ndarray)):
            leaves[path] = jnp.asarray(node, dtype=jnp.float32)
            return

    visit("", tree)
    return leaves


def get_layer_second_moment_rms_metrics(grads, opt_state, eps: float = 1e-30):
    """Per-leaf RMS diagnostics of normalized updates: sqrt(mean(g^2 / v))."""
    moment_state = _find_moment_state(opt_state)
    if moment_state is None:
        return {}

    second_moment = _get_state_component(moment_state, "nu")
    grad_leaves = _collect_array_leaves(grads)
    moment_leaves = _collect_array_leaves(second_moment)
    common_paths = sorted(set(grad_leaves) & set(moment_leaves))

    if not common_paths:
        return {}

    metrics = {}
    eps_arr = jnp.asarray(eps, dtype=jnp.float32)
    whitened_vals = []

    for path in common_paths:
        grad = grad_leaves[path]
        moment = moment_leaves[path]
        if grad.shape != moment.shape:
            continue

        whitened_rms = jnp.sqrt(jnp.mean(jnp.square(grad) / jnp.maximum(moment, eps_arr)))

        metrics[f"rms_grad_sq_over_second_moment/{path}"] = float(whitened_rms)
        whitened_vals.append(float(whitened_rms))

    if whitened_vals:
        metrics["rms_grad_sq_over_second_moment/global_mean"] = float(np.mean(whitened_vals))

    return metrics


def get_output_embedding_group_metrics(opt_state, token_groups, eps: float):
    """Group-averaged output embedding diagnostics over selected token rows."""
    if not token_groups:
        return {}

    moment_state = _find_moment_state(opt_state)
    if moment_state is None:
        return {}

    try:
        output_embedding = _state_leaf_to_array(
            _get_nested_state_item(opt_state.model, ("token_embed_out", "embedding"))
        )
        first_moment = _state_leaf_to_array(
            _get_first_nested_state_item(
                _get_state_component(moment_state, "mu"),
                (
                    ("token_embed_out", "embedding"),
                    ("model", "token_embed_out", "embedding"),
                ),
            )
        )
        second_moment = _state_leaf_to_array(
            _get_first_nested_state_item(
                _get_state_component(moment_state, "nu"),
                (
                    ("token_embed_out", "embedding"),
                    ("model", "token_embed_out", "embedding"),
                ),
            )
        )
    except (AttributeError, KeyError, TypeError):
        return {}

    if (
        output_embedding.ndim != 2
        or first_moment.shape != output_embedding.shape
        or second_moment.shape != output_embedding.shape
    ):
        return {}

    sqrt_second_moment = jnp.sqrt(jnp.maximum(second_moment, 0.0))
    adam_ratio = first_moment / (sqrt_second_moment + jnp.asarray(eps, dtype=jnp.float32))
    vocab_size = int(output_embedding.shape[0])
    metrics = {}
    group_rows_by_name = {}
    group_token_ids_by_name = {}

    for group in token_groups:
        token_ids = np.asarray(group["token_ids"], dtype=np.int32)
        valid_token_ids = token_ids[(token_ids >= 0) & (token_ids < vocab_size)]
        if valid_token_ids.size == 0:
            continue

        metric_group_name = _sanitize_metric_name(group["name"])
        group_rows = output_embedding[valid_token_ids]
        group_rows_by_name[metric_group_name] = group_rows
        group_token_ids_by_name[metric_group_name] = valid_token_ids
        group_sqrt_second_moment = sqrt_second_moment[valid_token_ids]
        group_adam_ratio = adam_ratio[valid_token_ids]

        row_l2_norm = jnp.linalg.norm(group_rows, axis=-1)
        row_sqrt_second_moment_mean = jnp.mean(group_sqrt_second_moment, axis=-1)
        row_sqrt_second_moment_max = jnp.max(group_sqrt_second_moment, axis=-1)
        row_adam_ratio_mean = jnp.mean(group_adam_ratio, axis=-1)

        metrics[f"output_embedding_groups/{metric_group_name}/row_l2_norm_mean"] = jnp.mean(row_l2_norm)
        metrics[f"output_embedding_groups/{metric_group_name}/sqrt_second_moment_mean"] = jnp.mean(
            row_sqrt_second_moment_mean
        )
        metrics[f"output_embedding_groups/{metric_group_name}/sqrt_second_moment_max"] = jnp.mean(
            row_sqrt_second_moment_max
        )
        metrics[f"output_embedding_groups/{metric_group_name}/adam_ratio_mean"] = jnp.mean(row_adam_ratio_mean)

    most_frequent_rows = group_rows_by_name.get("most_frequent")
    least_frequent_rows = group_rows_by_name.get("least_frequent")
    least_frequent_token_ids = group_token_ids_by_name.get("least_frequent")
    if most_frequent_rows is not None and least_frequent_rows is not None and least_frequent_token_ids is not None:
        most_frequent_mean = jnp.mean(most_frequent_rows, axis=0)
        most_frequent_mean_norm = jnp.linalg.norm(most_frequent_mean)
        least_frequent_norms = jnp.linalg.norm(least_frequent_rows, axis=-1)
        cosine_denoms = jnp.maximum(most_frequent_mean_norm * least_frequent_norms, jnp.asarray(eps, dtype=jnp.float32))
        cosine_sims = jnp.sum(least_frequent_rows * most_frequent_mean[None, :], axis=-1) / cosine_denoms

        for token_id, cosine_sim in zip(least_frequent_token_ids.tolist(), cosine_sims):
            metrics[
                f"output_embedding_groups/most_frequent_to_least_frequent/cosine_similarity/token_{int(token_id)}"
            ] = cosine_sim
        metrics["output_embedding_groups/most_frequent_to_least_frequent/cosine_similarity_mean"] = jnp.mean(
            cosine_sims
        )

    return metrics

def compute_qkv_stats(qkv_dict):
    stats = {}
    for i, (q, k, v) in qkv_dict.items():
        # Using simple mean/std here. Could add min/max/norm if needed.
        stats[f'q/layer_{i}/mean'] = q.mean()
        stats[f'k/layer_{i}/mean'] = k.mean()
        stats[f'v/layer_{i}/mean'] = v.mean()
        stats[f'q/layer_{i}/std'] = q.std()
        stats[f'k/layer_{i}/std'] = k.std()
        stats[f'v/layer_{i}/std'] = v.std()
        stats[f'q/layer_{i}/max'] = q.max()
        stats[f'k/layer_{i}/max'] = k.max()
        stats[f'v/layer_{i}/max'] = v.max()
        stats[f'q/layer_{i}/fro_norm'] = get_l2_norm(q)
        stats[f'k/layer_{i}/fro_norm'] = get_l2_norm(k)
        stats[f'v/layer_{i}/fro_norm'] = get_l2_norm(v)
    return stats

@partial(jax.jit, static_argnames=('model_graphdef', 'vocab_size'))
def _per_token_loss_sum_and_count(model_state, model_graphdef, x, vocab_size):
    """Per-target-token loss sum [V] and count [V] for a single batch."""
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x, return_qkv=False).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)

    # remove wraparound last position
    losses = losses[:, :-1]
    targets = y[:, :-1]
    # both [B·(T-1)]
    losses_flat = losses.reshape(-1)
    targets_flat = targets.reshape(-1)

    # scatter-add each (target_id, loss) pair into per-token bins
    loss_sum = jnp.zeros(vocab_size, dtype=jnp.float32).at[targets_flat].add(losses_flat)
    count = jnp.zeros(vocab_size, dtype=jnp.int32).at[targets_flat].add(1)
    return loss_sum, count


@partial(jax.jit, static_argnames=('model_graphdef',))
def _per_batch_grad_W_U(model_state, model_graphdef, x):
    """[V, D] gradient of summed CE loss w.r.t. W_U for a single batch."""
    def loss_fn(state):
        model = nnx.merge(model_graphdef, state)
        y = jnp.roll(x, -1, axis=1)
        logits = model(x, return_qkv=False).astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        return losses[:, :-1].sum()

    grads = jax.grad(loss_fn)(model_state)
    g_W_U = _get_nested_state_item(grads, ("token_embed_out", "embedding"))
    return _state_leaf_to_array(g_W_U)  # [V, D]


def _dist_stats(x):
    """Return (median, p90, p99, max, mean) as Python floats for a [V] array."""
    return (
        float(jnp.median(x)),
        float(jnp.quantile(x, 0.9)),
        float(jnp.quantile(x, 0.99)),
        float(jnp.max(x)),
        float(jnp.mean(x)),
    )


def compute_spike_token_diagnostics(
    model_state, model_graphdef, ds_valid, step: int,
    top_k: int = 20, min_count: int = 5,
):
    """Rank target tokens by mean eval loss; for the top-k tokens, show
    W_U row norms and ||∂L/∂W_U[i,:]|| alongside the vocab-wide distributions.
    Returns wandb-loggable metrics + prints summary."""
    W_U = _state_leaf_to_array(
        _get_nested_state_item(model_state, ("token_embed_out", "embedding"))
    )
    vocab_size, D = int(W_U.shape[0]), int(W_U.shape[1])

    # Single eval pass: accumulate (per-token loss sum, count) AND the gradient
    # of summed CE loss w.r.t. W_U across all batches. Grads sum by linearity.
    loss_sum_total = jnp.zeros(vocab_size, dtype=jnp.float32)
    count_total = jnp.zeros(vocab_size, dtype=jnp.int32)
    grad_sum_total = jnp.zeros((vocab_size, D), dtype=jnp.float32)
    for i in range(len(ds_valid)):
        batch = ds_valid[i]
        ls, cnt = _per_token_loss_sum_and_count(
            model_state, model_graphdef, batch, vocab_size
        )
        loss_sum_total = loss_sum_total + ls
        count_total = count_total + cnt
        grad_sum_total = grad_sum_total + _per_batch_grad_W_U(
            model_state, model_graphdef, batch
        )

    # only rank tokens seen >= min_count times (else one-shot rare tokens dominate)
    mean_loss = jnp.where(
        count_total >= min_count,
        loss_sum_total / jnp.maximum(count_total, 1),
        -jnp.inf,
    )

    # vocab-wide distributions: W_U row norms and per-batch-avg grad row norms
    row_norms = jnp.linalg.norm(W_U, axis=-1)                                # [V]
    grad_row_norms = jnp.linalg.norm(grad_sum_total / len(ds_valid), axis=-1)  # [V]

    median_rn, p90, p99, max_rn, _ = _dist_stats(row_norms)
    grad_median, grad_p90, grad_p99, grad_max, grad_mean = _dist_stats(grad_row_norms)

    eligible = int(jnp.sum(count_total >= min_count))
    k = min(top_k, eligible) if eligible > 0 else 0
    if k == 0:
        print(f"\n=== Spike analysis @ step {step}: no tokens with count >= {min_count} in eval ===")
        return {}

    top_ids = jnp.argsort(-mean_loss)[:k]
    top_ids_list = [int(x) for x in top_ids.tolist()]
    counts_top = [int(x) for x in count_total[top_ids].tolist()]
    mean_loss_top = [float(x) for x in mean_loss[top_ids].tolist()]
    row_norms_top = [float(x) for x in row_norms[top_ids].tolist()]
    grad_rn_top = [float(x) for x in grad_row_norms[top_ids].tolist()]

    print(f"\n=== Spike analysis @ step {step} ===")
    print(
        f"W_U row_norm dist: median={median_rn:.3f}  p90={p90:.3f}  p99={p99:.3f}  max={max_rn:.3f}"
        f"   (eligible tokens: {eligible}/{vocab_size}, min_count={min_count})"
    )
    print(
        f"grad row_norm dist: median={grad_median:.3e}  p90={grad_p90:.3e}  p99={grad_p99:.3e}  max={grad_max:.3e}"
    )
    print(f"Top-{k} target tokens by mean eval loss:")
    print(
        f"  {'rank':>4} {'token_id':>9} {'count':>6} {'mean_loss':>10}"
        f" {'row_norm':>10} {'rn/median':>10}"
        f" {'grad_rn':>12} {'g_rn/median':>12}"
    )
    for rank in range(k):
        print(
            f"  {rank+1:>4d} {top_ids_list[rank]:>9d} {counts_top[rank]:>6d}"
            f" {mean_loss_top[rank]:>10.4f} {row_norms_top[rank]:>10.4f}"
            f" {row_norms_top[rank] / max(median_rn, 1e-9):>10.3f}"
            f" {grad_rn_top[rank]:>12.3e}"
            f" {grad_rn_top[rank] / max(grad_median, 1e-12):>12.3f}"
        )

    grad_row_mean = jnp.mean(grad_sum_total / len(ds_valid), axis=-1)
    top_k_grad = min(top_k, vocab_size)
    top_grad_ids = jnp.argsort(-jnp.abs(grad_row_mean))[:top_k_grad]
    top_grad_ids_list = [int(x) for x in top_grad_ids.tolist()]
    counts_grad_top = [int(x) for x in count_total[top_grad_ids].tolist()]
    grad_mean_grad_top = [float(x) for x in grad_row_mean[top_grad_ids].tolist()]
    grad_rn_grad_top = [float(x) for x in grad_row_norms[top_grad_ids].tolist()]
    row_norms_grad_top = [float(x) for x in row_norms[top_grad_ids].tolist()]
    raw_mean_loss = loss_sum_total / jnp.maximum(count_total, 1)
    raw_mean_loss_grad_top = [float(x) for x in raw_mean_loss[top_grad_ids].tolist()]

    print(f"\nTop-{top_k_grad} tokens by |grad row mean|:")
    print(
        f"  {'rank':>4} {'token_id':>9} {'count':>6} {'grad_mean':>12} {'grad_rn':>12}"
        f" {'g_rn/median':>12} {'row_norm':>10} {'rn/median':>10}"
        f" {'mean_loss':>10}"
    )
    for rank in range(top_k_grad):
        ml_str = (
            f"{raw_mean_loss_grad_top[rank]:>10.4f}"
            if counts_grad_top[rank] > 0 else f"{'—':>10}"
        )
        print(
            f"  {rank+1:>4d} {top_grad_ids_list[rank]:>9d} {counts_grad_top[rank]:>6d}"
            f" {grad_mean_grad_top[rank]:>12.3e}"
            f" {grad_rn_grad_top[rank]:>12.3e}"
            f" {grad_rn_grad_top[rank] / max(grad_median, 1e-12):>12.3f}"
            f" {row_norms_grad_top[rank]:>10.4f}"
            f" {row_norms_grad_top[rank] / max(median_rn, 1e-9):>10.3f}"
            f" {ml_str}"
        )

    bot_grad_ids = jnp.argsort(jnp.abs(grad_row_mean))[:top_k_grad]
    bot_grad_ids_list = [int(x) for x in bot_grad_ids.tolist()]
    counts_grad_bot = [int(x) for x in count_total[bot_grad_ids].tolist()]
    grad_mean_grad_bot = [float(x) for x in grad_row_mean[bot_grad_ids].tolist()]
    grad_rn_grad_bot = [float(x) for x in grad_row_norms[bot_grad_ids].tolist()]
    row_norms_grad_bot = [float(x) for x in row_norms[bot_grad_ids].tolist()]
    raw_mean_loss_grad_bot = [float(x) for x in raw_mean_loss[bot_grad_ids].tolist()]

    print(f"\nBottom-{top_k_grad} tokens by |grad row mean|:")
    print(
        f"  {'rank':>4} {'token_id':>9} {'count':>6} {'grad_mean':>12} {'grad_rn':>12}"
        f" {'g_rn/median':>12} {'row_norm':>10} {'rn/median':>10}"
        f" {'mean_loss':>10}"
    )
    for rank in range(top_k_grad):
        ml_str = (
            f"{raw_mean_loss_grad_bot[rank]:>10.4f}"
            if counts_grad_bot[rank] > 0 else f"{'—':>10}"
        )
        print(
            f"  {rank+1:>4d} {bot_grad_ids_list[rank]:>9d} {counts_grad_bot[rank]:>6d}"
            f" {grad_mean_grad_bot[rank]:>12.3e}"
            f" {grad_rn_grad_bot[rank]:>12.3e}"
            f" {grad_rn_grad_bot[rank] / max(grad_median, 1e-12):>12.3f}"
            f" {row_norms_grad_bot[rank]:>10.4f}"
            f" {row_norms_grad_bot[rank] / max(median_rn, 1e-9):>10.3f}"
            f" {ml_str}"
        )

    counts_f32 = count_total.astype(jnp.float32)
    c_min = int(jnp.min(counts_f32))
    c_q1 = int(jnp.quantile(counts_f32, 0.25))
    c_med = int(jnp.quantile(counts_f32, 0.50))
    c_q3 = int(jnp.quantile(counts_f32, 0.75))
    c_max = int(jnp.max(counts_f32))
    nonzero_mask = count_total > 0
    n_nonzero = int(jnp.sum(nonzero_mask))
    counts_nz = counts_f32[nonzero_mask] if n_nonzero > 0 else counts_f32
    nz_min = int(jnp.min(counts_nz))
    nz_q1 = int(jnp.quantile(counts_nz, 0.25))
    nz_med = int(jnp.quantile(counts_nz, 0.50))
    nz_q3 = int(jnp.quantile(counts_nz, 0.75))
    nz_max = int(jnp.max(counts_nz))
    print(
        f"\nTarget-count 5-number summary (all {vocab_size} tokens):"
        f" min={c_min:_}  Q1={c_q1:_}  median={c_med:_}  Q3={c_q3:_}  max={c_max:_}"
    )
    print(
        f"Target-count 5-number summary ({n_nonzero} tokens with count>0):"
        f" min={nz_min:_}  Q1={nz_q1:_}  median={nz_med:_}  Q3={nz_q3:_}  max={nz_max:_}"
    )

    metrics = {
        "spike_analysis/row_norm/median": median_rn,
        "spike_analysis/row_norm/p90": p90,
        "spike_analysis/row_norm/p99": p99,
        "spike_analysis/row_norm/max": max_rn,
        "spike_analysis/grad_row_norm/mean": grad_mean,
        "spike_analysis/grad_row_norm/median": grad_median,
        "spike_analysis/grad_row_norm/p90": grad_p90,
        "spike_analysis/grad_row_norm/p99": grad_p99,
        "spike_analysis/grad_row_norm/max": grad_max,
        "spike_analysis/eligible_token_count": eligible,
    }
    for rank in range(k):
        prefix = f"spike_analysis/top_{rank+1:02d}"
        metrics[f"{prefix}/token_id"] = top_ids_list[rank]
        metrics[f"{prefix}/mean_loss"] = mean_loss_top[rank]
        metrics[f"{prefix}/count"] = counts_top[rank]
        metrics[f"{prefix}/row_norm"] = row_norms_top[rank]
        metrics[f"{prefix}/row_norm_over_median"] = row_norms_top[rank] / max(median_rn, 1e-9)
        metrics[f"{prefix}/grad_row_norm"] = grad_rn_top[rank]
        metrics[f"{prefix}/grad_row_norm_over_median"] = grad_rn_top[rank] / max(grad_median, 1e-12)
    return metrics
