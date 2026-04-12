import jax
import jax.numpy as jnp
from flax import nnx
from collections.abc import Mapping
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
