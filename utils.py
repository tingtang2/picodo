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


def get_layer_moment_norms(opt_state):
    def find_adam_state(tree):
        # Recursively search for a node containing 'mu' and 'nu'
        if isinstance(tree, Mapping):
            if 'mu' in tree and 'nu' in tree:
                return tree
            for v in tree.values():
                found = find_adam_state(v)
                if found: return found
        elif hasattr(tree, 'mu') and hasattr(tree, 'nu'):
            return tree
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                found = find_adam_state(v)
                if found: return found
        return None

    adam_state = find_adam_state(opt_state)
    if adam_state is None:
        return {}

    def get_component(state, key):
        if isinstance(state, Mapping):
            return state[key]
        return getattr(state, key)

    metrics = {}
    
    # Process First Moment (mu)
    mu = get_component(adam_state, 'mu')
    metrics['moment1_norm/global'] = get_l2_norm(mu)
    if isinstance(mu, (Mapping, nnx.State)):
        for key, value in mu.items():
            if key == 'blocks':
                for i, block_params in value.items():
                    metrics[f'moment1_norm/blocks.{i}'] = get_l2_norm(block_params)
            else:
                metrics[f'moment1_norm/{key}'] = get_l2_norm(value)

    # Process Second Moment (nu)
    nu = get_component(adam_state, 'nu')
    metrics['moment2_norm/global'] = get_l2_norm(nu)
    if isinstance(nu, (Mapping, nnx.State)):
        for key, value in nu.items():
            if key == 'blocks':
                for i, block_params in value.items():
                    metrics[f'moment2_norm/blocks.{i}'] = get_l2_norm(block_params)
            else:
                metrics[f'moment2_norm/{key}'] = get_l2_norm(value)
                
    return metrics
