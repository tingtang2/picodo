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


def get_global_grad_norm(grads):
    sq_sum = jax.tree_util.tree_reduce(lambda acc, g: acc + jnp.sum(g * g), grads, initializer=0.)
    return jnp.sqrt(sq_sum)
