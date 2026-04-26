"""
Offline spike-token diagnostic: loads a saved checkpoint, runs the per-token
eval-loss ranking + W_U row-norm comparison, and saves a JSON report.

Usage:
    python3 analyze_spike_tokens.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        run_name=<run_name> \
        checkpoint.start_step=<step>
"""

import hydra
import jax
import jax.numpy as jnp
import json
import math
import numpy as np
import optax
import orbax.checkpoint as ocp
import os
import time
from etils import epath
from omegaconf import DictConfig, OmegaConf
from flax import nnx
from functools import partial

from configs import resolver_setup
import data
import model as model_lib
import utils
from train import _StandardNameFormatHNS


def _cfg_get(c: DictConfig, path: str, default):
    node = c
    for part in path.split('.'):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c)

    run_name = c.run_name if c.run_name else 'picodo_run'
    top_k = int(_cfg_get(c, 'diagnostics.spike_analysis_top_k', 20))
    min_count = int(_cfg_get(c, 'diagnostics.spike_analysis_min_count', 5))

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(c))
    print("---------------------")

    jax.distributed.initialize()
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    jax.set_mesh(mesh)
    print(f"Sharding mesh: {mesh.shape} (data, model)")

    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    print(f"Model vocab size rounded to: {c.model.V}")

    print('Initializing model...')
    model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(model)

    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')

    print('Loading datasets...')
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (
            n_params['n_param_nonembed'] + n_params['n_param_embed']
        )

    ds_train, ds_valid = data.load_ds(
        key_dataset,
        mesh,
        c.ds_path,
        c.model.T,
        c.opt.batch_size,
        c.num_tokens_valid,
        c.num_tokens_train,
    )
    if c.num_tokens_train is None:
        c.num_tokens_train = ds_train.size

    print(f"Loaded {len(ds_train)} training batches.")
    print(f"Loaded {len(ds_valid)} validation batches.")

    print('Initializing abstract optimizer structure for restore...')
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
        0,
        c.opt.peak_lr,
        warmup_steps,
        num_opt_steps,
    )
    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        c.opt.b2,
        eps=float(_cfg_get(c, 'opt.eps', 1e-8)),
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )
    if c.opt.clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(c.opt.clip_by_global_norm), tx)
    optimizer = nnx.ModelAndOptimizer(model, tx)
    _, opt_state = nnx.split(optimizer)
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    gcp_bucket = getattr(c.checkpoint, 'gcp_bucket', None)
    if gcp_bucket:
        if not gcp_bucket.startswith('gs://'):
            gcp_bucket = f'gs://{gcp_bucket}'
        ckpt_dir = f'{gcp_bucket.rstrip("/")}/{run_name}'
    else:
        ckpt_dir = os.path.join(c.checkpoint.workdir, run_name)
    print(f'Checkpoint source: {ckpt_dir}')

    step_prefix = getattr(c.checkpoint, 'step_prefix', None)
    if gcp_bucket:
        name_format = _StandardNameFormatHNS(step_prefix=step_prefix)
        mngr_options = ocp.CheckpointManagerOptions(create=False, step_name_format=name_format)
    else:
        mngr_options = ocp.CheckpointManagerOptions(create=False)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=mngr_options)

    step_to_load = (
        c.checkpoint.start_step
        if c.checkpoint.start_step is not None
        else ckpt_mngr.latest_step()
    )
    if step_to_load is None:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        return

    print(f"Restoring checkpoint from step {step_to_load}...")
    restored_data = ckpt_mngr.restore(
        step_to_load,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_opt_state),
            training_metadata=ocp.args.JsonRestore(),
        ),
    )
    opt_state = restored_data['state']
    ckpt_mngr.close()
    model_state = opt_state.model
    print('Checkpoint restored successfully.')

    if gcp_bucket:
        default_save_dir = f"{ckpt_dir.rstrip('/')}/evaluation_metrics/step_{step_to_load}"
    else:
        local_base = c.checkpoint.workdir if c.checkpoint.workdir else os.getcwd()
        default_save_dir = os.path.join(local_base, 'evaluation_metrics', run_name, f'step_{step_to_load}')
    save_dir = str(_cfg_get(c, 'analysis.save_dir', default_save_dir))
    save_dir_path = epath.Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    print(
        f"\nRunning spike-token diagnostic over {len(ds_valid)} eval batches "
        f"(top_k={top_k}, min_count={min_count})..."
    )
    t0 = time.time()

    #which batch
    use_val = True
    num_samps = len(ds_valid)

    metrics = utils.compute_spike_token_diagnostics(
        model_state,
        model_graphdef,
        #ds_valid,
        ds_train[:num_samps] if use_val == False else ds_valid[:num_samps],
        step=step_to_load,
        top_k=top_k,
        min_count=min_count,
    )

    print(f"Diagnostic complete in {time.time() - t0:.1f}s")

    report_path = save_dir_path / f'spike_token_diagnostic_step_{step_to_load}.json'
    with report_path.open('w') as f:
        json.dump(
            {
                'run_name': run_name,
                'checkpoint_step': int(step_to_load),
                'checkpoint_dir': ckpt_dir,
                'top_k': top_k,
                'min_count': min_count,
                'num_eval_batches': int(num_samps),
                'metrics': metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved diagnostic report to: {report_path}")


if __name__ == '__main__':
    main()
