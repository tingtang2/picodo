import heapq
import hydra
import io
import jax
import jax.numpy as jnp
import math
import numpy as np
import optax
import orbax.checkpoint as ocp
import os
import time
from etils import epath
from orbax.checkpoint._src.path import gcs_utils, step as step_lib
from omegaconf import DictConfig, OmegaConf
from flax import nnx
from functools import partial
import json

from configs import resolver_setup
import data
import model as model_lib
import utils


class _StandardNameFormatHNS(step_lib._StandardNameFormat):
    """Fixes GCS HNS listing when step_prefix is None."""

    def _glob_step_paths(self, base_path: epath.PathLike) -> list[epath.Path]:
        base_path = epath.Path(base_path)
        if gcs_utils.is_hierarchical_namespace_enabled(base_path):
            bucket_name, path_prefix = gcs_utils.parse_gcs_path(base_path)
            bucket = gcs_utils.get_bucket(bucket_name)
            result = bucket.list_blobs(
                prefix=path_prefix,
                delimiter='/',
                include_folders_as_prefixes=True,
            )
            for _ in result.pages:
                pass
            step_prefix = self.step_prefix or ''
            return [
                epath.Path(f'gs://{bucket_name}/{folder}')
                for folder in result.prefixes
                if folder.startswith(os.path.join(path_prefix, step_prefix))
            ]
        return list(
            epath.Path(base_path).glob(
                f'{step_lib.step_prefix_with_underscore(self.step_prefix)}*'
            )
        )


@partial(jax.jit, static_argnames=('model_graphdef',))
def batch_loss_stats(model_state, model_graphdef, x):
    """Returns batch mean loss, per-example mean loss, and per-token loss [B, T-1]."""
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    logits = model(x).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_losses = -jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
    token_losses = token_losses[:, :-1]
    per_example_loss = token_losses.mean(axis=1)
    batch_mean_loss = per_example_loss.mean()
    return batch_mean_loss, per_example_loss, token_losses


@partial(jax.jit, static_argnames=('model_graphdef',))
def forward_logits(model_state, model_graphdef, x):
    model = nnx.merge(model_graphdef, model_state)
    return model(x).astype(jnp.float32)


def forward_logits_mesh_safe(model_state, model_graphdef, x, data_axis_size: int):
    """Pads batch dimension so it is divisible by mesh data axis, then slices back."""
    batch_size = int(x.shape[0])
    if batch_size % data_axis_size == 0:
        return forward_logits(model_state, model_graphdef, x)

    pad = data_axis_size - (batch_size % data_axis_size)
    x_padded = jnp.concatenate([x, jnp.repeat(x[-1:, :], pad, axis=0)], axis=0)
    logits_padded = forward_logits(model_state, model_graphdef, x_padded)
    return logits_padded[:batch_size]


def _cfg_get(c: DictConfig, path: str, default):
    node = c
    for part in path.split('.'):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


def _decode_token(tokenizer, token_id: int):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(token_id)]).replace('\n', '\\n')
    except Exception:
        return None


def _decode_span(tokenizer, token_ids):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(t) for t in token_ids]).replace('\n', '\\n')
    except Exception:
        return None


def _push_topk(heap, item, k):
    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _top_token_candidates(token_losses_np: np.ndarray, k: int):
    if k <= 0:
        return []
    flat = token_losses_np.reshape(-1)
    if flat.size == 0:
        return []
    kk = min(k, flat.size)
    idx = np.argpartition(flat, -kk)[-kk:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    t_minus_1 = token_losses_np.shape[1]
    out = []
    for i in idx.tolist():
        loss = float(flat[i])
        ex_idx = int(i // t_minus_1)
        pos = int(i % t_minus_1)
        out.append((loss, ex_idx, pos))
    return out


def _topk_logits(logits_vec: np.ndarray, top_k: int):
    top_k = max(1, min(int(top_k), logits_vec.shape[-1]))
    idx = np.argpartition(logits_vec, -top_k)[-top_k:]
    idx = idx[np.argsort(logits_vec[idx])[::-1]]
    return [(int(i), float(logits_vec[i])) for i in idx.tolist()]


def _get_tokenizer(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as e:
        print(f"Tokenizer unavailable ({name}): {e}. Continuing with token IDs only.")
        return None


def _numpy_dtype_from_name(name: str):
    normalized = str(name).lower()
    if normalized in ('float16', 'fp16', 'f16'):
        return np.float16
    if normalized in ('float32', 'fp32', 'f32'):
        return np.float32
    raise ValueError(f'Unsupported analysis.full_logits_dtype={name!r}. Use float16 or float32.')


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):
    OmegaConf.resolve(c)

    run_name = c.run_name if c.run_name else 'picodo_run'
    split = str(_cfg_get(c, 'analysis.split', 'valid')).lower()
    top_batches = int(_cfg_get(c, 'analysis.top_batches', 10))
    top_token_events = int(_cfg_get(c, 'analysis.top_token_events', 50))
    context_window = int(_cfg_get(c, 'analysis.context_window', 32))
    top_logits_k = int(_cfg_get(c, 'analysis.top_logits_k', 10))
    tokenizer_name = str(_cfg_get(c, 'analysis.tokenizer_name', 'gpt2'))
    save_full_logits = bool(_cfg_get(c, 'analysis.save_full_logits', False))
    full_logits_dtype_name = str(_cfg_get(c, 'analysis.full_logits_dtype', 'float16'))
    swap_relative_positions = [int(x) for x in _cfg_get(c, 'analysis.swap_relative_positions', [])]
    swap_replacement_token_ids = [int(x) for x in _cfg_get(c, 'analysis.swap_replacement_token_ids', [])]
    full_logits_dtype = _numpy_dtype_from_name(full_logits_dtype_name)

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(c))
    print("---------------------")

    if swap_relative_positions and len(swap_relative_positions) != len(swap_replacement_token_ids):
        raise ValueError(
            'analysis.swap_relative_positions and analysis.swap_replacement_token_ids must have same length.'
        )

    if any(rel > 0 for rel in swap_relative_positions):
        raise ValueError('Only non-positive relative positions are supported (must edit context, not target/future).')

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
        print(f'Checkpoint source: {ckpt_dir}')
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

    tokenizer = _get_tokenizer(tokenizer_name)

    if split == 'train':
        dataset = ds_train
    else:
        split = 'valid'
        dataset = ds_valid

    data_axis_size = int(mesh.shape['data'])

    print(
        f"Running eval-only analysis on {split} split with {len(dataset)} batches "
        f"(top_batches={top_batches}, top_token_events={top_token_events})"
    )

    top_batch_heap = []
    top_token_heap = []

    t0 = time.time()
    for batch_idx in range(len(dataset)):
        batch = dataset[batch_idx]
        batch_mean_loss, _, token_losses = batch_loss_stats(model_state, model_graphdef, batch)

        batch_mean_loss_f = float(jax.device_get(batch_mean_loss))
        token_losses_np = np.asarray(jax.device_get(token_losses), dtype=np.float32)
        max_token_loss_f = float(token_losses_np.max()) if token_losses_np.size > 0 else float('-inf')

        _push_topk(top_batch_heap, (batch_mean_loss_f, batch_idx, max_token_loss_f), top_batches)

        for loss, ex_idx, pos in _top_token_candidates(token_losses_np, top_token_events):
            _push_topk(top_token_heap, (loss, batch_idx, ex_idx, pos), top_token_events)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataset):
            elapsed = time.time() - t0
            print(f"Processed {batch_idx + 1}/{len(dataset)} batches in {elapsed:.1f}s")

    top_batches_sorted = sorted(top_batch_heap, key=lambda x: x[0], reverse=True)
    top_tokens_sorted = sorted(top_token_heap, key=lambda x: x[0], reverse=True)

    top_batch_records = []
    for rank, (mean_loss, batch_idx, max_token_loss) in enumerate(top_batches_sorted, start=1):
        top_batch_records.append(
            {
                'rank': rank,
                'batch_index': int(batch_idx),
                'mean_loss': float(mean_loss),
                'max_token_loss': float(max_token_loss),
            }
        )

    token_event_records = []
    full_logits_before = []
    full_logits_after = []
    full_logits_before_rank = []
    full_logits_after_rank = []
    full_logits_before_batch_idx = []
    full_logits_after_batch_idx = []
    full_logits_before_example_idx = []
    full_logits_after_example_idx = []
    full_logits_before_pos = []
    full_logits_after_pos = []
    for rank, (loss, batch_idx, ex_idx, pos) in enumerate(top_tokens_sorted, start=1):
        batch_np = np.asarray(jax.device_get(dataset[batch_idx]), dtype=np.int32)
        seq = batch_np[ex_idx]

        if pos + 1 >= seq.shape[0]:
            continue

        target_id = int(seq[pos + 1])
        input_token_id = int(seq[pos])

        logits_np = np.asarray(
            jax.device_get(
                forward_logits_mesh_safe(
                    model_state,
                    model_graphdef,
                    jnp.asarray(seq[None, :], dtype=jnp.int32),
                    data_axis_size=data_axis_size,
                )
            ),
            dtype=np.float32,
        )[0, pos]
        log_probs = np.asarray(jax.device_get(jax.nn.log_softmax(jnp.asarray(logits_np))), dtype=np.float32)
        base_loss = float(-log_probs[target_id])

        top_logits = []
        for token_id, logit in _topk_logits(logits_np, top_logits_k):
            top_logits.append(
                {
                    'token_id': token_id,
                    'logit': logit,
                    'token': _decode_token(tokenizer, token_id),
                }
            )

        context_start = max(0, pos - context_window + 1)
        context_ids = seq[context_start:pos + 1].tolist()

        record = {
            'rank': rank,
            'loss': float(loss),
            'recomputed_loss': base_loss,
            'batch_index': int(batch_idx),
            'example_index': int(ex_idx),
            'position': int(pos),
            'input_token_id': input_token_id,
            'input_token': _decode_token(tokenizer, input_token_id),
            'target_token_id': target_id,
            'target_token': _decode_token(tokenizer, target_id),
            'context_start': int(context_start),
            'context_ids': [int(x) for x in context_ids],
            'context_text': _decode_span(tokenizer, context_ids),
            'top_logits': top_logits,
        }

        if save_full_logits:
            full_logits_before.append(logits_np.astype(full_logits_dtype))
            full_logits_before_rank.append(int(rank))
            full_logits_before_batch_idx.append(int(batch_idx))
            full_logits_before_example_idx.append(int(ex_idx))
            full_logits_before_pos.append(int(pos))

        if swap_relative_positions:
            edited_seq = seq.copy()
            applied_swaps = []
            for rel, new_token in zip(swap_relative_positions, swap_replacement_token_ids):
                token_pos = pos + rel
                if token_pos < 0 or token_pos >= edited_seq.shape[0]:
                    continue
                old_token = int(edited_seq[token_pos])
                edited_seq[token_pos] = int(new_token)
                applied_swaps.append(
                    {
                        'relative_position': int(rel),
                        'absolute_position': int(token_pos),
                        'old_token_id': old_token,
                        'old_token': _decode_token(tokenizer, old_token),
                        'new_token_id': int(new_token),
                        'new_token': _decode_token(tokenizer, int(new_token)),
                    }
                )

            if applied_swaps:
                edited_logits_np = np.asarray(
                    jax.device_get(
                        forward_logits_mesh_safe(
                            model_state,
                            model_graphdef,
                            jnp.asarray(edited_seq[None, :], dtype=jnp.int32),
                            data_axis_size=data_axis_size,
                        )
                    ),
                    dtype=np.float32,
                )[0, pos]
                edited_log_probs = np.asarray(
                    jax.device_get(jax.nn.log_softmax(jnp.asarray(edited_logits_np))),
                    dtype=np.float32,
                )
                edited_loss = float(-edited_log_probs[target_id])

                edited_top_logits = []
                for token_id, logit in _topk_logits(edited_logits_np, top_logits_k):
                    edited_top_logits.append(
                        {
                            'token_id': token_id,
                            'logit': logit,
                            'token': _decode_token(tokenizer, token_id),
                        }
                    )

                record['swap_analysis'] = {
                    'applied_swaps': applied_swaps,
                    'loss_after_swap': edited_loss,
                    'loss_delta': edited_loss - base_loss,
                    'target_logit_before': float(logits_np[target_id]),
                    'target_logit_after': float(edited_logits_np[target_id]),
                    'target_logit_delta': float(edited_logits_np[target_id] - logits_np[target_id]),
                    'top_logits_after_swap': edited_top_logits,
                }
                if save_full_logits:
                    full_logits_after.append(edited_logits_np.astype(full_logits_dtype))
                    full_logits_after_rank.append(int(rank))
                    full_logits_after_batch_idx.append(int(batch_idx))
                    full_logits_after_example_idx.append(int(ex_idx))
                    full_logits_after_pos.append(int(pos))

        token_event_records.append(record)

    report = {
        'run_name': run_name,
        'checkpoint_step': int(step_to_load),
        'checkpoint_dir': ckpt_dir,
        'split': split,
        'num_batches': int(len(dataset)),
        'analysis': {
            'top_batches': int(top_batches),
            'top_token_events': int(top_token_events),
            'context_window': int(context_window),
            'top_logits_k': int(top_logits_k),
            'tokenizer_name': tokenizer_name,
            'save_full_logits': bool(save_full_logits),
            'full_logits_dtype': full_logits_dtype_name,
            'swap_relative_positions': [int(x) for x in swap_relative_positions],
            'swap_replacement_token_ids': [int(x) for x in swap_replacement_token_ids],
        },
        'top_batches_by_mean_loss': top_batch_records,
        'top_token_events': token_event_records,
    }

    report_path = save_dir_path / f'loss_trigger_analysis_step_{step_to_load}_{split}.json'
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)

    full_logits_path = None
    if save_full_logits:
        full_logits_path = save_dir_path / f'loss_trigger_full_logits_step_{step_to_load}_{split}.npz'
        npz_payload = {
            'before_logits': np.stack(full_logits_before, axis=0)
            if full_logits_before
            else np.zeros((0, c.model.V), dtype=full_logits_dtype),
            'before_rank': np.asarray(full_logits_before_rank, dtype=np.int32),
            'before_batch_index': np.asarray(full_logits_before_batch_idx, dtype=np.int32),
            'before_example_index': np.asarray(full_logits_before_example_idx, dtype=np.int32),
            'before_position': np.asarray(full_logits_before_pos, dtype=np.int32),
            'after_logits': np.stack(full_logits_after, axis=0)
            if full_logits_after
            else np.zeros((0, c.model.V), dtype=full_logits_dtype),
            'after_rank': np.asarray(full_logits_after_rank, dtype=np.int32),
            'after_batch_index': np.asarray(full_logits_after_batch_idx, dtype=np.int32),
            'after_example_index': np.asarray(full_logits_after_example_idx, dtype=np.int32),
            'after_position': np.asarray(full_logits_after_pos, dtype=np.int32),
        }
        with io.BytesIO() as buf:
            np.savez_compressed(buf, **npz_payload)
            with full_logits_path.open('wb') as f:
                f.write(buf.getvalue())

    print('\nTop batches (mean loss):')
    for row in top_batch_records[: min(10, len(top_batch_records))]:
        print(
            f"  rank={row['rank']:<3} batch={row['batch_index']:<6} "
            f"mean_loss={row['mean_loss']:.6f} max_token_loss={row['max_token_loss']:.6f}"
        )

    print('\nTop token events:')
    for row in token_event_records[: min(10, len(token_event_records))]:
        print(
            f"  rank={row['rank']:<3} loss={row['loss']:.6f} "
            f"batch={row['batch_index']:<6} ex={row['example_index']:<4} pos={row['position']:<4} "
            f"input={row['input_token_id']} target={row['target_token_id']}"
        )
        if row.get('context_text'):
            print(f"    context: {row['context_text']!r}")

    print(f"\nSaved analysis report to: {report_path}")
    if full_logits_path is not None:
        print(f"Saved full-logits artifact to: {full_logits_path}")


if __name__ == '__main__':
    main()
