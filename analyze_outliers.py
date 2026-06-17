"""
Offline outlier diagnostic: restore one or more checkpoints, run the same batch
set through each checkpoint, and compare activation, attention, and weight
outlier summaries before/after a spike.

Example:
    python3 analyze_outliers.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        run_name=<run_name> \
        analysis.checkpoint_steps=[1999,2000,2001] \
        analysis.split=train \
        analysis.num_batches=3
"""

from __future__ import annotations

import json
import math
import os
import time
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from etils import epath
from flax import nnx
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm

from configs import resolver_setup
import data
import model as model_lib
import outlier_analysis
import utils
from train import _StandardNameFormatHNS

def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, jax.Array):
        return _to_jsonable(jax.device_get(value))
    if isinstance(value, jnp.ndarray):
        return _to_jsonable(jax.device_get(value))
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _to_jsonable(value.tolist())
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    return value


def _cfg_get(c: DictConfig, path: str, default):
    node = c
    for part in path.split('.'):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


def _normalize_int_list(values):
    if values is None:
        return None
    return [int(v) for v in values]


def _build_optimizer_state(model, c: DictConfig, num_opt_steps: int, resolved_lm_head_target_rms):
    utils.validate_row_oblique_lm_head_options(c.opt)
    opt_eps = float(getattr(c.opt, "eps", 1e-8))

    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    lm_head_peak_lr = utils.get_lm_head_peak_lr(c.opt)
    lm_head_tx_lr_schedule = lr_schedule
    if not math.isclose(lm_head_peak_lr, float(c.opt.peak_lr)):
        lm_head_tx_lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
            0,
            lm_head_peak_lr,
            warmup_steps,
            num_opt_steps,
        )

    b2_schedule = None
    b2_hparam = c.opt.b2
    b2_cosine_cfg = getattr(c.opt, "b2_cosine_anneal", None)
    if bool(getattr(b2_cosine_cfg, "enabled", False)):
        final_b2 = float(getattr(b2_cosine_cfg, "final_b2", c.opt.b2))
        b2_schedule = optax.schedules.warmup_cosine_decay_schedule(
            init_value=c.opt.b2,
            peak_value=c.opt.b2,
            warmup_steps=0,
            decay_steps=num_opt_steps,
            end_value=final_b2,
        )
        b2_hparam = b2_schedule

    wd_mask = utils.build_weight_decay_mask(model, c.opt.exclude_input_embedding_weight_decay)
    adamw_tx = optax.inject_hyperparams(optax.adamw)(
        lr_schedule,
        c.opt.b1,
        b2_hparam,
        eps=opt_eps,
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )

    pre_transforms, post_transforms, output_embedding_mask, _ = utils.build_lm_head_update_transforms(model, c.opt)
    lm_head_optimizer_cfg = getattr(c.opt, "lm_head_optimizer", None)
    lm_head_optimizer_type = utils.get_lm_head_optimizer_type(c.opt)

    if lm_head_optimizer_type == "adamw":
        if math.isclose(lm_head_peak_lr, float(c.opt.peak_lr)):
            base_optimizer_tx = adamw_tx
        else:
            if output_embedding_mask is None:
                output_embedding_mask = utils.build_output_embedding_mask(model)
            non_output_embedding_mask = jax.tree_util.tree_map(
                lambda is_output_embedding: not is_output_embedding,
                output_embedding_mask,
            )
            if wd_mask is None:
                rest_wd_mask = non_output_embedding_mask
                lm_head_wd_mask = output_embedding_mask
            else:
                rest_wd_mask = jax.tree_util.tree_map(
                    lambda use_wd, is_non_output_embedding: bool(use_wd and is_non_output_embedding),
                    wd_mask,
                    non_output_embedding_mask,
                )
                lm_head_wd_mask = jax.tree_util.tree_map(
                    lambda use_wd, is_output_embedding: bool(use_wd and is_output_embedding),
                    wd_mask,
                    output_embedding_mask,
                )
            lm_head_adamw_tx = optax.inject_hyperparams(optax.adamw)(
                lm_head_tx_lr_schedule,
                c.opt.b1,
                b2_hparam,
                eps=opt_eps,
                weight_decay=c.opt.weight_decay,
                mask=lm_head_wd_mask,
            )
            rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
                lr_schedule,
                c.opt.b1,
                b2_hparam,
                eps=opt_eps,
                weight_decay=c.opt.weight_decay,
                mask=rest_wd_mask,
            )
            base_optimizer_tx = optax.chain(
                optax.masked(rest_adamw_tx, non_output_embedding_mask),
                optax.masked(lm_head_adamw_tx, output_embedding_mask),
            )
    elif lm_head_optimizer_type == "sgd_momentum":
        if output_embedding_mask is None:
            output_embedding_mask = utils.build_output_embedding_mask(model)
        non_output_embedding_mask = jax.tree_util.tree_map(
            lambda is_output_embedding: not is_output_embedding,
            output_embedding_mask,
        )
        if wd_mask is None:
            rest_wd_mask = non_output_embedding_mask
        else:
            rest_wd_mask = jax.tree_util.tree_map(
                lambda use_wd, is_non_output_embedding: bool(use_wd and is_non_output_embedding),
                wd_mask,
                non_output_embedding_mask,
            )
        lm_head_sgd_tx = optax.inject_hyperparams(optax.sgd)(
            learning_rate=lm_head_tx_lr_schedule,
            momentum=float(getattr(lm_head_optimizer_cfg, "momentum", 0.9)),
            nesterov=bool(getattr(lm_head_optimizer_cfg, "nesterov", False)),
        )
        rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lr_schedule,
            c.opt.b1,
            b2_hparam,
            eps=opt_eps,
            weight_decay=c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_output_embedding_mask),
            optax.masked(lm_head_sgd_tx, output_embedding_mask),
        )
    elif lm_head_optimizer_type == "adamw_b1":
        if output_embedding_mask is None:
            output_embedding_mask = utils.build_output_embedding_mask(model)
        non_output_embedding_mask = jax.tree_util.tree_map(
            lambda is_output_embedding: not is_output_embedding,
            output_embedding_mask,
        )
        if wd_mask is None:
            rest_wd_mask = non_output_embedding_mask
        else:
            rest_wd_mask = jax.tree_util.tree_map(
                lambda use_wd, is_non_output_embedding: bool(use_wd and is_non_output_embedding),
                wd_mask,
                non_output_embedding_mask,
            )
        lm_head_b1 = float(getattr(lm_head_optimizer_cfg, "b1", c.opt.b1))
        lm_head_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lm_head_tx_lr_schedule,
            lm_head_b1,
            b2_hparam,
            eps=opt_eps,
            weight_decay=c.opt.weight_decay,
        )
        rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lr_schedule,
            c.opt.b1,
            b2_hparam,
            eps=opt_eps,
            weight_decay=c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_output_embedding_mask),
            optax.masked(lm_head_adamw_tx, output_embedding_mask),
        )
    elif lm_head_optimizer_type in {"row_oblique", "column_oblique"}:
        if output_embedding_mask is None:
            output_embedding_mask = utils.build_output_embedding_mask(model)
        non_output_embedding_mask = jax.tree_util.tree_map(
            lambda is_output_embedding: not is_output_embedding,
            output_embedding_mask,
        )
        if wd_mask is None:
            rest_wd_mask = non_output_embedding_mask
        else:
            rest_wd_mask = jax.tree_util.tree_map(
                lambda use_wd, is_non_output_embedding: bool(use_wd and is_non_output_embedding),
                wd_mask,
                non_output_embedding_mask,
            )
        lm_head_momentum = float(getattr(lm_head_optimizer_cfg, "momentum", 0.9))
        lm_head_target_rms = (
            resolved_lm_head_target_rms
            if resolved_lm_head_target_rms is not None
            else utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
        )
        lm_head_oblique_eps = float(getattr(lm_head_optimizer_cfg, "eps", opt_eps))
        lm_head_oblique_tx = optax.chain(
            utils.scale_by_ema_momentum(lm_head_momentum),
            (
                utils.row_oblique_steepest_descent
                if lm_head_optimizer_type == "row_oblique"
                else utils.column_oblique_steepest_descent
            )(
                learning_rate=lm_head_tx_lr_schedule,
                target_rms=lm_head_target_rms,
                eps=lm_head_oblique_eps,
            )
        )
        rest_adamw_tx = optax.inject_hyperparams(optax.adamw)(
            lr_schedule,
            c.opt.b1,
            b2_hparam,
            eps=opt_eps,
            weight_decay=c.opt.weight_decay,
            mask=rest_wd_mask,
        )
        base_optimizer_tx = optax.chain(
            optax.masked(rest_adamw_tx, non_output_embedding_mask),
            optax.masked(lm_head_oblique_tx, output_embedding_mask),
        )
    else:
        raise ValueError(
            "Expected `opt.lm_head_optimizer.type` to be one of "
            "{'adamw', 'sgd_momentum', 'adamw_b1', 'row_oblique', 'column_oblique'}, "
            f"got {lm_head_optimizer_type!r}."
        )

    tx_chain = [*pre_transforms, base_optimizer_tx, *post_transforms]
    tx = tx_chain[0] if len(tx_chain) == 1 else optax.chain(*tx_chain)
    clip_by_global_norm = getattr(c.opt, "clip_by_global_norm", None)
    if clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(clip_by_global_norm), tx)

    optimizer = nnx.ModelAndOptimizer(model, tx)
    _, opt_state = nnx.split(optimizer)
    return opt_state


def _resolve_target_rms(model, c: DictConfig):
    lm_head_optimizer_type = utils.get_lm_head_optimizer_type(c.opt)
    if lm_head_optimizer_type not in {"row_oblique", "column_oblique"}:
        return None

    learn_target_rms = utils.lm_head_uses_learned_target_rms(c.opt)
    target_rms_from_random_init = utils.lm_head_target_rms_from_random_init(c.opt)
    if learn_target_rms:
        resolved = utils.get_lm_head_oblique_optimizer_target_rms(c.opt)
    elif target_rms_from_random_init:
        resolved = (
            float(model_lib.get_average_output_embedding_row_rms(model))
            if lm_head_optimizer_type == "row_oblique"
            else float(model_lib.get_average_output_embedding_col_rms(model))
        )
    else:
        resolved = utils.get_lm_head_oblique_optimizer_target_rms(c.opt)

    eps = float(getattr(getattr(c.opt, "lm_head_optimizer", None), "eps", getattr(c.opt, "eps", 1e-8)))
    if lm_head_optimizer_type == "row_oblique":
        model_lib.row_normalize_output_embeddings(model, target_rms=resolved, eps=eps)
    else:
        model_lib.column_normalize_output_embeddings(model, target_rms=resolved, eps=eps)
    return resolved


@partial(
    jax.jit,
    static_argnames=(
        "model_graphdef",
        "use_z_loss",
        "collect_activation_tensors",
        "collect_attention_tensors",
        "analysis_layer_indices",
    ),
)
def forward_analysis(
    model_state,
    model_graphdef,
    x,
    use_z_loss: bool = False,
    collect_activation_tensors: bool = True,
    collect_attention_tensors: bool = True,
    analysis_layer_indices: tuple[int, ...] | None = None,
):
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    if collect_activation_tensors or collect_attention_tensors:
        logits, analysis_outputs = model(
            x,
            return_analysis_tensors=True,
            collect_activation_tensors=collect_activation_tensors,
            collect_attention_tensors=collect_attention_tensors,
            analysis_layer_indices=analysis_layer_indices,
        )
    else:
        logits = model(x)
        analysis_outputs = {}
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = utils.cross_entropy_losses_from_log_probs(log_probs, y)[:, :-1]
    batch_loss = losses.mean()
    if use_z_loss:
        z = jax.nn.logsumexp(logits[:, :-1], axis=-1)
        batch_loss = batch_loss + 1e-4 * (z**2).mean()
    return batch_loss, analysis_outputs


def _build_checkpoint_manager(ckpt_dir, gcp_bucket, step_prefix):
    if gcp_bucket:
        name_format = _StandardNameFormatHNS(step_prefix=step_prefix)
        options = ocp.CheckpointManagerOptions(create=False, step_name_format=name_format)
    else:
        options = ocp.CheckpointManagerOptions(create=False)
    return ocp.CheckpointManager(ckpt_dir, options=options)


def _resolve_batch_indices(c: DictConfig, ds_train, ds_valid, checkpoint_steps):
    split = str(_cfg_get(c, "analysis.split", "valid")).lower()
    if split not in {"train", "valid", "val"}:
        raise ValueError("analysis.split must be one of: 'train', 'valid', 'val'.")
    dataset = ds_train if split == "train" else ds_valid

    explicit = _normalize_int_list(_cfg_get(c, "analysis.batch_indices", None))
    if explicit is not None:
        return dataset, explicit

    align_train = bool(_cfg_get(c, "analysis.align_train_batches_to_checkpoints", True))
    if split == "train" and align_train and checkpoint_steps:
        aligned = [step for step in checkpoint_steps if 0 <= step < len(ds_train)]
        if aligned:
            return dataset, aligned

    start_batch = int(_cfg_get(c, "analysis.start_batch", 0))
    num_batches = int(_cfg_get(c, "analysis.num_batches", 4))
    end_batch = min(start_batch + num_batches, len(dataset))
    return dataset, list(range(start_batch, end_batch))


def _ckpt_dir_for_run(c: DictConfig, run_name: str):
    gcp_bucket = getattr(c.checkpoint, "gcp_bucket", None)
    if gcp_bucket:
        if not gcp_bucket.startswith("gs://"):
            gcp_bucket = f"gs://{gcp_bucket}"
        return f"{gcp_bucket.rstrip('/')}/{run_name}", gcp_bucket
    return os.path.join(c.checkpoint.workdir, run_name), None


def _restore_model_state(ckpt_mngr, step, abstract_opt_state):
    restored = ckpt_mngr.restore(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_opt_state),
            training_metadata=ocp.args.JsonRestore(),
        ),
    )
    return restored["state"].model


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(c: DictConfig):
    OmegaConf.resolve(c)

    run_name = c.run_name if c.run_name else "picodo_run"
    tau = float(_cfg_get(c, "analysis.tau", 6.0))
    use_mean = bool(_cfg_get(c, "analysis.use_mean", True))
    top_k = int(_cfg_get(c, "analysis.top_k", 8))
    include_weights = bool(_cfg_get(c, "analysis.include_weights", True))
    include_activations = bool(_cfg_get(c, "analysis.include_activations", True))
    include_attention = bool(_cfg_get(c, "analysis.include_attention", True))
    requested_layer_indices = _normalize_int_list(_cfg_get(c, "analysis.layer_indices", None))
    save_per_batch = bool(_cfg_get(c, "analysis.save_per_batch", False))
    compare_reference_index = int(_cfg_get(c, "analysis.compare_reference_index", 0))
    checkpoint_steps_cfg = _normalize_int_list(_cfg_get(c, "analysis.checkpoint_steps", None))
    force_no_flash_attn = bool(_cfg_get(c, "analysis.force_no_flash_attn", False))
    use_z_loss = bool(getattr(c.opt, "use_z_loss", False))

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(c))
    print("---------------------")

    jax.distributed.initialize()
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ("data", "model"))
    jax.set_mesh(mesh)
    print(f"Sharding mesh: {mesh.shape} (data, model)")

    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    utils.sync_lm_head_oblique_model_config(c)
    if force_no_flash_attn and bool(getattr(c.model, "use_flash_attn", False)):
        with open_dict(c.model):
            c.model.use_flash_attn = False
        print("Disabled flash attention for offline outlier analysis.")
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    target_layer_indices = (
        tuple(requested_layer_indices)
        if requested_layer_indices is not None
        else tuple(range(int(c.model.L)))
    )

    print("Initializing model...")
    model = model_lib.create_sharded_model(c.model, key_model)
    resolved_lm_head_target_rms = _resolve_target_rms(model, c)
    model_graphdef = nnx.graphdef(model)

    print("Loading datasets...")
    n_params = {
        "n_param_nonembed": 12 * c.model.L * c.model.D**2,
        "n_param_embed": c.model.D * c.model.V,
        "n_param_actual": utils.get_num_model_params(model),
    }
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params["n_param_nonembed"] + n_params["n_param_embed"])
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

    opt_state = _build_optimizer_state(model, c, len(ds_train), resolved_lm_head_target_rms)
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    ckpt_dir, gcp_bucket = _ckpt_dir_for_run(c, run_name)
    step_prefix = getattr(c.checkpoint, "step_prefix", None)
    ckpt_mngr = _build_checkpoint_manager(ckpt_dir, gcp_bucket, step_prefix)

    if checkpoint_steps_cfg is None:
        fallback_step = c.checkpoint.start_step if c.checkpoint.start_step is not None else ckpt_mngr.latest_step()
        if fallback_step is None:
            print(f"ERROR: No checkpoint found in {ckpt_dir}")
            ckpt_mngr.close()
            return
        checkpoint_steps = [int(fallback_step)]
    else:
        checkpoint_steps = checkpoint_steps_cfg

    dataset, batch_indices = _resolve_batch_indices(c, ds_train, ds_valid, checkpoint_steps)
    if not batch_indices:
        raise ValueError("No batch indices selected for outlier analysis.")

    output_dir_cfg = _cfg_get(c, "analysis.output_dir", None)
    if output_dir_cfg is None:
        output_dir = os.path.join(c.checkpoint.workdir or os.getcwd(), "outlier_analysis", run_name)
    else:
        output_dir = str(output_dir_cfg)
    output_dir_path = epath.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint source: {ckpt_dir}")
    print(f"Analyzing split={_cfg_get(c, 'analysis.split', 'valid')} batches={batch_indices} steps={checkpoint_steps}")

    checkpoint_reports = []
    t0 = time.time()
    checkpoint_pbar = checkpoint_steps
    if jax.process_index() == 0:
        checkpoint_pbar = tqdm(checkpoint_steps, desc="checkpoints")
    for step in checkpoint_pbar:
        print(f"Restoring checkpoint {step}...")
        model_state = _restore_model_state(ckpt_mngr, step, abstract_opt_state)

        activation_per_batch = []
        attention_per_batch = []
        batch_loss_values = []
        batch_reports = []

        batch_pbar = batch_indices
        if jax.process_index() == 0:
            batch_pbar = tqdm(batch_indices, desc=f"step {step} batches", leave=False)
        for batch_idx in batch_pbar:
            batch = dataset[batch_idx]
            batch_loss, _ = forward_analysis(
                model_state,
                model_graphdef,
                batch,
                use_z_loss=use_z_loss,
                collect_activation_tensors=False,
                collect_attention_tensors=False,
                analysis_layer_indices=None,
            )
            batch_loss_values.append(float(jax.device_get(batch_loss)))

            if include_activations or include_attention:
                layer_activation_summary = {}
                layer_attention_summary = {}
                layer_pbar = target_layer_indices
                if jax.process_index() == 0:
                    layer_pbar = tqdm(target_layer_indices, desc=f"step {step} batch {batch_idx} layers", leave=False)
                for layer_idx in layer_pbar:
                    _, layer_analysis_outputs = forward_analysis(
                        model_state,
                        model_graphdef,
                        batch,
                        use_z_loss=use_z_loss,
                        collect_activation_tensors=include_activations,
                        collect_attention_tensors=include_attention,
                        analysis_layer_indices=(int(layer_idx),),
                    )
                    layer_analysis_outputs_host = jax.device_get(layer_analysis_outputs)
                    activation_summary, attention_summary = outlier_analysis.summarize_analysis_outputs(
                        layer_analysis_outputs_host,
                        tau=tau,
                        use_mean=use_mean,
                        top_k=top_k,
                    )
                    layer_activation_summary.update(activation_summary)
                    layer_attention_summary.update(attention_summary)
                activation_per_batch.append(layer_activation_summary)
                attention_per_batch.append(layer_attention_summary)
            if save_per_batch:
                batch_report = {
                    "batch_index": int(batch_idx),
                    "batch_loss": batch_loss_values[-1],
                }
                if include_activations:
                    batch_report["activations"] = activation_summary
                if include_attention:
                    batch_report["attention"] = attention_summary
                batch_reports.append(batch_report)

        report = {
            "step": int(step),
            "batch_indices": [int(i) for i in batch_indices],
            "batch_loss_mean": float(sum(batch_loss_values) / len(batch_loss_values)),
            "batch_loss_min": float(min(batch_loss_values)),
            "batch_loss_max": float(max(batch_loss_values)),
        }
        if include_activations:
            report["activations"] = outlier_analysis.aggregate_named_summaries(activation_per_batch)
        if include_attention:
            report["attention"] = outlier_analysis.aggregate_named_summaries(attention_per_batch)
        if include_weights:
            report["weights"] = outlier_analysis.summarize_model_weights(
                model_state, tau=tau, use_mean=use_mean, top_k=top_k
            )
        if save_per_batch:
            report["per_batch"] = batch_reports
        checkpoint_reports.append(report)

    ckpt_mngr.close()

    if checkpoint_reports:
        reference_idx = max(0, min(compare_reference_index, len(checkpoint_reports) - 1))
        reference_report = checkpoint_reports[reference_idx]
        for report in checkpoint_reports:
            if report["step"] == reference_report["step"]:
                continue
            deltas = {}
            for section in ("activations", "attention", "weights"):
                if section in reference_report and section in report:
                    deltas[section] = outlier_analysis.diff_named_summaries(
                        reference_report[section], report[section]
                    )
            report["delta_vs_reference"] = {
                "reference_step": int(reference_report["step"]),
                **deltas,
            }

    final_report = {
        "run_name": run_name,
        "checkpoint_dir": ckpt_dir,
        "steps": checkpoint_steps,
        "split": str(_cfg_get(c, "analysis.split", "valid")),
        "batch_indices": [int(i) for i in batch_indices],
        "tau": tau,
        "use_mean": use_mean,
        "top_k": top_k,
        "generated_in_seconds": float(time.time() - t0),
        "checkpoints": checkpoint_reports,
    }

    report_path = output_dir_path / "outlier_analysis_report.json"
    print(type(final_report["checkpoints"][0]["weights"]))
    with report_path.open("w") as f:
        json.dump(_to_jsonable(final_report), f, indent=2)
    print(f"Saved outlier analysis report to: {report_path}")


if __name__ == "__main__":
    main()
