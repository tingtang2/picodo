"""Replay selected Adam steps and compare empirical logit changes with theory.

This restores post-update checkpoints. A checkpoint with ID ``s`` contains the
state immediately before training update ``s + 1``. For each restored state we
select 64 positions from that update's batch (21 low-loss, 22 median-loss, and
21 high-loss), replay the exact training step, and save three scatter plots.

Example:
    python analyze_adam_logit_probes.py \
      --config-name=wortsman_default \
      +dataset=fw_gpt2_100B \
      model.D=256 model.L=3 model.T=512 \
      opt.peak_lr=0.1 opt.batch_size=256 opt.weight_decay=0 \
      opt.use_z_loss=False model.rmsnorm_use_scale=True \
      num_tokens_train=16000000000 \
      run_name=small_scale_logit_repro_16B_tokens \
      checkpoint.gcp_bucket=gs://BUCKET/PREFIX \
      +analysis.checkpoint_steps='[6102,61034,115965]' \
      +analysis.save_dir=./adam_logit_probe_results \
      use_single_vm_among_multiple=True \
      ds_path=/dev/shm/datasets/fineweb100B_gpt2.bin
"""

import json
import math
import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import resolver_setup
import data
import model as model_lib
import train as train_lib
import utils


NUM_PROBES = 64
STRATUM_SIZES = {"low": 21, "median": 22, "high": 21}


def _cfg_get(c: DictConfig, path: str, default):
    node = c
    for part in path.split("."):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


def _validate_supported_adam_config(c: DictConfig):
    """Keep the replay optimizer structurally identical to this Adam run."""
    unsupported = []
    if utils.get_lm_head_optimizer_type(c.opt) != "adamw":
        unsupported.append("opt.lm_head_optimizer.type must be adamw")
    if not math.isclose(utils.get_lm_head_peak_lr(c.opt), float(c.opt.peak_lr)):
        unsupported.append("opt.lm_head_optimizer.peak_lr must equal opt.peak_lr")
    if bool(getattr(getattr(c.opt, "adamc", None), "enabled", False)):
        unsupported.append("opt.adamc.enabled must be false")
    if bool(getattr(getattr(c.opt, "b2_cosine_anneal", None), "enabled", False)):
        unsupported.append("opt.b2_cosine_anneal.enabled must be false")
    if str(getattr(c.opt, "lm_head_gradient_centering", "off")) != "off":
        unsupported.append("opt.lm_head_gradient_centering must be off")
    if str(getattr(c.opt, "lm_head_weighted_columnwise_gradient_centering", "off")) != "off":
        unsupported.append("opt.lm_head_weighted_columnwise_gradient_centering must be off")
    if bool(getattr(getattr(c.opt, "lm_head_adaptive_tuc", None), "enabled", False)):
        unsupported.append("opt.lm_head_adaptive_tuc.enabled must be false")
    if bool(getattr(getattr(c.opt, "out_ln_scale_sgd", None), "enabled", False)):
        unsupported.append("opt.out_ln_scale_sgd.enabled must be false")
    if bool(getattr(getattr(c.opt, "all_rmsnorm_scales_sgd", None), "enabled", False)):
        unsupported.append("opt.all_rmsnorm_scales_sgd.enabled must be false")
    if bool(getattr(c.opt, "mucentering", False)):
        unsupported.append("opt.mucentering must be false")
    if bool(getattr(c.opt, "use_z_loss", False)):
        unsupported.append("opt.use_z_loss must be false")
    for objective_name in ("loss_skip", "loss_cap", "loss_rewrite"):
        if bool(getattr(getattr(c.opt, objective_name, None), "enabled", False)):
            unsupported.append(f"opt.{objective_name}.enabled must be false")
    if unsupported:
        raise ValueError("Unsupported replay configuration: " + "; ".join(unsupported))


def _build_optimizer(model, c: DictConfig, lr_schedule):
    wd_mask = utils.build_weight_decay_mask(
        model, c.opt.exclude_input_embedding_weight_decay
    )
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=lr_schedule,
        b1=c.opt.b1,
        b2=c.opt.b2,
        eps=c.opt.eps,
        weight_decay=c.opt.weight_decay,
        mask=wd_mask,
    )
    if c.opt.clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(c.opt.clip_by_global_norm), tx)
    return nnx.ModelAndOptimizer(model, tx)


@jax.jit
def _copy_array(x):
    return jnp.array(x, copy=True)


@jax.jit
def _global_grad_norm(grads):
    return optax.global_norm(grads)


@jax.jit
def _head_array(model_state):
    return utils._state_leaf_to_array(
        utils._get_nested_state_item(model_state, ("token_embed_out", "embedding"))
    )


@jax.jit
def _head_grad_array(grads):
    return utils._state_leaf_to_array(
        utils._get_nested_state_item(grads, ("token_embed_out", "embedding"))
    )


@jax.jit
def _token_losses(model_state, model_graphdef, batch):
    model = nnx.merge(model_graphdef, model_state)
    targets = batch[:, 1:]
    logits = model(batch).astype(jnp.float32)[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)


@jax.jit
def _probe_hidden_logits(model_state, model_graphdef, batch, example_ids, positions):
    model = nnx.merge(model_graphdef, model_state)
    hidden, _ = model(batch, return_update_inputs=True, skip_output_logits=True)
    probe_hidden_native = hidden[example_ids, positions]
    # Match the training forward's LM-head dtype for the empirical logits. The
    # float32 hidden copy below is used only by the update decompositions.
    probe_logits = model.token_embed_out.attend(probe_hidden_native).astype(jnp.float32)
    probe_hidden = jnp.asarray(probe_hidden_native, dtype=jnp.float32)
    return probe_hidden, probe_logits


def _select_stratified_probes(losses: np.ndarray):
    flat = np.asarray(losses, dtype=np.float32).reshape(-1)
    if flat.size < NUM_PROBES:
        raise ValueError(f"Need at least {NUM_PROBES} token positions, got {flat.size}.")

    order = np.argsort(flat, kind="stable")
    low = order[: STRATUM_SIZES["low"]]
    high = order[-STRATUM_SIZES["high"] :]

    median_value = np.median(flat)
    median_order = np.argsort(np.abs(flat - median_value), kind="stable")
    excluded = set(low.tolist()) | set(high.tolist())
    median = np.asarray(
        [idx for idx in median_order.tolist() if idx not in excluded][
            : STRATUM_SIZES["median"]
        ],
        dtype=np.int64,
    )

    indices = np.concatenate([low, median, high]).astype(np.int64)
    strata = np.asarray(
        ["low"] * len(low) + ["median"] * len(median) + ["high"] * len(high)
    )
    sequence_length = losses.shape[1]
    example_ids = indices // sequence_length
    positions = indices % sequence_length
    return example_ids, positions, strata, flat[indices]


def _find_head_adam_state(opt_state):
    candidate_paths = (
        ("token_embed_out", "embedding"),
        ("model", "token_embed_out", "embedding"),
    )
    for state in utils._find_moment_states(opt_state):
        for path in candidate_paths:
            try:
                mu = utils._state_leaf_to_array(
                    utils._get_nested_state_item(utils._get_state_component(state, "mu"), path)
                )
                nu = utils._state_leaf_to_array(
                    utils._get_nested_state_item(utils._get_state_component(state, "nu"), path)
                )
                count = utils._state_leaf_to_array(utils._get_state_component(state, "count"))
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if mu.ndim == 2 and nu.ndim == 2:
                return mu, nu, count
    raise KeyError("Could not locate LM-head Adam moments in restored optimizer state.")


def _scatter_metrics(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return {"count": 0, "pearson": float("nan"), "slope": float("nan"), "r2": float("nan")}
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    xx = np.dot(x_centered, x_centered)
    yy = np.dot(y_centered, y_centered)
    xy = np.dot(x_centered, y_centered)
    slope = xy / xx if xx > 0 else float("nan")
    pearson = xy / math.sqrt(xx * yy) if xx > 0 and yy > 0 else float("nan")
    intercept = y.mean() - slope * x.mean()
    residual = y - (slope * x + intercept)
    r2 = 1.0 - np.dot(residual, residual) / yy if yy > 0 else float("nan")
    return {
        "count": int(x.size),
        "pearson": float(pearson),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def _make_scatter(x, y, xlabel, ylabel, title, path: Path, max_points: int, seed: int):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(x.size, size=max_points, replace=False)
        x, y = x[keep], y[keep]

    metrics = _scatter_metrics(x, y)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=2, alpha=0.18, rasterized=True)
    if x.size:
        lo = float(min(np.quantile(x, 0.001), np.quantile(y, 0.001)))
        hi = float(max(np.quantile(x, 0.999), np.quantile(y, 0.999)))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\nPearson={metrics['pearson']:.4f}, "
        f"slope={metrics['slope']:.4f}, R²={metrics['r2']:.4f}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _analyze_checkpoint(
    c,
    checkpoint_manager,
    abstract_opt_state,
    opt_graphdef,
    model_graphdef,
    ds_train,
    lr_schedule,
    checkpoint_step,
    output_dir,
    original_vocab_size,
    max_scatter_points,
):
    restored = checkpoint_manager.restore(
        checkpoint_step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_opt_state),
            training_metadata=ocp.args.JsonRestore(),
        ),
    )
    opt_state = restored["state"]
    metadata = restored.get("training_metadata", {})
    update_step = int(metadata.get("next_step", checkpoint_step + 1))
    if update_step != checkpoint_step + 1:
        raise ValueError(
            f"Checkpoint {checkpoint_step} metadata says next_step={update_step}; expected {checkpoint_step + 1}."
        )
    if update_step >= len(ds_train):
        raise IndexError(f"Update step {update_step} is outside dataset length {len(ds_train)}.")

    print(f"Analyzing checkpoint {checkpoint_step} -> update {update_step}")
    batch = ds_train[update_step]
    losses = np.asarray(jax.device_get(_token_losses(opt_state.model, model_graphdef, batch)))
    example_ids, positions, strata, probe_losses = _select_stratified_probes(losses)
    example_ids_device = jnp.asarray(example_ids, dtype=jnp.int32)
    positions_device = jnp.asarray(positions, dtype=jnp.int32)

    hidden_before, logits_before = _probe_hidden_logits(
        opt_state.model,
        model_graphdef,
        batch,
        example_ids_device,
        positions_device,
    )
    hidden_before = _copy_array(hidden_before)
    logits_before = _copy_array(logits_before)
    embedding_before = _copy_array(_head_array(opt_state.model))
    mu_before, nu_before, count_before = _find_head_adam_state(opt_state)
    mu_before = _copy_array(mu_before)
    nu_before = _copy_array(nu_before)
    count_before = _copy_array(count_before)
    jax.block_until_ready((hidden_before, logits_before, embedding_before, mu_before, nu_before))

    opt_state, _, _, grads, _ = train_lib.train_step(
        opt_state,
        opt_graphdef,
        model_graphdef,
        batch,
        False,
        True,
        False,
    )
    hidden_after, logits_after = _probe_hidden_logits(
        opt_state.model,
        model_graphdef,
        batch,
        example_ids_device,
        positions_device,
    )
    embedding_after = _head_array(opt_state.model)
    head_grad = _head_grad_array(grads)
    grad_norm = _global_grad_norm(grads)

    learning_rate = jnp.asarray(lr_schedule(update_step), dtype=jnp.float32)
    clip_threshold = float(c.opt.clip_by_global_norm or 0.0)
    if clip_threshold > 0:
        clip_scale = jnp.minimum(1.0, clip_threshold / grad_norm)
    else:
        clip_scale = jnp.asarray(1.0, dtype=jnp.float32)
    clipped_head_grad = head_grad * clip_scale

    count_after = count_before.astype(jnp.float32) + 1.0
    beta1 = jnp.asarray(c.opt.b1, dtype=jnp.float32)
    beta2 = jnp.asarray(c.opt.b2, dtype=jnp.float32)
    mu_after = beta1 * mu_before + (1.0 - beta1) * clipped_head_grad
    nu_after = beta2 * nu_before + (1.0 - beta2) * jnp.square(clipped_head_grad)
    mu_hat = mu_after / (1.0 - jnp.power(beta1, count_after))
    nu_hat = nu_after / (1.0 - jnp.power(beta2, count_after))
    adam_delta_embedding = -learning_rate * (
        mu_hat / (jnp.sqrt(nu_hat) + float(c.opt.eps))
        + float(c.opt.weight_decay) * embedding_before
    )
    sign_delta_embedding = -learning_rate * (
        jnp.sign(clipped_head_grad) + float(c.opt.weight_decay) * embedding_before
    )

    actual_delta_embedding = embedding_after - embedding_before
    exact_head_delta = jnp.dot(hidden_before, actual_delta_embedding.T)
    sign_predicted_delta = jnp.dot(hidden_before, sign_delta_embedding.T)
    adam_predicted_delta = jnp.dot(hidden_before, adam_delta_embedding.T)
    total_empirical_delta = logits_after - logits_before

    arrays = {
        "example_ids": np.asarray(example_ids, dtype=np.int32),
        "positions": np.asarray(positions, dtype=np.int32),
        "target_ids": np.asarray(batch)[example_ids, positions + 1].astype(np.int32),
        "strata": strata,
        "probe_losses": np.asarray(probe_losses, dtype=np.float32),
        "hidden_before": np.asarray(jax.device_get(hidden_before), dtype=np.float32),
        "hidden_after": np.asarray(jax.device_get(hidden_after), dtype=np.float32),
        "logits_before": np.asarray(jax.device_get(logits_before[:, :original_vocab_size]), dtype=np.float32),
        "logits_after": np.asarray(jax.device_get(logits_after[:, :original_vocab_size]), dtype=np.float32),
        "total_empirical_delta": np.asarray(
            jax.device_get(total_empirical_delta[:, :original_vocab_size]), dtype=np.float32
        ),
        "exact_head_delta": np.asarray(
            jax.device_get(exact_head_delta[:, :original_vocab_size]), dtype=np.float32
        ),
        "sign_predicted_delta": np.asarray(
            jax.device_get(sign_predicted_delta[:, :original_vocab_size]), dtype=np.float32
        ),
        "adam_predicted_delta": np.asarray(
            jax.device_get(adam_predicted_delta[:, :original_vocab_size]), dtype=np.float32
        ),
    }

    step_dir = output_dir / f"checkpoint_{checkpoint_step}_update_{update_step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(step_dir / "probe_results.npz", **arrays)

    plot_specs = (
        (
            "sign_predicted_delta",
            "exact_head_delta",
            "Adam sign approximation",
            "Exact LM-head contribution",
            "Adam sign approximation vs exact LM-head contribution",
            "01_sign_vs_exact_head.png",
        ),
        (
            "exact_head_delta",
            "total_empirical_delta",
            "Exact LM-head contribution",
            "Total empirical logit change",
            "LM-head contribution vs total model change",
            "02_exact_head_vs_total.png",
        ),
        (
            "adam_predicted_delta",
            "exact_head_delta",
            "Moment-derived Adam prediction",
            "Exact LM-head contribution",
            "Exact Adam formula vs observed LM-head contribution",
            "03_adam_formula_vs_exact_head.png",
        ),
    )
    metrics = {}
    for plot_index, (x_key, y_key, xlabel, ylabel, title, filename) in enumerate(plot_specs):
        metrics[f"{x_key}_vs_{y_key}"] = _scatter_metrics(arrays[x_key], arrays[y_key])
        _make_scatter(
            arrays[x_key],
            arrays[y_key],
            xlabel,
            ylabel,
            title,
            step_dir / filename,
            max_scatter_points,
            seed=int(c.seed) + update_step + plot_index,
        )

    delta_error = np.asarray(
        jax.device_get(adam_delta_embedding - actual_delta_embedding), dtype=np.float32
    )
    report = {
        "checkpoint_step": int(checkpoint_step),
        "update_step": int(update_step),
        "learning_rate": float(jax.device_get(learning_rate)),
        "global_gradient_norm": float(jax.device_get(grad_norm)),
        "gradient_clip_scale": float(jax.device_get(clip_scale)),
        "adam_count_before": int(np.asarray(jax.device_get(count_before))),
        "num_probes": NUM_PROBES,
        "stratum_sizes": STRATUM_SIZES,
        "original_vocab_size": int(original_vocab_size),
        "adam_delta_embedding_max_abs_error": float(np.max(np.abs(delta_error))),
        "adam_delta_embedding_rms_error": float(np.sqrt(np.mean(np.square(delta_error)))),
        "plots": metrics,
    }
    with (step_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved analysis to {step_dir}")


@hydra.main(version_base=None, config_path="configs", config_name="wortsman_default")
def main(c: DictConfig):
    OmegaConf.resolve(c)
    _validate_supported_adam_config(c)

    if not c.use_single_vm_among_multiple:
        jax.distributed.initialize()
    mesh = jax.make_mesh(
        (jax.device_count() // c.num_tp_devices, c.num_tp_devices), ("data", "model")
    )
    jax.set_mesh(mesh)

    key = jax.random.key(c.seed)
    _, key_model, key_dataset = jax.random.split(key, 3)
    original_vocab_size = int(c.model.V)
    utils.sync_lm_head_oblique_model_config(c)
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count())
    model = model_lib.create_sharded_model(c.model, key_model)
    model_graphdef = nnx.graphdef(model)

    ds_train, _ = data.load_ds(
        key_dataset,
        mesh,
        c.ds_path,
        c.model.T,
        c.opt.batch_size,
        c.num_tokens_valid,
        c.num_tokens_train,
    )
    num_opt_steps = len(ds_train)
    warmup_steps = int(c.opt.warmup_frac * num_opt_steps)
    lr_schedule = utils.build_learning_rate_schedule(
        c.opt, c.opt.peak_lr, warmup_steps, num_opt_steps
    )

    optimizer = _build_optimizer(model, c, lr_schedule)
    opt_graphdef, opt_state = nnx.split(optimizer)
    abstract_opt_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)

    run_name = c.run_name or "picodo_run"
    gcp_bucket = getattr(c.checkpoint, "gcp_bucket", None)
    if gcp_bucket:
        if not gcp_bucket.startswith("gs://"):
            gcp_bucket = f"gs://{gcp_bucket}"
        checkpoint_dir = f"{gcp_bucket.rstrip('/')}/{run_name}"
        manager_options = ocp.CheckpointManagerOptions(
            create=False,
            step_name_format=train_lib._StandardNameFormatHNS(
                step_prefix=getattr(c.checkpoint, "step_prefix", None)
            ),
        )
    else:
        checkpoint_dir = os.path.join(c.checkpoint.workdir, run_name)
        manager_options = ocp.CheckpointManagerOptions(create=False)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=manager_options)

    configured_steps = _cfg_get(c, "analysis.checkpoint_steps", None)
    if configured_steps is None:
        if c.checkpoint.start_step is None:
            raise ValueError(
                "Set +analysis.checkpoint_steps=[...] or checkpoint.start_step=<step>."
            )
        checkpoint_steps = [int(c.checkpoint.start_step)]
    else:
        checkpoint_steps = [int(step) for step in configured_steps]

    output_dir = Path(str(_cfg_get(c, "analysis.save_dir", "adam_logit_probe_results")))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_scatter_points = int(_cfg_get(c, "analysis.max_scatter_points", 250_000))
    print(OmegaConf.to_yaml(c))
    print(f"Checkpoint source: {checkpoint_dir}")

    for checkpoint_step in checkpoint_steps:
        _analyze_checkpoint(
            c,
            checkpoint_manager,
            abstract_opt_state,
            opt_graphdef,
            model_graphdef,
            ds_train,
            lr_schedule,
            checkpoint_step,
            output_dir,
            original_vocab_size,
            max_scatter_points,
        )
    checkpoint_manager.close()


if __name__ == "__main__":
    main()
