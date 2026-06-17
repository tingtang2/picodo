from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import utils


EPS = 1e-12


def _as_float32_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_float(x) -> float:
    return float(np.asarray(x, dtype=np.float64))


def _topk_flattened(x, k: int):
    arr = _as_float32_array(x)
    flat = arr.reshape(-1)
    if flat.size == 0 or k <= 0:
        return [], []
    kk = min(int(k), int(flat.size))
    idx = np.argpartition(flat, -kk)[-kk:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    values = [float(flat[i]) for i in idx.tolist()]
    indices = [[int(v) for v in np.unravel_index(int(i), arr.shape)] for i in idx.tolist()]
    return values, indices


def _kurtosis_along_last(x, fisher: bool = True, eps: float = EPS) -> float:
    arr = _as_float32_array(x)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 0:
        return 0.0
    mean = arr.mean(axis=-1, keepdims=True)
    centered = arr - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    fourth = np.mean(centered**4, axis=-1, keepdims=True)
    kurt = fourth / np.maximum(var**2, eps)
    if fisher:
        kurt = kurt - 3.0
    return _safe_float(np.mean(kurt))


def _max_to_median_ratio(x, eps: float = EPS) -> float:
    arr = _as_float32_array(x)
    if arr.size == 0:
        return 0.0
    abs_arr = np.abs(arr)
    return _safe_float(abs_arr.max() / np.maximum(np.median(abs_arr), eps))


def summarize_activation_tensor(x, tau: float = 6.0, use_mean: bool = True, top_k: int = 8):
    arr = np.abs(_as_float32_array(x))
    if arr.size == 0:
        return {}
    baseline = arr.mean() if use_mean else np.median(arr)
    threshold = float(tau * baseline)
    mask = arr > threshold
    top_values, top_indices = _topk_flattened(arr, top_k)
    return {
        "outliers_pct": float(mask.mean() * 100.0),
        "threshold": threshold,
        "max_abs": _safe_float(arr.max()),
        "max_to_median": _max_to_median_ratio(arr),
        "kurtosis_mean": _kurtosis_along_last(arr),
        "top_values": top_values,
        "top_indices": top_indices,
    }


def summarize_attention_probs(attn, tau: float = 6.0, use_mean: bool = True, top_k: int = 8):
    arr = _as_float32_array(attn)
    if arr.size == 0:
        return {}
    if arr.ndim != 4:
        raise ValueError(f"Expected attention probs with shape [B, N, T, T], got {arr.shape}.")

    cumulative = arr.sum(axis=2)
    baseline = cumulative.mean(axis=-1, keepdims=True) if use_mean else np.median(cumulative, axis=-1, keepdims=True)
    threshold = tau * baseline
    mask = cumulative > threshold

    max_per_head = cumulative.max(axis=-1)
    median_per_head = np.median(cumulative, axis=-1)
    top_values, top_indices = _topk_flattened(cumulative, top_k)

    centered = cumulative - cumulative.mean(axis=-1, keepdims=True)
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    fourth = np.mean(centered**4, axis=-1, keepdims=True)
    kurt = fourth / np.maximum(var**2, EPS) - 3.0

    return {
        "outliers_pct": float(mask.mean() * 100.0),
        "threshold_mean": _safe_float(np.mean(threshold)),
        "max_prob": _safe_float(arr.max()),
        "max_to_median": _safe_float(np.mean(max_per_head / np.maximum(median_per_head, EPS))),
        "kurtosis_mean": _safe_float(np.mean(kurt)),
        "top_values": top_values,
        "top_indices": top_indices,
    }


def _reshape_weight_matrix(path: str, value):
    arr = _as_float32_array(value)
    if arr.ndim < 2:
        return None
    if "qkv_proj" in path and arr.ndim == 4:
        return np.transpose(arr, (0, 1, 3, 2)).reshape(-1, arr.shape[2])
    if "gate_proj" in path and arr.ndim == 3:
        return np.transpose(arr, (0, 2, 1)).reshape(-1, arr.shape[1])
    if "out_proj" in path and arr.ndim == 3:
        return np.transpose(arr, (2, 0, 1)).reshape(arr.shape[2], -1)
    if ("up_proj" in path or "down_proj" in path) and arr.ndim == 2:
        return arr.T
    return arr.reshape(arr.shape[0], -1)


def summarize_weight_matrix(matrix, tau: float = 6.0, use_mean: bool = True, top_k: int = 8):
    arr = _as_float32_array(matrix)
    if arr.size == 0:
        return {}
    abs_arr = np.abs(arr)
    baseline = abs_arr.mean(axis=-1, keepdims=True) if use_mean else np.median(abs_arr, axis=-1, keepdims=True)
    threshold = tau * baseline
    mask = abs_arr > threshold
    row_outlier_pct = mask.mean(axis=-1) * 100.0
    max_per_row = abs_arr.max(axis=-1)
    median_per_row = np.median(abs_arr, axis=-1)
    top_values, top_indices = _topk_flattened(abs_arr, top_k)
    return {
        "outliers_pct_mean": _safe_float(np.mean(row_outlier_pct)),
        "outliers_pct_max": _safe_float(np.max(row_outlier_pct)),
        "threshold_mean": _safe_float(np.mean(threshold)),
        "max_abs": _safe_float(np.max(abs_arr)),
        "max_to_median": _safe_float(np.mean(max_per_row / np.maximum(median_per_row, EPS))),
        "kurtosis_mean": _kurtosis_along_last(arr),
        "top_values": top_values,
        "top_indices": top_indices,
    }


def summarize_analysis_outputs(analysis_outputs, tau: float = 6.0, use_mean: bool = True, top_k: int = 8):
    activation_summaries = {}
    attention_summaries = {}
    for layer_idx, tensors in analysis_outputs.items():
        layer_name = f"layer_{layer_idx}"
        attn_probs = tensors.get("attention_probs")
        if attn_probs is not None:
            attention_summaries[layer_name] = summarize_attention_probs(
                attn_probs, tau=tau, use_mean=use_mean, top_k=top_k
            )
        for tensor_name in ("attn_input", "mlp_input", "mlp_hidden"):
            value = tensors.get(tensor_name)
            if value is None:
                continue
            activation_summaries[f"{layer_name}/{tensor_name}"] = summarize_activation_tensor(
                value, tau=tau, use_mean=use_mean, top_k=top_k
            )
    return activation_summaries, attention_summaries


def summarize_model_weights(model_state, tau: float = 6.0, use_mean: bool = True, top_k: int = 8):
    summaries = {}
    leaves = utils._collect_weight_leaves_with_effective_lm_head(model_state)
    for path, value in leaves.items():
        matrix = _reshape_weight_matrix(path, value)
        if matrix is None:
            continue
        summary = summarize_weight_matrix(matrix, tau=tau, use_mean=use_mean, top_k=top_k)
        summary["source_shape"] = list(np.shape(value))
        summary["analysis_shape"] = list(matrix.shape)
        summaries[path] = summary
    return summaries


def aggregate_named_summaries(named_summaries):
    buckets = defaultdict(list)
    for summary_map in named_summaries:
        for name, summary in summary_map.items():
            buckets[name].append(summary)

    aggregated = {}
    for name, summaries in buckets.items():
        numeric_fields = defaultdict(list)
        representative = {}
        representative_max = -math.inf
        for summary in summaries:
            summary_max = float(summary.get("max_abs", summary.get("max_prob", 0.0)))
            if summary_max > representative_max:
                representative = summary
                representative_max = summary_max
            for key, value in summary.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_fields[key].append(float(value))
        aggregated[name] = {key: float(np.mean(values)) for key, values in numeric_fields.items()}
        aggregated[name]["num_batches"] = len(summaries)
        if "top_values" in representative:
            aggregated[name]["representative_top_values"] = representative["top_values"]
        if "top_indices" in representative:
            aggregated[name]["representative_top_indices"] = representative["top_indices"]
        for shape_key in ("source_shape", "analysis_shape"):
            if shape_key in representative:
                aggregated[name][shape_key] = representative[shape_key]
    return aggregated


def diff_named_summaries(reference, current):
    delta = {}
    shared_names = sorted(set(reference).intersection(current))
    for name in shared_names:
        ref_summary = reference[name]
        cur_summary = current[name]
        metric_delta = {}
        for key, value in cur_summary.items():
            if not isinstance(value, (int, float)):
                continue
            if key not in ref_summary or not isinstance(ref_summary[key], (int, float)):
                continue
            metric_delta[key] = float(value - ref_summary[key])
        if metric_delta:
            delta[name] = metric_delta
    return delta
