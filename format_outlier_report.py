from __future__ import annotations

import argparse
import json
from pathlib import Path


SECTION_SPECS = {
    "activations": {
        "label": "Activations",
        "metric": "outliers_pct",
        "secondary": "kurtosis_mean",
        "magnitude": "max_abs",
    },
    "attention": {
        "label": "Attention",
        "metric": "outliers_pct",
        "secondary": "kurtosis_mean",
        "magnitude": "max_prob",
    },
    "weights": {
        "label": "Weights",
        "metric": "outliers_pct_mean",
        "secondary": "kurtosis_mean",
        "magnitude": "max_abs",
    },
}


def _safe_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _load_report(path: Path):
    return json.loads(path.read_text())


def _resolve_spike_step(report, explicit_spike_step: int | None):
    checkpoints = report["checkpoints"]
    if explicit_spike_step is not None:
        return explicit_spike_step
    return max(checkpoints, key=lambda ck: ck.get("batch_loss_mean", float("-inf")))["step"]


def _resolve_baseline_steps(report, spike_step: int, explicit_baseline_steps: list[int] | None):
    checkpoints = report["checkpoints"]
    steps = [ck["step"] for ck in checkpoints]
    if explicit_baseline_steps:
        return explicit_baseline_steps
    baseline_steps = [step for step in steps if step < spike_step]
    if baseline_steps:
        return baseline_steps
    return [step for step in steps if step != spike_step]


def _index_checkpoints(report):
    return {ck["step"]: ck for ck in report["checkpoints"]}


def _collect_item_names(step_to_ckpt, steps, section):
    names = set()
    for step in steps:
        ck = step_to_ckpt.get(step)
        if ck is None or section not in ck:
            continue
        names.update(ck[section].keys())
    return sorted(names)


def _build_section_rows(report, section: str, spike_step: int, baseline_steps: list[int]):
    spec = SECTION_SPECS[section]
    metric = spec["metric"]
    secondary = spec["secondary"]
    magnitude = spec["magnitude"]
    step_to_ckpt = _index_checkpoints(report)
    rows = []
    for name in _collect_item_names(step_to_ckpt, baseline_steps + [spike_step], section):
        spike_ck = step_to_ckpt.get(spike_step, {})
        spike_stats = spike_ck.get(section, {}).get(name)
        if not isinstance(spike_stats, dict):
            continue
        baseline_stats = [
            step_to_ckpt[step][section][name]
            for step in baseline_steps
            if step in step_to_ckpt and section in step_to_ckpt[step] and name in step_to_ckpt[step][section]
        ]
        baseline_metric = _mean([_safe_float(stats.get(metric)) for stats in baseline_stats if _safe_float(stats.get(metric)) is not None])
        baseline_secondary = _mean([_safe_float(stats.get(secondary)) for stats in baseline_stats if _safe_float(stats.get(secondary)) is not None])
        baseline_magnitude = _mean([_safe_float(stats.get(magnitude)) for stats in baseline_stats if _safe_float(stats.get(magnitude)) is not None])
        spike_metric = _safe_float(spike_stats.get(metric))
        spike_secondary = _safe_float(spike_stats.get(secondary))
        spike_magnitude = _safe_float(spike_stats.get(magnitude))
        delta_metric = None if baseline_metric is None or spike_metric is None else spike_metric - baseline_metric
        delta_secondary = None if baseline_secondary is None or spike_secondary is None else spike_secondary - baseline_secondary
        delta_magnitude = None if baseline_magnitude is None or spike_magnitude is None else spike_magnitude - baseline_magnitude
        rows.append({
            "name": name,
            "baseline_metric": baseline_metric,
            "spike_metric": spike_metric,
            "delta_metric": delta_metric,
            "baseline_secondary": baseline_secondary,
            "spike_secondary": spike_secondary,
            "delta_secondary": delta_secondary,
            "baseline_magnitude": baseline_magnitude,
            "spike_magnitude": spike_magnitude,
            "delta_magnitude": delta_magnitude,
        })
    return rows


def _format_num(value, digits: int = 3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _markdown_table(rows, section: str, top_k: int):
    spec = SECTION_SPECS[section]
    if not rows:
        return "_No data._"

    top_baseline = sorted(
        [row for row in rows if row["baseline_metric"] is not None],
        key=lambda row: row["baseline_metric"],
        reverse=True,
    )[:top_k]
    top_delta = sorted(
        [row for row in rows if row["delta_metric"] is not None],
        key=lambda row: row["delta_metric"],
        reverse=True,
    )[:top_k]

    def render(title, selected_rows):
        lines = [f"**{title}**", "", "| Item | Pre-spike | Spike | Delta | Pre-spike Kurtosis | Spike Kurtosis | Delta Kurtosis |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in selected_rows:
            lines.append(
                "| "
                + row["name"]
                + " | "
                + _format_num(row["baseline_metric"])
                + " | "
                + _format_num(row["spike_metric"])
                + " | "
                + _format_num(row["delta_metric"])
                + " | "
                + _format_num(row["baseline_secondary"])
                + " | "
                + _format_num(row["spike_secondary"])
                + " | "
                + _format_num(row["delta_secondary"])
                + " |"
            )
        if len(selected_rows) == 0:
            lines.append("| _none_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    return "\n\n".join([
        render(f"Top {top_k} Pre-spike {spec['label']} Outliers", top_baseline),
        render(f"Top {top_k} Spike-vs-Pre Delta ({spec['label']})", top_delta),
    ])


def _overview(report, spike_step: int, baseline_steps: list[int]):
    checkpoints = _index_checkpoints(report)
    lines = [
        "# Outlier Report Summary",
        "",
        f"- Run: `{report['run_name']}`",
        f"- Split: `{report['split']}`",
        f"- Batches: `{report['batch_indices']}`",
        f"- Steps analyzed: `{report['steps']}`",
        f"- Pre-spike baseline steps: `{baseline_steps}`",
        f"- Spike step: `{spike_step}`",
        "",
        "| Step | Batch Loss Mean | Batch Loss Min | Batch Loss Max |",
        "|---|---:|---:|---:|",
    ]
    for step in report["steps"]:
        ck = checkpoints[step]
        lines.append(
            f"| {step} | {_format_num(ck.get('batch_loss_mean'))} | "
            f"{_format_num(ck.get('batch_loss_min'))} | {_format_num(ck.get('batch_loss_max'))} |"
        )
    return "\n".join(lines)


def build_markdown(report, spike_step: int, baseline_steps: list[int], top_k: int):
    parts = [_overview(report, spike_step, baseline_steps)]
    for section in ("activations", "attention", "weights"):
        has_section = any(section in ck for ck in report["checkpoints"])
        if not has_section:
            continue
        rows = _build_section_rows(report, section, spike_step, baseline_steps)
        parts.extend([
            "",
            f"## {SECTION_SPECS[section]['label']}",
            "",
            _markdown_table(rows, section, top_k),
        ])
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Format an outlier analysis report into a presentable markdown summary.")
    parser.add_argument("--report", required=True, help="Path to outlier_analysis_report.json")
    parser.add_argument("--output", default=None, help="Where to write the markdown summary")
    parser.add_argument("--spike-step", type=int, default=None, help="Override the step treated as the spike step")
    parser.add_argument(
        "--baseline-steps",
        type=int,
        nargs="*",
        default=None,
        help="Override which steps are averaged into the pre-spike baseline",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Rows to show per table")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = _load_report(report_path)
    spike_step = _resolve_spike_step(report, args.spike_step)
    baseline_steps = _resolve_baseline_steps(report, spike_step, args.baseline_steps)
    markdown = build_markdown(report, spike_step, baseline_steps, args.top_k)

    output_path = Path(args.output) if args.output else report_path.with_suffix(".summary.md")
    output_path.write_text(markdown)
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
