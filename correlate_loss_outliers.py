from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_loss_report_paths(explicit_paths: list[str], glob_patterns: list[str]):
    seen = set()
    for raw_path in explicit_paths:
        path = Path(raw_path)
        if path.exists() and path not in seen:
            seen.add(path)
            yield path
    for pattern in glob_patterns:
        for raw_path in sorted(glob.glob(pattern, recursive=True)):
            path = Path(raw_path)
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _extract_loss_events(loss_report: dict, source_path: Path):
    checkpoint_step = int(loss_report["checkpoint_step"])
    base_events = []
    for event in loss_report.get("top_token_events", []):
        base_events.append({
            "checkpoint_step": checkpoint_step,
            "source": "top_token_events",
            "split": str(loss_report.get("split", "unknown")),
            "report_path": str(source_path),
            **event,
        })

    next_train = loss_report.get("next_train_batch_analysis")
    if isinstance(next_train, dict):
        for event in next_train.get("top_token_events", []):
            base_events.append({
                "checkpoint_step": checkpoint_step,
                "source": "next_train_batch_analysis",
                "split": "train",
                "report_path": str(source_path),
                **event,
            })
    return base_events


def _find_per_batch_outliers(checkpoint_report: dict, batch_index: int):
    for batch_report in checkpoint_report.get("per_batch", []):
        if int(batch_report.get("batch_index", -1)) == int(batch_index):
            return batch_report
    return None


def _match_activation_outliers(section: dict, example_index: int, position: int):
    matches = []
    for name, stats in section.items():
        if not isinstance(stats, dict):
            continue
        indices = stats.get("top_indices", stats.get("representative_top_indices", []))
        hit_dims = []
        for idx in indices:
            if len(idx) >= 3 and int(idx[0]) == int(example_index) and int(idx[1]) == int(position):
                hit_dims.append(int(idx[2]))
        if hit_dims:
            matches.append({
                "item": name,
                "outliers_pct": float(stats.get("outliers_pct", 0.0)),
                "kurtosis_mean": float(stats.get("kurtosis_mean", 0.0)),
                "matched_dims": sorted(set(hit_dims)),
            })
    matches.sort(key=lambda row: row["outliers_pct"], reverse=True)
    return matches


def _match_attention_outliers(section: dict, example_index: int, position: int):
    matches = []
    for name, stats in section.items():
        if not isinstance(stats, dict):
            continue
        indices = stats.get("top_indices", stats.get("representative_top_indices", []))
        hit_heads = []
        for idx in indices:
            if len(idx) >= 3 and int(idx[0]) == int(example_index) and int(idx[2]) == int(position):
                hit_heads.append(int(idx[1]))
        if hit_heads:
            matches.append({
                "item": name,
                "outliers_pct": float(stats.get("outliers_pct", 0.0)),
                "kurtosis_mean": float(stats.get("kurtosis_mean", 0.0)),
                "matched_heads": sorted(set(hit_heads)),
            })
    matches.sort(key=lambda row: row["outliers_pct"], reverse=True)
    return matches


def _match_embedding_weight_outliers(weights_section: dict, input_token_id: int, target_token_id: int):
    matches = []
    embedding_specs = [
        ("token_embed_in.embedding", "input_token_id", int(input_token_id)),
        ("token_embed_out.embedding", "target_token_id", int(target_token_id)),
    ]
    for name, token_kind, token_id in embedding_specs:
        stats = weights_section.get(name)
        if not isinstance(stats, dict):
            continue
        hit_features = []
        for idx in stats.get("top_indices", []):
            if len(idx) >= 2 and int(idx[0]) == token_id:
                hit_features.append(int(idx[1]))
        if hit_features:
            matches.append({
                "item": name,
                "token_kind": token_kind,
                "token_id": token_id,
                "outliers_pct_mean": float(stats.get("outliers_pct_mean", 0.0)),
                "kurtosis_mean": float(stats.get("kurtosis_mean", 0.0)),
                "matched_features": sorted(set(hit_features)),
            })
    return matches


def _summarize_correlations(correlations: list[dict]):
    aligned_batch_modes = {"exact", "single_batch_aggregate"}
    summary = {
        "num_events": len(correlations),
        "events_with_activation_match": 0,
        "events_with_attention_match": 0,
        "events_with_embedding_weight_match": 0,
        "events_with_batch_aligned_activation_attention": 0,
        "events_missing_batch_alignment": 0,
        "batch_alignment_counts": {},
        "activation_match_counts_by_item": {},
        "attention_match_counts_by_item": {},
        "weight_match_counts_by_item": {},
    }
    activation_counter = Counter()
    attention_counter = Counter()
    weight_counter = Counter()
    alignment_counter = Counter()

    for row in correlations:
        batch_alignment = str(row.get("batch_alignment", "unknown"))
        if batch_alignment in aligned_batch_modes:
            summary["events_with_batch_aligned_activation_attention"] += 1
        else:
            summary["events_missing_batch_alignment"] += 1
        alignment_counter[batch_alignment] += 1
        if row["activation_matches"]:
            summary["events_with_activation_match"] += 1
        if row["attention_matches"]:
            summary["events_with_attention_match"] += 1
        if row["weight_matches"]:
            summary["events_with_embedding_weight_match"] += 1
        for match in row["activation_matches"]:
            activation_counter[match["item"]] += 1
        for match in row["attention_matches"]:
            attention_counter[match["item"]] += 1
        for match in row["weight_matches"]:
            weight_counter[match["item"]] += 1

    summary["batch_alignment_counts"] = dict(alignment_counter.most_common())
    summary["activation_match_counts_by_item"] = dict(activation_counter.most_common())
    summary["attention_match_counts_by_item"] = dict(attention_counter.most_common())
    summary["weight_match_counts_by_item"] = dict(weight_counter.most_common())
    return summary


def _build_markdown(outlier_report: dict, correlations: list[dict], summary: dict):
    lines = [
        "# Loss/Outlier Correlation Summary",
        "",
        f"- Run: `{outlier_report['run_name']}`",
        f"- Outlier steps: `{outlier_report['steps']}`",
        f"- Outlier split: `{outlier_report['split']}`",
        f"- Event count: `{summary['num_events']}`",
        f"- Events with batch-aligned activation/attention analysis: `{summary['events_with_batch_aligned_activation_attention']}`",
        f"- Events missing batch alignment for activation/attention: `{summary['events_missing_batch_alignment']}`",
        f"- Events with activation-position matches: `{summary['events_with_activation_match']}`",
        f"- Events with attention-key matches: `{summary['events_with_attention_match']}`",
        f"- Events with token-embedding weight matches: `{summary['events_with_embedding_weight_match']}`",
        "",
        "Batch alignment meanings:",
        "- `exact`: matched via `per_batch`.",
        "- `single_batch_aggregate`: matched from a report that only analyzed one batch.",
        "- `missing_per_batch`: multi-batch outlier report without `per_batch`, so activation/attention joins are unavailable.",
        "- `missing_checkpoint_step`: no outlier checkpoint entry for that loss event step.",
        "",
        "## Top Matched Events",
        "",
        "| Step | Source | Split | Rank | Loss | Batch | Ex | Pos | Input | Target | Alignment | Activation Matches | Attention Matches | Weight Matches |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    ranked = sorted(
        correlations,
        key=lambda row: (
            len(row["activation_matches"]) + len(row["attention_matches"]) + len(row["weight_matches"]),
            row.get("loss", 0.0),
        ),
        reverse=True,
    )
    for row in ranked[:20]:
        lines.append(
            f"| {row['checkpoint_step']} | {row['source']} | {row['split']} | {row['rank']} | "
            f"{row['loss']:.3f} | {row['batch_index']} | {row['example_index']} | {row['position']} | "
            f"{row['input_token_id']} | {row['target_token_id']} | {row['batch_alignment']} | {len(row['activation_matches'])} | "
            f"{len(row['attention_matches'])} | {len(row['weight_matches'])} |"
        )

    def append_counter_section(title: str, counter_map: dict[str, int]):
        lines.extend(["", f"## {title}", ""])
        if not counter_map:
            lines.append("_No matches._")
            return
        lines.extend([
            "| Item | Match Count |",
            "|---|---:|",
        ])
        for item, count in list(counter_map.items())[:20]:
            lines.append(f"| {item} | {count} |")

    append_counter_section("Activation Match Hotspots", summary["activation_match_counts_by_item"])
    append_counter_section("Attention Match Hotspots", summary["attention_match_counts_by_item"])
    append_counter_section("Embedding Weight Match Hotspots", summary["weight_match_counts_by_item"])
    return "\n".join(lines) + "\n"


def correlate_reports(outlier_report: dict, loss_reports: list[dict]):
    checkpoints_by_step = {int(ck["step"]): ck for ck in outlier_report["checkpoints"]}
    correlations = []

    for loss_report, source_path in loss_reports:
        for event in _extract_loss_events(loss_report, source_path):
            checkpoint_step = int(event["checkpoint_step"])
            checkpoint_report = checkpoints_by_step.get(checkpoint_step)
            if checkpoint_report is None:
                correlations.append({
                    **event,
                    "batch_alignment": "missing_checkpoint_step",
                    "activation_matches": [],
                    "attention_matches": [],
                    "weight_matches": [],
                })
                continue

            per_batch = _find_per_batch_outliers(checkpoint_report, int(event["batch_index"]))
            if per_batch is not None:
                activation_section = per_batch.get("activations", {})
                attention_section = per_batch.get("attention", {})
                batch_alignment = "exact"
            elif checkpoint_report.get("batch_indices") == [int(event["batch_index"])]:
                activation_section = checkpoint_report.get("activations", {})
                attention_section = checkpoint_report.get("attention", {})
                batch_alignment = "single_batch_aggregate"
            else:
                activation_section = {}
                attention_section = {}
                batch_alignment = "missing_per_batch"

            correlations.append({
                **event,
                "batch_alignment": batch_alignment,
                "activation_matches": _match_activation_outliers(
                    activation_section,
                    int(event["example_index"]),
                    int(event["position"]),
                ),
                "attention_matches": _match_attention_outliers(
                    attention_section,
                    int(event["example_index"]),
                    int(event["position"]),
                ),
                "weight_matches": _match_embedding_weight_outliers(
                    checkpoint_report.get("weights", {}),
                    int(event["input_token_id"]),
                    int(event["target_token_id"]),
                ),
            })
    return correlations


def main():
    parser = argparse.ArgumentParser(
        description="Correlate highest-loss token events from evaluate_checkpoint.py with outlier positions from analyze_outliers.py."
    )
    parser.add_argument("--outlier-report", required=True, help="Path to outlier_analysis_report.json")
    parser.add_argument(
        "--loss-report",
        action="append",
        default=[],
        help="Path to a loss_trigger_analysis_step_*_*.json report. Repeatable.",
    )
    parser.add_argument(
        "--loss-report-glob",
        action="append",
        default=[],
        help="Glob for one or more loss-trigger reports, e.g. 'picodo/**/loss_trigger_analysis_step_*_valid.json'. Repeatable.",
    )
    parser.add_argument("--output-json", default=None, help="Where to write the correlation JSON")
    parser.add_argument("--output-md", default=None, help="Where to write the markdown summary")
    args = parser.parse_args()

    outlier_report_path = Path(args.outlier_report)
    outlier_report = _load_json(outlier_report_path)
    loss_report_paths = list(_iter_loss_report_paths(args.loss_report, args.loss_report_glob))
    if not loss_report_paths:
        raise SystemExit("No loss-trigger reports found. Pass --loss-report or --loss-report-glob.")

    loss_reports = [( _load_json(path), path) for path in loss_report_paths]
    correlations = correlate_reports(outlier_report, loss_reports)
    summary = _summarize_correlations(correlations)

    payload = {
        "outlier_report": str(outlier_report_path),
        "loss_reports": [str(path) for _, path in loss_reports],
        "summary": summary,
        "event_correlations": correlations,
    }

    output_json = Path(args.output_json) if args.output_json else outlier_report_path.with_name(
        outlier_report_path.stem + ".loss_event_correlation.json"
    )
    output_md = Path(args.output_md) if args.output_md else outlier_report_path.with_name(
        outlier_report_path.stem + ".loss_event_correlation.md"
    )

    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(_build_markdown(outlier_report, correlations, summary))
    print(f"Wrote JSON correlation report to {output_json}")
    print(f"Wrote markdown summary to {output_md}")


if __name__ == "__main__":
    main()
