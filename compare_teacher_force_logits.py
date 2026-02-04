"""
Utility to compare saved teacher-forced logits between two checkpoints.

Expected input files (produced by inference_teacher_force.py):
  {diagnostics_dir}/logits_step_{step}_teacher_force_checkpoint_{ckpt}.npy
where each file stores a logits array with shape [1, V] or [B, V].
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with a scipy fallback; ranks with numpy if scipy is unavailable."""
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception:
        spearmanr = None

    if spearmanr is not None:
        corr, _ = spearmanr(x, y)
        return float(corr)

    # Numpy-only fallback: rank via argsort-of-argsort (ties get arbitrary ordering).
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    num = np.sum((rx - rx_mean) * (ry - ry_mean))
    den = np.sqrt(np.sum((rx - rx_mean) ** 2) * np.sum((ry - ry_mean) ** 2))
    return float(num / den) if den != 0 else 0.0


def load_logits(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    # Flatten to 1D logits vector.
    return arr.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="Compare teacher-forced logits between checkpoints.")
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        required=True,
        help="Directory containing logits_step_*_teacher_force_checkpoint_*.npy files.",
    )
    parser.add_argument("--checkpoint-a", type=int, required=True, help="First checkpoint number, e.g. 1293.")
    parser.add_argument("--checkpoint-b", type=int, required=True, help="Second checkpoint number, e.g. 1500.")
    parser.add_argument("--step-start", type=int, default=0, help="First decoding step (inclusive).")
    parser.add_argument("--step-end", type=int, default=0, help="Last decoding step (inclusive).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save scatter plots. Defaults to {diagnostics-dir}/spearman_scatter_{a}_vs_{b}.",
    )
    args = parser.parse_args()

    diag_dir: Path = args.diagnostics_dir
    diag_dir.mkdir(parents=True, exist_ok=True)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = diag_dir / f"spearman_scatter_{args.checkpoint_a}_vs_{args.checkpoint_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    corrs = []
    print(f"Comparing checkpoints {args.checkpoint_a} vs {args.checkpoint_b} for steps {args.step_start}-{args.step_end}")
    for step in range(args.step_start, args.step_end + 1):
        file_a = diag_dir / f"logits_step_{step}_teacher_force_checkpoint_{args.checkpoint_a}.npy"
        file_b = diag_dir / f"logits_step_{step}_teacher_force_checkpoint_{args.checkpoint_b}.npy"
        if not file_a.exists() or not file_b.exists():
            print(f"Skipping step {step}: missing file(s) {file_a if not file_a.exists() else ''} {file_b if not file_b.exists() else ''}")
            continue

        logits_a = load_logits(file_a)
        logits_b = load_logits(file_b)
        corr = spearman_corr(logits_a, logits_b)
        corrs.append((step, corr))
        print(f"Step {step + 1}: spearman={corr:.4f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(logits_a, logits_b, s=4, alpha=0.5)
        plt.xlabel(f"Checkpoint {args.checkpoint_a} logits")
        plt.ylabel(f"Checkpoint {args.checkpoint_b} logits")
        plt.title(f"Step {step + 1} Spearman {corr:.4f}")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_step_{step}.png")
        plt.close()

    if not corrs:
        print("No overlapping steps compared (files missing?).")
        return

    avg_corr = sum(c for _, c in corrs) / len(corrs)
    print(f"\nAverage Spearman across {len(corrs)} steps: {avg_corr:.4f}")


if __name__ == "__main__":
    main()
