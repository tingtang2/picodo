#!/usr/bin/env bash
# Global-eps sweep (eps applied to *all* parameters) with tier-1 v_t
# evidence logging turned on. Hypothesis-test: as eps grows, the fraction
# of W_U / W_E cells below eps shrinks, spike rate drops, and the
# rare-token bucket mean of sqrt(v_hat) converges toward the common
# buckets (protection regime shifts).
#
# Prereq on the TPU: run compute_token_frequencies.py once to produce
# ~/datasets/fineweb_gpt2_freqs.npy.
#
# Runs in series, foreground; failures log and continue.

set -uo pipefail

cd "$(dirname "$0")"
source ../env-loss-spikes/bin/activate

EPS_VALUES=(1e-8 1e-6 1e-4 1e-2)
FREQ_PATH="${HOME}/datasets/fineweb_gpt2_freqs.npy"

for eps in "${EPS_VALUES[@]}"; do
    run_name="gpt2s-v_t_evidence-eps${eps}-lr0.17"
    echo "============================================================"
    echo "[sweep] starting run: ${run_name}  (global eps=${eps})"
    echo "============================================================"

    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        seed=0 \
        opt.peak_lr=0.17 \
        opt.batch_size=256 \
        opt.use_z_loss=False \
        opt.eps="${eps}" \
        checkpoint.turn_on=false \
        wu_diagnostics.enabled=true \
        v_t_diagnostics.enabled=true \
        v_t_diagnostics.freq_path="${FREQ_PATH}" \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${run_name}" \
    || echo "=== eps=${eps} FAILED, continuing ==="

    echo "=== Finished eps=${eps} ==="
done

echo "[sweep] all runs complete."
