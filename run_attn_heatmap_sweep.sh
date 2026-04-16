#!/usr/bin/env bash
# Attention heatmap sweep: 3 runs with global eps in {1e-8, 1e-4, 1e-2}.
# Logs per-layer [T, T] attention weight heatmaps to W&B every 100 steps,
# plus wu_diagnostics. Spike-inducing config: gpt2s + fw_gpt2 at lr=0.17.

set -uo pipefail

cd "$(dirname "$0")"
source ../env-loss-spikes/bin/activate

EPS_VALUES=(1e-8 1e-4 1e-2)

for eps in "${EPS_VALUES[@]}"; do
    run_name="gpt2s-attn_heatmap-eps${eps}-lr0.17"
    echo "============================================================"
    echo "[sweep] starting run: ${run_name}  (eps=${eps})"
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
        attention_heatmaps.enabled=true \
        attention_heatmaps.every_n_steps=100 \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${run_name}" \
    || echo "=== eps=${eps} FAILED, continuing ==="

    echo "=== Finished eps=${eps} ==="
done

echo "[sweep] all 3 runs complete."
