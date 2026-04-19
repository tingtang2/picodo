#!/usr/bin/env bash
# Unembed-only eps sweep: global eps stays at 1e-8, only token_embed_out
# gets a different eps via opt.unembed_eps. Tests whether the spike
# mitigation is localized to the lm_head (rules out "eps is just smaller LR").
#
# Per Aditya's suggestion: 1e-8 (control, no split), 1e-5, 1e-4.

set -uo pipefail

cd "$(dirname "$0")"
source ../env-loss-spikes/bin/activate

#UNEMBED_EPS_VALUES=(1e-8 1e-5 1e-4)
UNEMBED_EPS_VALUES=(1e-6 1e-7)
FREQ_PATH="${HOME}/datasets/finewebedu_gpt2_freqs.npy"

for ueps in "${UNEMBED_EPS_VALUES[@]}"; do
    run_name="gpt2s-unembed_eps${ueps}-rest1e-8-lr0.17"
    echo "============================================================"
    echo "[sweep] starting run: ${run_name}  (unembed eps=${ueps}, rest eps=1e-8)"
    echo "============================================================"

    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fwedu_gpt2 \
        seed=0 \
        opt.peak_lr=0.17 \
        opt.batch_size=256 \
        opt.use_z_loss=False \
        opt.eps=1e-8 \
        opt.unembed_eps="${ueps}" \
        checkpoint.turn_on=false \
        wu_diagnostics.enabled=true \
        v_t_diagnostics.enabled=true \
        v_t_diagnostics.freq_path="${FREQ_PATH}" \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${run_name}" \
    || echo "=== unembed_eps=${ueps} FAILED, continuing ==="

    echo "=== Finished unembed_eps=${ueps} ==="
done

echo "[sweep] all runs complete."
