#!/usr/bin/env bash
# Sweeps token_embed_out.embedding (W_U) Adam epsilon over {1e-6, 1e-4, 1e-2}
# while keeping every other parameter on c.opt.eps (1e-8 in wortsman_default).
# Spike-inducing config: peak_lr=0.17, use_z_loss=False (mirrors prior
# gpt2m-v6e16 sweep at lr=0.12 that exhibited spikes; gpt2s + higher lr
# expected to spike similarly under small global eps).
# Runs in series, foreground; failures log and continue to the next eps.

set -uo pipefail

cd "$(dirname "$0")"
source ../env-loss-spikes/bin/activate

UNEMBED_EPS_VALUES=(1e-6 1e-4 1e-2)

for eps in "${UNEMBED_EPS_VALUES[@]}"; do
    run_name="gpt2s-unembed_eps${eps}-lr0.17"
    echo "============================================================"
    echo "[sweep] starting run: ${run_name}  (unembed_eps=${eps})"
    echo "============================================================"

    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        seed=0 \
        opt.peak_lr=0.17 \
        opt.batch_size=256 \
        opt.use_z_loss=False \
        opt.unembed_eps="${eps}" \
        checkpoint.turn_on=false \
        cosine.enabled=true \
        hessian.enabled=true \
        hessian.k=5 \
        hessian.every_n_steps=50 \
        wu_diagnostics.enabled=true \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${run_name}" \
    || echo "=== unembed_eps=${eps} FAILED, continuing ==="

    echo "=== Finished unembed_eps=${eps} ==="
done

echo "[sweep] all runs complete."
