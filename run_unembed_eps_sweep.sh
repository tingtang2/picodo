#!/usr/bin/env bash
# Sweeps token_embed_out.embedding (W_U) Adam epsilon over {1e-6, 1e-4, 1e-2}
# while keeping every other parameter on c.opt.eps (1e-8 in wortsman_default).
# Runs in series, foreground; first failure halts the sweep.

set -euo pipefail

cd "$(dirname "$0")"

EPS_VALUES=(1e-6 1e-4 1e-2)

for eps in "${EPS_VALUES[@]}"; do
    run_name="unembed_eps_sweep_eps_${eps}"
    echo "============================================================"
    echo "[sweep] starting run: ${run_name}  (unembed_eps=${eps})"
    echo "============================================================"

    python main.py \
        --config-name=wortsman_default \
        model=gpt2s \
        dataset=fw_gpt2 \
        seed=0 \
        wandb_mode=online \
        run_name="${run_name}" \
        opt.unembed_eps="${eps}" \
        checkpoint.turn_on=false \
        cosine.enabled=true \
        hessian.enabled=true \
        hessian.k=5 \
        hessian.every_n_steps=50 \
        wu_diagnostics.enabled=true
done

echo "[sweep] all runs complete."
