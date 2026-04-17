#!/usr/bin/env bash
# 12-run grid over Adam nu initialisation (v_init) x epsilon. Outer loop
# is v_init, inner is eps; runs back-to-back with v_t evidence + wu
# diagnostics enabled. Spike-inducing config: gpt2s + fw_gpt2 at lr=0.17,
# use_z_loss=False. Prereq: ~/datasets/fineweb_gpt2_freqs.npy produced by
# compute_token_frequencies.py.

set -uo pipefail

cd "$(dirname "$0")"
source ../env-loss-spikes/bin/activate

V_INIT_VALUES=(1e-6)
EPS_VALUES=(1e-8 1e-4 1e-2 1e-6)
FREQ_PATH="${HOME}/datasets/fineweb_gpt2_freqs.npy"

for v_init in "${V_INIT_VALUES[@]}"; do
    for eps in "${EPS_VALUES[@]}"; do
        run_name="gpt2s-v_init_experiment-v_init${v_init}-eps${eps}-lr0.17"
        echo "============================================================"
        echo "[sweep] starting run: ${run_name}"
        echo "[sweep]   v_init=${v_init}  eps=${eps}"
        echo "============================================================"

        python3 main.py \
            --config-name=wortsman_default \
            +model=gpt2s +dataset=fw_gpt2 \
            seed=0 \
            opt.peak_lr=0.17 \
            opt.batch_size=256 \
            opt.use_z_loss=False \
            opt.eps="${eps}" \
            opt.v_init="${v_init}" \
            checkpoint.turn_on=false \
            wu_diagnostics.enabled=true \
            v_t_diagnostics.enabled=true \
            v_t_diagnostics.freq_path="${FREQ_PATH}" \
            log_metrics_per_step=true \
            wandb_mode=online \
            run_name="${run_name}" \
        || echo "=== v_init=${v_init} eps=${eps} FAILED, continuing ==="

        echo "=== Finished v_init=${v_init} eps=${eps} ==="
    done
done

echo "[sweep] all 12 runs complete."
