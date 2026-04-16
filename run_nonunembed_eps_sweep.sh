#!/usr/bin/env bash
# Negative-control sweep: raise Adam eps on every parameter EXCEPT the
# unembedding W_U. W_U stays at opt.eps=1e-8 (the baseline small value).
# If spikes still occur, localizes the mechanism to W_U specifically
# (confirms the shift-invariance story). If spikes disappear, refutes it.
#
# Includes wu_diagnostics + v_t evidence so mu_W/norm growth can be
# compared across runs.

set -uo pipefail

cd "$(dirname "$0")"
source ../env-loss-spikes/bin/activate

NONUNEMBED_EPS_VALUES=(1e-6 1e-4 1e-2)
FREQ_PATH="${HOME}/datasets/fineweb_gpt2_freqs.npy"

for eps in "${NONUNEMBED_EPS_VALUES[@]}"; do
    run_name="gpt2s-nonunembed_eps${eps}-lr0.17"
    echo "============================================================"
    echo "[sweep] starting run: ${run_name}  (nonunembed_eps=${eps}, W_U stays at 1e-8)"
    echo "============================================================"

    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        seed=0 \
        opt.peak_lr=0.17 \
        opt.batch_size=256 \
        opt.use_z_loss=False \
        opt.eps=1e-8 \
        opt.nonunembed_eps="${eps}" \
        checkpoint.turn_on=false \
        wu_diagnostics.enabled=true \
        v_t_diagnostics.enabled=true \
        v_t_diagnostics.freq_path="${FREQ_PATH}" \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${run_name}" \
    || echo "=== nonunembed_eps=${eps} FAILED, continuing ==="

    echo "=== Finished nonunembed_eps=${eps} ==="
done

echo "[sweep] all 3 runs complete."
