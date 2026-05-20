#!/usr/bin/env bash
# U-curve LR sensitivity sweep.
#
# ACTIVE condition (pivot — Ting's z-loss-on-attn-logits as a QK-Norm replacement):
#   nothing_attnz = QK-Norm OFF, lm_head AdamW (trunk LR), qkv AdamW, attn z-loss ON, LM-head z-loss OFF
#
# Direct analog of the earlier "nothing_qkv*" experiments but with z-loss
# (a soft magnitude penalty) instead of qkv-SGD-M as the QK-Norm replacement.
#
# Trunk LR is swept; everything else fixed.

set -uo pipefail

cd "$(dirname "$0")"

#LRS=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)
#LRS=(1e-4 3e-4 1e-3 3e-3)
LRS=(1e-4 3e-4)

SEED=0

for lr in "${LRS[@]}"; do
    echo "===================== LR=$lr ====================="
    
    '''
    name_method="ucurve_method_lr${lr}-s${SEED}"
    echo "--- METHOD: ${name_method} ---"
    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        seed=${SEED} \
        opt.peak_lr=${lr} \
        opt.batch_size=256 \
        model.use_qk_norm=false \
        opt.lm_head_optimizer.type=sgd_momentum \
        opt.lm_head_optimizer.peak_lr=3e-1 \
        opt.lm_head_optimizer.momentum=0.99 \
        opt.use_qkv_opt=true \
        opt.qkv_optimizer.type=sgd_momentum \
        opt.qkv_optimizer.peak_lr=3e-1 \
        opt.qkv_optimizer.momentum=0.9 \
        checkpoint.turn_on=false \
        checkpoint.workdir=/home/sophieli/picodo/checkpoints \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${name_method}" \
    || echo "=== ${name_method} FAILED, continuing ==="

    name_sgd_head="ucurve_sgd_head_lr${lr}-s${SEED}"
    echo "--- SGD_HEAD: ${name_sgd_head} ---"
    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        seed=${SEED} \
        opt.peak_lr=${lr} \
        opt.batch_size=256 \
        model.use_qk_norm=true \
        opt.lm_head_optimizer.type=sgd_momentum \
        opt.lm_head_optimizer.peak_lr=3e-1 \
        opt.lm_head_optimizer.momentum=0.99 \
        opt.use_qkv_opt=false \
        checkpoint.turn_on=false \
        checkpoint.workdir=/home/sophieli/picodo/checkpoints \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${name_sgd_head}" \
    || echo "=== ${name_sgd_head} FAILED, continuing ==="

    name_baseline="ucurve_baseline_lr${lr}-s${SEED}"
    echo "--- BASELINE: ${name_baseline} ---"
    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        seed=${SEED} \
        opt.peak_lr=${lr} \
        opt.batch_size=256 \
        model.use_qk_norm=true \
        opt.lm_head_optimizer.type=adamw \
        opt.lm_head_optimizer.peak_lr=${lr} \
        opt.use_qkv_opt=false \
        checkpoint.turn_on=false \
        checkpoint.workdir=/home/sophieli/picodo/checkpoints \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${name_baseline}" \
    || echo "=== ${name_baseline} FAILED, continuing ==="
    '''

    name_nothing_attnz="ucurve_nothing_attnz_lr${lr}-s${SEED}"
    echo "--- NOTHING + ATTN Z-LOSS (QK-Norm OFF, lm_head AdamW, attn z-loss ON): ${name_nothing_attnz} ---"
    python3 main.py \
        --config-name=wortsman_default \
        +model=gpt2s +dataset=fw_gpt2 \
        seed=${SEED} \
        opt.peak_lr=${lr} \
        opt.batch_size=256 \
        model.use_qk_norm=false \
        model.use_attn_z_loss=true \
        opt.lm_head_optimizer.type=adamw \
        opt.lm_head_optimizer.peak_lr=${lr} \
        opt.use_qkv_opt=false \
        opt.use_z_loss=false \
        checkpoint.turn_on=false \
        checkpoint.workdir=/home/sophieli/picodo/checkpoints \
        log_metrics_per_step=true \
        wandb_mode=online \
        run_name="${name_nothing_attnz}" \
    || echo "=== ${name_nothing_attnz} FAILED, continuing ==="
done

echo "===================== SWEEP COMPLETE ====================="
