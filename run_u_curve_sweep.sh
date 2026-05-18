#!/usr/bin/env bash
# U-curve LR sensitivity sweep — QK-Norm removal experiment (todos #6 + #7).
#
# Three conditions per trunk LR, interleaved:
#   method   = QK-Norm OFF, lm_head SGD-M (3e-1, β=0.99), qkv SGD-M (3e-1, β=0.9)
#   sgd_head = QK-Norm ON,  lm_head SGD-M (3e-1, β=0.99), qkv AdamW (the prior method)
#   baseline = QK-Norm ON,  lm_head AdamW, qkv AdamW (pure AdamW everywhere)
#
# Ablation chain the three conditions let you read off:
#   baseline   vs sgd_head : does SGD-M-on-LM-head help?       (prior contribution)
#   sgd_head   vs method   : does removing QK-Norm + qkv SGD-M help on top?
#   baseline   vs method   : does the full new stack beat vanilla?
#
# Trunk LRs span 1e-5 .. 1e-1 to capture BOTH sides of the U.
# Interleaved ordering: each LR gets method THEN sgd_head THEN baseline.

set -uo pipefail

cd "$(dirname "$0")"

#LRS=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)
LRS=(1e-4 3e-4 1e-3 3e-3)

SEED=0

for lr in "${LRS[@]}"; do
    echo "===================== LR=$lr ====================="

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
done

echo "===================== SWEEP COMPLETE ====================="
