#!/bin/bash

CUDA_VISIBLE_DEVICES='3' python main_qm9.py \
    --output-dir 'models/qm9/equiformer/se_l2/target@5/' \
    --model-name 'graph_attention_transformer_nonlinear_bessel_l2_drop01' \
    --input-irreps '5x0e' \
    --target 5 \
    --data-path 'data/qm9' \
    --feature-type 'one_hot' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 8 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp \
    --ssp
