#!/bin/bash

CUDA_VISIBLE_DEVICES='1' python main_qm9.py \
    --output-dir 'models/qm9/equiformer/se_l2/target@4/v2/' \
    --model-name 'graph_attention_transformer_nonlinear_l2' \
    --input-irreps '5x0e' \
    --target 4 \
    --data-path 'data/qm9' \
    --feature-type 'one_hot' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp \
    --ssp
