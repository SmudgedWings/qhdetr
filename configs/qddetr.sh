#!/usr/bin/env bash

set -x

EXP_DIR=exps/qddetr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries_one2one 300 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 12 \
    --lr_drop 11 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --quant \
    ${PY_ARGS}