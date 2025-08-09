#!/usr/bin/env bash

set -x
# EXP_DIR=exps/ddetr_n300
EXP_DIR=exps/w32a32/ddetr_n1800
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries_one2one 1800 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 12 \
    --lr_drop 11 \
    ${PY_ARGS}
