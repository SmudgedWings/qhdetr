#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 \

# CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/ddetr.sh \
#     --dataset_file voc \
#     --coco_path /data/nvme7/zhangbilang/datasets/VOCdevkit \
#     --batch_size 1 \
#     --resume ./exps/w32a32/hdetr/checkpoint0011.pth \
#     --resume_weight_only \
#     --load_q_RN50 \
#     --eval \
#     --quant  \

# CUDA_VISIBLE_DEVICES=0 ./configs/ddetr.sh \
#     --dataset_file voc \
#     --coco_path /data/nvme7/zhangbilang/datasets/VOCdevkit \
#     --batch_size 1 \
#     --resume ./exps/w32a32/hdetr/checkpoint0011.pth \
#     --resume_weight_only \
#     --load_q_RN50 \
#     --quant  \

CUDA_VISIBLE_DEVICES=0 ./configs/ddetr.sh \
    --dataset_file voc \
    --coco_path /data/nvme7/zhangbilang/datasets/VOCdevkit \
    --batch_size 1 \