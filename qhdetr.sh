#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 \

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/qhdetr.sh \
    --dataset_file voc \
    --coco_path /data/nvme7/zhangbilang/datasets/VOCdevkit \
    --batch_size 1 \
    --resume ./exps/w32a32/hdetr/checkpoint0011.pth \
    --resume_weight_only \
    --load_q_RN50 \
#    --eval
# --resume ./checkpoint_437.pth 
# --resume ./exps/12eps/one_stage/exp09/checkpoint0011.pth
# --resume ./pretrain/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth

# CUDA_VISIBLE_DEVICES=1 \
#     GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/qhdetr.sh \
#     --dataset_file DroneVehicle \
#     --coco_path /data/code/zbl/datasets/DroneVehicle/DroneVehicle_coco \
#     --resume ./exps/32bit/r50_hybrid_branch_deformable_detr_DV/checkpoint0011.pth \
#     --resume_weight_only \
#     --load_q_RN50 \
#     --batch_size 1

# CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/qhdetr.sh \