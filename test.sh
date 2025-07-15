# CUDA_VISIBLE_DEVICES=0 \
#     GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/qddetr.sh \
#     --coco_path /data/nvme8/zhangbilang/datasets/coco2017 \
#     --eval \
#     --resume /data/nvme8/zhangbilang/qhdetr/exps/qddetr/checkpoint0001.pth \
#     --batch_size 1

CUDA_VISIBLE_DEVICES=0 ./configs/qddetr.sh \
    --coco_path /data/nvme8/zhangbilang/datasets/coco2017 \
    --eval \
    --resume /data/nvme8/zhangbilang/qhdetr/exps/qhdetr/checkpoint.pth \
    --batch_size 1
# CUDA_VISIBLE_DEVICES=0 \
#     GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/ddetr.sh \
#     --coco_path /data/nvme8/zhangbilang/datasets/coco2017 \
#     --eval \
#     --resume /data/nvme8/zhangbilang/qhdetr/exps/qddetr/checkpoint0001.pth \
#     --batch_size 1