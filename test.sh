CUDA_VISIBLE_DEVICES=0 \
    GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/qhdetr.sh \
    --coco_path /data/zhangbilang/datasets/coco2017 \
    --eval \
    --resume ./LIEM/checkpoint0003.pth \
    --batch_size 1