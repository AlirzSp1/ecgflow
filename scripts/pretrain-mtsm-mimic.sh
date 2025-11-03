#!/bin/bash
# Pretrain with ssl
EXPHOME=~/ecgflow/experiments/
DATAROOT=~/ecgflow/data/mimic-iv-ecg
DATANAME=mimic
EXPNAME=mtsm-p50-d12-h8
MODELNAME=mtsm_base_patch50
$(dirname "$0")/train.py \
    --data-dir $DATAROOT \
    --ssl \
    --ssl-mask-ratio 0.6 \
    --dataset ecgflow/$DATANAME \
    --no-aug --no-prefetcher \
    --model $MODELNAME \
    --model-kwargs img_size=5000 \
    --input-size 1 8 5000 \
    --num-classes -1 \
    --opt adamw --epochs 300 --bce-loss --smoothing 0.01 \
    --sched cosine --lr 1e-3 --lr-k-decay 7 --sched-on-updates \
    --batch-size 384 \
    --workers 12 \
    --checkpoint-hist 3 \
    --output $EXPHOME/$DATANAME \
    --experiment $EXPNAME
