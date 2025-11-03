#!/bin/bash
# Fine-tune
EXPHOME=~/ecgflow/experiments
PRETRAINED_IN=mimic
PRETRAINED_EXP=mtsm-p50-d12-h8
DATAROOT=~/ecgflow/data/ptb-xl
DATANAME=ptbxl_diag
EXPNAME=mvtst-p50-d12-h8.mimic-1
MODELNAME=mvtst_base_patch50
SEED=20241153
$(dirname "$0")/train.py \
    --seed $SEED \
    --data-dir $DATAROOT \
    --early-stop 10 \
    --ft-top 4 \
    --pretrained \
    --pretrained-path $EXPHOME/$PRETRAINED_IN/$PRETRAINED_EXP/last.pth.tar \
    --dataset ecgflow/$DATANAME \
    --no-aug --no-prefetcher \
    --model $MODELNAME \
    --model-kwargs img_size=5000 \
    --input-size 1 8 5000 \
    --num-classes 44 \
    --warmup-epochs 5 \
    --opt adamw --epochs 100 --bce-loss --smoothing 0.001 \
    --sched cosine --lr 5e-5 --lr-k-decay 1 --sched-on-updates \
    --weight-decay 0 \
    --batch-size 256 \
    --eval-metric AUROC \
    --workers 12 \
    --checkpoint-hist 3 \
    --output $EXPHOME/$DATANAME \
    --experiment $EXPNAME
