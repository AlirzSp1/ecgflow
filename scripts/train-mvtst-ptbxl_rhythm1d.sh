#!/bin/bash
# Train from scratch
EXPHOME=~/ecgflow/experiments
DATAROOT=~/ecgflow/data/ptb-xl
DATANAME=ptbxl_rhythm
EXPNAME=mvtst-p50-d12-h8-1
MODELNAME=mvtst_base_patch50
SEED=20301153
$(dirname "$0")/train.py \
    --seed $SEED \
    --data-dir $DATAROOT \
    --early-stop 10 \
    --dataset ecgflow/$DATANAME \
    --no-aug --no-prefetcher \
    --model $MODELNAME \
    --model-kwargs img_size=5000 \
    --input-size 1 8 5000 \
    --num-classes 12 \
    --warmup-epochs 5 \
    --opt adamw --epochs 100 --bce-loss --smoothing 0.001 \
    --sched cosine --lr 1e-3 --lr-k-decay 1 --sched-on-updates \
    --weight-decay 0 \
    --batch-size 256 \
    --eval-metric AUROC \
    --workers 12 \
    --checkpoint-hist 3 \
    --output $EXPHOME/$DATANAME \
    --experiment $EXPNAME
