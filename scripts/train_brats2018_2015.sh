#!/usr/bin/env bash
# Train RASC-Net on BraTS 2018 (split 3) or BraTS 2015.
# Usage: bash scripts/train_brats2018_2015.sh [BRATS2018|BRATS2015]

export CUDA_VISIBLE_DEVICES=0

DATANAME=${1:-BRATS2018}
DATAPATH=${DATAPATH:-/path/to/${DATANAME}_Training_none_npy}
SAVEPATH=${SAVEPATH:-output/rascnet_${DATANAME,,}}

python train.py \
    --dataname "$DATANAME" \
    --batch_size 2 \
    --num_epochs 500 \
    --iter_per_epoch -1 \
    --region_fusion_start_epoch 20 \
    --crop_size 112 \
    --lr 2e-4 \
    --alpha 0.5 \
    --beta 0.4 \
    --tau 0.1 \
    --use_reg_loss \
    --datapath "$DATAPATH" \
    --savepath "$SAVEPATH"
