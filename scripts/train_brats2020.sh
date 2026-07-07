#!/usr/bin/env bash
# Train RASC-Net on BraTS 2020 (all 15 missing-modality combinations).
# Edit DATAPATH / SAVEPATH to your environment before running.

export CUDA_VISIBLE_DEVICES=0

DATAPATH=${DATAPATH:-/path/to/BRATS2020_Training_none_npy}
SAVEPATH=${SAVEPATH:-output/rascnet_brats2020}

python train.py \
    --dataname BRATS2020 \
    --batch_size 2 \
    --num_epochs 500 \
    --iter_per_epoch -1 \
    --region_fusion_start_epoch 20 \
    --crop_size 112 \
    --lr 2e-4 \
    --alpha 0.5 \
    --beta 0.4 \
    --lambda_ana 1.0 \
    --lambda_mod 1.0 \
    --lambda_rec 1.0 \
    --tau 0.1 \
    --use_reg_loss \
    --datapath "$DATAPATH" \
    --savepath "$SAVEPATH"
