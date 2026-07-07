#!/usr/bin/env bash
# Evaluate a trained RASC-Net checkpoint across all 15 missing-modality
# combinations on the test set.
# Usage: bash scripts/evaluate.sh /path/to/model_last.pth [BRATS2020]

export CUDA_VISIBLE_DEVICES=0

WEIGHTS=${1:?"provide checkpoint path"}
DATANAME=${2:-BRATS2020}
DATAPATH=${DATAPATH:-/path/to/${DATANAME}_Training_none_npy}
SAVEPATH=${SAVEPATH:-output/eval_${DATANAME,,}}

python train.py \
    --resume "$WEIGHTS" \
    --dataname "$DATANAME" \
    --datapath "$DATAPATH" \
    --savepath "$SAVEPATH" \
    --crop_size 112
