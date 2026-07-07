#!/usr/bin/env bash
# Reproduces the component ablation in the manuscript (Table IV):
#   MSD  = multi-scale disentanglement (L_dis)
#   RAG  = reliability-aware gating
#   SCL  = subset consistency learning
#
# Baseline (all off) approximates DC-Seg; each row adds one component.
# Edit DATAPATH before running.

export CUDA_VISIBLE_DEVICES=0
DATAPATH=${DATAPATH:-/path/to/BRATS2020_Training_none_npy}
COMMON="--dataname BRATS2020 --datapath $DATAPATH --crop_size 112 --num_epochs 500 --use_reg_loss"

# Row 1: baseline (no MSD, no RAG, no SCL)
python train.py $COMMON --no_disentangle_loss --no_gating --no_fusion --no_consistency \
    --savepath output/ablation/baseline

# Row 2: + MSD
python train.py $COMMON --no_gating --no_fusion --no_consistency \
    --savepath output/ablation/msd

# Row 3: + MSD + RAG
python train.py $COMMON --no_consistency \
    --savepath output/ablation/msd_rag

# Row 4: + MSD + RAG + SCL (full RASC-Net)
python train.py $COMMON \
    --savepath output/ablation/msd_rag_scl
