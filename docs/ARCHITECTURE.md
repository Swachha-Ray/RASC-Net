# Architecture & Code Map

This document maps each RASC-Net component to the manuscript sections and
equations, and to the file that implements it.

## Pipeline overview (Fig. 2)

```
Input X (B,4,H,W,Z)
   │
   ▼  rascnet/encoder.py :: Encoder  (shared multi-scale encoder, per modality)
{F^(1), F^(2), F^(3), F^(4)}  per modality
   │
   ▼  rascnet/disentangle.py :: MultiScaleDisentangler   (Sec. III-B, i∈{3,4})
(A_m^(i), S_m^(i)) + reconstruction F̂_m^(i)
   │
   ▼  rascnet/reliability.py :: ReliabilityAwareGating   (Sec. III-C, Eqs. 12-14)
Ã_m^(i) = (δ_m · q_m) · A_m^(i)
   │
   ▼  rascnet/fusion.py :: StabilityConstrainedFusion     (Sec. III-D, Eqs. 15-17)
Z^(i) = Σ_m w_m^(i) · Ã_m^(i)
   │
   ▼  rascnet/encoder.py :: Decoder_fuse (region-aware) + low-level skips
Ŷ = D(Z^(4), Z^(3), F_skip^(2), F_skip^(1))              (Eq. 20)
```

## Component ↔ equation ↔ file

| Component | Manuscript | File / class |
|-----------|------------|--------------|
| Shared multi-scale encoder | Sec. III-A, Eqs. 1-2 | `encoder.py :: Encoder` |
| Anatomy/style projection heads (1×1×1) | Sec. III-B, Eqs. 3-4 | `disentangle.py :: ProjectionHead`, `ScaleDisentangler` |
| Feature reconstruction R^(i) | Eq. 9 | `disentangle.py :: ReconHead` |
| Anatomical contrastive loss L_ana^(i) | Eq. 6 | `losses.py :: AnatomicalContrastiveLoss` |
| Modality contrastive loss L_mod^(i) | Eq. 8 | `losses.py :: ModalityContrastiveLoss` |
| Reconstruction loss L_rec^(i) | Eq. 10 | `losses.py :: reconstruction_loss` |
| Disentanglement objective L_dis | Eq. 11 | `losses.py :: DisentanglementObjective` |
| Reliability estimator q_m | Eqs. 12-13 | `reliability.py :: ReliabilityEstimator` |
| Gated feature modulation | Eq. 14 | `reliability.py :: ReliabilityAwareGating` |
| Spatial attention Ψ_m^(i) | Eq. 15 | `fusion.py :: StabilityConstrainedFusion.attn` |
| Availability-masked fusion weights | Eqs. 16-17 | `fusion.py :: StabilityConstrainedFusion.forward` |
| Subset consistency L_cons | Eqs. 18-19 | `losses.py :: subset_consistency_loss`, `model.py :: _fused_embedding` |
| Segmentation decoder | Eq. 20 | `encoder.py :: Decoder_fuse` |
| Segmentation loss (Dice + CE) | Eq. 21 | `utils/criterions.py` |
| Joint objective L_total | Eq. 22 | `train.py` |

## Design notes / deviations from DC-Seg

RASC-Net is built on the DC-Seg (MICCAI 2025) code base. The key changes:

1. **Multi-scale disentanglement.** DC-Seg disentangles anatomy/style only at
   the deepest stage via VAE-style style encoders. RASC-Net adds lightweight
   1×1×1 anatomy/style projection heads and a reconstruction head at *both*
   scales i∈{3,4}, with hierarchical InfoNCE contrastive objectives
   (`losses.py`). This matches the manuscript's InfoNCE formulation (Eqs. 6, 8)
   rather than DC-Seg's BCE-style similarity target.

2. **Reliability-aware gating (new).** A per-modality scalar quality score q_m
   gates each anatomical feature before fusion (Eq. 14). This module has no
   counterpart in DC-Seg.

3. **Stability-constrained fusion (new).** Replaces uniform / attention fusion
   with an availability-masked spatial-attention softmax (Eqs. 15-17), so
   missing modalities are provably excluded and per-voxel weights sum to one
   over the available set.

4. **Subset consistency (new).** Two random modality subsets of the same
   subject are encoded/gated/fused, and their pooled embeddings are pulled
   together (Eq. 19), enforcing invariance across arbitrary subsets rather than
   only across full/single modalities.

5. **Removed code.** DC-Seg's WMH (white-matter-hyperintensity) branch,
   VAE product-of-experts / KL machinery, and unused loss variants have been
   dropped to keep the release focused on the BraTS incomplete-modality task
   described in the manuscript.

The region-aware modal fusion (RFM) segmentation back-end from RFNet/DC-Seg is
retained as the decoder, with the reliability-gated features replacing the raw
high-level features at scales 3 and 4.

## Inference contract

With `model.is_training = False`, `forward(x, mask)` returns **only** the fused
softmax prediction, so the model is a drop-in for the sliding-window evaluator
in `predict.py`.
