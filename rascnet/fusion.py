"""Stability-constrained fusion (manuscript Sec. III-D, Eqs. 15-17).

Given reliability-gated anatomical features A_gated_m, a lightweight conv maps
each to a spatial attention map (Eq. 15):

    Psi_m^(i) = phi(A_gated_m^(i)).

Availability-masked softmax over modalities produces per-voxel fusion weights so
only available modalities contribute (Eq. 16):

    w_m^(i) = exp(Psi_m) / sum_{k: delta_k=1} exp(Psi_k).

The fused representation is the weighted sum (Eq. 17):

    Z^(i) = sum_m w_m^(i) * A_gated_m^(i).
"""
import torch
import torch.nn as nn


class StabilityConstrainedFusion(nn.Module):
    """Attention-based, availability-masked fusion of gated anatomical features."""

    def __init__(self, ana_ch):
        super().__init__()
        # phi(.) : maps an anatomical feature to a 1-channel spatial attention logit
        self.attn = nn.Conv3d(ana_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, gated_list, mask):
        """
        gated_list : list of length M of tensors (B, C, H, W, Z)
        mask       : (B, M) boolean availability
        returns fused tensor (B, C, H, W, Z) and fusion weights (B, M, 1, H, W, Z)
        """
        B, C, H, W, Z = gated_list[0].shape
        M = len(gated_list)

        logits = torch.stack([self.attn(g) for g in gated_list], dim=1)  # (B, M, 1, H, W, Z)

        # availability mask -> set logits of missing modalities to -inf before softmax
        avail = mask.view(B, M, 1, 1, 1, 1).to(logits.dtype)
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = torch.where(avail > 0, logits, torch.full_like(logits, neg_inf))

        weights = torch.softmax(masked_logits, dim=1)                    # (B, M, 1, H, W, Z)
        # guard: if a sample had all-missing (should not happen), zero the weights
        weights = torch.nan_to_num(weights, nan=0.0)

        feats = torch.stack(gated_list, dim=1)                           # (B, M, C, H, W, Z)
        fused = torch.sum(weights * feats, dim=1)                        # (B, C, H, W, Z)
        return fused, weights
