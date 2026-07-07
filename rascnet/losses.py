"""RASC-Net auxiliary losses.

Implements the manuscript objectives:

* Anatomical contrastive loss  L_ana^(i)  (Eq. 6) -- InfoNCE that pulls
  anatomical features of the *same subject across modalities* together and
  pushes different subjects apart.
* Modality contrastive loss    L_mod^(i)  (Eq. 8) -- InfoNCE that pulls style
  features of the *same modality across subjects* together and pushes different
  modalities apart.
* Feature reconstruction loss  L_rec^(i)  (Eq. 10) -- L1 between F and F_hat.
* Subset consistency loss      L_cons     (Eq. 19) -- L2 between the pooled fused
  embeddings of two random modality subsets of the same subject.

The multi-scale disentanglement objective L_dis aggregates the first three over
scales i in {3,4} with weights lambda_ana / lambda_mod / lambda_rec (Eq. 11).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _info_nce(anchor, positive, negatives, tau):
    """Single-positive InfoNCE.

    anchor, positive : (D,)
    negatives        : (N, D)
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    pos = torch.exp((anchor * positive).sum(-1) / tau)
    neg = torch.exp((negatives @ anchor) / tau).sum()
    return -torch.log(pos / (pos + neg + 1e-8) + 1e-8)


class AnatomicalContrastiveLoss(nn.Module):
    """Eq. (6): align anatomy across modalities of the same subject.

    Input `anatomy_vecs`: (B, M, D) global-pooled anatomical features.
    Positives: same subject, different modality.
    Negatives: different subjects (any modality).
    """

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, anatomy_vecs):
        B, M, D = anatomy_vecs.shape
        if B < 2:
            return anatomy_vecs.sum() * 0.0
        flat = anatomy_vecs.reshape(B * M, D)
        subj_id = torch.arange(B, device=flat.device).repeat_interleave(M)
        loss = 0.0
        count = 0
        for idx in range(B * M):
            b = subj_id[idx].item()
            # positives: same subject, other modality
            pos_mask = (subj_id == b)
            pos_mask[idx] = False
            neg_mask = (subj_id != b)
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            anchor = flat[idx]
            # average over available positives
            for p in torch.where(pos_mask)[0]:
                loss = loss + _info_nce(anchor, flat[p], flat[neg_mask], self.tau)
                count += 1
        return loss / max(count, 1)


class ModalityContrastiveLoss(nn.Module):
    """Eq. (8): align style within a modality across subjects.

    Input `style_vecs`: (B, M, D) global style features.
    Positives: same modality, different subject.
    Negatives: different modalities.
    """

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, style_vecs):
        B, M, D = style_vecs.shape
        if B < 2:
            return style_vecs.sum() * 0.0
        flat = style_vecs.reshape(B * M, D)
        mod_id = torch.arange(M, device=flat.device).repeat(B)
        loss = 0.0
        count = 0
        for idx in range(B * M):
            m = mod_id[idx].item()
            pos_mask = (mod_id == m)
            pos_mask[idx] = False
            neg_mask = (mod_id != m)
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            anchor = flat[idx]
            for p in torch.where(pos_mask)[0]:
                loss = loss + _info_nce(anchor, flat[p], flat[neg_mask], self.tau)
                count += 1
        return loss / max(count, 1)


def reconstruction_loss(feat, recon):
    """Eq. (10): L1 feature reconstruction."""
    return F.l1_loss(recon, feat)


def subset_consistency_loss(z1, z2):
    """Eq. (19): squared L2 between two pooled subset embeddings.

    z1, z2 : (B, C) pooled fused representations of two modality subsets.
    """
    return ((z1 - z2) ** 2).sum(dim=1).mean()


class DisentanglementObjective(nn.Module):
    """Aggregates L_ana / L_mod / L_rec over scales (Eq. 11)."""

    def __init__(self, lambda_ana=1.0, lambda_mod=1.0, lambda_rec=1.0, tau=0.1):
        super().__init__()
        self.lambda_ana = lambda_ana
        self.lambda_mod = lambda_mod
        self.lambda_rec = lambda_rec
        self.ana_loss = AnatomicalContrastiveLoss(tau=tau)
        self.mod_loss = ModalityContrastiveLoss(tau=tau)

    def forward(self, anatomy_vecs_by_scale, style_vecs_by_scale,
                feats_by_scale, recon_by_scale):
        """Each argument is a dict {scale: tensor}. Anatomy/style vecs are
        (B, M, D); feats/recon are (B, M, C, H, W, Z)."""
        total = 0.0
        logs = {}
        for s in anatomy_vecs_by_scale.keys():
            l_ana = self.ana_loss(anatomy_vecs_by_scale[s])
            l_mod = self.mod_loss(style_vecs_by_scale[s])
            l_rec = reconstruction_loss(feats_by_scale[s], recon_by_scale[s])
            total = total + (self.lambda_ana * l_ana
                             + self.lambda_mod * l_mod
                             + self.lambda_rec * l_rec)
            logs[f'ana_{s}'] = float(l_ana)
            logs[f'mod_{s}'] = float(l_mod)
            logs[f'rec_{s}'] = float(l_rec)
        return total, logs
