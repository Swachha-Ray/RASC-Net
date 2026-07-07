"""RASC-Net full model (manuscript Fig. 2 pipeline).

Flow:
  1. Shared multi-scale Encoder per modality -> {F^(1..4)}.
  2. Multi-scale anatomy/style disentanglement at scales {3,4} (Sec. III-B).
  3. Reliability-aware anatomical gating on the anatomical features (Sec. III-C).
  4. Stability-constrained fusion -> refined per-modality anatomy that replaces
     the raw high-level features fed to the region-aware segmentation decoder
     (Sec. III-D). The gated modality features are stacked and passed to the
     RFM-based decoder together with the low-level skip features.
  5. During training, two random modality subsets are additionally encoded /
     gated / fused to compute the subset-consistency embeddings (Eqs. 18-19).

Design choice: the reliability gate re-weights each modality's anatomical
feature map *before* it enters the region-aware fusion decoder. This keeps the
proven RFNet/DC-Seg segmentation back-end while injecting the manuscript's
reliability and stability mechanisms. Missing modalities (delta_m = 0) are gated
to zero, so the decoder's own availability mask and the gate agree.

`is_training = False` returns only `fuse_pred` so the model is a drop-in for the
sliding-window evaluator in predict.py.
"""
import random

import torch
import torch.nn as nn

from .encoder import Encoder, Decoder_sep, Decoder_fuse, basic_dims
from .disentangle import MultiScaleDisentangler
from .reliability import ReliabilityAwareGating
from .fusion import StabilityConstrainedFusion

# BraTS modality order used throughout: [FLAIR, T1ce, T1, T2]
NUM_MODAL = 4
DISENTANGLE_SCALES = (3, 4)   # manuscript i in {3,4}
# encoder channel widths by scale index
SCALE_CH = {1: basic_dims, 2: basic_dims * 2, 3: basic_dims * 4, 4: basic_dims * 8}


class RASC_Net(nn.Module):
    def __init__(self, num_cls=4, disentangle_scales=DISENTANGLE_SCALES,
                 use_gating=True, use_fusion=True):
        super().__init__()
        self.num_cls = num_cls
        self.num_modal = NUM_MODAL
        self.disentangle_scales = tuple(disentangle_scales)
        self.use_gating = use_gating
        self.use_fusion = use_fusion

        # ---- shared encoders (one weight set per modality stream) ----
        self.encoders = nn.ModuleList([Encoder() for _ in range(NUM_MODAL)])

        # ---- multi-scale disentanglement (shared across modalities) ----
        feat_ch = {s: SCALE_CH[s] for s in self.disentangle_scales}
        self.disentangler = MultiScaleDisentangler(feat_ch)

        # ---- reliability gating + stability fusion, one per disentangled scale ----
        self.gating = nn.ModuleDict({
            str(s): ReliabilityAwareGating(SCALE_CH[s]) for s in self.disentangle_scales
        })
        self.fusion = nn.ModuleDict({
            str(s): StabilityConstrainedFusion(SCALE_CH[s]) for s in self.disentangle_scales
        })

        # ---- segmentation back-end ----
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls, num_modal=NUM_MODAL)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    # ------------------------------------------------------------------ #
    def encode_all(self, x):
        """Encode every modality. Returns list over modalities of (x1,x2,x3,x4)."""
        feats = []
        for m in range(self.num_modal):
            feats.append(self.encoders[m](x[:, m:m + 1, :, :, :]))
        return feats

    def _gate_and_refine(self, feats, mask):
        """Apply disentanglement + gating on the high scales.

        Returns:
            refined : dict {scale: (B, M, C, H, W, Z)} gated anatomical features,
                      used as decoder inputs (replacing raw features at scales 3,4).
            aux     : dict of intermediate tensors for losses (train only).
        """
        # gather per-scale, per-modality features into {scale: [feat_m,...]}
        per_scale = {s: [feats[m][s - 1] for m in range(self.num_modal)]
                     for s in self.disentangle_scales}

        refined = {}
        aux = {'anatomy_vec': {}, 'style_vec': {}, 'feat': {}, 'recon': {},
               'q': {}, 'fused': {}}

        for s in self.disentangle_scales:
            anatomy_list, style_map_list, style_vec_list, recon_list = [], [], [], []
            for m in range(self.num_modal):
                a, _sm, sv, r = self.disentangler.disentanglers[str(s)](per_scale[s][m])
                anatomy_list.append(a)
                style_vec_list.append(sv)
                recon_list.append(r)

            # reliability-aware gating
            if self.use_gating:
                gated_list, q = self.gating[str(s)](anatomy_list, mask)
            else:
                gated_list, q = anatomy_list, None

            refined[s] = torch.stack(gated_list, dim=1)  # (B, M, C, H, W, Z)

            if self.is_training:
                B = per_scale[s][0].size(0)
                aux['anatomy_vec'][s] = torch.stack(
                    [torch.mean(a, dim=(2, 3, 4)) for a in anatomy_list], dim=1)  # (B,M,D)
                aux['style_vec'][s] = torch.stack(style_vec_list, dim=1)          # (B,M,D)
                aux['feat'][s] = torch.stack(per_scale[s], dim=1)                 # (B,M,C,...)
                aux['recon'][s] = torch.stack(recon_list, dim=1)
                aux['q'][s] = q
        return refined, aux

    def _fused_embedding(self, feats, mask):
        """Compute a pooled fused embedding at the deepest scale for a subset
        (used by subset consistency, Eqs. 18-19)."""
        s = max(self.disentangle_scales)
        anatomy_list = []
        for m in range(self.num_modal):
            a, _sm, _sv, _r = self.disentangler.disentanglers[str(s)](feats[m][s - 1])
            anatomy_list.append(a)
        if self.use_gating:
            gated_list, _ = self.gating[str(s)](anatomy_list, mask)
        else:
            gated_list = anatomy_list
        if self.use_fusion:
            fused, _ = self.fusion[str(s)](gated_list, mask)
        else:
            fused = torch.stack(gated_list, dim=1).mean(1)
        return torch.mean(fused, dim=(2, 3, 4))  # (B, C)

    # ------------------------------------------------------------------ #
    def forward(self, x, mask, subset_masks=None):
        """
        x            : (B, 4, H, W, Z)
        mask         : (B, 4) boolean modality availability
        subset_masks : optional tuple (mask_s1, mask_s2) for subset consistency;
                       if None during training, two random subsets are sampled.
        """
        feats = self.encode_all(x)  # list_m of (x1,x2,x3,x4)

        # stack raw features per scale for the decoder
        x1 = torch.stack([feats[m][0] for m in range(self.num_modal)], dim=1)
        x2 = torch.stack([feats[m][1] for m in range(self.num_modal)], dim=1)
        x3 = torch.stack([feats[m][2] for m in range(self.num_modal)], dim=1)
        x4 = torch.stack([feats[m][3] for m in range(self.num_modal)], dim=1)

        # disentangle + gate high scales, replacing x3/x4 with refined anatomy
        refined, aux = self._gate_and_refine(feats, mask)
        if 3 in refined:
            x3 = refined[3]
        if 4 in refined:
            x4 = refined[4]

        fuse_pred, prm_preds, _fusion_x4 = self.decoder_fuse(x1, x2, x3, x4, mask)

        if not self.is_training:
            return fuse_pred

        # ---- auxiliary training outputs ----
        sep_preds = tuple(
            self.decoder_sep(feats[m][0], feats[m][1], feats[m][2], feats[m][3])
            for m in range(self.num_modal)
        )

        # subset consistency embeddings (Eqs. 18-19)
        if subset_masks is None:
            subset_masks = self._sample_subset_masks(mask)
        m_s1, m_s2 = subset_masks
        z_s1 = self._fused_embedding(feats, m_s1)
        z_s2 = self._fused_embedding(feats, m_s2)

        return {
            'fuse_pred': fuse_pred,
            'sep_preds': sep_preds,
            'prm_preds': prm_preds,
            'anatomy_vec': aux['anatomy_vec'],
            'style_vec': aux['style_vec'],
            'feat': aux['feat'],
            'recon': aux['recon'],
            'q': aux['q'],
            'z_s1': z_s1,
            'z_s2': z_s2,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_subset_masks(mask):
        """Sample two random, non-empty modality subsets of the available set.

        Each subset is a boolean (B, M) tensor that is a subset of `mask`.
        """
        B, M = mask.shape
        device = mask.device

        def sample_one():
            out = torch.zeros_like(mask)
            for b in range(B):
                avail = torch.where(mask[b])[0].tolist()
                if len(avail) == 0:
                    continue
                k = random.randint(1, len(avail))
                chosen = random.sample(avail, k)
                out[b, chosen] = True
            return out

        s1 = sample_one()
        s2 = sample_one()
        return s1.to(device), s2.to(device)


if __name__ == '__main__':
    model = RASC_Net(num_cls=4)
    model.is_training = True
    x = torch.randn(2, 4, 64, 64, 64)
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
    out = model(x, mask)
    print('fuse_pred', out['fuse_pred'].shape)
    print('sep_preds', [p.shape for p in out['sep_preds']])
    print('z_s1', out['z_s1'].shape, 'z_s2', out['z_s2'].shape)
    model.is_training = False
    print('infer', model(x, mask).shape)
