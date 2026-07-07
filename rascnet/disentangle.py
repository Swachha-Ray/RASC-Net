"""Multi-scale anatomy-style disentanglement (manuscript Sec. III-B).

For each modality m and each selected scale i in {3,4} we map the encoder
feature F_m^(i) through two lightweight 1x1x1 projection heads into an
anatomical feature A_m^(i) and a style feature S_m^(i) (Eq. 4):

    A_m^(i) = phi_a^(i)(F_m^(i)),   S_m^(i) = phi_s^(i)(F_m^(i)).

A lightweight reconstruction head R^(i) rebuilds F_m^(i) from (A, S) so the
decomposition stays information-preserving (Eqs. 9-10).

The 1x1x1 convolutions re-weight / linearly project channels without mixing
spatial dimensions, which keeps the anatomical branch spatially aligned for the
downstream reliability gating and segmentation decoder.
"""
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """1x1x1 conv projection (channel re-weighting, no spatial mixing)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.proj(x)


class ReconHead(nn.Module):
    """Lightweight reconstruction R^(i): (A, S) -> F_hat (Eq. 9)."""

    def __init__(self, ana_ch, style_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(ana_ch + style_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, anatomy, style):
        # style is a channel vector broadcast over the spatial grid of anatomy
        if style.dim() == 2:
            style = style[:, :, None, None, None]
        style = style.expand(-1, -1, *anatomy.shape[2:])
        return self.fuse(torch.cat((anatomy, style), dim=1))


class ScaleDisentangler(nn.Module):
    """Anatomy / style / recon heads for a single scale."""

    def __init__(self, feat_ch, ana_ch=None, style_ch=None):
        super().__init__()
        ana_ch = ana_ch or feat_ch
        style_ch = style_ch or feat_ch
        self.anatomy_head = ProjectionHead(feat_ch, ana_ch)
        self.style_head = ProjectionHead(feat_ch, style_ch)
        self.recon_head = ReconHead(ana_ch, style_ch, feat_ch)

    def forward(self, feat):
        anatomy = self.anatomy_head(feat)                       # A_m^(i)
        style_map = self.style_head(feat)                       # S_m^(i) (spatial)
        style_vec = torch.mean(style_map, dim=(2, 3, 4))        # \tilde S_m^(i) (Eq. 7)
        recon = self.recon_head(anatomy, style_vec)             # F_hat_m^(i) (Eq. 9)
        return anatomy, style_map, style_vec, recon


class MultiScaleDisentangler(nn.Module):
    """Runs disentanglement at each requested scale for one modality stream.

    Expects a dict {scale_index: feature_tensor}. Returns dicts keyed by scale.
    """

    def __init__(self, feat_channels, ana_channels=None, style_channels=None):
        """feat_channels: dict {scale_idx: channels}."""
        super().__init__()
        ana_channels = ana_channels or {}
        style_channels = style_channels or {}
        self.scales = sorted(feat_channels.keys())
        self.disentanglers = nn.ModuleDict({
            str(s): ScaleDisentangler(
                feat_channels[s],
                ana_ch=ana_channels.get(s),
                style_ch=style_channels.get(s),
            ) for s in self.scales
        })

    def forward(self, feats):
        anatomy, style_map, style_vec, recon = {}, {}, {}, {}
        for s in self.scales:
            a, sm, sv, r = self.disentanglers[str(s)](feats[s])
            anatomy[s], style_map[s], style_vec[s], recon[s] = a, sm, sv, r
        return anatomy, style_map, style_vec, recon
