"""Shared multi-scale encoder, segmentation decoders and region-aware modal fusion.

The encoder / decoders follow the RFNet / DC-Seg lineage. The region-aware modal
fusion (RFM) and probability-map (prm) generators are reused as the segmentation
back-end so that the four low/high-level skip features feed the decoder, matching
Eq. (20) of the manuscript:

    Y_hat = D(Z^(4), Z^(3), F_skip^(2), F_skip^(1)).

Manuscript Sec. III-A: mid/high-level scales i in {3,4} are used for
disentanglement, while low-level scales i in {1,2} are preserved as skip
connections.
"""
import torch
import torch.nn as nn

from .blocks import general_conv3d

basic_dims = 16


class Encoder(nn.Module):
    """Per-modality shared multi-scale encoder producing four feature scales."""

    def __init__(self):
        super().__init__()
        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims * 2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims * 2, basic_dims * 2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims * 2, basic_dims * 2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims * 2, basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims * 8, basic_dims * 8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4


class Decoder_sep(nn.Module):
    """Per-modality auxiliary segmentation decoder (regularizer branch)."""

    def __init__(self, num_cls=4):
        super().__init__()
        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        return self.softmax(self.seg_layer(de_x1))


# --------------------------------------------------------------------------- #
# Region-aware modal fusion back-end (probability-map guided).
# This is the segmentation decoder that consumes the four skip scales.
# --------------------------------------------------------------------------- #
class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, num_cls=4, num_modal=4):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel // 4, k_size=1, padding=0, stride=1),
            general_conv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
            general_conv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1))
        self.prm_layer = nn.Sequential(
            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1))

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)
        return self.prm_layer(self.embedding_layer(y))


class prm_generator(nn.Module):
    def __init__(self, in_channel=64, num_cls=4, num_modal=4):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel // 4, k_size=1, padding=0, stride=1),
            general_conv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
            general_conv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1))
        self.prm_layer = nn.Sequential(
            general_conv3d(in_channel * 2, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1))

    def forward(self, x1, x2, mask):
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        y = y.view(B, -1, H, W, Z)
        return self.prm_layer(torch.cat((x1, self.embedding_layer(y)), dim=1))


class modal_fusion(nn.Module):
    def __init__(self, in_channel=64, num_modal=4):
        super().__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv3d(num_modal * in_channel + 1, 128, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(128, num_modal, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm):
        B, K, C, H, W, Z = x.size()
        prm_avg = torch.mean(prm, dim=(3, 4, 5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg
        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)
        return torch.sum(x * weight, dim=1)


class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, _, _, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)


class region_aware_modal_fusion(nn.Module):
    """Region-aware fusion of four modality feature maps guided by a prob map."""

    def __init__(self, in_channel=64, num_cls=4, num_modal=4):
        super().__init__()
        self.num_cls = num_cls
        self.num_modal = num_modal
        self.modal_fusion = nn.ModuleList([modal_fusion(in_channel=in_channel, num_modal=num_modal)
                                           for _ in range(num_cls)])
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
            general_conv3d(in_channel * num_modal, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1))

    def forward(self, x, prm, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        modal_feat = torch.stack([y[:, m:m + 1, ...] * prm for m in range(self.num_modal)], dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        region_fused_feat = []
        for i in range(self.num_cls):
            region_fused_feat.append(self.modal_fusion[i](region_feat[i], prm[:, i:i + 1, ...]))
        region_fused_feat = torch.stack(region_fused_feat, dim=1)

        final_feat = torch.cat((self.region_fusion(region_fused_feat),
                                self.short_cut(y.view(B, -1, H, W, Z))), dim=1)
        return final_feat


class Decoder_fuse(nn.Module):
    """Segmentation decoder that fuses multi-scale skip features (Eq. 20)."""

    def __init__(self, num_cls=4, num_modal=4):
        super().__init__()
        self.d3_c1 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims * 8, num_cls=num_cls, num_modal=num_modal)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims * 4, num_cls=num_cls, num_modal=num_modal)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims * 2, num_cls=num_cls, num_modal=num_modal)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims * 1, num_cls=num_cls, num_modal=num_modal)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims * 8, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator3 = prm_generator(in_channel=basic_dims * 4, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator2 = prm_generator(in_channel=basic_dims * 2, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator1 = prm_generator(in_channel=basic_dims * 1, num_cls=num_cls, num_modal=num_modal)

    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        fusion_x4 = de_x4
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        pred = self.softmax(self.seg_layer(de_x1))
        prm_preds = (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4))
        return pred, prm_preds, fusion_x4
