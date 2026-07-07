"""Low-level 3D building blocks.

Adapted from the DC-Seg reference implementation
(https://github.com/CuCl-2/DC-Seg, MICCAI 2025) with unused variants removed.
Only the blocks that RASC-Net actually uses are kept here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization(planes, norm='in'):
    if norm == 'bn':
        return nn.BatchNorm3d(planes)
    if norm == 'gn':
        return nn.GroupNorm(4, planes)
    if norm == 'in':
        return nn.InstanceNorm3d(planes)
    raise ValueError('normalization type {} is not supported'.format(norm))


class general_conv3d(nn.Module):
    """Conv3d -> Norm -> Activation."""

    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1,
                 pad_type='reflect', norm='in', act_type='lrelu', relufactor=0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            raise ValueError('act type {} is not supported'.format(act_type))

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class BasicConv(nn.Module):
    """Conv3d with optional dropout / norm / relu (used by style encoder & decoder)."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, norm=True, bias=False, drop_rate=0.0):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.drop = nn.Dropout3d(p=drop_rate) if drop_rate != 0 else None
        self.norm = nn.InstanceNorm3d(out_planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.drop is not None:
            x = self.drop(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
