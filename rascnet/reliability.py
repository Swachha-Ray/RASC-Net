"""Reliability-aware anatomical gating (manuscript Sec. III-C, Eqs. 12-14).

For each modality the anatomical feature A_m^(i) is pooled (GAP), passed through
an MLP Q(.) and a sigmoid to yield a scalar reliability score q_m in [0,1]:

    A_tilde_m = GAP(A_m^(i))                                   (Eq. 12)
    q_m       = sigma(Q(A_tilde_m))                            (Eq. 13)

The gated anatomical feature multiplies A by the availability indicator delta_m
and the reliability q_m (Eq. 14):

    A_gated_m = (delta_m * q_m) * A_m^(i).

delta_m = 0 for a missing modality guarantees no contribution; q_m lets the
network down-weight noisy / corrupted-but-present modalities.
"""
import torch
import torch.nn as nn


class ReliabilityEstimator(nn.Module):
    """MLP quality estimator Q(.) producing a scalar reliability per modality."""

    def __init__(self, in_ch, hidden_ch=None):
        super().__init__()
        hidden_ch = hidden_ch or max(in_ch // 4, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, 1),
        )

    def forward(self, anatomy):
        # anatomy: (B, C, H, W, Z) -> GAP -> (B, C)
        pooled = torch.mean(anatomy, dim=(2, 3, 4))
        score = self.mlp(pooled)               # (B, 1)
        return torch.sigmoid(score)            # q_m in [0,1]


class ReliabilityAwareGating(nn.Module):
    """Estimates q_m for every modality and applies availability-aware gating."""

    def __init__(self, ana_ch):
        super().__init__()
        self.estimator = ReliabilityEstimator(ana_ch)

    def forward(self, anatomy_list, mask):
        """
        anatomy_list : list of length M of tensors (B, C, H, W, Z)
        mask         : (B, M) boolean availability (delta_m)
        returns gated_list (same shapes) and q (B, M)
        """
        B = anatomy_list[0].size(0)
        M = len(anatomy_list)
        q = []
        gated = []
        for m in range(M):
            q_m = self.estimator(anatomy_list[m]).view(B)          # (B,)
            delta_m = mask[:, m].to(q_m.dtype)                     # (B,)
            gate = (delta_m * q_m).view(B, 1, 1, 1, 1)
            gated.append(gate * anatomy_list[m])
            q.append(q_m)
        q = torch.stack(q, dim=1)                                  # (B, M)
        return gated, q
