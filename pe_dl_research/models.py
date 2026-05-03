"""
Models:
- AttentionResidualUNet3d : segmentation (binary PE mask)
- EfficientNet classifier : PE vs no-PE (slice or pooled-volume level)
- Optional fusion hook (segmentation encoder embedding projected + classifier)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate3d(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Conv3d(f_g, f_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv3d(f_l, f_int, kernel_size=1, bias=True)
        self.psi = nn.Conv3d(f_int, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        q = self.relu(g1 + x1)
        q = self.sig(self.psi(q))
        return x * q


class ResidualConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class AttentionResidualUNet3d(nn.Module):
    """
    Attention + residual blocks for 3D PE segmentation.
    Dropout in bottleneck for regularization (reduce overfitting / FP noise).
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base: int = 16,
        dropout_p: float = 0.15,
    ):
        super().__init__()
        self.enc1 = ResidualConv3d(in_ch, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualConv3d(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResidualConv3d(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ResidualConv3d(base * 4, base * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bot = ResidualConv3d(base * 8, base * 16)
        self.drop = nn.Dropout3d(dropout_p)

        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate3d(base * 8, base * 8, base * 4)
        self.dec4 = ResidualConv3d(base * 16, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate3d(base * 4, base * 4, base * 2)
        self.dec3 = ResidualConv3d(base * 8, base * 4)

        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate3d(base * 2, base * 2, base)
        self.dec2 = ResidualConv3d(base * 4, base * 2)

        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate3d(base, base, base // 2)
        self.dec1 = ResidualConv3d(base * 2, base)

        self.out_conv = nn.Conv3d(base, out_ch, kernel_size=1)

        self.embed_dim = base * 16  # bottleneck channels for optional fusion

    def encode_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Global pooled bottleneck embedding for fusion classifier."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.drop(self.bot(self.pool4(e4)))
        return F.adaptive_avg_pool3d(b, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.drop(self.bot(self.pool4(e4)))

        d4 = self.up4(b)
        e4_c = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_c], dim=1))

        d3 = self.up3(d4)
        e3_c = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_c], dim=1))

        d2 = self.up2(d3)
        e2_c = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_c], dim=1))

        d1 = self.up1(d2)
        e1_c = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_c], dim=1))

        return self.out_conv(d1)


class EfficientNetPEClassifier(nn.Module):
    """2D EfficientNet-B0 backbone; input 3ch (repeat grayscale slice)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0

            try:
                from torchvision.models import EfficientNet_B0_Weights

                w = EfficientNet_B0_Weights.IMAGENET1K_V1
                self.backbone = efficientnet_b0(weights=w)
            except Exception:
                self.backbone = efficientnet_b0(weights=None)
        except ImportError as e:
            raise ImportError("torchvision required for classification model") from e
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class HybridSegCls(nn.Module):
    """
    Concatenate segmentation bottleneck embedding (projected) with EfficientNet features.
    Expects pre-sized tensors from training loop (two forward branches).
    """

    def __init__(self, seg_embed_dim: int = 256, eff_embed_dim: int = 1280, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.proj_seg = nn.Sequential(
            nn.Linear(seg_embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.proj_eff = nn.Sequential(
            nn.Linear(eff_embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, seg_vec: torch.Tensor, eff_vec: torch.Tensor) -> torch.Tensor:
        a = self.proj_seg(seg_vec)
        b = self.proj_eff(eff_vec)
        return self.head(torch.cat([a, b], dim=1))


def build_segmentation_model(device: torch.device) -> AttentionResidualUNet3d:
    m = AttentionResidualUNet3d(in_ch=1, out_ch=1, base=16, dropout_p=0.15)
    return m.to(device)


def build_classifier(device: torch.device) -> EfficientNetPEClassifier:
    return EfficientNetPEClassifier(num_classes=2, dropout=0.3).to(device)
