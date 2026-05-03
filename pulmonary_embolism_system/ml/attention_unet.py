"""
Attention U-Net style architecture for 2D CT slice segmentation.
Replace or extend to match your trained TSNet + Attention U-Net weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Conv2d(f_g, f_int, 1, bias=True)
        self.w_x = nn.Conv2d(f_l, f_int, 1, bias=True)
        self.psi = nn.Conv2d(f_int, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionUNet2D(nn.Module):
    """2-channel in (optional), 1-channel binary mask out — matches common PE slice setup."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(base, base * 2)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base * 2, base * 4)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base * 4, base * 8)

        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att2 = AttentionBlock(base * 4, base * 4, base * 2)
        self.up_conv2 = DoubleConv(base * 8, base * 4)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att1 = AttentionBlock(base * 2, base * 2, base)
        self.up_conv1 = DoubleConv(base * 4, base * 2)

        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att0 = AttentionBlock(base, base, base // 2)
        self.up_conv0 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.conv1(self.down1(x0))
        x2 = self.conv2(self.down2(x1))
        x3 = self.conv3(self.down3(x2))

        u2 = self.up2(x3)
        x2a = self.att2(u2, x2)
        u2 = self.up_conv2(torch.cat([u2, x2a], dim=1))

        u1 = self.up1(u2)
        x1a = self.att1(u1, x1)
        u1 = self.up_conv1(torch.cat([u1, x1a], dim=1))

        u0 = self.up0(u1)
        x0a = self.att0(u0, x0)
        u0 = self.up_conv0(torch.cat([u0, x0a], dim=1))

        return self.outc(u0)


def build_model(in_channels: int = 1, weights_path: str | None = None, device: str = "cpu"):
    """Load Attention U-Net; returns (model, loaded_ok)."""
    device_t = torch.device(device)
    model = AttentionUNet2D(in_channels=in_channels, out_channels=1, base=32).to(device_t)
    loaded = False
    if weights_path:
        import os

        if os.path.isfile(weights_path):
            try:
                state = torch.load(weights_path, map_location=device_t)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                    state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                loaded = True
            except Exception:
                loaded = False
    model.eval()
    return model, loaded
