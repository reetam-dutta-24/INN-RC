# inn_model.py
# Invertible network used in INN-RC (Haar DWT + affine coupling blocks).
# The same layers are re-implemented in MATLAB (matlab/inn_forward.m, inn_inverse.m),
# so keep the two in sync if you change anything here.

import torch
import torch.nn as nn
import torch.nn.functional as F


def dwt_haar(x):
    # x: (B,1,H,W) -> (B,4,H/2,W/2)  [LL, HL, LH, HH], orthonormal Haar
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) / 2
    hl = (-a + b - c + d) / 2
    lh = (-a - b + c + d) / 2
    hh = (a - b - c + d) / 2
    return torch.cat([ll, hl, lh, hh], dim=1)


def idwt_haar(y):
    ll, hl, lh, hh = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    a = (ll - hl - lh + hh) / 2
    b = (ll + hl - lh - hh) / 2
    c = (ll - hl + lh - hh) / 2
    d = (ll + hl + lh + hh) / 2
    B, _, h, w = ll.shape
    x = torch.zeros(B, 1, 2 * h, 2 * w, dtype=y.dtype, device=y.device)
    x[:, :, 0::2, 0::2] = a
    x[:, :, 0::2, 1::2] = b
    x[:, :, 1::2, 0::2] = c
    x[:, :, 1::2, 1::2] = d
    return x


class SubNet(nn.Module):
    # small 3-layer CNN, last layer starts at zero so every block starts as identity
    def __init__(self, cin, cout, hid=32):
        super().__init__()
        self.c1 = nn.Conv2d(cin, hid, 3, padding=1)
        self.c2 = nn.Conv2d(hid, hid, 3, padding=1)
        self.c3 = nn.Conv2d(hid, cout, 3, padding=1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.c2(x), 0.2)
        return self.c3(x)


class CouplingBlock(nn.Module):
    # y1 = x1 + phi(x2)
    # y2 = x2 * exp(s(y1)) + eta(y1),   s = clamp * tanh(rho(y1))
    def __init__(self, c1, c2, hid=32, clamp=2.0):
        super().__init__()
        self.phi = SubNet(c2, c1, hid)
        self.rho = SubNet(c1, c2, hid)
        self.eta = SubNet(c1, c2, hid)
        self.clamp = clamp

    def scale(self, y1):
        return self.clamp * torch.tanh(self.rho(y1))

    def forward(self, x1, x2):
        y1 = x1 + self.phi(x2)
        y2 = x2 * torch.exp(self.scale(y1)) + self.eta(y1)
        return y1, y2

    def inverse(self, y1, y2):
        x2 = (y2 - self.eta(y1)) * torch.exp(-self.scale(y1))
        x1 = y1 - self.phi(x2)
        return x1, x2


class INN(nn.Module):
    def __init__(self, msg_ch=1, n_blocks=4, hid=32):
        super().__init__()
        self.msg_ch = msg_ch
        self.blocks = nn.ModuleList([CouplingBlock(4, msg_ch, hid) for _ in range(n_blocks)])

    def embed(self, cover, msg):
        # cover (B,1,H,W) in [0,1], msg (B,msg_ch,H/2,W/2) in {-0.5,+0.5}
        x1, x2 = dwt_haar(cover), msg
        for blk in self.blocks:
            x1, x2 = blk(x1, x2)
        return idwt_haar(x1), x2          # stego (float), lost latent z

    def reveal(self, stego, z=None):
        y1 = dwt_haar(stego)
        y2 = torch.zeros(y1.shape[0], self.msg_ch, *y1.shape[2:], dtype=y1.dtype) if z is None else z
        for blk in reversed(self.blocks):
            y1, y2 = blk.inverse(y1, y2)
        return idwt_haar(y1), y2          # recovered cover estimate, recovered message
