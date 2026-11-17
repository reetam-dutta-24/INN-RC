# icinn_model.py
# Integer message-conditioned coupling network (IC-INN) - proposed model.
#
# Cover x (integers 0..255) is split by a 2x2 space-to-depth into
#   xa = (top-left, bottom-right)  and  xb = (top-right, bottom-left).
# Coupling k (k = 1..K) updates one half from the other half and the message:
#   odd  k : xb <- xb + round(phi_k([xa/255, M]))
#   even k : xa <- xa + round(phi_k([xb/255, M]))
# Every step is an integer addition, so if the receiver knows M the cover is
# recovered EXACTLY by subtracting in reverse order (no float error at all).
# A separate CNN decoder reads the message back from the stego.

import torch
import torch.nn as nn
import torch.nn.functional as F


def s2d(x):     # (B,1,H,W) -> a,b,c,d each (B,1,H/2,W/2)
    return x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2], x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]


def d2s(a, b, c, d):
    B, _, h, w = a.shape
    x = torch.zeros(B, 1, 2 * h, 2 * w, dtype=a.dtype, device=a.device)
    x[:, :, 0::2, 0::2] = a; x[:, :, 0::2, 1::2] = b
    x[:, :, 1::2, 0::2] = c; x[:, :, 1::2, 1::2] = d
    return x


def round_ste(v):
    return v + (torch.round(v) - v).detach()


def round_noise(v):
    # training proxy for rounding: additive uniform noise in [-0.5, 0.5]
    return v + (torch.rand_like(v) - 0.5)


class Phi(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.c1 = nn.Conv2d(3, hid, 3, padding=1)
        self.c2 = nn.Conv2d(hid, hid, 3, padding=1)
        self.c3 = nn.Conv2d(hid, 2, 3, padding=1)
        # small (not zero) init: with rounding in the loop a zero output never moves
        nn.init.normal_(self.c3.weight, 0, 0.05); nn.init.zeros_(self.c3.bias)

    def forward(self, half, M):
        h = torch.cat([half / 255.0, M], 1)
        h = F.leaky_relu(self.c1(h), 0.2)
        h = F.leaky_relu(self.c2(h), 0.2)
        # offsets in grey levels, bounded to +-16 so training can never blow up;
        # ~8*c3 for small values so the message is visible from the start
        return 16.0 * torch.tanh(0.5 * self.c3(h))


class Decoder(nn.Module):
    # stego (B,1,H,W) -> per-position message logits at (H/2, W/2)
    def __init__(self, hid=32):
        super().__init__()
        self.c1 = nn.Conv2d(4, hid, 3, padding=1)
        self.c2 = nn.Conv2d(hid, hid, 3, padding=1)
        self.c3 = nn.Conv2d(hid, hid, 3, padding=1)
        self.c4 = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, y):
        a, b, c, d = s2d(y / 255.0)
        h = torch.cat([a, b, c, d], 1)
        h = F.leaky_relu(self.c1(h), 0.2)
        h = F.leaky_relu(self.c2(h), 0.2)
        h = F.leaky_relu(self.c3(h), 0.2)
        return self.c4(h)


def msg_map(bits, s):
    # bits (B,1,h,w) in {0,1} -> +-1 map repeated s x s times (on the H/2 grid)
    return F.interpolate(bits * 2 - 1, scale_factor=s, mode='nearest')


def pool_logits(L, s):
    return F.avg_pool2d(L, s) if s > 1 else L


class ICINN(nn.Module):
    def __init__(self, n_blocks=4, hid=32):
        super().__init__()
        self.phis = nn.ModuleList([Phi(hid) for _ in range(n_blocks)])
        self.dec = Decoder(hid)

    def embed(self, x, M, mode='round'):
        # x: integer-valued float tensor 0..255; returns the unclipped stego
        # mode: 'round' (inference, exact integers), 'ste', or 'noise' (training)
        a, b, c, d = s2d(x)
        xa, xb = torch.cat([a, d], 1), torch.cat([b, c], 1)
        rnd = {'round': torch.round, 'ste': round_ste, 'noise': round_noise}[mode]
        for k, phi in enumerate(self.phis):
            if k % 2 == 0:
                xb = xb + rnd(phi(xa, M))
            else:
                xa = xa + rnd(phi(xb, M))
        return d2s(xa[:, 0:1], xb[:, 0:1], xb[:, 1:2], xa[:, 1:2])

    def invert(self, y, M):
        a, b, c, d = s2d(y)
        ya, yb = torch.cat([a, d], 1), torch.cat([b, c], 1)
        for k in reversed(range(len(self.phis))):
            if k % 2 == 0:
                yb = yb - torch.round(self.phis[k](ya, M))
            else:
                ya = ya - torch.round(self.phis[k](yb, M))
        return d2s(ya[:, 0:1], yb[:, 0:1], yb[:, 1:2], ya[:, 1:2])
