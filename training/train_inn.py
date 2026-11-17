# train_inn.py
# Trains the INN used by INN-RC and exports the weights to a .mat file for MATLAB.
#
#   python train_inn.py --msg_ch 2 --mode ours     --out ../models/inn_c2_ours
#   python train_inn.py --msg_ch 2 --mode baseline --out ../models/inn_c2_base
#
# mode = ours      : rounding in the loop (STE) + reveal with z = 0
#                    + cover recovery loss + latent loss  (proposed)
# mode = baseline  : HiNet style training, float stego, z ~ N(0,1) at reveal,
#                    no cover recovery loss (the network never learns to give
#                    back the cover, it is only asked to hide and reveal)
# mode = no_ste    : ours but without rounding in the loop        (ablation)
# mode = no_cov    : ours but without cover recovery/latent loss   (ablation)

import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat
from inn_model import INN, dwt_haar

p = argparse.ArgumentParser()
p.add_argument('--msg_ch', type=int, default=1)       # message channels on the half-resolution grid
p.add_argument('--s', type=int, default=2)            # each bit repeated s x s times (same payloads as IC-INN)
p.add_argument('--mode', default='ours')
p.add_argument('--iters', type=int, default=6000)
p.add_argument('--batch', type=int, default=12)
p.add_argument('--patch', type=int, default=64)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--hid', type=int, default=32)
p.add_argument('--blocks', type=int, default=4)
p.add_argument('--w_stego', type=float, default=1.0)
p.add_argument('--w_low', type=float, default=1.0)
p.add_argument('--w_msg', type=float, default=0.02)
p.add_argument('--warm', type=float, default=0.3)       # fraction of iterations with boosted message weight
p.add_argument('--w_cov', type=float, default=4.0)
p.add_argument('--w_z', type=float, default=0.05)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--data', default='../data/train_urban100')
p.add_argument('--out', default='../models/inn')
p.add_argument('--threads', type=int, default=0)
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(a.threads or os.cpu_count())

files = sorted(glob.glob(os.path.join(a.data, '*.png')))
imgs = [np.array(Image.open(f).convert('L'), dtype=np.float32) / 255 for f in files]
val_imgs, train_imgs = imgs[:8], imgs[8:]          # first 8 Urban100 images = validation
print(f'{len(train_imgs)} train / {len(val_imgs)} val images, mode={a.mode}, msg_ch={a.msg_ch}')


def batch(n, s):
    out = np.zeros((n, 1, s, s), np.float32)
    for i in range(n):
        im = train_imgs[np.random.randint(len(train_imgs))]
        r = np.random.randint(im.shape[0] - s + 1); c = np.random.randint(im.shape[1] - s + 1)
        q = im[r:r + s, c:c + s]
        if np.random.rand() < 0.5: q = q[:, ::-1]
        if np.random.rand() < 0.5: q = q[::-1, :]
        out[i, 0] = q
    return torch.from_numpy(out)


def rand_bits(n, s):
    return torch.randint(0, 2, (n, a.msg_ch, s // 2 // a.s, s // 2 // a.s)).float()


def bits_to_map(b):
    return F.interpolate(b - 0.5, scale_factor=a.s, mode='nearest')      # +-0.5


def map_to_soft(mr):
    return F.avg_pool2d(mr, a.s) if a.s > 1 else mr


def round_ste(x):                      # quantise to 8 bit, straight-through gradient
    q = torch.clamp(torch.round(x * 255), 0, 255) / 255
    return x + (q - x).detach()


net = INN(a.msg_ch, a.blocks, a.hid)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr * 0.02)
use_ste = a.mode in ('ours', 'no_cov')
use_cov = a.mode in ('ours', 'no_ste')
log = []


@torch.no_grad()
def validate():
    net.eval(); ps, bers, nz = [], [], []
    g = torch.Generator().manual_seed(123)
    for im in val_imgs:
        h, w = (im.shape[0] // 2) * 2, (im.shape[1] // 2) * 2
        x = torch.from_numpy(im[:h, :w].copy())[None, None].double()
        h, w = (h // (2 * a.s)) * 2 * a.s, (w // (2 * a.s)) * 2 * a.s
        bits = torch.randint(0, 2, (1, a.msg_ch, h // 2 // a.s, w // 2 // a.s), generator=g).double()
        x = x[:, :, :h, :w]
        m = bits_to_map(bits)
        net.double()
        st, _ = net.embed(x, m)
        stq = torch.clamp(torch.round(st * 255), 0, 255) / 255
        xr, mr = net.reveal(stq)
        mse = ((stq - x) ** 2).mean().item()
        ps.append(10 * math.log10(1 / mse))
        bers.append(((map_to_soft(mr) > 0).double() != bits).double().mean().item())
        res = torch.round(x * 255) - torch.clamp(torch.round(xr * 255), 0, 255)
        nz.append((res != 0).double().mean().item())
        net.float()
    net.train()
    return float(np.mean(ps)), float(np.mean(bers)), float(np.mean(nz))


t0 = time.time()
for it in range(1, a.iters + 1):
    x = batch(a.batch, a.patch)
    bits = rand_bits(a.batch, a.patch)
    m = bits_to_map(bits)
    stego, z = net.embed(x, m)
    l_stego = F.mse_loss(stego, x)
    l_low = F.mse_loss(dwt_haar(stego)[:, :1], dwt_haar(x)[:, :1])
    if a.mode == 'baseline':
        xr, mr = net.reveal(stego, torch.randn_like(z))
    else:
        st_in = round_ste(stego) if use_ste else stego
        xr, mr = net.reveal(st_in)                        # z = 0
    l_msg = F.mse_loss(map_to_soft(mr), bits - 0.5)
    # message weight starts at 1 and decays to w_msg, otherwise the net never starts hiding anything
    w_msg = a.w_msg + (1 - a.w_msg) * max(0.0, 1 - it / (a.warm * a.iters))
    loss = a.w_stego * l_stego + a.w_low * l_low + w_msg * l_msg
    if use_cov:
        l_cov = F.mse_loss(xr, x)
        l_z = (z ** 2).mean()
        loss = loss + a.w_cov * l_cov + a.w_z * l_z
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step()
    if it % 250 == 0 or it == 1:
        ps = 10 * math.log10(1 / max(l_stego.item(), 1e-12))
        msg = f'it {it:5d}  loss {loss.item():.5f}  stegoPSNR {ps:5.2f}  msgMSE {l_msg.item():.4f}  {time.time()-t0:6.0f}s'
        if it % 1000 == 0 or it == a.iters:
            vp, vb, vn = validate()
            msg += f'  | val PSNR {vp:.2f}  BER {vb:.4f}  residual!=0 {vn:.4f}'
            log.append(dict(it=it, loss=loss.item(), val_psnr=vp, val_ber=vb, val_nz=vn))
        else:
            log.append(dict(it=it, loss=loss.item()))
        print(msg, flush=True)

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
torch.save(net.state_dict(), a.out + '.pt')
W = {}
for k, v in net.state_dict().items():
    W[k.replace('.', '_')] = v.double().numpy()
W['msg_ch'] = a.msg_ch; W['s'] = a.s; W['n_blocks'] = a.blocks; W['hid'] = a.hid; W['clamp'] = 2.0
savemat(a.out + '.mat', W)
json.dump(dict(args=vars(a), log=log), open(a.out + '_log.json', 'w'), indent=1)
print('saved', a.out)
