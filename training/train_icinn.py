# train_icinn.py
# Trains the proposed IC-INN (integer message-conditioned coupling + message decoder)
# and exports the weights for MATLAB.
#
#   python train_icinn.py --w_msg 10 --out ../models/icinn
#
# One model serves three payloads: the message map is repeated s x s times on the
# half-resolution grid, s is drawn at random from {1,2,4} for every batch.

import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat
from icinn_model import ICINN, msg_map, pool_logits

p = argparse.ArgumentParser()
p.add_argument('--iters', type=int, default=8000)
p.add_argument('--batch', type=int, default=12)
p.add_argument('--patch', type=int, default=64)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--hid', type=int, default=32)
p.add_argument('--blocks', type=int, default=4)
p.add_argument('--w_img', type=float, default=1.0)
p.add_argument('--w_msg', type=float, default=10.0)
p.add_argument('--w_of', type=float, default=5.0)      # overflow penalty
p.add_argument('--warm', type=float, default=0.25)
p.add_argument('--noise', type=int, default=1)         # 1 = train with the noise layer
p.add_argument('--scales', default='1,2,4')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--data', default='../data/train_urban100')
p.add_argument('--out', default='../models/icinn')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(os.cpu_count())
scales = [int(s) for s in a.scales.split(',')]

files = sorted(glob.glob(os.path.join(a.data, '*.png')))
imgs = [np.array(Image.open(f).convert('L'), dtype=np.float32) for f in files]
val_imgs, train_imgs = imgs[:8], imgs[8:]
print(f'{len(train_imgs)} train / {len(val_imgs)} val, scales {scales}, noise {a.noise}')


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


def ste(v, f):
    return v + (f(v) - v).detach()


def noise_layer(y):
    # y: stego in grey levels. Returns a distorted copy (decoder input during training).
    r = np.random.rand()
    if r < 0.45 or not a.noise:
        return y
    if r < 0.75:                                    # additive Gaussian noise, sigma 0..4 grey levels
        return y + torch.randn_like(y) * np.random.uniform(0, 4)
    if r < 0.9:                                     # salt & pepper, density up to 2 %
        m = (torch.rand_like(y) < np.random.uniform(0, 0.02)).float()
        sp = (torch.rand_like(y) < 0.5).float() * 255
        return y * (1 - m) + sp * m
    k = torch.tensor([1., 2., 1.]); k = (k[:, None] * k[None, :]) / 16   # mild blur
    return F.conv2d(F.pad(y, (1, 1, 1, 1), mode='replicate'), k[None, None])


net = ICINN(a.blocks, a.hid)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr * 0.02)
log = []


@torch.no_grad()
def validate():
    net.eval(); res = {}
    g = torch.Generator().manual_seed(7)
    for s in scales:
        ps, bers, ofs, exact = [], [], [], 0
        for im in val_imgs:
            h, w = (im.shape[0] // (2 * s * 4)) * 2 * s * 4, (im.shape[1] // (2 * s * 4)) * 2 * s * 4
            x = torch.from_numpy(im[:h, :w].copy())[None, None].double()
            bits = torch.randint(0, 2, (1, 1, h // 2 // s, w // 2 // s), generator=g).double()
            net.double()
            yt = net.embed(x, msg_map(bits, s), mode='round')
            y = torch.clamp(yt, 0, 255)
            ps.append(10 * math.log10(255 ** 2 / max(((y - x) ** 2).mean().item(), 1e-6)))
            L = pool_logits(net.dec(y), s)
            bers.append(((L > 0).double() != bits).double().mean().item())
            ofs.append((yt != y).double().mean().item())
            exact += int(torch.equal(net.invert(yt, msg_map(bits, s)), x))
            net.float()
        res[s] = (float(np.mean(ps)), float(np.mean(bers)), float(np.mean(ofs)), exact)
    net.train()
    return res


t0 = time.time()
for it in range(1, a.iters + 1):
    s = scales[np.random.randint(len(scales))]
    x = batch(a.batch, a.patch)
    bits = torch.randint(0, 2, (a.batch, 1, a.patch // 2 // s, a.patch // 2 // s)).float()
    # first half of training uses the noise proxy, then real rounding (STE)
    yt = net.embed(x, msg_map(bits, s), mode='noise' if it < a.iters // 2 else 'ste')
    y = ste(yt, lambda v: torch.clamp(v, 0, 255))
    yn = ste(noise_layer(y), lambda v: torch.clamp(torch.round(v), 0, 255))
    L = pool_logits(net.dec(yn), s)
    l_msg = F.binary_cross_entropy_with_logits(L, bits)
    l_img = ((y - x) ** 2).mean()
    l_of = (F.relu(-yt) + F.relu(yt - 255)).mean()
    w_msg = a.w_msg
    # curriculum: the decoder first learns to read a strong signal (no image loss),
    # then the image and overflow losses are brought in slowly and the stego gets cleaner
    ramp = min(1.0, max(0.0, (it - 0.1 * a.iters) / ((a.warm + 0.25) * a.iters)))
    w_img = a.w_img * ramp ** 2
    loss = w_img * l_img + w_msg * l_msg + a.w_of * ramp * l_of
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step(); sched.step()
    if it % 250 == 0 or it == 1:
        line = (f'it {it:5d} s={s} loss {loss.item():.4f}  mse {l_img.item():.3f} '
                f'(PSNR {10*math.log10(255**2/max(l_img.item(),1e-9)):.2f})  bce {l_msg.item():.4f}  of {l_of.item():.4f}  {time.time()-t0:5.0f}s')
        entry = dict(it=it, loss=loss.item(), mse=l_img.item(), bce=l_msg.item())
        if it % 1000 == 0 or it == a.iters:
            v = validate()
            line += '\n   val ' + '  '.join(f's={k}: PSNR {p:.2f} BER {b:.4f} ovf {o:.4f} exact {e}/8' for k, (p, b, o, e) in v.items())
            entry['val'] = {str(k): vv for k, vv in v.items()}
        log.append(entry)
        print(line, flush=True)

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
torch.save(net.state_dict(), a.out + '.pt')
W = {k.replace('.', '_'): v.double().numpy() for k, v in net.state_dict().items()}
W['n_blocks'] = a.blocks; W['hid'] = a.hid
savemat(a.out + '.mat', W)
json.dump(dict(args=vars(a), log=log), open(a.out + '_log.json', 'w'), indent=1)
print('saved', a.out)
