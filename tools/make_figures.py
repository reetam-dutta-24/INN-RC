# make_figures.py
# Builds every table (LaTeX) and plot used in the paper straight from the CSV files
# written by the MATLAB scripts, so no number is typed by hand.
#   python tools/make_figures.py

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'paper', 'figs')
TAB = os.path.join(ROOT, 'paper')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8,
                     'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'PEE': '#7f7f7f', 'BASE': '#d62728', 'PROP': '#1f77b4', 'PROPR': '#2ca02c'}
NAME = {'PEE': 'Rhombus PEE [4]', 'BASE': 'Float INN + RC (baseline)',
        'PROP': 'IC-INN-Q (ours)', 'PROPR': 'IC-INN-R (ours)'}
SETNAME = {'test_bsd100': 'BSD100', 'test_kodak': 'Kodak', 'test_medical': 'DICOM'}

frames = [pd.read_csv(os.path.join(RES, f)) for f in sorted(os.listdir(RES))
          if f.startswith('bench_') and f.endswith('.csv')]
B = pd.concat(frames, ignore_index=True)
B.to_csv(os.path.join(RES, 'benchmark_all.csv'), index=False)
B['set'] = B['set'].map(SETNAME)
summary = {}


def fmt(v, d=2):
    return '--' if pd.isna(v) else f'{v:.{d}f}'


# ---------- Table: proposed (both models) and PEE per dataset and payload ----------
rows = []
for st in ['BSD100', 'Kodak', 'DICOM']:
    for s in [8, 4, 2]:
        sub = B[(B.set == st) & (B.s == s)]
        r = [st if s == 8 else '', f'{int(sub.bits.iloc[0])}', f'{sub.bpp.iloc[0]:.4f}']
        for m in ['PEE', 'PROP', 'PROPR']:
            q = sub[sub.method == m]
            ok = q[q.ok == 1]
            r += [fmt(ok.psnr.mean()), fmt(ok.ssim.mean(), 4)]
            if m != 'PEE':
                r += [fmt(ok.aux_bits.mean(), 0)]
            r += [f'{int(q.exact_cover.sum())}/{len(q)}']
        rows.append(r)
with open(os.path.join(TAB, 'tab_main.tex'), 'w') as f:
    f.write('\\begin{table*}[t]\n\\centering\n\\caption{Lossless results per test set and payload (mean over images). '
            'Aux = side-information bits embedded by PEE; Exact = images whose cover \\emph{and} message were '
            'recovered bit for bit. Lossless BER was 0 and NC was 1 for every exact case.}\n\\label{tab:main}\n')
    f.write('\\setlength{\\tabcolsep}{3.2pt}\n\\begin{tabular}{lcc|ccc|cccc|cccc}\n\\toprule\n')
    f.write('& & & \\multicolumn{3}{c|}{Rhombus PEE \\cite{sachnev2009}} & \\multicolumn{4}{c|}{IC-INN-Q (ours)} & '
            '\\multicolumn{4}{c}{IC-INN-R (ours)}\\\\\n')
    f.write('Set & Bits & bpp & PSNR & SSIM & Exact & PSNR & SSIM & Aux & Exact & PSNR & SSIM & Aux & Exact\\\\\n\\midrule\n')
    for r in rows:
        f.write(' & '.join(r) + '\\\\\n')
        if r[1] == '4096' and r is not rows[-1]:
            f.write('\\midrule\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table*}\n')

# overall numbers used in the text
for m in ['PEE', 'PROP', 'PROPR', 'BASE']:
    q = B[B.method == m]
    if len(q) == 0:
        continue
    ok = q[q.ok == 1]
    summary[m] = dict(n=len(q), ok=int(q.ok.sum()), exact=int(q.exact_cover.sum()),
                      ber_max=float(ok.ber.max()), nc_min=float(ok.nc.min()))
    for s in [8, 4, 2]:
        qs = ok[ok.s == s]
        if len(qs):
            summary[m][f's{s}'] = dict(psnr=float(qs.psnr.mean()), ssim=float(qs.ssim.mean()),
                                       aux=float(qs.aux_bits.mean()), psnr_inn=float(qs.psnr_inn.mean()),
                                       ber_inn=float(qs.ber_inn.mean()), te=float(qs.t_embed.mean()),
                                       tx=float(qs.t_extract.mean()), n=len(qs),
                                       psnr_by_set={k: float(v) for k, v in qs.groupby('set').psnr.mean().items()},
                                       aux_by_set={k: float(v) for k, v in qs.groupby('set').aux_bits.mean().items()})

# ---------- PSNR vs payload ----------
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.55))
for m in ['PEE', 'PROP', 'PROPR']:
    q = B[(B.method == m) & (B.ok == 1)].groupby('bpp')
    ax[0].plot(q.psnr.mean().index, q.psnr.mean().values, 'o-', ms=3, color=COL[m], label=NAME[m])
    if m != 'PEE':
        ax[1].plot(q.aux_bits.mean().index, q.aux_bits.mean().values / 1000, 'o-', ms=3, color=COL[m])
        bits = B[(B.method == m)].groupby('bpp').bits.first()
        if m == 'PROP':
            ax[1].plot(bits.index, bits.values / 1000, 'k--', lw=0.8, label='message bits')
qb = B[(B.method == 'BASE') & (B.ok == 1)]
if len(qb):
    ax[0].plot(qb.bpp.mean(), qb.psnr.mean(), 's', ms=4, color=COL['BASE'], label=NAME['BASE'])
for a in ax:
    a.set_xscale('log', base=2); a.set_xlabel('payload (bpp)')
    a.set_xticks([1/256, 1/64, 1/16]); a.set_xticklabels(['1/256', '1/64', '1/16'])
ax[0].set_ylabel('PSNR (dB)'); ax[1].set_ylabel('bits (x1000)')
ax[0].set_title('(a) stego quality'); ax[1].set_title('(b) side info vs payload')
ax[1].legend(frameon=False, loc='upper left')
fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.17))
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, 'psnr_bpp.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- training curves ----------
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.45))
for tag, col, lab in [('icinn', COL['PROP'], 'IC-INN-Q'), ('icinn_r', COL['PROPR'], 'IC-INN-R')]:
    p = os.path.join(ROOT, 'models', tag + '_log.json')
    if not os.path.exists(p):
        continue
    L = json.load(open(p))['log']
    it = [e['it'] for e in L]
    ax[0].plot(it, [10 * np.log10(255 ** 2 / max(e['mse'], 1e-6)) for e in L], color=col, lw=0.9, label=lab)
    ax[1].plot(it, [e['bce'] for e in L], color=col, lw=0.9)
ax[0].set_ylabel('stego PSNR (dB)'); ax[1].set_ylabel('message BCE'); ax[1].set_yscale('log')
for a in ax: a.set_xlabel('iteration')
ax[0].legend(frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, 'training.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- visual examples ----------
vis = os.path.join(RES, 'visual')
if os.path.isdir(vis):
    names = ['kodim23', 'bsd_010', 'med05_CR', 'med03_MR']
    cols = ['cover', 'stego', 'diff20', 'recovered']
    titles = ['cover', 'stego', '|stego-cover| x20', 'recovered']
    fig, ax = plt.subplots(len(names), 4, figsize=(3.45, 3.55))
    for i, n in enumerate(names):
        for j, c in enumerate(cols):
            im = np.array(Image.open(os.path.join(vis, f'{n}_{c}.png')))
            ax[i, j].imshow(im, cmap='gray', vmin=0, vmax=255); ax[i, j].axis('off')
            if i == 0: ax[i, j].set_title(titles[j], fontsize=7)
    fig.tight_layout(pad=0.15, w_pad=0.1, h_pad=0.1)
    fig.savefig(os.path.join(FIG, 'visual.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)

# ---------- robustness table ----------
ap = os.path.join(RES, 'attacks.csv')
if os.path.exists(ap):
    A = pd.read_csv(ap)
    A['key'] = A.attack + ':' + A.param.astype(str)
    order = [('none', '0', 'none'), ('gauss', '1', 'GN $\\sigma{=}1$'), ('gauss', '2', 'GN $\\sigma{=}2$'),
             ('gauss', '3', 'GN $\\sigma{=}3$'), ('sp', '0.005', 'S\\&P 0.5\\%'), ('sp', '0.01', 'S\\&P 1\\%'),
             ('jpeg', '95', 'JPEG 95'), ('jpeg', '90', 'JPEG 90'), ('jpeg', '80', 'JPEG 80')]
    rob = {}
    with open(os.path.join(TAB, 'tab_robust.tex'), 'w') as f:
        f.write('\\begin{table}[t]\n\\centering\n\\caption{Message BER (\\%) on a lossy channel, payload 1024 bits, '
                f'{A.image.nunique()} images. The cover is not recoverable here; the question is only how many '
                'message bits survive.}\n\\label{tab:robust}\n\\setlength{\\tabcolsep}{4pt}\n'
                '\\begin{tabular}{lcccc}\n\\toprule\nAttack & PEE & Float INN & IC-INN-Q & IC-INN-R\\\\\n\\midrule\n')
        for at, pa, lab in order:
            q = A[(A.attack == at) & (np.isclose(A.param, float(pa)))]
            vals = [q[q.method == m].ber.mean() * 100 for m in ['PEE', 'BASE', 'PROP', 'PROPR']]
            rob[f'{at}{pa}'] = vals
            f.write(lab + ' & ' + ' & '.join(fmt(v, 2) for v in vals) + '\\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    summary['robust'] = rob

json.dump(summary, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1)[:4000])
