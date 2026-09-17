"""Metric-matched patch-support controls for tab:gan_revert_patch (thesis ch4 revert subsection).

Computes the two cells missing from the revert job 3930515: SSIM on the worst 16x16 patch
(stride 4, worst = lowest mean of the skimage SSIM map, win border cropped) and reads the
full-frame L1 from gan_revert_out/scores.npz. Flow fields come from job 3740842
(gan_flowmetric_out/flows.npz, float16, same checkpoint/samples as the mid60 pass of the
revert job -- ordering verified by reproducing the per-sample headline flow-SSIM exactly).
Also prints paired patient-cluster bootstrap deltas (2000 draws, seed 0) and writes
gan_revert_out/patch_ssim_control.json with every table cell.

Run with the derisk env (needs numpy + scikit-image); submit via gan_patch_ssim_control.pbs.
First produced offline 2026-08-31:
  SSIM worst patch  0.9375 / 0.7244 / 0.574   (ACDC / MM-Test / MM-Val, patient mean)
  L1 full frame     0.9250 / 0.6715 / 0.583
"""
import json
import os
import time

import numpy as np
from skimage.metrics import structural_similarity as ssim

REPO = os.path.dirname(os.path.abspath(__file__))
FLOWS = os.path.join(REPO, 'gan_flowmetric_out', 'flows.npz')
SCORES = os.path.join(REPO, 'gan_revert_out', 'scores.npz')
OUT_JSON = os.path.join(REPO, 'gan_revert_out', 'patch_ssim_control.json')

F = np.load(FLOWS)
S = np.load(SCORES, allow_pickle=True)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - cnt + csum + 1) / 2.0
    r = avg[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def win_means(m, size=16, step=4):
    H, W = m.shape
    ii = np.arange(0, H - size + 1, step)
    jj = np.arange(0, W - size + 1, step)
    c = np.zeros((H + 1, W + 1))
    c[1:, 1:] = m.cumsum(0).cumsum(1)
    out = np.empty((len(ii), len(jj)))
    for a, i in enumerate(ii):
        out[a, :] = (c[i + size, jj + size] - c[i, jj + size] - c[i + size, jj] + c[i, jj]) / (size * size)
    return out


results = {}
order_check = {}
for ds in ['ACDC', 'MM', 'MM_VAL']:
    gt = F[f'gt__{ds}'].astype(np.float32)
    pr = F[f'pred__{ds}'].astype(np.float32)
    n = len(gt)
    full_ssim = np.zeros(n)
    patch_ssim = np.zeros(n)
    patch_ssim_msesel = np.zeros(n)
    for k in range(n):
        g, p = gt[k], pr[k]
        dr = float(np.max([g, p]) - np.min([g, p])) or 1.0
        s_val, s_map = ssim(g, p, data_range=dr, channel_axis=-1, full=True)
        full_ssim[k] = 1.0 - s_val
        m = s_map.mean(-1)
        pad = 3  # win_size 7 border
        wm = win_means(m[pad:-pad, pad:-pad])
        patch_ssim[k] = 1.0 - wm.min()            # worst-SSIM patch
        # paper-style selection: patch of max MSE, read SSIM there
        mse = ((g - p) ** 2).mean(-1)
        wmse = win_means(mse[pad:-pad, pad:-pad])
        idx = np.unravel_index(wmse.argmax(), wmse.shape)
        patch_ssim_msesel[k] = 1.0 - wm[idx]
    ref = S[f'mid60__{ds}__flow_ssim']
    corr = float(np.corrcoef(ref, full_ssim)[0, 1])
    maxdiff = float(np.abs(ref - full_ssim).max())
    order_check[ds] = {'corr': corr, 'max_abs_diff': maxdiff}
    results[ds] = dict(full_ssim=full_ssim, patch_ssim=patch_ssim, patch_ssim_msesel=patch_ssim_msesel,
                       pids=S[f'mid60__{ds}__pids'], labels=S[f'mid60__{ds}__labels'],
                       full_l1=S[f'mid60__{ds}__flow_l1'], patch_mse=S[f'mid60__{ds}__patch_flow'])
    print(f'{ds}: order check corr={corr:.5f} max|diff|={maxdiff:.4f} (float16 gt/pred, so small diff expected)')
    assert maxdiff < 0.01, f'{ds}: flows.npz ordering does not match scores.npz mid60 pass'


def patient_auc(r, key):
    pids, labels, s = r['pids'], r['labels'], r[key]
    up = np.unique(pids)
    pm = np.array([s[pids == p].mean() for p in up])
    pl = np.array([labels[pids == p][0] for p in up])
    return auc(pm[pl != 'NOR'], pm[pl == 'NOR']), up, pm, pl


NAMES = [('full_ssim', 'SSIM, full frame (headline)'),
         ('patch_ssim', 'SSIM, worst patch'),
         ('patch_ssim_msesel', 'SSIM at MSE-argmax patch'),
         ('full_l1', 'L1, full frame'),
         ('patch_mse', 'MSE, worst patch (paper)')]

table = {}
print()
print(f"{'score':28s} {'ACDC':>7s} {'MM':>7s} {'MM_VAL':>7s}")
for key, name in NAMES:
    row = {ds: float(patient_auc(results[ds], key)[0]) for ds in ['ACDC', 'MM', 'MM_VAL']}
    table[key] = {'name': name, 'auc': row}
    print(f"{name:28s} {row['ACDC']:7.4f} {row['MM']:7.4f} {row['MM_VAL']:7.4f}")

rng = np.random.default_rng(0)
N_BOOT = 2000


def paired_boot(r, k1, k2, nb=N_BOOT):
    a1, up, pm1, pl = patient_auc(r, k1)
    a2, _, pm2, _ = patient_auc(r, k2)
    pos = np.where(pl != 'NOR')[0]
    neg = np.where(pl == 'NOR')[0]
    ds_ = []
    for _ in range(nb):
        bp = rng.choice(pos, len(pos))
        bn = rng.choice(neg, len(neg))
        ds_.append(auc(pm2[bp], pm2[bn]) - auc(pm1[bp], pm1[bn]))
    ds_ = np.array(ds_)
    return a2 - a1, float(np.percentile(ds_, 2.5)), float(np.percentile(ds_, 97.5))


boot = {}
print()
for ds in ['ACDC', 'MM']:
    boot[ds] = {}
    for k1, k2, lbl in [('full_ssim', 'patch_ssim', 'SSIM: full -> worst patch'),
                        ('full_l1', 'patch_mse', 'L1/MSE: full -> worst patch'),
                        ('full_ssim', 'full_l1', 'full frame: SSIM -> L1')]:
        dlt, lo, hi = paired_boot(results[ds], k1, k2)
        sig = 'EXCLUDES 0' if lo * hi > 0 else 'ns'
        boot[ds][f'{k1}->{k2}'] = {'label': lbl, 'delta': float(dlt), 'ci': [lo, hi], 'excludes_zero': lo * hi > 0}
        print(f'{ds:6s} {lbl:30s} delta={dlt:+.4f} CI=[{lo:+.4f},{hi:+.4f}] {sig}')

summary = {
    'produced': time.strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'gan_patch_ssim_control.py',
    'inputs': {'flows': FLOWS, 'scores': SCORES},
    'patch': {'size': 16, 'step': 4, 'ssim_win': 7, 'border_crop': 3},
    'n_boot': N_BOOT, 'seed': 0,
    'order_check': order_check,
    'auc': table,
    'paired_bootstrap_deltas': boot,
}
with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nWrote {OUT_JSON}')
