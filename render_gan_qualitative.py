#!/usr/bin/env python
"""Render fig:gan_qualitative from gan_flowmetric_out/qualitative_panels.npz.

2x4 panel: [ED input, reconstruction, GT flow magnitude, predicted flow magnitude]
for one healthy and one DCM ACDC patient (median of each group by patient-mean
flow-SSIM). The slice shown is the one whose flow-SSIM is closest to the patient
mean. One shared magnitude scale across all four flow panels so contraction is
comparable within and across rows. Prints the central-crop GT-vs-predicted mean
magnitudes that the caption's claim rests on.

Runs in the `derisk` env (numpy + matplotlib only).
Output: thesis/figures/gan_qualitative.{png,pdf}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = np.load('gan_flowmetric_out/qualitative_panels.npz', allow_pickle=True)
ROWS = [('patient139', 'Healthy (NOR)'), ('patient117', 'DCM')]
INK, INK2 = '#0b0b0b', '#52514e'

pids = D['pids'].astype(str)


def pick_slice(pid):
    m = np.where(pids == pid)[0]
    fs = D['flow_ssim'][m]
    return m[np.argmin(np.abs(fs - fs.mean()))]


idx = {pid: pick_slice(pid) for pid, _ in ROWS}

# shared magnitude scale across all four flow panels
mags = [D[k][idx[pid]][..., 2] for pid, _ in ROWS for k in ('gt_flow', 'pred_flow')]
vmax = max(np.percentile(m, 99.5) for m in mags)

# caption-verification stats: mean |flow| in the central crop (LV is centred by
# the orientation normalisation, so this is a heart-region proxy)
c0, c1 = 40, 88
print('central-crop (48x48) mean |flow|, px:')
for pid, name in ROWS:
    i = idx[pid]
    g = D['gt_flow'][i][c0:c1, c0:c1, 2].mean()
    p = D['pred_flow'][i][c0:c1, c0:c1, 2].mean()
    print(f'  {name:14s} slice z={int(D["slices"][i])}: GT {g:.3f}  pred {p:.3f}  '
          f'pred/GT {p/g:.2f}  1-SSIM(flow) {D["flow_ssim"][i]:.3f}')

fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.4))
cols = ['ED input', 'Reconstruction', 'GT flow magnitude', 'Predicted flow magnitude']
im = None
for r, (pid, name) in enumerate(ROWS):
    i = idx[pid]
    panels = [
        (D['images'][i][..., 0], dict(cmap='gray', vmin=0, vmax=1)),
        (np.clip((D['recon'][i][..., 0] + 1) / 2, 0, 1), dict(cmap='gray', vmin=0, vmax=1)),
        (D['gt_flow'][i][..., 2], dict(cmap='inferno', vmin=0, vmax=vmax)),
        (D['pred_flow'][i][..., 2], dict(cmap='inferno', vmin=0, vmax=vmax)),
    ]
    for c, (img, kw) in enumerate(panels):
        ax = axes[r, c]
        h = ax.imshow(img, **kw, interpolation='nearest')
        if c >= 2:
            im = h
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(cols[c], fontsize=11, color=INK, pad=6)
        if c == 0:
            ax.set_ylabel(name, fontsize=11, color=INK)

fig.subplots_adjust(left=0.045, right=0.90, top=0.93, bottom=0.03,
                    wspace=0.04, hspace=0.06)
cax = fig.add_axes([0.915, 0.06, 0.015, 0.84])
cb = fig.colorbar(im, cax=cax)
cb.set_label('displacement magnitude (px)', fontsize=9, color=INK2)
cb.ax.tick_params(labelsize=8, colors=INK2)
cb.outline.set_visible(False)

for ext in ('png', 'pdf'):
    fig.savefig(f'thesis/figures/gan_qualitative.{ext}', dpi=300)
print('Saved thesis/figures/gan_qualitative.png and .pdf')
