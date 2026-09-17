"""fusion_sweep_2d.py — does adding the (dead) APPEARANCE stream to the MOTION stream help,
measured ON THIS model, with the two streams put on a common scale first?

Appearance = embedding-space reconstruction distance (arbitrary units, unbounded).
Motion (flow_SSIM) = 1 - SSIM (bounded [0,2]). Different scales, so a raw A + w*F sweep is
meaningless. We therefore NORMALISE each stream first (percentile-rank primary; z-score cross-
check) *within each dataset*, then fuse. For a single stream AUC is invariant to this (rank is
monotonic) so appearance-alone / flow-alone AUCs are unchanged — only the fusion sees the scale fix.

Pre-committed reduction rule = 'mean' (the recipe val-selection actually picked in the confirmatory);
'middle60' shown as a secondary robustness row. Streams: appearance + flow_SSIM (the val-selected
motion stream). We report, per dataset, on the HELD-OUT M&Ms-Test + ACDC-50:
  appearance-alone | flow-alone | EQUAL-WEIGHT rank fusion (no tuning, pre-committed)
  | oracle best-w (ceiling, peeks at test labels) | honest val-selected-w (select on M&Ms-Val).
If equal-weight fusion doesn't beat flow-alone on M&Ms, appearance genuinely doesn't help here.
"""

import numpy as np
from qfae_report import _load, _fast_auc, _reduce_slices

# (name, fmt, test_npz, val_npz). fmt: 'slice' = 2-D per-slice-sample; 'stack' = CineMA-SP per-stack.
CONFIGS = [
    ("MAE@224",    "slice",
     "qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz",  "qfae_dino2d_mae224_out_mmval/qfae_dino_arrays.npz"),
    ("DINOv2@224", "slice",
     "qfae_dino2d_dino224_out_mmtest/qfae_dino_arrays.npz", "qfae_dino2d_dino224_out_mmval/qfae_dino_arrays.npz"),
    ("DINOv2@518", "slice",
     "qfae_dino2d_dino518_out_mmtest/qfae_dino_arrays.npz", "qfae_dino2d_dino518_out_mmval/qfae_dino_arrays.npz"),
    ("CineMA-SP",  "stack",
     "qfae_flow_sp_out_mmtest/qfae_arrays.npz",             "qfae_flow_sp_out_mmval/qfae_arrays.npz"),
]

APPE_KEY = "appearance"
FLOW_KEY = "flow_SSIM"
WGRID = np.linspace(0.0, 1.0, 21)   # w = weight on APPEARANCE (w=0 -> flow only, w=1 -> appe only)


def per_patient(npz, fmt, rule):
    """Return {ds: dict(y, appe, flow)} of per-patient scalars (reduced by `rule`) for one file."""
    a = _load(npz)
    pids, labels, datasets = a["pids"], a["labels"], a["datasets"]
    out = {}
    if fmt == "stack":
        nreal = a["n_real_slices"]
        am, fm = a["appe_slices"], a["flow_SSIM_slices"]           # (n,16)
        appe = np.array([_reduce_slices(am[i], nreal[i], rule) for i in range(len(am))])
        flow = np.array([_reduce_slices(fm[i], nreal[i], rule) for i in range(len(fm))])
        plab, pds = labels, datasets                               # 1 stack == 1 patient
        for ds in ("ACDC", "MM"):
            m = pds == ds
            out[ds] = dict(y=(plab[m] != "NOR").astype(int), appe=appe[m], flow=flow[m])
    else:
        slcs = a["slcs"]
        uniq = np.unique(pids)
        plab = np.array([labels[pids == p][0] for p in uniq])
        pds = np.array([datasets[pids == p][0] for p in uniq])
        av, fv = a[APPE_KEY], a[FLOW_KEY]
        appe = np.empty(len(uniq)); flow = np.empty(len(uniq))
        for i, p in enumerate(uniq):
            m = pids == p
            order = np.argsort(slcs[m])
            appe[i] = _reduce_slices(av[m][order], m.sum(), rule)
            flow[i] = _reduce_slices(fv[m][order], m.sum(), rule)
        for ds in ("ACDC", "MM"):
            dm = pds == ds
            out[ds] = dict(y=(plab[dm] != "NOR").astype(int), appe=appe[dm], flow=flow[dm])
    return out


def norm(v, method):
    v = np.asarray(v, float)
    if method == "rank":
        return v.argsort().argsort() / max(1, len(v) - 1)          # percentile rank in [0,1]
    return (v - v.mean()) / (v.std() + 1e-9)                        # zscore


def blend(appe, flow, method, w):
    return (1.0 - w) * norm(flow, method) + w * norm(appe, method)  # w = appearance weight


def _boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    b = []
    for _ in range(n):
        bi = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        if len(np.unique(y[bi])) > 1:
            b.append(_fast_auc(y[bi], s[bi]))
    return np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan)


def cell(y, s, ci=True):
    if len(np.unique(y)) < 2:
        return f"{'n/a':>22}"
    auc = _fast_auc(y, s)
    if not ci:
        return f"{auc:.3f}".rjust(22)
    lo, hi = _boot_ci(y, s)
    return f"{auc:.3f} [{lo:.2f},{hi:.2f}] {int((y==0).sum())}/{int((y==1).sum())}".rjust(22)


def run(name, fmt, test_npz, val_npz, rule, method):
    T = per_patient(test_npz, fmt, rule)
    V = per_patient(val_npz, fmt, rule)
    print(f"\n  [{name}] rule={rule}  norm={method}")
    print(f"    {'variant':<24}{'ACDC-test':>22}{'M&Ms-test':>22}")
    # appearance alone / flow alone (norm is monotonic -> same as raw single-stream AUC)
    for lab, key in (("appearance alone", "appe"), ("flow_SSIM alone", "flow")):
        row = f"    {lab:<24}"
        for ds in ("ACDC", "MM"):
            d = T[ds]; row += cell(d["y"], norm(d[key], method))
        print(row)
    # equal-weight rank fusion (pre-committed, NO tuning)
    row = f"    {'equal-weight fuse':<24}"
    for ds in ("ACDC", "MM"):
        d = T[ds]; row += cell(d["y"], blend(d["appe"], d["flow"], method, 0.5))
    print(row)
    # oracle best-w on the test set itself (CEILING, peeks at test labels)
    row = f"    {'oracle best-w':<24}"
    for ds in ("ACDC", "MM"):
        d = T[ds]
        aucs = [_fast_auc(d["y"], blend(d["appe"], d["flow"], method, w)) for w in WGRID]
        j = int(np.nanargmax(aucs)); row += f"{aucs[j]:.3f} (w={WGRID[j]:.2f})".rjust(22)
    print(row)
    # honest: select w on M&Ms-VAL, report on M&Ms-TEST (ACDC has no separate val -> n/a)
    dv = V["MM"]
    vaucs = [_fast_auc(dv["y"], blend(dv["appe"], dv["flow"], method, w)) for w in WGRID]
    wv = WGRID[int(np.nanargmax(vaucs))]
    dt = T["MM"]
    test_at_wv = _fast_auc(dt["y"], blend(dt["appe"], dt["flow"], method, wv))
    print(f"    {'val-selected-w':<24}{'(no ACDC-val)':>22}"
          f"{f'{test_at_wv:.3f} (w={wv:.2f}, val={max(vaucs):.3f})':>22}")


def main():
    print("=" * 72)
    print("NORMALISED APPEARANCE + MOTION FUSION — held-out (ACDC-50 / M&Ms-Test)")
    print("Question: does the appearance stream ADD anything to flow_SSIM once scale is fixed?")
    print("GAN bar: ACDC 0.8125 / M&Ms-test 0.731.  w = weight on APPEARANCE (w=0 = flow only)")
    print("=" * 72)
    for method in ("rank", "zscore"):
        for rule in ("mean", "middle60"):
            print(f"\n{'#'*72}\n# norm={method}  rule={rule}")
            for name, fmt, tn, vn in CONFIGS:
                try:
                    run(name, fmt, tn, vn, rule, method)
                except FileNotFoundError as e:
                    print(f"  [skip] {name}: {e}")


if __name__ == "__main__":
    main()
