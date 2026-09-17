"""confirm_offline.py — offline confirmatory for the 2-D/single-pass middle-60% result.

The Phase-A/B sweep chose (config, stream, rule) BY LOOKING AT M&Ms-Test → optimistic. This does two
honest, no-GPU checks on the already-saved held-out arrays (ACDC-50 + M&Ms-Test, disjoint patients):

  1. SELECT-ON-ACDC, REPORT-ON-M&Ms  — pick the recipe that maximises ACDC AUC (M&Ms never peeked),
     then report its M&Ms-Test AUC. Done over ALL streams, and restricted to MOTION streams
     (motion is pre-committed on domain grounds: only motion crosses the vendor gap).
  2. RULE ROBUSTNESS — for each model/stream, is middle-60% the best rule on ACDC *and* on M&Ms
     independently? A rule that wins on both, across 4 independently-trained models, is not cherry-picked.

The gold-standard (select on M&Ms-Validation, report on M&Ms-Test) needs a GPU re-eval on the
Validation folder — launched separately.
"""

import numpy as np
from qfae_report import _load, _reduce_slices, _fast_auc, REDUCTION_RULES

RULES = ["mean", "top10", "top20", "top30", "max", "middle60"]      # drop 'mean_all' (padded, legacy)
MOTION = ("flow_SSIM", "mag_SSIM", "flow_L1")

# (name, npz, format). 'stack' = CineMA single-pass per-stack (appe_slices..(n,16)+n_real_slices);
# 'slice' = dino2d per-slice-sample (stream vals + pids/slcs).
RUNS = [
    ("CineMA-SP",   "qfae_flow_sp_out_mmtest/qfae_arrays.npz",           "stack"),
    ("DINOv2@224",  "qfae_dino2d_dino224_out_mmtest/qfae_dino_arrays.npz", "slice"),
    ("DINOv2@518",  "qfae_dino2d_dino518_out_mmtest/qfae_dino_arrays.npz", "slice"),
    ("MAE@224",     "qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz",  "slice"),
]
SLICE_KEYS = {"appearance": "appe_slices", "flow_SSIM": "flow_SSIM_slices",
              "mag_SSIM": "mag_SSIM_slices", "flow_L1": "flow_L1_slices"}


def _auc_by_ds(patient_score, plab, pds):
    y = (plab != "NOR").astype(int)
    out = {}
    for ds in ("ACDC", "MM"):
        m = pds == ds
        out[ds] = _fast_auc(y[m], patient_score[m]) if len(np.unique(y[m])) > 1 else np.nan
    return out


def grid_for_run(name, npz, fmt):
    """Return {(stream, rule): {'ACDC':auc,'MM':auc}} for one model."""
    a = _load(npz)
    pids, labels, datasets = a["pids"], a["labels"], a["datasets"]
    res = {}
    if fmt == "stack":
        nreal = a["n_real_slices"]
        streams = {s: k for s, k in SLICE_KEYS.items() if k in a}
        plab, pds = labels, datasets                        # single-pass: 1 stack == 1 patient
        for s, key in streams.items():
            mat = a[key]
            for rule in RULES:
                sc = np.array([_reduce_slices(mat[i], nreal[i], rule) for i in range(len(mat))])
                res[(s, rule)] = _auc_by_ds(sc, plab, pds)
    else:
        uniq = np.unique(pids)
        plab = np.array([labels[pids == p][0] for p in uniq])
        pds = np.array([datasets[pids == p][0] for p in uniq])
        streams = [s for s in ("appearance", "flow_SSIM", "mag_SSIM", "flow_L1") if s in a]
        for s in streams:
            vals = a[s]
            for rule in RULES:
                sc = np.empty(len(uniq))
                for i, p in enumerate(uniq):
                    m = pids == p
                    v = vals[m][np.argsort(slcs_of(a)[m])]
                    sc[i] = _reduce_slices(v, len(v), rule)
                res[(s, rule)] = _auc_by_ds(sc, plab, pds)
    return res


def slcs_of(a):
    return a["slcs"]


def main():
    grids = {}
    for name, npz, fmt in RUNS:
        try:
            grids[name] = grid_for_run(name, npz, fmt)
        except FileNotFoundError:
            print(f"[skip] {name}: {npz} missing")
    print("=" * 78)
    print("OFFLINE CONFIRMATORY — held-out ACDC-50 + M&Ms-Test (32 NOR/104). GAN = ACDC 0.8125 / MM 0.731")
    print("=" * 78)

    # ── 1. middle-60% motion table across all models (the pre-committed rule+stream family) ──
    print("\n[1] middle-60% MOTION streams across all models (ACDC / M&Ms):")
    print(f"    {'model':<12}{'flow_SSIM':>18}{'mag_SSIM':>18}{'flow_L1':>18}")
    for name in grids:
        row = f"    {name:<12}"
        for s in MOTION:
            g = grids[name].get((s, "middle60"))
            row += (f"{g['ACDC']:.3f}/{g['MM']:.3f}".rjust(18)) if g else f"{'--':>18}"
        print(row)

    # ── 2. SELECT-ON-ACDC → REPORT-ON-M&Ms (no M&Ms peeking) ──
    def select_report(pool, label):
        best = max(pool, key=lambda r: (r[3] if not np.isnan(r[3]) else -1))   # r=(model,stream,rule,acdc,mm)
        print(f"\n[2] {label}")
        print(f"    picked on ACDC: {best[0]} / {best[1]} / {best[2]}  (ACDC={best[3]:.3f})")
        print(f"    -> reported held-out M&Ms-Test = {best[4]:.3f}")

    allrec = [(name, s, rule, g["ACDC"], g["MM"])
              for name, grid in grids.items() for (s, rule), g in grid.items()]
    select_report(allrec, "select over ALL streams+rules on ACDC:")
    select_report([r for r in allrec if r[1] in MOTION],
                  "select over MOTION streams (pre-committed) + all rules, on ACDC:")
    select_report([r for r in allrec if r[1] in MOTION and r[2] == "middle60"],
                  "select over MOTION + middle-60% (both pre-committed), config on ACDC:")

    # ── 3. RULE ROBUSTNESS — is middle-60% the best rule, on each dataset independently? ──
    print("\n[3] Is middle-60% the BEST rule (per model × motion stream), independently per dataset?")
    for ds in ("ACDC", "MM"):
        wins = tot = 0
        for name, grid in grids.items():
            for s in MOTION:
                cells = {rule: grid[(s, rule)][ds] for rule in RULES if (s, rule) in grid}
                if not cells or any(np.isnan(v) for v in cells.values()):
                    continue
                tot += 1
                if max(cells, key=cells.get) == "middle60":
                    wins += 1
        print(f"    {ds}: middle-60% is the top rule in {wins}/{tot} (model × motion-stream) cells")


if __name__ == "__main__":
    main()
