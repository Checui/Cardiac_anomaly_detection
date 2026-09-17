"""analyze_abcd.py — offline table for the A/B/C/D batch (no GPU).

Collates every arm of the batch onto ONE held-out protocol — patient-Mean AUC on
ACDC-50 and M&Ms-Testing, with patient-stratified bootstrap 95% CIs — so the numbers
sit directly beside POST_ICCV2019_EXPERIMENTS_SUMMARY.md.

  (A) bottleneck   qfae_abcd_nq{8,16,32,64,196}_out_mmtest
  (B) tricks       qfae_abcd_{drop02,linattn,hardmine,dinomaly}_out_mmtest
  (C) metrics      every arm emits flow_EPE / flow_EPEn / flow_ang / flow_magr;
                   the 'maha' rescore adds flow_maha
  (D) ROI          qfae_abcd_rescore_<enc>_{base,roi_center,roi_motion,roi_lv}_mmtest

READ IT FOR DIRECTION, NOT FOR WINNERS. The CIs are roughly +-0.16 on ACDC (10 NOR)
and +-0.09 on M&Ms (32 NOR); run-to-run jitter alone is ~+-0.03 on ACDC. A single
cell beating the baseline by 0.02 is noise. What counts is a lever that moves the
same way across arms, streams and both rules.

The 'base' rescore arm is a REGRESSION GATE: with --roi none nothing in the scoring
path has changed, so its flow_SSIM/mag_SSIM/flow_L1 must reproduce the stored
qfae_dino2d_<enc>_out_mmtest numbers exactly. A mismatch means the batch silently
altered the established pipeline and every other number here is suspect.
"""

import json
import os

import numpy as np

from qfae_report import _load, _reduce_slices, _patient_auc

BAR_ACDC, BAR_MM = 0.8125, 0.7310                      # the GAN Flow-SSIM baseline

MOTION = ["flow_SSIM", "mag_SSIM", "flow_L1"]
NEW_METRICS = ["flow_EPE", "flow_EPEn", "flow_ang", "flow_magr", "flow_maha"]
GT_ONLY = ["gt_mag_mean", "gt_mag_std", "gt_mag_cv"]
RULES = ["mean", "middle60"]

A_ARMS = [("nq8", "8 latents"), ("nq16", "16"), ("nq32", "32"), ("nq64", "64"),
          ("nq196", "196 (control: no compression)")]
B_ARMS = [("drop02", "dropout 0.2"), ("linattn", "linear attention"),
          ("hardmine", "hard-mining 25%"), ("dinomaly", "all three")]
D_ARMS = [("base", "whole frame"), ("roi_center", "central disc"),
          ("roi_motion", "moving pixels"), ("roi_lv", "heart segmentation")]


def _npz(path):
    return os.path.join(path, "qfae_dino_arrays.npz")


def patient_reduce(a, stream, rule):
    """Per-slice stream -> (per-patient score, labels, datasets) under a reduction rule."""
    vals, pids, slcs = np.asarray(a[stream], float), a["pids"], a["slcs"]
    labels, ds = a["labels"], a["datasets"]
    sc, pl, pd, kept = [], [], [], []
    for p in np.unique(pids):
        m = pids == p
        v = vals[m][np.argsort(slcs[m])]
        v = v[np.isfinite(v)]
        if v.size == 0:                     # keep pid alignment: drop the patient, not just its score
            continue
        sc.append(_reduce_slices(v, len(v), rule))
        pl.append(labels[m][0]); pd.append(ds[m][0]); kept.append(p)
    return np.array(sc), np.array(pl), np.array(pd), np.array(kept)


def auc_cell(a, stream, rule, n_boot=2000):
    """{'ACDC': (auc, lo, hi), 'MM': ...} for one (stream, rule)."""
    sc, pl, pd, pids = patient_reduce(a, stream, rule)
    out = {}
    for dsn in ("ACDC", "MM"):
        m = pd == dsn
        if m.sum() == 0 or len(np.unique(pl[m] != "NOR")) < 2:
            out[dsn] = (np.nan, np.nan, np.nan)
            continue
        r = _patient_auc(sc[m], pids[m], pl[m], n_boot=n_boot)
        out[dsn] = (r["auc"], r["lo"], r["hi"])
    return out


def fmt(cell):
    a, m = cell["ACDC"][0], cell["MM"][0]
    return f"{a:.3f}/{m:.3f}" if np.isfinite(a) and np.isfinite(m) else "   --    "


def load_arm(path):
    try:
        return _load(_npz(path))
    except (FileNotFoundError, OSError):
        return None


def table(title, rows, streams, rule, note=""):
    """rows = [(label, arrays_or_None)]."""
    print(f"\n{title}   [rule = {rule}]")
    if note:
        print(f"    {note}")
    head = "    " + f"{'arm':<34}" + "".join(s.rjust(14) for s in streams)
    print(head)
    print("    " + "-" * (len(head) - 4))
    for label, a in rows:
        if a is None:
            print(f"    {label:<34}{'(not run)':>14}")
            continue
        line = f"    {label:<34}"
        for s in streams:
            line += (fmt(auc_cell(a, s, rule)).rjust(14) if s in a else f"{'--':>14}")
        print(line)


def regression_gate(enc):
    """The roi=none rescore must reproduce the stored baseline results.json exactly."""
    base = load_arm(f"./qfae_abcd_rescore_{enc}_base_mmtest")
    ref_path = f"./qfae_dino2d_{enc}_out_mmtest/qfae_dino_results.json"
    print(f"\n[GATE] regression check ({enc}): roi=none rescore vs the stored baseline")
    if base is None or not os.path.exists(ref_path):
        print("    SKIPPED — rescore or reference results.json missing")
        return
    ref = json.load(open(ref_path))["streams"]
    ok = True
    for s in MOTION:
        if s not in ref or s not in base:
            continue
        got = auc_cell(base, s, "mean")
        for dsn in ("ACDC", "MM"):
            want = ref[s]["per_dataset"][dsn].get("overall", float("nan"))
            d = abs(got[dsn][0] - want)
            flag = "OK " if d < 5e-3 else "MISMATCH"
            if d >= 5e-3:
                ok = False
            print(f"    {flag} {s:<10} {dsn:<5} stored={want:.4f}  rescored={got[dsn][0]:.4f}  "
                  f"|d|={d:.4f}")
    print("    => scoring path unchanged" if ok else
          "    => DIVERGED: the batch altered the established pipeline. Fix before reading anything below.")


def main():
    print("=" * 92)
    print("A/B/C/D BATCH — held-out ACDC-50 / M&Ms-Testing, patient-Mean AUC (ACDC/M&Ms per cell)")
    print(f"GAN bar = {BAR_ACDC:.3f}/{BAR_MM:.3f}   |   CI ~ +-0.16 ACDC, +-0.09 M&Ms — read direction, not winners")
    print("=" * 92)

    for enc in ("mae224", "dino224"):
        if os.path.exists(f"./qfae_abcd_rescore_{enc}_base_mmtest"):
            regression_gate(enc)

    baseline = load_arm("./qfae_dino2d_mae224_out_mmtest")

    # ── (A) the bottleneck ──
    rows = [("direct decoder (baseline, 196 latents)", baseline)]
    rows += [(f"perceiver, {lbl}", load_arm(f"./qfae_abcd_{arm}_out_mmtest")) for arm, lbl in A_ARMS]
    for rule in RULES:
        table("(A) BOTTLENECK — Q-Former latent count, MAE@224", rows, MOTION, rule,
              note="nq196 is the control: perceiver decode with NO compression. If nq196 ~ baseline "
                   "and small-N differs, the effect is compression, not the decoder." if rule == "mean" else "")

    # ── (B) Dinomaly tricks ──
    rows = [("baseline (softmax, dropout 0.01)", baseline)]
    rows += [(lbl, load_arm(f"./qfae_abcd_{arm}_out_mmtest")) for arm, lbl in B_ARMS]
    for rule in RULES:
        table("(B) DINOMALY TRICKS — one at a time on the direct decoder", rows, MOTION, rule)

    # ── (C) the flow score-way sweep, on every arm that ran ──
    all_rows = [("baseline MAE@224", baseline)]
    all_rows += [(f"A/{arm}", load_arm(f"./qfae_abcd_{arm}_out_mmtest")) for arm, _ in A_ARMS]
    all_rows += [(f"B/{arm}", load_arm(f"./qfae_abcd_{arm}_out_mmtest")) for arm, _ in B_ARMS]
    all_rows += [(f"C-D/{enc}/{arm}", load_arm(f"./qfae_abcd_rescore_{enc}_{arm}_mmtest"))
                 for enc in ("mae224", "dino224")
                 for arm, _ in D_ARMS + [("maha", "")]]
    all_rows = [(l, a) for l, a in all_rows if a is not None]
    for rule in RULES:
        table("(C) FLOW SCORE-WAY — new metrics vs the two established ones",
              all_rows, ["flow_SSIM", "mag_SSIM"] + NEW_METRICS, rule,
              note="flow_EPEn/ang/magr are scale- or direction-only: the ones expected to survive "
                   "the vendor shift." if rule == "mean" else "")

    # ── (D) ROI pooling ──
    for enc in ("mae224", "dino224"):
        rows = [(lbl, load_arm(f"./qfae_abcd_rescore_{enc}_{arm}_mmtest")) for arm, lbl in D_ARMS]
        if all(a is None for _, a in rows):
            continue
        for rule in RULES:
            table(f"(D) ROI POOLING — {enc}", rows, MOTION + ["flow_EPEn", "flow_ang"], rule,
                  note="'whole frame' is the current behaviour; the rest restrict pooling to the heart."
                       if rule == "mean" else "")

    # NOTE: the ROI circularity controls (qfae_roictrl.pbs) are collated by analyze_roictrl.py,
    # which owns the paired statistics that actually decide the question.

    # ── headline: best cell per dataset, and what beats the bar ──
    print("\n" + "=" * 92)
    print("BEST CELLS (every arm x stream x rule that ran)")
    best = []
    for label, a in all_rows:
        for s in MOTION + NEW_METRICS:
            if s not in a:
                continue
            for rule in RULES:
                c = auc_cell(a, s, rule, n_boot=200)          # cheap CI for the scan
                if np.isfinite(c["ACDC"][0]) and np.isfinite(c["MM"][0]):
                    best.append((label, s, rule, c["ACDC"][0], c["MM"][0]))
    if best:
        for key, name in ((4, "M&Ms-Test"), (3, "ACDC-50")):
            top = sorted(best, key=lambda r: -r[key])[:5]
            print(f"\n  top 5 by {name}:")
            for lbl, s, rule, acdc, mm in top:
                mark = " <- beats the GAN bar" if (mm > BAR_MM if key == 4 else acdc > BAR_ACDC) else ""
                print(f"    {lbl:<22} {s:<10} {rule:<9} ACDC={acdc:.3f}  M&Ms={mm:.3f}{mark}")
    print("\nReminder: M&Ms-Val (9 NOR) cannot select these recipes — anything read off this table")
    print("is a test-set read unless it was pre-committed. See conclusion 9 of the summary.")


if __name__ == "__main__":
    main()
