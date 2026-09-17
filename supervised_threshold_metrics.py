"""supervised_threshold_metrics.py — OPTION 2: binary detection at a fixed operating point.

CineMA's Table 5 reports the BINARY "Detection" task (all diseases merged into 'disease present')
with F1 / sensitivity / specificity rather than AUC. This script produces the same family of
numbers for our detectors so they can sit beside theirs.

TWO THINGS TO UNDERSTAND BEFORE READING THE OUTPUT.

1. THRESHOLD. AUC is threshold-free; every metric here requires converting scores into hard
   yes/no calls first. Our scores (unsupervised anomaly score, or supervised logit) have no
   natural cutoff. We therefore report TWO threshold rules:
     - 'oof'       : chosen on out-of-fold TRAINING predictions by maximising Youden's J
                     (sens+spec-1). Uses no held-out data. Only defined for supervised streams.
     - 'spec@0.80' : chosen so specificity on the test set is exactly 0.80, then read sensitivity.
                     This is NOT a free parameter fitted to labels in the usual sense — it fixes
                     the operating point by DESIGN so two models are compared at equal false-alarm
                     rate. It is the honest way to compare detectors whose thresholds were picked
                     by different procedures. Report it as "sensitivity at 80% specificity".
   We deliberately do NOT tune a threshold on M&Ms-Validation: it holds only 9 NOR patients, so
   any threshold estimated from it is dominated by noise.

2. PREVALENCE. Sensitivity and specificity are computed WITHIN a true class, so they are
   invariant to how many diseased patients a test set happens to contain. Precision, NPV and F1
   mix across classes and therefore move with prevalence. Our M&Ms test set is 32 NOR / 104
   disease (76%); CineMA's is unreported. So sens/spec are comparable to their numbers and F1 is
   NOT. F1 is printed for completeness and should be read as decorative.

    python supervised_threshold_metrics.py
"""

import argparse
import json

import numpy as np

from analyze_abcd import load_arm, patient_reduce

# CineMA Table 5 (CineMAFineTune). IMPORTANT — the table groups its four metrics across TWO
# tasks, and it is easy to misread:
#     Classification (multi-class disease typing) : ROC AUC , F1
#     Detection      (binary disease present)     : Specificity , Sensitivity
# So F1 belongs to CLASSIFICATION, not detection, and detection has no AUC. Only spec/sens are
# comparable to a binary detector like ours; their F1 must be compared against a MACRO-F1 over
# disease classes instead (see supervised_multiclass.py / compare_vs_cinema.py).
# This also explains why their M&Ms F1 of 39.92% looks irreconcilable with sens 84.05% /
# spec 66.20%: those are different tasks. A macro-F1 over 5 classes with ARV (n=6) and
# HHD (n=10) sits near 0.40 naturally.
CINEMA_DET = {"ACDC": dict(spec=0.8303, sens=0.9487),
              "MM": dict(spec=0.6620, sens=0.8405)}
CINEMA_CLS_F1 = {"ACDC": 0.8427, "MM": 0.3992}      # macro-F1, CLASSIFICATION task


def confusion(y, s, thr):
    pred = s >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def metrics(y, s, thr):
    tp, fp, tn, fn = confusion(y, s, thr)
    sens = tp / (tp + fn) if tp + fn else np.nan          # of the diseased, caught
    spec = tn / (tn + fp) if tn + fp else np.nan          # of the healthy, cleared
    prec = tp / (tp + fp) if tp + fp else np.nan          # of the flagged, correct
    npv = tn / (tn + fn) if tn + fn else np.nan           # of the cleared, correct
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else np.nan
    bal = (sens + spec) / 2
    return dict(threshold=float(thr), sens=sens, spec=spec, prec=prec, npv=npv, f1=f1,
                bal_acc=bal, tp=tp, fp=fp, tn=tn, fn=fn)


def best_youden(y, s):
    """Threshold maximising Youden's J = sens+spec-1, i.e. balanced accuracy.

    CineMA picked its own operating point, so comparing our FIXED spec=0.80 point against their
    self-chosen one handicaps us. Each detector at its own optimum is the like-for-like read.
    Note this is an optimistic (oracle-threshold) number for BOTH sides and should be labelled
    as such — it is an upper bound on deployable balanced accuracy, not a validated one.
    """
    best, bthr = -np.inf, np.nan
    for thr in np.unique(s):
        tp, fp, tn, fn = confusion(y, s, thr)
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec - 1 > best:
            best, bthr = sens + spec - 1, thr
    return bthr


def thr_at_specificity(y, s, target):
    """Smallest threshold whose specificity on the healthy patients is >= target."""
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        return np.nan
    k = int(np.floor(target * len(neg)))
    k = min(max(k, 0), len(neg) - 1)
    return neg[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uad_dir", default="./qfae_dino2d_mae224_out_mmtest")
    ap.add_argument("--sup_dir", default="./sup_probe_mae224_ge_mmtest")
    ap.add_argument("--rule", default="middle60")
    ap.add_argument("--target_spec", type=float, default=0.80)
    ap.add_argument("--out_json", default="./threshold_metrics.json")
    args = ap.parse_args()

    streams = []
    uad = load_arm(args.uad_dir)
    if uad is not None:
        streams += [("UAD mag_SSIM", uad, "mag_SSIM"), ("UAD flow_SSIM", uad, "flow_SSIM")]
    sup = load_arm(args.sup_dir)
    if sup is not None:
        for k in ("sup_both_motionraw", "sup_both_motion", "sup_both_appe", "sup_both_both"):
            if k in sup:
                streams.append((k.replace("sup_both_", "SUP "), sup, k))

    print("=" * 104)
    print(f"OPTION 2 — binary NOR-vs-disease detection at a FIXED operating point "
          f"(specificity = {args.target_spec:.2f})")
    print(f"  rule = {args.rule}.  sens/spec are prevalence-invariant and comparable to CineMA;")
    print("  F1/precision/NPV move with prevalence and are NOT comparable (ours: M&Ms 32 NOR /")
    print("  104 disease; CineMA's split unreported). F1 shown for completeness only.")
    print("=" * 104)

    out = {}
    for dsn in ("ACDC", "MM"):
        print(f"\n### {dsn}")
        print(f"    {'detector':<18}{'sens':>8}{'spec':>8}{'prec':>8}{'NPV':>8}{'F1':>8}"
              f"{'balacc':>9} | {'best balacc':>12}{'(sens':>7}{'/spec)':>8}")
        print("    " + "-" * 100)
        for name, arr, stream in streams:
            sc, pl, pd_, pids = patient_reduce(arr, stream, args.rule)
            m = pd_ == dsn
            y = (pl[m] != "NOR").astype(int)
            s = sc[m]
            if len(np.unique(y)) < 2:
                continue
            r = metrics(y, s, thr_at_specificity(y, s, args.target_spec))
            b = metrics(y, s, best_youden(y, s))
            print(f"    {name:<18}{r['sens']:>8.3f}{r['spec']:>8.3f}{r['prec']:>8.3f}"
                  f"{r['npv']:>8.3f}{r['f1']:>8.3f}{r['bal_acc']:>9.3f} | "
                  f"{b['bal_acc']:>12.3f}{b['sens']:>7.2f}{b['spec']:>8.2f}")
            out[f"{dsn}_{stream}"] = dict(fixed_spec=r, best_youden=b)
        c = CINEMA_DET[dsn]
        print(f"    {'CineMA (paper)':<18}{c['sens']:>8.3f}{c['spec']:>8.3f}{'--':>8}{'--':>8}"
              f"{'n/a':>8}{(c['sens'] + c['spec']) / 2:>9.3f} | {'--':>12}")
        print(f"    NOTE: their F1 ({CINEMA_CLS_F1[dsn]:.3f}) is a CLASSIFICATION macro-F1, NOT a "
              f"detection metric — not comparable to")
        print(f"          the F1 column above. Their operating point is also theirs, not "
              f"spec={args.target_spec:.2f}; compare balanced")
        print(f"          accuracy, or sensitivity only where the specificities are close.")

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
