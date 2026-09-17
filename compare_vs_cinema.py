"""compare_vs_cinema.py — sensitivity / specificity / F1, ours vs CineMA, on IDENTICAL patients.

This supersedes supervised_threshold_metrics.py's comparison against the paper's printed
Detection numbers. Those carried an unfixable ambiguity: CineMA's split is unreported, and their
threshold was chosen by them. Here we instead use the per-patient probabilities CineMA's OWN
released classifiers produced on OUR held-out patients (cinema_finetuned_eval.py), so both
models are scored on the same people with the same rules.

BINARY SCORE FROM A 5-WAY SOFTMAX. CineMA's CVD head is multi-class; their "Detection" task
merges all diseases into 'disease present'. The corresponding score is therefore

    s_disease = 1 - P(NOR)

which is well defined even for a patient whose specific class CineMA never trained on (M&Ms
IHD / LVNC / Other): such a patient IS abnormal, and the model only has to avoid calling them
NOR. We therefore report the binary comparison on BOTH cohorts:
  * full     — all held-out patients (ACDC 50, M&Ms 136). This is our real detection task.
  * in-class — only patients whose label is inside CineMA's label space (ACDC 50, M&Ms 106),
               which is the cohort most favourable to them.

THRESHOLDS. Every metric here needs one, and AUC does not, so two rules are reported:
  * spec@0.80   — threshold set so specificity is 0.80 for BOTH models. Equal false-alarm rate,
                  so sensitivities are directly comparable. This is the headline.
  * best-Youden — each model at its own sens+spec-1 optimum, chosen ON THE TEST SET. Optimistic
                  for both, and an upper bound rather than a deployable number.

PREVALENCE. Sensitivity and specificity are computed within a true class and so are unaffected
by how many diseased patients the cohort contains. Precision, NPV and F1 are not. Since both
models are scored on the SAME cohort here, all of them are comparable — unlike the earlier
comparison against the paper.

    python compare_vs_cinema.py
"""

import argparse
import json
import os

import numpy as np

from analyze_abcd import load_arm, patient_reduce
from qfae_report import _fast_auc
from supervised_threshold_metrics import metrics, thr_at_specificity, best_youden
from supervised_multiclass import macro_f1


def cinema_binary(dsn, seed_reduce="mean"):
    """-> (pids, labels, disease_score) from the saved CineMA probabilities, or None."""
    p = f"cinema_ft_probs_{dsn}.npz"
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    classes = list(d["classes"])
    nor = classes.index("NOR")
    seeds = sorted(k for k in d.files if k.startswith("probs_seed"))
    s = np.stack([1.0 - d[k][:, nor] for k in seeds])          # (n_seeds, n_patients)
    score = s.mean(axis=0) if seed_reduce == "mean" else s[0]
    return d["pids"], d["labels"], score, d["in_class"], len(seeds)


def ours(dsn, arr, stream, rule):
    sc, pl, pd_, pids = patient_reduce(arr, stream, rule)
    m = pd_ == dsn
    return pids[m], pl[m], sc[m]


def row(name, y, s, target_spec):
    a = _fast_auc(y, s)
    f = metrics(y, s, thr_at_specificity(y, s, target_spec))
    b = metrics(y, s, best_youden(y, s))
    print(f"    {name:<22}{a:>7.3f}{f['sens']:>8.3f}{f['spec']:>8.3f}{f['prec']:>8.3f}"
          f"{f['npv']:>7.3f}{f['f1']:>8.3f}{f['bal_acc']:>8.3f} |{b['bal_acc']:>8.3f}"
          f"{b['sens']:>7.2f}{b['spec']:>7.2f}")
    return dict(auc=float(a), fixed_spec=f, best_youden=b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uad_dir", default="./qfae_dino2d_mae224_out_mmtest")
    ap.add_argument("--sup_dir", default="./sup_probe_mae224_ge_mmtest")
    ap.add_argument("--rule", default="middle60")
    ap.add_argument("--target_spec", type=float, default=0.80)
    ap.add_argument("--cohorts", nargs="+", default=["full", "in-class"])
    ap.add_argument("--out_json", default="./compare_vs_cinema.json")
    args = ap.parse_args()

    uad, sup = load_arm(args.uad_dir), load_arm(args.sup_dir)
    OURS = []
    if uad is not None:
        OURS += [("UAD mag_SSIM", uad, "mag_SSIM"), ("UAD flow_SSIM", uad, "flow_SSIM")]
    if sup is not None:
        OURS += [(f"SUP {s}", sup, f"sup_both_{s}")
                 for s in ("motionraw", "motion", "appe", "both")
                 if f"sup_both_{s}" in sup]

    print("=" * 112)
    print("OURS vs CineMA — binary NOR-vs-disease on IDENTICAL held-out patients")
    print(f"  CineMA score = 1 - P(NOR) from its own released classifiers, averaged over seeds.")
    print(f"  rule={args.rule}; 'spec@{args.target_spec:.2f}' fixes an equal false-alarm rate for")
    print("  both models, so sensitivities are directly comparable. best-balacc is oracle-threshold.")
    print("=" * 112)

    out = {}
    for dsn in ("ACDC", "MM"):
        cin = cinema_binary(dsn)
        if cin is None:
            print(f"\n### {dsn}: cinema_ft_probs_{dsn}.npz missing — run cinema_finetuned_eval.py")
            continue
        cpids, clabels, cscore, in_class, n_seeds = cin
        for cohort in args.cohorts:
            mask = np.ones(len(cpids), bool) if cohort == "full" else in_class.astype(bool)
            keep_pids = set(cpids[mask])
            y_c = (clabels[mask] != "NOR").astype(int)
            print(f"\n### {dsn}  [{cohort}]  {int(mask.sum())} patients "
                  f"({int((y_c == 0).sum())} NOR / {int(y_c.sum())} disease), "
                  f"CineMA averaged over {n_seeds} seeds")
            print(f"    {'model':<22}{'AUC':>7}{'sens':>8}{'spec':>8}{'prec':>8}{'NPV':>7}"
                  f"{'F1':>8}{'balacc':>8} |{'best':>8}{'sens':>7}{'spec':>7}")
            print("    " + "-" * 104)
            out[f"{dsn}_{cohort}_CineMA"] = row("CineMA fine-tuned", y_c, cscore[mask],
                                                args.target_spec)
            for name, arr, stream in OURS:
                pids, labels, sc = ours(dsn, arr, stream, args.rule)
                sel = np.array([p in keep_pids for p in pids])
                order = {p: i for i, p in enumerate(cpids[mask])}
                idx = np.argsort([order[p] for p in pids[sel]])   # align to CineMA's patient order
                y = (labels[sel][idx] != "NOR").astype(int)
                if len(y) != int(mask.sum()) or len(np.unique(y)) < 2:
                    print(f"    {name:<22} (patient mismatch: {len(y)} vs {int(mask.sum())})")
                    continue
                out[f"{dsn}_{cohort}_{stream}"] = row(name, y, sc[sel][idx], args.target_spec)

    # ── CineMA's Table 5 F1 is a CLASSIFICATION macro-F1, so the only like-for-like F1
    #    comparison is against our multi-class probe, not against any binary detector.
    print("\n" + "=" * 112)
    print("CLASSIFICATION macro-F1 — the metric CineMA's Table 5 F1 column actually reports")
    print("  (their F1 sits under Classification, NOT Detection; detection has no AUC/F1)")
    print("=" * 112)
    for dsn, paper_f1 in (("ACDC", 0.8427), ("MM", 0.3992)):
        p = f"cinema_ft_probs_{dsn}.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        classes = list(d["classes"])
        ic = d["in_class"].astype(bool)
        y = d["labels"][ic]
        seeds = sorted(k for k in d.files if k.startswith("probs_seed"))
        f1s = [macro_f1(d[k][ic], y, classes) for k in seeds]
        print(f"\n### {dsn}  ({int(ic.sum())} patients, CineMA classes {classes})")
        print(f"    CineMA on OUR patients : macro-F1 {np.mean(f1s):.3f} "
              f"(sd {np.std(f1s):.3f}, seeds {[round(f, 3) for f in f1s]})")
        print(f"    CineMA reported (paper): macro-F1 {paper_f1:.3f}")
        mc = "./sup_probe_mae224_ge_mmtest/multiclass_cinema_matched.json"
        if os.path.exists(mc):
            j = json.load(open(mc))
            ours_f1 = {k: v.get("macro_f1") for k, v in j.items()
                       if k.startswith(dsn + "_") and v.get("macro_f1") is not None}
            if ours_f1:
                best = max(ours_f1, key=ours_f1.get)
                print(f"    our probe (best of {len(ours_f1)}): macro-F1 "
                      f"{ours_f1[best]:.3f}  [{best}]")
        out[f"{dsn}_classification_macroF1_cinema"] = dict(
            mean=float(np.mean(f1s)), sd=float(np.std(f1s)), seeds=f1s, paper=paper_f1)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
