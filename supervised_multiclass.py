"""supervised_multiclass.py — multi-class disease typing, to match CineMA's Table 5 metric.

WHY THIS EXISTS
---------------
Our supervised ceiling probe (supervised_fit.py) scores BINARY NOR-vs-disease AUC, because
that is what the unsupervised detector does and what the rest of this project reports. CineMA's
headline Table 5 number ("Classification", ROC AUC 97.98% ACDC / 77.41% M&Ms) is instead
MULTI-CLASS disease typing, one-vs-rest. Comparing our binary number to their multi-class one
is a category error, so this script recomputes ours on their task using the SAME cached
features, so at least the metric and the task line up.

WHAT IT STILL DOES NOT FIX: the split. CineMA does not report which patients it held out, so
this is "same metric, same task, unknown-vs-known split" — closer, but not apples-to-apples.
Only re-running CineMA's own fine-tuning on OUR held-out patients would be that.

CLASS-SPACE PROBLEM (real, and CineMA does not say how they handled it): M&Ms Testing contains
labels absent from M&Ms Training — 'Other' (24 patients) and 'LVNC' (2). A classifier cannot
predict a class it never saw. We therefore report macro AUC over the classes present in BOTH
train and test, and state explicitly how many test patients that excludes.

    python supervised_multiclass.py --cache_dir ./sup_cache_mae224_ge
"""

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qfae_report import _fast_auc, _middle_slice_range
from supervised_fit import (STREAM_BLOCK, load_cache, stack_splits, stream_matrix,
                            slice_weights)

# CineMA Table 5, CineMAFineTune "Classification" ROC AUC (arXiv 2506.00679).
CINEMA = {"ACDC": 0.9798, "MM": 0.7741}

# The exact label sets CineMA's released CVD classifiers use, read from
# hf://mathpluscode/CineMA finetuned/classification_cvd/{acdc,mnms}_sax/config.yaml.
# ACDC matches our full label space; for M&Ms they keep only 5 classes and DROP
# AHS / IHD / LVNC / Other — which is how they sidestep the unseen-class problem.
# Matching this exactly is what makes our number comparable to theirs.
CINEMA_CLASSES = {"ACDC": ["DCM", "HCM", "MINF", "NOR", "RV"],
                  "MM": ["DCM", "HCM", "NOR", "ARV", "HHD"]}

# Table 5's F1 column sits under CLASSIFICATION, not Detection — so the like-for-like
# counterpart is a MACRO-F1 over disease classes from argmax predictions, not a binary F1.
CINEMA_F1 = {"ACDC": 0.8427, "MM": 0.3992}


def macro_f1(P, y, classes):
    """Macro-F1 over classes from argmax predictions — matches CineMA's Table 5 F1 column."""
    pred = np.asarray(classes)[np.argmax(P, axis=1)]
    f1s = []
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fn == 0:                       # class absent from this cohort
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else np.nan


def patient_probs(P, pids, slcs, rule):
    """(N_slices, K) class probabilities -> (N_patients, K) by averaging over a patient's slices."""
    out, keep = [], []
    for p in np.unique(pids):
        m = pids == p
        v = P[m][np.argsort(slcs[m])]
        if rule == "middle60":
            idx = list(_middle_slice_range(len(v)))
            v = v[idx] if idx else v
        out.append(v.mean(axis=0)); keep.append(p)
    return np.asarray(out), np.asarray(keep)


def macro_ovr_auc(P, y, classes):
    """One-vs-rest macro ROC AUC. Returns (macro, weighted, per-class dict)."""
    per, sup = {}, {}
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.min() == yy.max():                    # class absent from the test set
            continue
        per[c] = float(_fast_auc(yy, P[:, i]))
        sup[c] = int(yy.sum())
    if not per:
        return np.nan, np.nan, {}
    macro = float(np.mean(list(per.values())))
    tot = sum(sup.values())
    weighted = float(sum(per[c] * sup[c] for c in per) / tot)
    return macro, weighted, per


def make_clf(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=5000,
                           class_weight="balanced"))


def select_C(X, y, pids, slcs, classes, cs, n_splits, seed, rule):
    """Grouped-CV pick of C by out-of-fold multi-class macro AUC (patients never straddle folds)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups=pids))
    best, best_s, table = cs[len(cs) // 2], -np.inf, []
    for C in cs:
        oof = np.zeros((len(y), len(classes)))
        for tr, te in folds:
            assert not (set(pids[tr]) & set(pids[te])), "patient straddles a fold"
            clf = make_clf(C)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(X[tr], y[tr],
                        logisticregression__sample_weight=slice_weights(pids[tr]))
            cls_i = {c: i for i, c in enumerate(clf.classes_)}
            for j, c in enumerate(classes):
                if c in cls_i:
                    oof[te, j] = clf.predict_proba(X[te])[:, cls_i[c]]
        Pp, kp = patient_probs(oof, pids, slcs, rule)
        yp = np.array([y[pids == p][0] for p in kp])
        s, _, _ = macro_ovr_auc(Pp, yp, classes)
        table.append(dict(C=float(C), oof_macro_auc=s))
        if np.isfinite(s) and s > best_s:
            best, best_s = C, s
    return best, best_s, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="./sup_cache_mae224_ge")
    ap.add_argument("--out_json", default="./sup_probe_mae224_ge_mmtest/multiclass_results.json")
    ap.add_argument("--streams", nargs="+",
                    default=["appe", "motion", "motionstat", "motionraw", "both"])
    ap.add_argument("--rules", nargs="+", default=["mean", "middle60"])
    ap.add_argument("--cs", nargs="+", type=float, default=list(np.logspace(-4, 2, 13)))
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cinema_classes", action="store_true",
                    help="Restrict BOTH train and test to CineMA's exact released label sets "
                         "(M&Ms: DCM/HCM/NOR/ARV/HHD only). Required for the numbers to be "
                         "comparable to their Table 5.")
    args = ap.parse_args()

    print("=" * 100)
    print("MULTI-CLASS DISEASE TYPING — one-vs-rest macro ROC AUC, to match CineMA Table 5")
    print(f"  CineMA CineMAFineTune 'Classification': ACDC {CINEMA['ACDC']:.4f} | "
          f"M&Ms {CINEMA['MM']:.4f}")
    print("  NOTE: same metric + same task as CineMA, but their SPLIT is unreported —")
    print("        this is not a like-for-like benchmark. See the module docstring.")
    print("=" * 100)

    results = {}
    for dsn, tr_split, te_split in (("ACDC", "acdc_train", "acdc_test"),
                                    ("MM", "mm_train", "mm_test")):
        tr, te = stack_splits(args.cache_dir, (tr_split,)), stack_splits(args.cache_dir, (te_split,))
        if args.cinema_classes:                       # match their released label space exactly
            allow = CINEMA_CLASSES[dsn]
            n0_tr, n0_te = len(np.unique(tr["pids"])), len(np.unique(te["pids"]))
            tr = {k: v[np.isin(tr["labels"], allow)] for k, v in tr.items()}
            te = {k: v[np.isin(te["labels"], allow)] for k, v in te.items()}
            print(f"\n[cinema_classes] {dsn}: restricted to {allow} — "
                  f"train {n0_tr}->{len(np.unique(tr['pids']))} patients, "
                  f"test {n0_te}->{len(np.unique(te['pids']))}")
        tr_cls = sorted(set(tr["labels"]))
        te_cls = sorted(set(te["labels"]))
        classes = sorted(set(tr_cls) & set(te_cls))
        dropped = sorted(set(te_cls) - set(tr_cls))
        keep = np.isin(te["labels"], classes)
        n_drop_pat = len(np.unique(te["pids"][~keep]))

        print(f"\n### {dsn}")
        print(f"    train classes {tr_cls}  ({len(np.unique(tr['pids']))} patients)")
        print(f"    test  classes {te_cls}  ({len(np.unique(te['pids']))} patients)")
        print(f"    evaluated on {classes}")
        if dropped:
            print(f"    ** EXCLUDED from the metric: {dropped} = {n_drop_pat} test patients "
                  f"never seen in training (a classifier cannot predict an unseen class) **")

        te_k = {k: v[keep] for k, v in te.items()}
        yte = te_k["labels"]
        for stream in args.streams:
            X, Xte = stream_matrix(tr, stream), stream_matrix(te_k, stream)
            for rule in args.rules:
                C, oof, table = select_C(X, tr["labels"], tr["pids"], tr["slcs"], classes,
                                         args.cs, args.n_splits, args.seed, rule)
                clf = make_clf(C)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clf.fit(X, tr["labels"],
                            logisticregression__sample_weight=slice_weights(tr["pids"]))
                cls_i = {c: i for i, c in enumerate(clf.classes_)}
                proba = clf.predict_proba(Xte)
                P = np.zeros((len(yte), len(classes)))
                for j, c in enumerate(classes):
                    if c in cls_i:
                        P[:, j] = proba[:, cls_i[c]]
                Pp, kp = patient_probs(P, te_k["pids"], te_k["slcs"], rule)
                yp = np.array([yte[te_k["pids"] == p][0] for p in kp])
                macro, weighted, per = macro_ovr_auc(Pp, yp, classes)
                f1 = macro_f1(Pp, yp, classes)
                bar, barf1 = CINEMA[dsn], CINEMA_F1[dsn]
                print(f"    {stream:<11}{rule:<10}C={C:<9.4g} macroAUC {macro:.3f} "
                      f"(CineMA {bar:.3f}, {macro - bar:+.3f})   macroF1 {f1:.3f} "
                      f"(CineMA {barf1:.3f}, {f1 - barf1:+.3f})")
                results[f"{dsn}_{stream}_{rule}"] = dict(
                    macro_auc=macro, weighted_auc=weighted, macro_f1=f1, per_class=per,
                    C=float(C), oof_macro_auc=oof, cinema_auc=bar, cinema_f1=barf1,
                    n_patients=int(len(kp)), excluded_classes=dropped,
                    excluded_patients=int(n_drop_pat))
        best = max((v["macro_auc"] for k, v in results.items() if k.startswith(dsn)
                    and np.isfinite(v["macro_auc"])), default=np.nan)
        print(f"    --> best {dsn} macro AUC {best:.3f}   vs CineMA {CINEMA[dsn]:.3f}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
