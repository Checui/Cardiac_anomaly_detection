"""supervised_fit.py — the supervised NOR-vs-disease ceiling probe (CPU, no GPU).

Stage 2. Loads the cached feature blocks from supervised_features.py, fits a regularised
linear probe per (training pool x feature stream), and writes per-slice held-out scores in
the SAME on-disk format as every unsupervised run, so analyze_abcd / analyze_roictrl read it
unmodified.

PROTOCOL (matched to the project's honest evaluation — no new splits):
  train  ACDC database/training (100 pts) and/or M&Ms Training (150, or 175 with recovery)
  report ACDC database/testing (50: 10 NOR/40 dis) + M&Ms Testing (136: 32/104)
Patients are disjoint by folder construction; asserted anyway. `C` is chosen by
StratifiedGroupKFold(5) grouped on PATIENT inside the training pool, so no held-out data and
no M&Ms-Validation data is touched.

WHY THE LOGIT IS THE PRIMARY SCORE, NOT predict_proba. The patient reduction is a mean over
slices, and a mean is NOT invariant to a per-slice monotone transform. On a near-separable
ACDC fit the probabilities saturate at ~0.999, the patient mean collapses into a narrow band,
and the AUC degrades through sigmoid saturation rather than through the features. Both are
written; a material disagreement between `sup_*` and `sup_*_prob` is a diagnostic, not noise.

    python supervised_fit.py --cache_dir ./sup_cache_mae224 --out_dir ./sup_probe_mae224_mmtest
"""

import argparse
import json
import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qfae_report import _fast_auc, _reduce_slices

STREAM_BLOCK = {"appe": ("emb_frame",), "motion": ("emb_flow",),
                "motionstat": ("stat_flow",), "appestat": ("stat_appe",),
                "motionraw": ("raw_flow",), "both": ("emb_frame", "emb_flow")}

POOLS = {"acdc": ("acdc_train",), "mm": ("mm_train",), "both": ("acdc_train", "mm_train"),
         "acdc_dh": ("acdc_train",), "mm_dh": ("mm_train",),
         "both_dh": ("acdc_train", "mm_train"),
         # No-recovery counterparts: the pool every existing model in this project actually
         # trained on. All 25 ED==ES cases in M&Ms Training are GE and GE-Training is exactly
         # those 25, so excluding vendor GE reproduces the historical 150-patient pool exactly
         # (asserted below). Lets one feature cache serve both arms — no second GPU pass.
         "mm_norec": ("mm_train",), "both_norec": ("acdc_train", "mm_train")}
POOL_EXCLUDE_VENDOR = {"mm_norec": ("GE",), "both_norec": ("GE",)}
MATCHED = ("NOR", "DCM", "HCM")          # the ACDC-and-M&Ms label intersection


def load_cache(cache_dir, split):
    d = np.load(os.path.join(cache_dir, f"{split}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def stack_splits(cache_dir, splits, label_subset=None, exclude_vendor=None):
    parts = [load_cache(cache_dir, s) for s in splits]
    keys = [k for k in parts[0] if not k.endswith("_names")]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if label_subset is not None:
        m = np.isin(out["labels"], label_subset)
        out = {k: v[m] for k, v in out.items()}
    if exclude_vendor:
        m = ~np.isin(out["vendors"], exclude_vendor)
        out = {k: v[m] for k, v in out.items()}
    return out


def stream_matrix(c, stream):
    return np.concatenate([c[b] for b in STREAM_BLOCK[stream]], axis=1).astype(np.float64)


def slice_weights(pids):
    """1 / n_slices(patient) — evaluation is patient-level, so thick stacks must not dominate."""
    u, inv, cnt = np.unique(pids, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def patient_auc_from_slices(scores, pids, slcs, labels, rule="mean"):
    sc, pl = [], []
    for p in np.unique(pids):
        m = pids == p
        v = np.asarray(scores[m], float)[np.argsort(slcs[m])]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        sc.append(_reduce_slices(v, len(v), rule)); pl.append(labels[m][0])
    sc, pl = np.asarray(sc), np.asarray(pl)
    y = (pl != "NOR").astype(int)
    return np.nan if y.min() == y.max() else float(_fast_auc(y, sc))


def make_probe(C):
    return make_pipeline(
        StandardScaler(),                                # inside the pipeline: refit per fold
        LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=5000,
                           class_weight="balanced"))


def _fit_predict(X, y, w, tr, te, C):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe = make_probe(C)
        pipe.fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
        n_conv = sum(1 for c in caught if issubclass(c.category, ConvergenceWarning))
    return pipe.decision_function(X[te]), n_conv


def grouped_cv_select(X, y, w, pids, slcs, labels, datasets, cs, n_splits, seed, rule, n_jobs):
    """Out-of-fold patient AUC per C. Grouped on patient so no patient straddles a fold."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups=pids))
    for tr, te in folds:                                   # GATE 7, asserted at runtime
        assert not (set(pids[tr]) & set(pids[te])), "patient straddles a CV fold"

    def one(C):
        oof = np.full(len(y), np.nan)
        conv = 0
        for tr, te in folds:
            s, n = _fit_predict(X, y, w, tr, te, C)
            oof[te] = s; conv += n
        uds = np.unique(datasets)
        if len(uds) > 1:                                   # mean of per-dataset AUCs, so the
            aucs = []                                      # easier dataset can't dominate
            for d in uds:
                m = datasets == d
                a = patient_auc_from_slices(oof[m], pids[m], slcs[m], labels[m], rule)
                if np.isfinite(a):
                    aucs.append(a)
            score = float(np.mean(aucs)) if aucs else np.nan
        else:
            score = patient_auc_from_slices(oof, pids, slcs, labels, rule)
        return dict(C=float(C), oof_auc=score, n_convergence_warnings=int(conv))

    table = Parallel(n_jobs=n_jobs)(delayed(one)(C) for C in cs)
    valid = [r for r in table if np.isfinite(r["oof_auc"])]
    best = max(valid, key=lambda r: r["oof_auc"])["C"] if valid else float(np.median(cs))
    return best, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="./sup_cache_mae224")
    ap.add_argument("--out_dir", default="./sup_probe_mae224_mmtest")
    ap.add_argument("--pools", nargs="+", default=list(POOLS))
    ap.add_argument("--streams", nargs="+", default=list(STREAM_BLOCK))
    ap.add_argument("--cs", nargs="+", type=float, default=list(np.logspace(-4, 2, 13)))
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--select_rule", default="mean")
    ap.add_argument("--lovo", action="store_true", help="Philips<->Siemens leave-one-vendor-out.")
    ap.add_argument("--shuffle_control", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test = stack_splits(args.cache_dir, ("acdc_test", "mm_test"))   # ACDC first — matches the eval
    n_test = len(test["pids"])
    print("=" * 92)
    print(f"SUPERVISED PROBE | test slices {n_test}, patients {len(np.unique(test['pids']))}")
    print("=" * 92)

    save = {k: test[k] for k in ("pids", "labels", "datasets", "slcs", "vendors", "centres")}
    results, coef_norms = {}, {}

    for pool in args.pools:
        subset = MATCHED if pool.endswith("_dh") else None
        tr = stack_splits(args.cache_dir, POOLS[pool], label_subset=subset,
                          exclude_vendor=POOL_EXCLUDE_VENDOR.get(pool))
        if pool in POOL_EXCLUDE_VENDOR:                 # must reproduce the historical pool
            n_mm = len(np.unique(tr["pids"][tr["datasets"] == "MM"]))
            assert n_mm == 150, (f"pool {pool} has {n_mm} M&Ms patients, expected the "
                                 f"historical 150 — the GE/ED==ES correspondence broke")
        y = (tr["labels"] != "NOR").astype(int)
        if len(np.unique(y)) < 2:
            print(f"\n### pool {pool}: single class, skipped"); continue
        assert not (set(tr["pids"]) & set(test["pids"])), f"GATE 3: {pool} overlaps the test set"
        w = slice_weights(tr["pids"])
        print(f"\n### pool {pool}: {len(np.unique(tr['pids']))} patients, {len(y)} slices, "
              f"{int((tr['labels'] == 'NOR').sum())} NOR slices, "
              f"vendors={dict(zip(*np.unique(tr['vendors'], return_counts=True)))}")

        for stream in args.streams:
            Xtr, Xte = stream_matrix(tr, stream), stream_matrix(test, stream)
            best, table = grouped_cv_select(Xtr, y, w, tr["pids"], tr["slcs"], tr["labels"],
                                            tr["datasets"], args.cs, args.n_splits, args.seed,
                                            args.select_rule, args.n_jobs)
            pipe = make_probe(best)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipe.fit(Xtr, y, logisticregression__sample_weight=w)
                nconv = sum(1 for c in caught if issubclass(c.category, ConvergenceWarning))
            lr = pipe.named_steps["logisticregression"]
            cn = float(np.linalg.norm(lr.coef_))
            key = f"sup_{pool}_{stream}"
            save[key] = pipe.decision_function(Xte).astype(np.float32)
            save[key + "_prob"] = pipe.predict_proba(Xte)[:, 1].astype(np.float32)
            coef_norms[key] = cn
            edge = "EDGE" if best in (min(args.cs), max(args.cs)) else ""
            oof = max((r["oof_auc"] for r in table if np.isfinite(r["oof_auc"])), default=np.nan)
            print(f"    {stream:<11} dim={Xtr.shape[1]:<5} C={best:<9.4g} {edge:<5}"
                  f"oof={oof:.3f}  |coef|={cn:8.3f}  conv_warn={nconv}")
            results[key] = dict(C=best, oof_auc=oof, coef_l2=cn, dim=int(Xtr.shape[1]),
                                n_train_patients=int(len(np.unique(tr["pids"]))),
                                n_train_slices=int(len(y)), convergence_warnings=int(nconv),
                                cv_table=table)

        # house z-scored-logit fusion (fusion_sweep_2d idiom), no extra fit
        ka, km = f"sup_{pool}_appe", f"sup_{pool}_motion"
        if ka in save and km in save:
            z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            save[f"sup_{pool}_fuse_appe_motion"] = (0.5 * (z(save[ka]) + z(save[km]))).astype(np.float32)

    # ── label-shuffle control: shuffled at PATIENT level (slice-level gives a false-tight null)
    if args.shuffle_control and "both" in args.pools:
        tr = stack_splits(args.cache_dir, POOLS["both"])
        up = np.unique(tr["pids"])
        plab = np.array([tr["labels"][tr["pids"] == p][0] for p in up]) != "NOR"
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(plab)
        lut = dict(zip(up, perm))
        ysh = np.array([lut[p] for p in tr["pids"]]).astype(int)
        X = stream_matrix(tr, "both")
        pipe = make_probe(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X, ysh, logisticregression__sample_weight=slice_weights(tr["pids"]))
        save["sup_shuffle_both_both"] = pipe.decision_function(stream_matrix(test, "both")).astype(np.float32)
        print("\n[CONTROL] patient-level label shuffle fitted (expect held-out AUC ~0.35-0.65)")

    # ── leave-one-vendor-out inside M&Ms Training (matched case mix, no held-out data used) ──
    lovo = {}
    if args.lovo:
        mm = stack_splits(args.cache_dir, ("mm_train",))
        print("\n### LOVO inside M&Ms-Training (Philips <-> Siemens)")
        for held in ("Philips", "Siemens"):
            te_m, tr_m = mm["vendors"] == held, mm["vendors"] != held
            if te_m.sum() == 0 or tr_m.sum() == 0:
                continue
            ytr = (mm["labels"][tr_m] != "NOR").astype(int)
            yte = (mm["labels"][te_m] != "NOR").astype(int)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                print(f"    train on !{held}: single class on one side "
                      f"(train NOR={int((ytr == 0).sum())}, test NOR={int((yte == 0).sum())})"
                      f" — skipped")
                continue
            for stream in args.streams:
                X = stream_matrix(mm, stream)
                # C selected by grouped CV WITHIN the training vendor(s) — never on the held-out
                # vendor. A fixed C would confound genuine vendor-specificity with overfitting,
                # which matters because these AUCs can land below chance.
                wtr = slice_weights(mm["pids"][tr_m])
                best, _ = grouped_cv_select(X[tr_m], ytr, wtr, mm["pids"][tr_m],
                                            mm["slcs"][tr_m], mm["labels"][tr_m],
                                            mm["datasets"][tr_m], args.cs, args.n_splits,
                                            args.seed, args.select_rule, args.n_jobs)
                pipe = make_probe(best)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipe.fit(X[tr_m], ytr, logisticregression__sample_weight=wtr)
                s = pipe.decision_function(X[te_m])
                a = patient_auc_from_slices(s, mm["pids"][te_m], mm["slcs"][te_m],
                                            mm["labels"][te_m], "mean")
                lovo[f"{held}_{stream}"] = dict(auc=a, C=best)
                print(f"    train on !{held:<8} -> test {held:<8} {stream:<11} "
                      f"C={best:<9.4g} AUC {a:.3f}")

    np.savez_compressed(os.path.join(args.out_dir, "qfae_dino_arrays.npz"), **save)
    with open(os.path.join(args.out_dir, "supervised_results.json"), "w") as f:
        json.dump(dict(cache_dir=args.cache_dir, pools=args.pools, streams=args.streams,
                       select_rule=args.select_rule, seed=args.seed,
                       fits=results, lovo=lovo), f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_dir}/qfae_dino_arrays.npz "
          f"({len([k for k in save if k.startswith('sup_')])} score streams)")
    print(f"Next: python analyze_supervised.py --sup_dir {args.out_dir}")


if __name__ == "__main__":
    main()
