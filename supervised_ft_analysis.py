"""Aggregate the CineMA supervised fine-tune grid into the three thesis deliverables.

(a) Label-efficiency curves — patient AUROC of the binary CineMA fine-tune vs label
    fraction, with the unsupervised detectors (GAN flow-SSIM, QFAE MAE@224 mag-SSIM
    middle-60%) and the authors' released 5-way fine-tune (1 − P(NOR)) as horizontal
    zero-label / full-label references. One CSV + PNG per dataset.
(b) LODO table — per held-out disease: supervised-unseen vs matched-N control vs
    full-label supervised vs the unsupervised detectors, one-vs-NOR AUC paired on
    the common patient subset.
(c) Threshold metrics — AUROC, AUPRC (+prevalence), Se@90%Sp, Youden-point
    sens/spec/F1/balanced-accuracy for every model family through one code path.
    Operating points are computed on the held-out ROC (no calibration split exists
    for the frozen artifacts) — comparable across models, optimistic in absolute
    terms; say so when quoting.

Tolerates missing cells (prints SKIP) so trimmed grids still analyze.
Inputs: supervised_ft_out/<ds>_<cell>_s<seed>.npz (supervised_ft_eval.py),
gan_flowmetric_out/scores.npz, qfae_mx_mae224_mae_out_mmtest/qfae_dino_arrays.npz,
cinema_ft_probs_{ACDC,MM}.npz. Outputs land in supervised_ft_out/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_abcd import patient_reduce
from qfae_report import _fast_auc, _load

REPO = Path(__file__).resolve().parent
OUT = REPO / "supervised_ft_out"
FRACTIONS = (10, 25, 50, 100)
SEEDS = (0, 1, 2)
LODO_FOLDS = [("acdc", d) for d in ("DCM", "HCM", "MINF", "RV")] + [("mnms", d) for d in ("DCM", "HCM")]
DS_NAME = {"acdc": "ACDC", "mnms": "MM"}

# dataviz reference palette (validated, light mode; WARN slots get direct labels)
C_SUP, C_GAN, C_QFAE, C_REL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def strip_prefix(pids) -> np.ndarray:
    return np.array([re.sub(r"^(ACDC_|MM_|MMVAL_)", "", str(p)) for p in pids])


# ── model families: each returns {ds: (pids, pathology, score)} with raw pid codes ──
def gan_scores() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    z = np.load(REPO / "gan_flowmetric_out" / "scores.npz", allow_pickle=True)
    out = {}
    for ds, skey, pkey, lkey in (("acdc", "ACDC__flow_ssim", "acdc_pids", "acdc_labels"),
                                 ("mnms", "MM__flow_ssim", "mm_pids", "mm_labels")):
        pids, labels, vals = strip_prefix(z[pkey]), z[lkey].astype(str), z[skey].astype(float)
        upids = np.unique(pids)
        score = np.array([vals[pids == p].mean() for p in upids])
        path = np.array([labels[pids == p][0] for p in upids])
        out[ds] = (upids, path, score)
    return out


def qfae_scores() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    a = _load(REPO / "qfae_mx_mae224_mae_out_mmtest" / "qfae_dino_arrays.npz")
    sc, pl, pds, pids = patient_reduce(a, "mag_SSIM", "middle60")
    pids = strip_prefix(pids)
    return {ds: (pids[pds == tag], pl[pds == tag].astype(str), sc[pds == tag])
            for ds, tag in (("acdc", "ACDC"), ("mnms", "MM"))}


def released_ft_scores() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-seed scores, shape (3, n): CineMA's convention is metrics per seed then
    averaged (never posterior ensembling), so seeds are kept separate here."""
    out = {}
    for ds, tag in (("acdc", "ACDC"), ("mnms", "MM")):
        z = np.load(REPO / f"cinema_ft_probs_{tag}.npz", allow_pickle=True)
        classes = [str(c) for c in z["classes"]]
        nor = classes.index("NOR")
        S = np.stack([1.0 - z[f"probs_seed{s}"][:, nor] for s in (0, 1, 2)])
        out[ds] = (strip_prefix(z["pids"]), z["labels"].astype(str), S)
    return out


def sup_cell(ds: str, cell: str, seed: int):
    """(pids, pathology, score=P(abnormal)) for one fine-tuned cell, or None if absent."""
    p = OUT / f"{ds}_{cell}_s{seed}.npz"
    if not p.exists():
        print(f"SKIP missing cell {p.name}")
        return None
    z = np.load(p, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    return z["pids"].astype(str), z["pathology"].astype(str), 1.0 - z["probs"][:, classes.index("NOR")]


def sup_seeds(ds: str, cell: str, seeds=SEEDS):
    """(pids, pathology, S) with S of shape (n_seeds, n) — seeds kept separate,
    matching CineMA's metrics-per-seed-then-average reporting convention."""
    per_seed = [x for x in (sup_cell(ds, cell, s) for s in seeds) if x is not None]
    if not per_seed:
        return None
    ref_pids = per_seed[0][0]
    assert all(np.array_equal(ref_pids, x[0]) for x in per_seed[1:]), f"{ds}_{cell}: pid order differs across seeds"
    return ref_pids, per_seed[0][1], np.stack([x[2] for x in per_seed])


# ── metrics ──────────────────────────────────────────────────────────────────
# Convention (CineMA's): a multi-seed model is scored per seed and the metric is
# averaged over seeds; each bootstrap resample also averages the per-seed metrics,
# mirroring the paper's "metrics were averaged across the three trained model
# checkpoints for each bootstrap sample". Scores may be (n,) or (n_seeds, n).
def _seed_mean_auc(y, S):
    S = np.atleast_2d(np.asarray(S, float))
    return float(np.nanmean([_fast_auc(y, s) for s in S]))


def boot_ci(y, S, n_boot=2000, seed=0):
    y = np.asarray(y)
    S = np.atleast_2d(np.asarray(S, float))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        boots.append(np.nanmean([_fast_auc(y[idx], s[idx]) for s in S]))
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def roc_points(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def threshold_metrics(y, s) -> dict:
    from sklearn.metrics import average_precision_score
    y, s = np.asarray(y), np.asarray(s, float)
    fpr, tpr, thr = roc_points(y, s)
    ok = fpr <= 0.10
    se_at_90sp = float(tpr[ok].max()) if ok.any() else 0.0
    j = int(np.argmax(tpr - fpr))
    pred = (s >= thr[j]).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(
        auroc=float(_fast_auc(y, s)),
        auprc=float(average_precision_score(y, s)),
        prevalence=float(y.mean()),
        se_at_90sp=se_at_90sp,
        youden_sens=sens,
        youden_spec=spec,
        youden_f1=2 * tp / max(2 * tp + fp + fn, 1),
        youden_bal_acc=(sens + spec) / 2,
        n=len(y),
    )


def restrict(model, keep_pids):
    pids, path, score = model
    m = np.isin(pids, list(keep_pids))
    score = score[..., m]  # works for (n,) and (n_seeds, n)
    return pids[m], path[m], score


def auc_of(model, subset_pathologies=None):
    """Seed-mean AUC + bootstrap CI; per-seed AUCs returned for multi-seed models."""
    pids, path, score = model
    if subset_pathologies is not None:
        m = np.isin(path, list(subset_pathologies))
        pids, path, score = pids[m], path[m], score[..., m]
    y = (path != "NOR").astype(int)
    if y.min() == y.max():
        return np.nan, (np.nan, np.nan), 0, []
    lo, hi = boot_ci(y, score)
    per_seed = [float(_fast_auc(y, s)) for s in np.atleast_2d(np.asarray(score, float))]
    return _seed_mean_auc(y, score), (lo, hi), len(y), per_seed


# ── (a) label-efficiency ─────────────────────────────────────────────────────
def label_efficiency(baselines) -> dict:
    results = {}
    for ds in ("acdc", "mnms"):
        rows = []
        for pct in FRACTIONS:
            model = sup_seeds(ds, f"frac{pct}")
            if model is None:
                continue
            auc, (lo, hi), n, per_seed = auc_of(model)
            _, path, S = model
            ens = float(_fast_auc((path != "NOR").astype(int), S.mean(axis=0)))
            n_train = int(np.mean([np.load(OUT / f"{ds}_frac{pct}_s{s}.npz")["n_train"]
                                   for s in SEEDS if (OUT / f"{ds}_frac{pct}_s{s}.npz").exists()]))
            rows.append(dict(fraction_pct=pct, n_train=n_train, auc=auc, ci_lo=lo, ci_hi=hi,
                             auc_sd_of_seeds=float(np.nanstd(per_seed)),
                             seed_aucs=";".join(f"{a:.4f}" for a in per_seed),
                             auc_posterior_ensemble=ens, n_test=n))
        base_rows = []
        for name, model in baselines.items():
            auc, (lo, hi), n, _ = auc_of(model[ds])
            base_rows.append(dict(model=name, auc=auc, ci_lo=lo, ci_hi=hi, n_test=n))
        df, bdf = pd.DataFrame(rows), pd.DataFrame(base_rows)
        df.to_csv(OUT / f"label_efficiency_{DS_NAME[ds]}.csv", index=False)
        bdf.to_csv(OUT / f"label_efficiency_{DS_NAME[ds]}_baselines.csv", index=False)
        results[ds] = dict(curve=rows, baselines=base_rows)
        if rows:
            plot_label_efficiency(ds, df, bdf)
    return results


def plot_label_efficiency(ds: str, df: pd.DataFrame, bdf: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)

    base_colors = {"GAN flow-SSIM": C_GAN, "QFAE mag-SSIM mid60": C_QFAE, "CineMA released FT": C_REL}
    # label placement: top reference on the right; the rest stack on the left,
    # alternating above/below their lines — the curve occupies the right half,
    # so closely spaced reference labels stay collision-free on the left
    for i, (_, r) in enumerate(bdf.sort_values("auc", ascending=False).iterrows()):
        c = base_colors.get(r["model"], INK2)
        ax.axhline(r["auc"], color=c, linewidth=2, linestyle=(0, (5, 3)))
        txt = f"{r['model']}  {r['auc']:.3f}"
        if i == 0:
            near_top = r["auc"] > 0.97  # y-axis tops out at 1.0; drop the label below the line
            ax.annotate(txt, xy=(100, r["auc"]), xytext=(0, -4 if near_top else 3),
                        textcoords="offset points", color=INK2, fontsize=8, ha="right",
                        va="top" if near_top else "baseline")
        else:
            below = i % 2 == 0
            ax.annotate(txt, xy=(10, r["auc"]), xytext=(0, -4 if below else 3),
                        textcoords="offset points", color=INK2, fontsize=8,
                        ha="left", va="top" if below else "baseline")

    x = df["fraction_pct"].to_numpy()
    ax.fill_between(x, df["ci_lo"], df["ci_hi"], color=C_SUP, alpha=0.12, linewidth=0)
    # primary line: mean of per-seed AUCs (CineMA's reporting convention, same as
    # the ceiling table). Individual seeds stay visible as faint points so seed
    # instability (ACDC 100%) is not hidden.
    for _, r in df.iterrows():
        seed_vals = [float(v) for v in str(r["seed_aucs"]).split(";") if v]
        ax.scatter([r["fraction_pct"]] * len(seed_vals), seed_vals, s=14, color=C_SUP,
                   alpha=0.4, linewidths=0, zorder=2)
    ax.plot(x, df["auc"], color=C_SUP, linewidth=2, marker="o", markersize=6, zorder=3)
    ax.annotate(f"CineMA binary FT (mean of 3 seeds)  {df['auc'].iloc[-1]:.3f}",
                xy=(x[-1], df["auc"].iloc[-1]), xytext=(0, 8),
                textcoords="offset points", color=INK, fontsize=8.5, ha="right")

    ax.set_xticks(list(FRACTIONS), [f"{p}%" for p in FRACTIONS])
    ax.set_xlabel("training labels used", color=INK2, fontsize=10)
    ax.set_ylabel("patient AUROC (NOR vs abnormal)", color=INK2, fontsize=10)
    ax.set_title(f"{DS_NAME[ds]} held-out — supervised label efficiency vs unsupervised", color=INK, fontsize=11)
    lo = min(0.5, float(df["ci_lo"].min()) - 0.02)
    ax.set_ylim(lo, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / f"label_efficiency_{DS_NAME[ds]}.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote label_efficiency_{DS_NAME[ds]}.png")


# ── (b) LODO ─────────────────────────────────────────────────────────────────
def lodo_table(baselines) -> pd.DataFrame:
    rows = []
    for ds, disease in LODO_FOLDS:
        cells = {"sup_lodo": f"lodo{disease}", "sup_ctrl": f"ctrl{disease}", "sup_full": "frac100"}
        models = {name: m for name, cell in cells.items() if (m := sup_seeds(ds, cell)) is not None}
        for name, fam in baselines.items():
            models[name] = fam[ds]
        if not models:
            continue
        common = set.intersection(*(set(m[0]) for m in models.values()))
        for name, model in models.items():
            auc, (lo, hi), n, per_seed = auc_of(restrict(model, common), {disease, "NOR"})
            rows.append(dict(dataset=DS_NAME[ds], held_out_disease=disease, model=name,
                             auc=auc, ci_lo=lo, ci_hi=hi, n_patients=n,
                             seed_aucs=";".join(f"{a:.4f}" for a in per_seed) if len(per_seed) > 1 else "",
                             n_common_pool=len(common)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lodo_table.csv", index=False)
    if not df.empty:
        print(df.pivot_table(index=["dataset", "held_out_disease"], columns="model",
                             values="auc").round(3).to_string())
    return df


# ── (c) threshold metrics ────────────────────────────────────────────────────
def extra_metrics(baselines) -> pd.DataFrame:
    rows = []
    for ds in ("acdc", "mnms"):
        fams = {name: fam[ds] for name, fam in baselines.items()}
        m = sup_seeds(ds, "frac100")
        if m is not None:
            fams["CineMA binary FT (100%)"] = m
        for name, model in fams.items():
            _, path, score = model
            y = (path != "NOR").astype(int)
            # multi-seed models: metrics per seed, then averaged (paper convention)
            per = [threshold_metrics(y, s) for s in np.atleast_2d(np.asarray(score, float))]
            avg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            avg["n"] = int(per[0]["n"])
            rows.append(dict(dataset=DS_NAME[ds], model=name, **avg))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "threshold_metrics.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


def _point_metrics(y, pred) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(sens=sens, spec=spec, f1=2 * tp / max(2 * tp + fp + fn, 1), bal_acc=(sens + spec) / 2)


def _sens_at_spec(y, s, spec_target: float) -> float:
    fpr, tpr, _ = roc_points(y, s)
    ok = fpr <= (1.0 - spec_target) + 1e-9
    return float(tpr[ok].max()) if ok.any() else 0.0


def _val_scores_mm() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """M&Ms-Validation patient scores for threshold selection (GAN + QFAE)."""
    out = {}
    z = np.load(REPO / "gan_flowmetric_out" / "scores.npz", allow_pickle=True)
    pids, labels = strip_prefix(z["mmval_pids"]), z["mmval_labels"].astype(str)
    vals = z["MM_VAL__flow_ssim"].astype(float)
    upids = np.unique(pids)
    sc = np.array([vals[pids == p].mean() for p in upids])
    y = np.array([labels[pids == p][0] != "NOR" for p in upids]).astype(int)
    out["GAN flow-SSIM"] = (sc, y)
    a = _load(REPO / "qfae_dino2d_mae224_out_mmval" / "qfae_dino_arrays.npz")
    sc, pl, pds, _ = patient_reduce(a, "mag_SSIM", "middle60")
    m = pds == "MM"
    out["QFAE mag-SSIM mid60"] = (sc[m], (pl[m] != "NOR").astype(int))
    return out


def operating_points(baselines) -> pd.DataFrame:
    """Sensitivity/specificity comparison at explicit operating points.

    Supervised models are binarized as P(healthy) < 0.5, i.e. detection score
    1 - P(NOR) > 0.5 — the released code computes sens/spec only for binary heads,
    so the paper's multiclass binarization is not in the code; this rule reproduces
    their printed M&Ms sensitivity to within 0.003 on the in-class subset (0.838 vs
    0.841; argmax != NOR gives 0.806) and is the natural threshold on the same score
    the AUC rows rank. Metrics per seed then averaged. Unsupervised scores have no
    built-in threshold, so three labelled policies are reported: matched-spec (sens
    at the released-FT operating specificity; rank-based), val-threshold (Youden
    picked on M&Ms-Validation, applied to test; M&Ms only — no ACDC validation
    scores exist), and test-Youden (optimistic ceiling).
    """
    rows = []
    val_mm = _val_scores_mm()
    for ds in ("acdc", "mnms"):
        rel = None
        for sup_name, model in (("CineMA released FT", baselines["CineMA released FT"][ds]),
                                ("CineMA binary FT (100%)", sup_seeds(ds, "frac100"))):
            if model is None:
                continue
            _, path, S = model
            y = (path != "NOR").astype(int)
            per = [_point_metrics(y, (s > 0.5).astype(int)) for s in np.atleast_2d(S)]
            avg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            rows.append(dict(dataset=DS_NAME[ds], model=sup_name, policy="P(healthy) < 0.5", **avg))
            if rel is None:
                rel = avg  # released FT sets the matched-spec target
        # unsupervised at three labelled policies
        for name in ("GAN flow-SSIM", "QFAE mag-SSIM mid60"):
            _, path, sc = baselines[name][ds]
            y = (path != "NOR").astype(int)
            sens = _sens_at_spec(y, sc, rel["spec"])
            rows.append(dict(dataset=DS_NAME[ds], model=name, policy=f"matched spec ({rel['spec']:.2f})",
                             sens=sens, spec=rel["spec"], f1=np.nan, bal_acc=(sens + rel["spec"]) / 2))
            if ds == "mnms":
                vs, vy = val_mm[name]
                fpr, tpr, thr = roc_points(vy, vs)
                t = thr[int(np.argmax(tpr - fpr))]
                rows.append(dict(dataset=DS_NAME[ds], model=name, policy="val threshold (MM-Val Youden)",
                                 **_point_metrics(y, (sc >= t).astype(int))))
            fpr, tpr, thr = roc_points(y, sc)
            t = thr[int(np.argmax(tpr - fpr))]
            rows.append(dict(dataset=DS_NAME[ds], model=name, policy="test Youden (ceiling)",
                             **_point_metrics(y, (sc >= t).astype(int))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "operating_points.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


def main() -> None:
    OUT.mkdir(exist_ok=True)
    baselines = {"GAN flow-SSIM": gan_scores(), "QFAE mag-SSIM mid60": qfae_scores(),
                 "CineMA released FT": released_ft_scores()}
    summary = dict(label_efficiency=label_efficiency(baselines))
    lodo_table(baselines)
    extra_metrics(baselines)
    operating_points(baselines)
    with open(OUT / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[done] outputs in {OUT}/")


if __name__ == "__main__":
    main()
