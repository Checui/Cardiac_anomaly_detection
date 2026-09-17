"""qfae_report.py -- offline per-dataset patient-level QFAE report with bootstrap CIs.

Rebuilds the anomaly-detection numbers for every saved QFAE run PURELY from the
`*_out*/qfae_arrays.npz` files (no GPU, no re-inference), on the SAME footing the
GAN notebook uses: patient-level AUC, aggregated Mean over a patient's stacks,
reported per dataset (ACDC / M&M) AND pooled, with stratified bootstrap 95% CIs
and paired-difference bootstrap between streams.

Why: the QFAE headline numbers mixed stack-level and patient-level AUCs and were
reported without CIs on a 9-NOR M&M set. This puts them in the GAN's units.

IMPORTANT cohort note (see the eval-audit memory): every saved QFAE array scores
ACDC `testing` (50 patients, IDENTICAL to the GAN's ACDC 50) + M&M **Validation
folder only** (34 patients, 9 NOR). The GAN's honest held-out table reports on
M&M **Testing** (which the QFAE has never scored). So:
  * the ACDC column here IS apples-to-apples with the GAN;
  * the M&M column here is the GAN's *selection* set (M&M-val, 9 NOR) -- NOT its
    held-out M&M-test. The held-out M&M-test QFAE number needs a re-run
    (`qfae_eval.py --mm_val_dir ../Dataset_1/Testing`); see --heldout below.

`aggregate_stacks_to_patient` is inlined (identical to derisk_cinema.py's) so the script
has no heavy imports; AUC uses a vectorised Mann-Whitney (`_fast_auc`) for a fast bootstrap.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _fast_auc(y, s):
    """AUC via the Mann-Whitney U statistic (vectorised; ties get 0.5 credit).

    Orders of magnitude faster than sklearn.roc_auc_score inside a bootstrap loop,
    and identical value. y in {0,1}; s float scores (higher = more anomalous).
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def aggregate_stacks_to_patient(scores, pids, labels):
    """Per-stack scores -> per-patient Mean over that patient's stacks (parity with derisk_cinema)."""
    scores = np.asarray(scores, dtype=float)
    pids = np.asarray(pids)
    labels = np.asarray(labels)
    uniq = np.unique(pids)
    mean = np.zeros(len(uniq))
    plabels = []
    for i, pid in enumerate(uniq):
        m = pids == pid
        mean[i] = float(np.mean(scores[m]))
        plabels.append(labels[m][0])
    return {"Mean": mean}, np.array(plabels)

# Every run's array dir + the score-stream keys it stores. Flow runs share 5 streams;
# the appearance-only and DINO runs store a single `scores`/`appearance` column.
RUNS = [
    ("qfae_appearance",  "qfae_out/qfae_arrays.npz",                    ["scores"]),
    ("qfae_flow",        "qfae_flow_out/qfae_arrays.npz",               ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
    ("qfae_flow_reg",    "qfae_flow_reg_out/qfae_arrays.npz",           ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
    ("qfae_masked_0.5",  "qfae_masked_out_motion_0.5/qfae_arrays.npz",  ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
    ("qfae_dino",        "qfae_dino_out/qfae_dino_arrays.npz",          ["scores"]),
    ("qfae_dino_mae",    "qfae_dino_mae_out/qfae_dino_arrays.npz",      ["scores"]),
]

# Held-out re-evals on M&Ms-TESTING (~31 NOR) + ACDC-50 (--mmtest). The appearance combos store
# `scores`; the CineMA motion runs already scored M&Ms-test store the 5 flow streams.
MMTEST_RUNS = [
    ("hybrid_a dino->cinema",  "qfae_hybrid_a_out_mmtest/qfae_arrays.npz",      ["scores"]),
    ("hybrid_b cinema->dino",  "qfae_hybrid_b_out_mmtest/qfae_arrays.npz",      ["scores"]),
    ("dino  dino->dino",       "qfae_dino_out_mmtest/qfae_dino_arrays.npz",     ["scores"]),
    ("dino_mae dino->mae",     "qfae_dino_mae_out_mmtest/qfae_dino_arrays.npz", ["scores"]),
    ("cinema motion (Farneback)", "qfae_flow_out_mmtest/qfae_arrays.npz",       ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
    ("cinema motion (reg)",       "qfae_flow_reg_out_mmtest/qfae_arrays.npz",   ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
    ("cinema motion (masked)",    "qfae_masked_out_mmtest/qfae_arrays.npz",     ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
    ("dino_flow (dino motion)",   "qfae_hybrid_dino_flow_out_mmtest/qfae_arrays.npz", ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
    ("mae_flow  (mae motion)",    "qfae_hybrid_mae_flow_out_mmtest/qfae_arrays.npz",  ["appearance", "flow_SSIM", "mag_SSIM", "flow_L1", "combined"]),
]


def _load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ── per-slice reduction rules (offline reduction sweep) ──────────────────────
def _middle_slice_range(Z, frac=0.2):
    """Middle (1-2*frac) of Z slices — verbatim copy of data_loader.py:202-211 (GAN's rule)."""
    n_drop = max(1, int(round(frac * Z)))
    stop = max(n_drop, Z - n_drop)
    return range(n_drop, stop)


def _reduce_slices(vec, z, rule):
    """Reduce a per-slice error vector (len 16, padded) to one scalar using the real slice count z.

    rule: 'mean' | 'top10'|'top20'|'top30' | 'max' | 'middle60'. Operates on the first z real slices
    (padding trailing), except 'mean_all' which mirrors the old scalar (mean over ALL 16 incl. pad).
    """
    z = max(1, int(z))
    real = np.asarray(vec[:z], dtype=float)
    if rule == "mean_all":
        return float(np.mean(vec))                      # == the legacy scalar (sanity gate)
    if rule == "mean":
        return float(np.mean(real))
    if rule == "max":
        return float(np.max(real))
    if rule == "middle60":
        keep = list(_middle_slice_range(z))
        return float(np.mean(real[keep])) if keep else float(np.mean(real))
    if rule.startswith("top"):
        frac = int(rule[3:]) / 100.0
        k = max(1, int(round(frac * real.size)))
        return float(np.sort(real)[-k:].mean())
    raise ValueError(rule)


REDUCTION_RULES = ["mean_all", "mean", "top10", "top20", "top30", "max", "middle60"]


def _patient_auc(scores, pids, labels, seed=0, n_boot=2000):
    """Patient-Mean AUC (overall) + stratified patient-bootstrap 95% CI + n per class."""
    pat, plabels = aggregate_stacks_to_patient(scores, pids, labels)
    y = (plabels != "NOR").astype(int)
    if y.min() == y.max():
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, n_nor=int((y == 0).sum()), n_dis=int((y == 1).sum()))
    s = pat["Mean"]
    auc = float(_fast_auc(y, s))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        yy = y[bi]
        if yy.min() != yy.max():
            boots.append(_fast_auc(yy, s[bi]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return dict(auc=auc, lo=float(lo), hi=float(hi), n_nor=int((y == 0).sum()), n_dis=int((y == 1).sum()))


def _paired_delta(sA, sB, pids, labels, seed=0, n_boot=2000):
    """Paired patient-bootstrap of AUC(sA) - AUC(sB) on the same patients."""
    patA, pl = aggregate_stacks_to_patient(sA, pids, labels)
    patB, _ = aggregate_stacks_to_patient(sB, pids, labels)
    y = (pl != "NOR").astype(int)
    if y.min() == y.max():
        return dict(delta=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    a, b = patA["Mean"], patB["Mean"]
    delta = _fast_auc(y, a) - _fast_auc(y, b)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ds = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        yy = y[bi]
        if yy.min() != yy.max():
            ds.append(_fast_auc(yy, a[bi]) - _fast_auc(yy, b[bi]))
    ds = np.array(ds)
    lo, hi = (np.percentile(ds, [2.5, 97.5]) if len(ds) else (np.nan, np.nan))
    p = float(2 * min((ds <= 0).mean(), (ds >= 0).mean())) if len(ds) else np.nan
    return dict(delta=float(delta), lo=float(lo), hi=float(hi), p=p)


def report_run(name, path, streams, min_dis=5):
    if not os.path.exists(path):
        print(f"\n### {name}: MISSING ({path})")
        return None
    a = _load(path)
    pids, labels, datasets = a["pids"], a["labels"], a["datasets"]
    print(f"\n{'='*78}\n### {name}   ({path})")
    print(f"    stacks={len(pids)}  patients={len(np.unique(pids))}  "
          f"datasets={sorted(set(datasets.tolist()))}")
    rows = []
    for st in streams:
        if st not in a:
            continue
        sc = a[st]
        line = {"stream": st}
        for ds in ["ACDC", "MM", "ALL"]:
            m = np.ones(len(pids), bool) if ds == "ALL" else (datasets == ds)
            r = _patient_auc(sc[m], pids[m], labels[m])
            line[ds] = r
        rows.append(line)
    # print table
    hdr = f"    {'stream':<12}" + "".join(f"{d:>26}" for d in ["ACDC", "M&Ms", "POOLED"])
    print(hdr)
    for line in rows:
        cells = []
        for ds in ["ACDC", "MM", "ALL"]:
            r = line[ds]
            if np.isnan(r["auc"]):
                cells.append(f"{'n/a':>26}")
            else:
                cells.append(f"{r['auc']:.3f} [{r['lo']:.2f},{r['hi']:.2f}] {r['n_nor']}/{r['n_dis']}".rjust(26))
        print(f"    {line['stream']:<12}" + "".join(cells))
    return dict(name=name, streams=streams, arrays=a, rows=rows)


def verify_against_json(name, path):
    """Sanity: reproduce the reported overall patient-Mean AUC from qfae_results.json if present."""
    jpath = os.path.join(os.path.dirname(path), "qfae_results.json")
    if not os.path.exists(jpath):
        return
    try:
        j = json.load(open(jpath))
        print(f"    [json] {os.path.basename(jpath)} keys: {list(j.keys())[:12]}")
    except Exception as e:  # noqa
        print(f"    [json] unreadable: {e}")


# ── held-out mode (mirrors run_model.ipynb cells 55/58 exactly) ──────────────
# GAN protocol: SELECT the (normalization, appe_metric, flow_metric, w) recipe by AUC on
# M&M-Validation (patient_mean), then REPORT the frozen recipe on held-out ACDC(50) +
# M&M-Testing(136), re-standardising per set. w = appearance weight (w=0 -> flow-only).
# For the QFAE the two blend streams are `appearance` (an error, higher=worse, ~"MSE"-like)
# and `flow_SSIM` (the motion stream, stored higher=worse). We sweep the same raw/zscore/logmu
# normalisations and w in linspace(0,1,21), matching the notebook.

_EPS = 1e-9


def _pos_error_qfae(x, kind):
    """Positive 'higher = worse' error for the QFAE streams (both already higher=worse)."""
    return np.maximum(np.asarray(x, float), _EPS)  # appearance and flow_SSIM are already errors


def _normalize(x, method):
    x = np.asarray(x, float)
    if method == "raw":
        return x
    if method == "logmu":
        return np.log(_pos_error_qfae(x, None))
    return (x - x.mean()) / (x.std() + _EPS)  # zscore


def _patient_streams(arr, ds_mask, appe_key="appearance", flow_key="flow_SSIM"):
    """Return per-patient (appe, flow, y, labels) for the masked dataset, patient-Mean."""
    pids, labels = arr["pids"][ds_mask], arr["labels"][ds_mask]
    pa, pl = aggregate_stacks_to_patient(arr[appe_key][ds_mask], pids, labels)
    pf, _ = aggregate_stacks_to_patient(arr[flow_key][ds_mask], pids, labels)
    y = (pl != "NOR").astype(int)
    return pa["Mean"], pf["Mean"], y, pl


def heldout(val_path, test_path, appe_key="appearance", flow_key="flow_SSIM"):
    """Replicate the GAN honest held-out sweep on QFAE streams.

    val_path  : arrays with M&M-Validation (SELECT set) + ACDC (ignored for select).
    test_path : arrays re-scored on M&M-Testing + ACDC (the REPORT sets).
    """
    val, test = _load(val_path), _load(test_path)
    # SELECT set = M&M patients in the val arrays
    a_sel, f_sel, y_sel, _ = _patient_streams(val, val["datasets"] == "MM", appe_key, flow_key)
    # REPORT sets from the test arrays
    a_ac, f_ac, y_ac, _ = _patient_streams(test, test["datasets"] == "ACDC", appe_key, flow_key)
    a_mt, f_mt, y_mt, _ = _patient_streams(test, test["datasets"] == "MM", appe_key, flow_key)
    a_po = np.concatenate([a_ac, a_mt]); f_po = np.concatenate([f_ac, f_mt]); y_po = np.concatenate([y_ac, y_mt])

    print(f"\n{'='*78}\n### HELD-OUT (select on M&M-val, report on ACDC + M&M-test)")
    print(f"    select M&M-val: {int((y_sel==0).sum())} NOR / {int((y_sel==1).sum())} dis")
    print(f"    report ACDC:    {int((y_ac==0).sum())} NOR / {int((y_ac==1).sum())} dis")
    print(f"    report M&M-test:{int((y_mt==0).sum())} NOR / {int((y_mt==1).sum())} dis")

    def blend(a, f, method, w):
        return w * _normalize(a, method) + (1.0 - w) * _normalize(f, method)

    best = None
    for method in ("raw", "logmu", "zscore"):
        for w in np.linspace(0, 1, 21):
            auc_sel = _fast_auc(y_sel, blend(a_sel, f_sel, method, w))
            if best is None or auc_sel > best["auc_sel"]:
                best = dict(method=method, w=float(w), auc_sel=float(auc_sel))
    m, w = best["method"], best["w"]
    rep = lambda a, f, y: _fast_auc(y, blend(a, f, m, w))  # noqa
    print(f"    BEST recipe on M&M-val: [{m}] w_appe={w:.2f} + flow_SSIM   AUC_sel={best['auc_sel']:.4f}")
    print(f"      -> held-out ACDC     = {rep(a_ac, f_ac, y_ac):.4f}")
    print(f"      -> held-out M&M-test = {rep(a_mt, f_mt, y_mt):.4f}")
    print(f"      -> held-out pooled   = {rep(a_po, f_po, y_po):.4f}")
    # pure flow baseline (w=0) for reference, per set
    print(f"    flow_SSIM alone (w=0):  ACDC={rep(a_ac,f_ac,y_ac) if w==0 else _fast_auc(y_ac,_normalize(f_ac,m)):.4f}  "
          f"M&M-test={_fast_auc(y_mt,_normalize(f_mt,m)):.4f}  pooled={_fast_auc(y_po,_normalize(f_po,m)):.4f}")
    print(f"    (compare GAN held-out: ACDC 0.8125 / M&M-test 0.7310 / pooled 0.7447 [flow-SSIM w=0])")


def _patient_stream(path, stream, ds):
    """Per-patient Mean of one stream for one dataset -> dict pid -> (score, y)."""
    a = _load(path)                                    # _load returns a plain dict of arrays
    key = stream if stream in a else ("scores" if "scores" in a else "appearance")
    m = a["datasets"] == ds
    pat, pl = aggregate_stacks_to_patient(a[key][m], a["pids"][m], a["labels"][m])
    pids = np.unique(a["pids"][m])
    return {p: (float(pat["Mean"][i]), int(pl[i] != "NOR")) for i, p in enumerate(pids)}


def two_file_fuse(specA, specB, n_boot=2000):
    """Tuning-free percentile-rank fusion of two streams from two npz files, aligned by patient.

    spec = 'path.npz:stream'. Rank-averages the two per-patient scores within each dataset (no weight
    to tune), then reports per-dataset + pooled AUC with bootstrap CIs. Use to fuse a DINOv2-scored
    APPEARANCE combo with the CineMA MOTION stream on the same held-out M&Ms-Test patients.
    """
    (pA, sA), (pB, sB) = (s.split(":") for s in (specA, specB))
    print(f"\n### RANK-FUSION  A={specA}  +  B={specB}  (tuning-free)")
    print(f"    {'':<8}{'ACDC':>26}{'M&Ms':>26}{'POOLED':>26}")
    pooled_scores, pooled_y = [], []
    per_ds = {}
    for ds in ["ACDC", "MM"]:
        dA, dB = _patient_stream(pA, sA, ds), _patient_stream(pB, sB, ds)
        common = sorted(set(dA) & set(dB))
        if not common:
            per_ds[ds] = None
            continue
        yA = np.array([dA[p][1] for p in common])
        va = np.array([dA[p][0] for p in common]); vb = np.array([dB[p][0] for p in common])
        # percentile rank within this set, then average (both already higher = more anomalous)
        ra = va.argsort().argsort() / max(1, len(va) - 1)
        rb = vb.argsort().argsort() / max(1, len(vb) - 1)
        fused = 0.5 * (ra + rb)
        per_ds[ds] = (fused, yA, va, vb, common)
        pooled_scores.append(fused); pooled_y.append(yA)

    def _cell(fused, y):
        if len(np.unique(y)) < 2:
            return f"{'n/a':>26}"
        auc = _fast_auc(y, fused)
        rng = np.random.default_rng(0); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; b = []
        for _ in range(n_boot):
            bi = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            if len(np.unique(y[bi])) > 1:
                b.append(_fast_auc(y[bi], fused[bi]))
        lo, hi = np.percentile(b, [2.5, 97.5])
        return f"{auc:.3f} [{lo:.2f},{hi:.2f}] {int((y==0).sum())}/{int((y==1).sum())}".rjust(26)

    # rank-fusion is per-set; pooled uses the concatenated per-set ranks
    fp = np.concatenate(pooled_scores) if pooled_scores else np.array([])
    yp = np.concatenate(pooled_y) if pooled_y else np.array([])
    ac = per_ds.get("ACDC"); mm = per_ds.get("MM")
    print("    fused   "
          + (_cell(ac[0], ac[1]) if ac else f"{'n/a':>26}")
          + (_cell(mm[0], mm[1]) if mm else f"{'n/a':>26}")
          + (_cell(fp, yp) if len(fp) else f"{'n/a':>26}"))
    print("    (compare GAN held-out: ACDC 0.8125 / M&Ms-test 0.7310)")


def _auc_cell(scalar, pids, labels, datasets, ds, n_boot):
    m = np.ones(len(pids), bool) if ds == "ALL" else (datasets == ds)
    r = _patient_auc(scalar[m], pids[m], labels[m], n_boot=n_boot)
    if np.isnan(r["auc"]):
        return f"{'n/a':>26}"
    return f"{r['auc']:.3f} [{r['lo']:.2f},{r['hi']:.2f}] {r['n_nor']}/{r['n_dis']}".rjust(26)


def reduction_sweep(path, n_boot=2000):
    """Sweep per-slice reduction rules on a per-slice npz (from qfae_eval, mask=none).

    For each stream and rule, reduce the per-slice error to a per-stack scalar → patient (single-pass
    = 1 stack/patient) → per-dataset AUC + bootstrap CI. Also the combined = z(appe)+z(flow_SSIM).
    Sanity: rule 'mean_all' must reproduce the eval's scalar stream AUC.
    """
    a = _load(path)
    if "n_real_slices" not in a:
        print(f"### {path}: no per-slice arrays (n_real_slices missing) — re-run qfae_eval with mask=none")
        return
    pids, labels, datasets, nreal = a["pids"], a["labels"], a["datasets"], a["n_real_slices"]
    slice_keys = {"appearance": "appe_slices", "flow_SSIM": "flow_SSIM_slices",
                  "mag_SSIM": "mag_SSIM_slices", "flow_L1": "flow_L1_slices"}
    present = {name: k for name, k in slice_keys.items() if k in a}
    print(f"\n{'='*90}\n### REDUCTION SWEEP  {path}")
    print(f"    stacks={len(pids)}  patients={len(np.unique(pids))}  streams={list(present)}  "
          f"(GAN M&Ms 0.731 / CineMA flow_L1 0.713)")

    reduced = {}   # (name, rule) -> per-stack scalar
    for name, key in present.items():
        mat = a[key]
        print(f"\n  stream = {name}")
        print(f"    {'rule':<10}{'ACDC':>26}{'M&Ms':>26}{'POOLED':>26}")
        for rule in REDUCTION_RULES:
            sc = np.array([_reduce_slices(mat[i], nreal[i], rule) for i in range(len(mat))])
            reduced[(name, rule)] = sc
            print(f"    {rule:<10}"
                  + "".join(_auc_cell(sc, pids, labels, datasets, ds, n_boot) for ds in ("ACDC", "MM", "ALL")))

    if "appearance" in present and "flow_SSIM" in present:
        print(f"\n  stream = combined (z[appe]+z[flow_SSIM], same rule both)")
        print(f"    {'rule':<10}{'ACDC':>26}{'M&Ms':>26}{'POOLED':>26}")
        for rule in REDUCTION_RULES:
            za = reduced[("appearance", rule)]; zf = reduced[("flow_SSIM", rule)]
            comb = (za - za.mean()) / (za.std() + 1e-8) + (zf - zf.mean()) / (zf.std() + 1e-8)
            print(f"    {rule:<10}"
                  + "".join(_auc_cell(comb, pids, labels, datasets, ds, n_boot) for ds in ("ACDC", "MM", "ALL")))


def agg_sweep(path, n_boot=2000):
    """Patient-level aggregation sweep for a 2-D per-slice-sample npz (qfae_dino2d flow eval).

    Each row is one slice (pids/slcs/labels/datasets + stream values). For each stream and rule,
    aggregate a patient's slice scores {mean/top-K%/max/middle-60%} → patient score → per-dataset AUC.
    """
    a = _load(path)
    pids, labels, datasets, slcs = a["pids"], a["labels"], a["datasets"], a["slcs"]
    stream_keys = [k for k in ("appearance", "flow_SSIM", "mag_SSIM", "flow_L1") if k in a]
    if not stream_keys and "scores" in a:
        a["appearance"] = a["scores"]; stream_keys = ["appearance"]
    uniq = np.unique(pids)
    plab = np.array([labels[pids == p][0] for p in uniq])
    pds = np.array([datasets[pids == p][0] for p in uniq])
    rules = ["mean", "top10", "top20", "top30", "max", "middle60"]
    print(f"\n{'='*90}\n### PATIENT-AGGREGATION SWEEP  {path}")
    print(f"    slices={len(pids)}  patients={len(uniq)}  streams={stream_keys}  (GAN M&Ms 0.731)")
    for st in stream_keys:
        vals = a[st]
        print(f"\n  stream = {st}")
        print(f"    {'rule':<10}{'ACDC':>26}{'M&Ms':>26}{'POOLED':>26}")
        for rule in rules:
            psc = np.empty(len(uniq))
            for i, p in enumerate(uniq):
                m = pids == p
                v = vals[m][np.argsort(slcs[m])]                 # patient's slices, ordered by index
                psc[i] = _reduce_slices(v, len(v), rule)
            cells = []
            for ds in ["ACDC", "MM", "ALL"]:
                dm = np.ones(len(uniq), bool) if ds == "ALL" else (pds == ds)
                y = (plab[dm] != "NOR").astype(int); s = psc[dm]
                if len(np.unique(y)) < 2:
                    cells.append(f"{'n/a':>26}"); continue
                auc = _fast_auc(y, s)
                rng = np.random.default_rng(0); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; b = []
                for _ in range(n_boot):
                    bi = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
                    if len(np.unique(y[bi])) > 1:
                        b.append(_fast_auc(y[bi], s[bi]))
                lo, hi = np.percentile(b, [2.5, 97.5])
                cells.append(f"{auc:.3f} [{lo:.2f},{hi:.2f}] {int((y==0).sum())}/{int((y==1).sum())}".rjust(26))
            print(f"    {rule:<10}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--paired", action="store_true",
                    help="also print paired flow_SSIM-vs-appearance bootstrap per dataset (flow runs)")
    ap.add_argument("--mmtest", action="store_true",
                    help="report the held-out M&Ms-Testing re-evals (MMTEST_RUNS) instead of the val runs")
    ap.add_argument("--heldout", nargs=2, metavar=("VAL_NPZ", "TEST_NPZ"),
                    help="run the GAN held-out protocol: select on VAL_NPZ M&M, report on TEST_NPZ ACDC+M&M")
    ap.add_argument("--fuse", nargs=2, metavar=("A_NPZ:STREAM", "B_NPZ:STREAM"),
                    help="tuning-free rank-fusion of two streams from two npz, aligned by patient")
    ap.add_argument("--reduction_sweep", metavar="PER_SLICE_NPZ",
                    help="sweep per-slice reduction rules (mean/topK%/max/middle-60%) on a per-slice npz")
    ap.add_argument("--agg_sweep", metavar="DINO2D_NPZ",
                    help="sweep patient aggregation rules over a 2-D per-slice-sample npz (qfae_dino2d)")
    args = ap.parse_args()

    if args.heldout:
        heldout(args.heldout[0], args.heldout[1])
        return
    if args.fuse:
        two_file_fuse(args.fuse[0], args.fuse[1], n_boot=args.n_boot)
        return
    if args.reduction_sweep:
        reduction_sweep(args.reduction_sweep, n_boot=args.n_boot)
        return
    if args.agg_sweep:
        agg_sweep(args.agg_sweep, n_boot=args.n_boot)
        return

    runs = MMTEST_RUNS if args.mmtest else RUNS
    print("QFAE offline patient-level report (patient-Mean AUC, per dataset, bootstrap 95% CI)")
    if args.mmtest:
        print("Cohort: ACDC testing 50 + M&Ms TESTING (~31 NOR / ~103 disease) — HELD-OUT, == GAN's report set.")
    else:
        print("Cohort: ACDC testing 50 (== GAN's ACDC 50) + M&M VALIDATION 34 (== GAN's SELECTION set, 9 NOR).")

    results = {}
    for name, path, streams in runs:
        res = report_run(name, path, streams, )
        if res is None:
            continue
        verify_against_json(name, path)
        results[name] = res
        if args.paired and "flow_SSIM" in streams and "appearance" in streams:
            a = res["arrays"]
            for ds in ["ACDC", "MM"]:
                m = a["datasets"] == ds
                d = _paired_delta(a["flow_SSIM"][m], a["appearance"][m], a["pids"][m], a["labels"][m])
                if not np.isnan(d["delta"]):
                    print(f"    [paired {ds}] flow_SSIM - appearance = {d['delta']:+.3f} "
                          f"[{d['lo']:+.2f},{d['hi']:+.2f}]  p={d['p']:.3f}")

    print("\nDone. NOTE: for the exact held-out M&M-Testing comparison to the GAN's 0.7310, "
          "re-run qfae_eval.py with --mm_val_dir ../Dataset_1/Testing (see plan).")


if __name__ == "__main__":
    main()
