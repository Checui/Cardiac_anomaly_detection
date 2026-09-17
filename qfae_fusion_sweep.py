"""qfae_fusion_sweep.py — compare score-fusion / normalisation schemes for the QFAE dual stream.

Loads a saved `qfae_arrays.npz` (per-stack appearance + flow scores + pids/labels/datasets) and
re-scores the appearance+motion fusion under three ways of putting the two streams on a common
scale, sweeping the stream weight:

    f(w) = (1-w) * T(appearance) + w * T(flow_stream)      w in [0, 1]

    T = 'zscore'      (x - mean) / std                      <- current qfae_eval.py fusion (baseline)
    T = 'rank'        rankdata(x) / N   in [0, 1]           <- percentile-rank
    T = 'logrobustz'  (log x - median(log x)) / IQR(log x)  <- log + robust standardise

For each method it reports patient-Mean AUC (overall) + per-dataset AUC, at:
  * equal weight w=0.5            (honest, no tuning)
  * best-overall-w               (ceiling; w tuned on the val set — like the [VAL-OPT-WEIGHT] pass)
  * each dataset's own best-w     (exposes the ACDC<->MM cross-vendor tension)
Plus a per-dataset-normalised oracle upper bound (needs the dataset label at test time).

Reporting convention MATCHES qfae_eval.py's report_stream so numbers line up with existing logs:
  overall = patient-Mean aggregation;  per-dataset ACDC/MM = stack-level (per_ds).
A patient-level per-dataset block is printed too for a cleaner internal view.

Pure numpy + sklearn; no torch/cinema import. Runs in <1s on CPU.
"""

import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


# ── AUC helpers (faithful re-impl of derisk_cinema.py:164-200) ───────────────────
def one_vs_nor_overall(scores, labels):
    """AUC with NOR=0, any disease=1. Returns nan if only one class present."""
    scores = np.asarray(scores, dtype=float)
    y = (np.asarray(labels) != "NOR").astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, scores))


def aggregate_to_patient(scores, pids, labels, datasets):
    """Per-stack -> per-patient Mean/Max, carrying each patient's label + dataset."""
    scores = np.asarray(scores, dtype=float)
    pids = np.asarray(pids)
    labels = np.asarray(labels)
    datasets = np.asarray(datasets)
    uniq = np.unique(pids)
    mean = np.zeros(len(uniq))
    mx = np.zeros(len(uniq))
    plabels, pdatasets = [], []
    for i, pid in enumerate(uniq):
        m = pids == pid
        mean[i] = float(np.mean(scores[m]))
        mx[i] = float(np.max(scores[m]))
        plabels.append(labels[m][0])
        pdatasets.append(datasets[m][0])
    return mean, mx, np.array(plabels), np.array(pdatasets)


def evaluate(scores, pids, labels, datasets):
    """Return the AUC views used throughout: dict of overall/per-dataset at stack & patient level."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    datasets = np.asarray(datasets)
    mean, mx, plabels, pdatasets = aggregate_to_patient(scores, pids, labels, datasets)
    out = {
        "stack_overall": one_vs_nor_overall(scores, labels),
        "patMean_overall": one_vs_nor_overall(mean, plabels),
        "patMax_overall": one_vs_nor_overall(mx, plabels),
    }
    for ds in ("ACDC", "MM"):
        sm = datasets == ds
        pm = pdatasets == ds
        out[f"stack_{ds}"] = one_vs_nor_overall(scores[sm], labels[sm])      # matches report_stream per_ds
        out[f"patMean_{ds}"] = one_vs_nor_overall(mean[pm], plabels[pm])     # cleaner patient-level view
    return out


# ── transforms ───────────────────────────────────────────────────────────────────
def T(x, method):
    x = np.asarray(x, dtype=float)
    if method == "zscore":
        return (x - x.mean()) / (x.std() + 1e-8)
    if method == "rank":
        return rankdata(x) / len(x)
    if method == "logrobustz":
        if x.min() <= 0:
            x = x - x.min() + 1e-6      # guard: shift to strictly positive before log
        lx = np.log(x)
        iqr = np.percentile(lx, 75) - np.percentile(lx, 25)
        return (lx - np.median(lx)) / (iqr + 1e-8)
    raise ValueError(method)


def T_per_dataset(x, datasets, method):
    """Same transform but stats estimated separately within each dataset (oracle: needs ds label)."""
    x = np.asarray(x, dtype=float)
    datasets = np.asarray(datasets)
    out = np.zeros_like(x, dtype=float)
    for ds in np.unique(datasets):
        m = datasets == ds
        out[m] = T(x[m], method)
    return out


# ── sweep ──────────────────────────────────────────────────────────────────────
def sweep(appe, flow, pids, labels, datasets, method, weights, per_dataset=False):
    """Evaluate the fused score across weights; return list of (w, eval-dict)."""
    ta = T_per_dataset(appe, datasets, method) if per_dataset else T(appe, method)
    tf = T_per_dataset(flow, datasets, method) if per_dataset else T(flow, method)
    rows = []
    for w in weights:
        fused = (1.0 - w) * ta + w * tf
        rows.append((float(w), evaluate(fused, pids, labels, datasets)))
    return rows


def pick(rows, key):
    """Row (w, ev) maximising ev[key] (ignoring nan)."""
    valid = [(w, ev) for (w, ev) in rows if not np.isnan(ev[key])]
    return max(valid, key=lambda r: r[1][key]) if valid else rows[0]


def at_weight(rows, target):
    return min(rows, key=lambda r: abs(r[0] - target))


def fmt(ev):
    """overall(patient-Mean) / ACDC / MM(stack-level) — the report_stream convention."""
    return (f"overall={ev['patMean_overall']:.4f}  "
            f"ACDC={ev['stack_ACDC']:.3f}  MM={ev['stack_MM']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="qfae_flow_out/qfae_arrays.npz")
    ap.add_argument("--flow_stream", default="flow_SSIM",
                    choices=["flow_SSIM", "mag_SSIM", "flow_L1"])
    ap.add_argument("--n_weights", type=int, default=21)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    appe = d["appearance"].astype(float)
    flow = d[args.flow_stream].astype(float)
    pids, labels, datasets = d["pids"], d["labels"], d["datasets"]
    weights = np.linspace(0.0, 1.0, args.n_weights)

    print("=" * 76)
    print(f"QFAE fusion sweep — {args.npz}")
    print(f"n_stacks={len(appe)}  patients={len(np.unique(pids))}  "
          f"flow_stream={args.flow_stream}  weights={args.n_weights}")
    print(f"label counts: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print("=" * 76)

    # ── single-stream references (my pipeline vs the known logs) ──
    ea, ef = evaluate(appe, pids, labels, datasets), evaluate(flow, pids, labels, datasets)
    print("\nSINGLE-STREAM (my pipeline; compare to logs appe ACDC 0.909 / flow MM 0.716):")
    print(f"  appearance  : {fmt(ea)}")
    print(f"  {args.flow_stream:11s} : {fmt(ef)}")

    # ── VALIDATION GATE: reproduce current fusion z(appe)+z(flow) == zscore @ w=0.5 ──
    saved = evaluate(d["combined"].astype(float), pids, labels, datasets) if "combined" in d else None
    z_rows = sweep(appe, flow, pids, labels, datasets, "zscore", weights)
    z_half = at_weight(z_rows, 0.5)[1]
    print("\nVALIDATION GATE (baseline z-sum @ w=0.5 must match saved 'combined'):")
    print(f"  zscore@0.5  : {fmt(z_half)}")
    if saved is not None:
        print(f"  saved combo : {fmt(saved)}")
        ok = (abs(z_half["patMean_overall"] - saved["patMean_overall"]) < 1e-3
              and abs(z_half["stack_ACDC"] - saved["stack_ACDC"]) < 1e-3
              and abs(z_half["stack_MM"] - saved["stack_MM"]) < 1e-3)
        print(f"  GATE: {'PASS' if ok else 'FAIL — pipeline does not match qfae_eval.py'}")
        if not ok:
            sys.exit("Validation gate failed; aborting.")

    # ── methods ──
    methods = [("zscore", "global z-sum (baseline)"),
               ("rank", "percentile-rank"),
               ("logrobustz", "log + robust-z")]

    record = {"npz": args.npz, "flow_stream": args.flow_stream,
              "single": {"appearance": ea, args.flow_stream: ef}, "methods": {}}

    for method, title in methods:
        rows = sweep(appe, flow, pids, labels, datasets, method, weights)
        eq = at_weight(rows, 0.5)
        bo = pick(rows, "patMean_overall")          # ceiling: best overall w
        ba = pick(rows, "stack_ACDC")               # ACDC champion
        bm = pick(rows, "stack_MM")                 # MM champion
        print("\n" + "-" * 76)
        print(f"METHOD: {title}")
        print(f"  equal   w=0.50 : {fmt(eq[1])}")
        print(f"  best-ovr w={bo[0]:.2f} : {fmt(bo[1])}      <- ceiling (w tuned on val)")
        print(f"  best-ACDC w={ba[0]:.2f}: ACDC={ba[1]['stack_ACDC']:.3f} "
              f"(MM here={ba[1]['stack_MM']:.3f})")
        print(f"  best-MM   w={bm[0]:.2f}: MM={bm[1]['stack_MM']:.3f} "
              f"(ACDC here={ba[1]['stack_ACDC']:.3f})")
        record["methods"][method] = {
            "equal": {"w": eq[0], **eq[1]},
            "best_overall": {"w": bo[0], **bo[1]},
            "best_ACDC": {"w": ba[0], **ba[1]},
            "best_MM": {"w": bm[0], **bm[1]},
        }

    # ── per-dataset-normalised oracle upper bound ──
    print("\n" + "=" * 76)
    print("ORACLE (per-dataset normalisation — needs dataset label at test time; upper bound):")
    for method, title in methods:
        rows = sweep(appe, flow, pids, labels, datasets, method, weights, per_dataset=True)
        bo = pick(rows, "patMean_overall")
        ba = pick(rows, "stack_ACDC")
        bm = pick(rows, "stack_MM")
        print(f"  {title:26s}: best-ovr w={bo[0]:.2f} {fmt(bo[1])} | "
              f"ACDC*={ba[1]['stack_ACDC']:.3f} MM*={bm[1]['stack_MM']:.3f}")
        record["methods"][method]["oracle_best_overall"] = {"w": bo[0], **bo[1]}
        record["methods"][method]["oracle_ACDC"] = ba[1]["stack_ACDC"]
        record["methods"][method]["oracle_MM"] = bm[1]["stack_MM"]

    print("\n" + "=" * 76)
    print("Convention: overall = patient-Mean; ACDC/MM = stack-level (matches qfae_eval.py logs).")
    print("best-* rows tune w on the val set => optimistic ceilings, not deployable operating points.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[wrote] {args.out_json}")


if __name__ == "__main__":
    main()
