#!/usr/bin/env python
"""gan_vendor_table.py -- per-vendor/centre breakdown of the baseline flow-GAN on M&Ms-Testing.

Reproduces the thesis table `tab:gan_vendor` (motion stream, flow-SSIM, patient mean,
middle-60% slices, AUC computed WITHIN each vendor) from the persisted per-slice GAN
scores, and extends it with

  * the APPEARANCE stream (loss_appe = MSE + gradient, same protocol), and
  * RECONSTRUCTION-QUALITY columns (SSIM / PSNR between the reconstruction head's output
    and its input frame) from `gan_recon_quality_eval.py`, summarised per vendor on
    healthy (NOR) patients -- the clean vendor-shift readout, free of the disease-mix
    confound -- with the all-patient and diseased-minus-NOR variants kept in the JSON.

Provenance (verified to the digit): cells = qfae_report._patient_auc(seed=0, n_boot=2000)
on `gan_revert_out/scores.npz` mid60__MM__*; seen-vs-unseen AUC contrast =
analyze_supervised.unpaired_delta (independent per-group bootstrap, n_boot=10000, seed=0).
The script GATES on the published motion cells so any drift in pass / rule / checkpoint is
caught before a new column is trusted. Offline, no TF: run with the `derisk` env python.

Usage
  python gan_vendor_table.py --scores gan_revert_out/scores.npz            # AUC columns only
  python gan_vendor_table.py --scores ... --recon gan_recon_quality_out/scores.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from qfae_report import _patient_auc
from analyze_supervised import attach_vendors, patient_frame, unpaired_delta, SEEN_NORECOVER

VENDORS = ["Siemens", "Philips", "GE", "Canon"]
LETTER = {"Siemens": "A", "Philips": "B", "GE": "C", "Canon": "D"}
STREAM_NAMES = {"flow_ssim": "Motion (flow SSIM)", "mag_ssim": "Motion (magnitude SSIM)",
                "appe": "Appearance"}
QUALITY_NAMES = {"recon_ssim_dr1": "Recon SSIM", "recon_psnr": "PSNR (dB)",
                 "recon_ssim_crop": "Recon SSIM (crop)", "recon_ssim_joint": "Recon SSIM (joint range)",
                 "recon_mse": "Recon MSE"}

# Published motion cells (thesis tab:gan_vendor) -- the provenance gate.
EXPECTED_MOTION = {
    "Siemens": (0.764, 5, 11), "Philips": (0.671, 7, 33), "GE": (0.692, 8, 30), "Canon": (0.784, 11, 29),
    "seen": (0.706, 12, 44), "unseen": (0.742, 19, 59),
    "delta": (-0.036, -0.233, 0.157),
}
GATE_TOL = 1.5e-3


# ── loading ──────────────────────────────────────────────────────────────────
def load_gan_arrays(path, pas, ds="MM"):
    """GAN scores.npz ({pass}__{ds}__{key}) -> analyze_abcd-style arrays dict for one dataset."""
    z = np.load(path, allow_pickle=False)
    pre = "%s__%s__" % (pas, ds)
    keys = [k[len(pre):] for k in z.files if k.startswith(pre)]
    if not keys:
        sys.exit("no keys with prefix %s in %s (available: %s)" % (pre, path, sorted(z.files)[:8]))
    a = {"pids": np.asarray(z[pre + "pids"]).astype(str),
         "labels": np.asarray(z[pre + "labels"]).astype(str),
         "slcs": np.asarray(z[pre + "slice_idx"]).astype(np.int64)}
    a["datasets"] = np.array([ds] * len(a["pids"]))
    for k in keys:
        if k not in ("pids", "labels", "slice_idx"):
            a[k] = np.asarray(z[pre + k], dtype=float)
    return a


def join_recon(a, recon_path, ds="MM", streams=None, tol=1e-3):
    """Attach the recon-quality streams to `a`, joined on (pid, slice_idx) -- never positional."""
    z = np.load(recon_path, allow_pickle=False)
    pre = "%s__" % ds
    rp = np.asarray(z[pre + "pids"]).astype(str)
    rs = np.asarray(z[pre + "slice_idx"]).astype(np.int64)
    row = {(p, int(s)): i for i, (p, s) in enumerate(zip(rp, rs))}
    idx = np.array([row.get((p, int(s)), -1) for p, s in zip(a["pids"], a["slcs"])])
    n_missing = int((idx < 0).sum())
    if n_missing:
        sys.exit("recon join: %d / %d (pid, slice) rows not found in %s" % (n_missing, len(idx), recon_path))
    # alignment proof: the anchor streams must agree slice-by-slice between the two files
    checks = {}
    for k in ("appe", "flow_ssim"):
        if k in a and (pre + k) in z.files:
            d = float(np.max(np.abs(np.asarray(z[pre + k], float)[idx] - a[k])))
            checks[k] = d
            if d > tol:
                sys.exit("recon join: per-slice %s differs between files (max |diff| = %.3g)" % (k, d))
    qkeys = [k[len(pre):] for k in z.files
             if k.startswith(pre) and k[len(pre):].startswith("recon_")]
    if streams:
        missing = [s for s in streams if s not in qkeys]
        if missing:
            sys.exit("recon file lacks quality streams %s (has %s)" % (missing, qkeys))
    b = dict(a)
    for k in qkeys:
        b[k] = np.asarray(z[pre + k], float)[idx]
    return b, qkeys, checks, len(idx)


# ── statistics ───────────────────────────────────────────────────────────────
def auc_cell(a, stream, mask_slices, n_boot, seed=0):
    r = _patient_auc(a[stream][mask_slices], a["pids"][mask_slices], a["labels"][mask_slices],
                     seed=seed, n_boot=n_boot)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}


def unpaired_mean_delta(sa, sb, n_boot=10000, seed=0):
    """mean(A) - mean(B) for disjoint patient groups; independent plain bootstrap per group."""
    sa, sb = np.asarray(sa, float), np.asarray(sb, float)
    if len(sa) == 0 or len(sb) == 0:
        return None
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ds[i] = rng.choice(sa, len(sa), True).mean() - rng.choice(sb, len(sb), True).mean()
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(a=float(sa.mean()), b=float(sb.mean()), delta=float(sa.mean() - sb.mean()),
                lo=float(lo), hi=float(hi),
                p=float(2 * min((ds <= 0).mean(), (ds >= 0).mean())),
                n_a=int(len(sa)), n_b=int(len(sb)))


def _msd(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()) if v.size else float("nan"),
                sd=float(v.std(ddof=1)) if v.size > 1 else float("nan"), n=int(v.size))


def quality_summary(a, metric, seen, unseen, n_boot):
    """Patient-mean quality per vendor (NOR / all / diseased) + seen-vs-unseen mean deltas."""
    sc, pl, _, pids, pv = patient_frame(a, metric, "mean")
    nor = pl == "NOR"
    out = {"per_vendor": {}, "seen_unseen": {}}
    for v in VENDORS:
        m = pv == v
        out["per_vendor"][v] = {"nor": _msd(sc[m & nor]), "all": _msd(sc[m]),
                                "dis": _msd(sc[m & ~nor]),
                                "gap_dis_minus_nor": float(sc[m & ~nor].mean() - sc[m & nor].mean())
                                if (m & nor).any() and (m & ~nor).any() else float("nan")}
    ms, mu = np.isin(pv, seen), np.isin(pv, unseen)
    for grp, m in (("seen", ms), ("unseen", mu)):
        out[grp] = {"nor": _msd(sc[m & nor]), "all": _msd(sc[m]), "dis": _msd(sc[m & ~nor])}
    out["seen_unseen"] = {
        "nor": unpaired_mean_delta(sc[ms & nor], sc[mu & nor], n_boot=n_boot),
        "all": unpaired_mean_delta(sc[ms], sc[mu], n_boot=n_boot),
    }
    return out


# ── formatting ───────────────────────────────────────────────────────────────
def f_auc(c, ci=True):
    if c is None or not np.isfinite(c.get("auc", np.nan)):
        return "--"
    return ("%.3f [%.2f, %.2f]" % (c["auc"], c["lo"], c["hi"])) if ci else "%.3f" % c["auc"]


def f_delta(d, digits=3):
    if d is None:
        return "--"
    fmt = "%%+.%df [%%+.%df, %%+.%df]" % (digits, digits, digits)
    return fmt % (d["delta"], d["lo"], d["hi"])


def f_msd(m, digits=3):
    if m is None or m["n"] == 0:
        return "--"
    fmt = "%%.%df $\\pm$ %%.%df" % (digits, digits)
    return fmt % (m["mean"], m["sd"])


def q_digits(metric):
    return 1 if metric == "recon_psnr" else 3


def build_latex(res, streams, qmetrics, cohort):
    L = []
    ncol = 3 + len(streams) + len(qmetrics)
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\caption[Breakdown by vendor]{TODO caption -- see printed notes.}")
    L.append("\\label{tab:gan_vendor}")
    L.append("\\small")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\begin{tabular}{llc" + "c" * (len(streams) + len(qmetrics)) + "}")
    L.append("\\toprule")
    head = ["Vendor", "Seen", "$n$ (NOR/dis)"]
    head += ["\\makecell{%s\\\\AUC {[95\\%% CI]}}" % STREAM_NAMES.get(s, s).replace(" (", "\\\\(") for s in streams]
    head += ["\\makecell{%s\\\\(%s)}" % (QUALITY_NAMES.get(q, q), "NOR" if cohort == "nor" else "all") for q in qmetrics]
    L.append(" & ".join(head) + " \\\\")
    L.append("\\midrule")
    for v in VENDORS:
        c0 = res["auc"][streams[0]]["per_vendor"][v]
        row = ["%s (%s)" % (LETTER[v], v), "yes" if v in res["seen"] else "no",
               "%d/%d" % (c0["n_nor"], c0["n_dis"])]
        row += [f_auc(res["auc"][s]["per_vendor"][v]) for s in streams]
        row += [f_msd(res["quality"][q]["per_vendor"][v][cohort], q_digits(q)) if q in res["quality"] else "--"
                for q in qmetrics]
        L.append(" & ".join(row) + " \\\\")
    L.append("\\midrule")
    for grp, lab in (("seen", "Seen (%s)" % " and ".join(LETTER[v] for v in VENDORS if v in res["seen"])),
                     ("unseen", "Unseen (%s)" % " and ".join(LETTER[v] for v in VENDORS if v not in res["seen"]))):
        c0 = res["auc"][streams[0]][grp]
        row = [lab, "yes" if grp == "seen" else "no", "%d/%d" % (c0["n_nor"], c0["n_dis"])]
        row += [f_auc(res["auc"][s][grp], ci=False) for s in streams]
        row += [f_msd(res["quality"][q][grp][cohort], q_digits(q)) if q in res["quality"] else "--"
                for q in qmetrics]
        L.append(" & ".join(row) + " \\\\")
    row = ["$\\Delta$ (seen $-$ unseen)", "", ""]
    row += [f_delta(res["auc"][s]["delta_seen_minus_unseen"]) for s in streams]
    row += [f_delta(res["quality"][q]["seen_unseen"][cohort], q_digits(q)) if q in res["quality"] else "--"
            for q in qmetrics]
    L.append(" & ".join(row) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="gan_revert_out/scores.npz")
    ap.add_argument("--recon", default=None, help="gan_recon_quality_out/scores.npz (optional)")
    ap.add_argument("--mm_csv", default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--pass", dest="pas", default="mid60", help="mid60 (headline) or all")
    ap.add_argument("--streams", nargs="+", default=["flow_ssim", "appe"])
    ap.add_argument("--quality", nargs="+", default=["recon_ssim_dr1", "recon_psnr"])
    ap.add_argument("--quality_extra", nargs="+", default=["recon_ssim_crop", "recon_ssim_joint", "recon_mse"])
    ap.add_argument("--quality_cohort", choices=["nor", "all"], default="nor")
    ap.add_argument("--seen", nargs="+", default=list(SEEN_NORECOVER))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_boot_delta", type=int, default=10000)
    ap.add_argument("--out_dir", default="gan_vendor_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seen = list(args.seen)
    unseen = [v for v in VENDORS if v not in seen]

    a = load_gan_arrays(args.scores, args.pas)
    a = attach_vendors(a, args.mm_csv)
    unknown = int((a["vendors"] == "UNKNOWN").sum())
    print("loaded %s pass=%s: %d slices / %d patients; vendors %s; unknown %d"
          % (args.scores, args.pas, len(a["pids"]), len(np.unique(a["pids"])),
             {v: int((a["vendors"] == v).sum()) for v in VENDORS}, unknown))
    for s in args.streams:
        if s not in a:
            sys.exit("stream %s not in %s" % (s, args.scores))

    res = {"config": vars(args), "seen": seen, "unseen": unseen,
           "n_slices": int(len(a["pids"])), "n_patients": int(len(np.unique(a["pids"]))),
           "auc": {}, "quality": {}, "recon_join": None}

    # ── AUC columns ───────────────────────────────────────────────────────────
    for s in args.streams:
        _, _, _, _, pv = patient_frame(a, s, "mean")
        r = {"per_vendor": {}}
        for v in VENDORS:
            r["per_vendor"][v] = auc_cell(a, s, a["vendors"] == v, args.n_boot)
        r["seen"] = auc_cell(a, s, np.isin(a["vendors"], seen), args.n_boot)
        r["unseen"] = auc_cell(a, s, np.isin(a["vendors"], unseen), args.n_boot)
        r["delta_seen_minus_unseen"] = unpaired_delta(
            a, s, "mean", np.isin(pv, seen), np.isin(pv, unseen), n_boot=args.n_boot_delta, seed=0)
        res["auc"][s] = r

    # ── gate on the published motion cells ────────────────────────────────────
    gate = []
    if "flow_ssim" in res["auc"]:
        m = res["auc"]["flow_ssim"]
        for v in VENDORS:
            e_auc, e_nor, e_dis = EXPECTED_MOTION[v]
            c = m["per_vendor"][v]
            gate.append((v, abs(c["auc"] - e_auc) < GATE_TOL and c["n_nor"] == e_nor and c["n_dis"] == e_dis,
                         "%.4f n=%d/%d (exp %.3f %d/%d)" % (c["auc"], c["n_nor"], c["n_dis"], e_auc, e_nor, e_dis)))
        for grp in ("seen", "unseen"):
            e_auc, e_nor, e_dis = EXPECTED_MOTION[grp]
            c = m[grp]
            gate.append((grp, abs(c["auc"] - e_auc) < GATE_TOL and c["n_nor"] == e_nor and c["n_dis"] == e_dis,
                         "%.4f n=%d/%d (exp %.3f %d/%d)" % (c["auc"], c["n_nor"], c["n_dis"], e_auc, e_nor, e_dis)))
        d = m["delta_seen_minus_unseen"]
        e_d, e_lo, e_hi = EXPECTED_MOTION["delta"]
        gate.append(("delta", all(abs(x - y) < GATE_TOL for x, y in ((d["delta"], e_d), (d["lo"], e_lo), (d["hi"], e_hi))),
                     "%+.4f [%+.4f, %+.4f] (exp %+.3f [%+.3f, %+.3f])" % (d["delta"], d["lo"], d["hi"], e_d, e_lo, e_hi)))
    gate_ok = all(ok for _, ok, _ in gate)
    res["gate"] = {"ok": gate_ok, "checks": [(k, bool(ok), msg) for k, ok, msg in gate]}
    print("\n=== provenance gate (published motion cells) ===")
    for k, ok, msg in gate:
        print("  [%s] %-8s %s" % ("OK " if ok else "BAD", k, msg))
    print("  [GATE] %s" % ("OK" if gate_ok else "** MISMATCH **"))

    # ── quality columns ───────────────────────────────────────────────────────
    qmetrics = list(args.quality)
    if args.recon:
        a, qkeys, checks, n = join_recon(a, args.recon, streams=args.quality)
        res["recon_join"] = {"file": args.recon, "n_matched": n, "anchor_max_absdiff": checks}
        print("\nrecon join: %d/%d rows matched on (pid, slice_idx); anchor agreement %s" % (n, n, checks))
        for q in qkeys:
            res["quality"][q] = quality_summary(a, q, seen, unseen, args.n_boot_delta)
    else:
        print("\n(no --recon given: quality columns left empty)")

    # ── report ────────────────────────────────────────────────────────────────
    print("\n=== within-vendor patient-mean AUC, M&Ms-Testing (%s pass) ===" % args.pas)
    hdr = "%-10s %-5s %-8s" % ("vendor", "seen", "n") + "".join("%26s" % STREAM_NAMES.get(s, s) for s in args.streams)
    print(hdr)
    for v in VENDORS:
        c0 = res["auc"][args.streams[0]]["per_vendor"][v]
        print("%-10s %-5s %-8s" % (v, "yes" if v in seen else "no", "%d/%d" % (c0["n_nor"], c0["n_dis"]))
              + "".join("%26s" % f_auc(res["auc"][s]["per_vendor"][v]) for s in args.streams))
    for grp in ("seen", "unseen"):
        c0 = res["auc"][args.streams[0]][grp]
        print("%-10s %-5s %-8s" % (grp, "", "%d/%d" % (c0["n_nor"], c0["n_dis"]))
              + "".join("%26s" % f_auc(res["auc"][s][grp]) for s in args.streams))
    print("%-25s" % "delta seen-unseen"
          + "".join("%26s" % f_delta(res["auc"][s]["delta_seen_minus_unseen"]) for s in args.streams))
    print("%-25s" % "  p (two-sided)"
          + "".join("%26.3f" % res["auc"][s]["delta_seen_minus_unseen"]["p"] for s in args.streams))

    if res["quality"]:
        for cohort in ("nor", "all", "dis"):
            print("\n=== reconstruction quality, patient mean, %s patients: mean +- SD (n) ===" % cohort.upper())
            qs = [q for q in qmetrics + list(args.quality_extra) if q in res["quality"]]
            print("%-10s" % "vendor" + "".join("%30s" % QUALITY_NAMES.get(q, q) for q in qs))
            for v in VENDORS + ["seen", "unseen"]:
                cells = []
                for q in qs:
                    m = (res["quality"][q]["per_vendor"][v] if v in VENDORS else res["quality"][q][v])[cohort]
                    cells.append("%s (%d)" % (f_msd(m, q_digits(q)).replace("$\\pm$", "+-"), m["n"]))
                print("%-10s" % v + "".join("%30s" % c for c in cells))
            if cohort != "dis":
                print("%-10s" % "delta s-u" + "".join(
                    "%30s" % (f_delta(res["quality"][q]["seen_unseen"][cohort], q_digits(q))
                              + " p=%.2f" % res["quality"][q]["seen_unseen"][cohort]["p"]) for q in qs))
        print("\n=== diseased minus NOR quality gap per vendor ===")
        qs = [q for q in qmetrics + list(args.quality_extra) if q in res["quality"]]
        print("%-10s" % "vendor" + "".join("%22s" % QUALITY_NAMES.get(q, q) for q in qs))
        for v in VENDORS:
            print("%-10s" % v + "".join("%+22.4f" % res["quality"][q]["per_vendor"][v]["gap_dis_minus_nor"] for q in qs))

    tex = build_latex(res, args.streams, qmetrics, args.quality_cohort)
    print("\n=== LaTeX (paste into the thesis; caption TODO) ===\n" + tex)
    with open(os.path.join(args.out_dir, "vendor_table.tex"), "w") as f:
        f.write(tex + "\n")
    with open(os.path.join(args.out_dir, "vendor_table.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\nSaved %s/vendor_table.{json,tex}" % args.out_dir)
    if not gate_ok:
        print("GATE FAILED -- numbers above do not reproduce the published motion cells", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
