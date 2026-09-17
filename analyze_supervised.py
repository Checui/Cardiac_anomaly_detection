"""analyze_supervised.py — offline tables for the supervised ceiling probe (no GPU).

Two jobs, both on the SAME held-out protocol as every other table in this project
(patient-Mean AUC on ACDC-50 + M&Ms-Testing, patient-stratified bootstrap 95% CIs):

  1. --uad_only : vendor-stratify the EXISTING unsupervised arrays. Needs no new
     experiment — `vendors` is derivable from `pids` + the M&Ms diagnosis CSV. This
     turns "motion crosses the vendor gap" from a dataset-level claim into a
     within-M&Ms one.

  2. full       : the supervised NOR-vs-disease probe (sup_* streams) beside the
     unsupervised numbers — the ceiling reference the supervisor asked for — plus the
     appearance-vs-motion disambiguation and the paired delta against the UAD.

READ FOR DIRECTION, NOT FOR WINNERS. Per-vendor cells are brutally underpowered: the
seen-minus-unseen contrast has SE ~ 0.11, so anything below a 0.20 AUC gap is invisible.
Report the pre-committed contrast; do not scan vendor x stream x rule for the biggest cell.

TWO CONFOUNDS THAT MAKE RAW PER-VENDOR AUC UNINTERPRETABLE ON M&Ms-Testing:
  * vendor == centre EXACTLY (Canon=C5, GE=C4, Siemens=C1, Philips=C2+C3), so scanner,
    site, protocol and population are one variable. Say "vendor/centre", never "vendor".
  * case mix is badly unbalanced -- Philips disease is 21/28 DCM (the easiest class),
    GE is 17/31 "Other" (a wastebasket label). Every vendor cell is therefore printed
    twice: on all disease, and on the matched {NOR, DCM, HCM} subset.
"""

import argparse
import os

import numpy as np
import pandas as pd

from qfae_report import _load, _fast_auc, _patient_auc
from analyze_abcd import patient_reduce, auc_cell, fmt, load_arm, BAR_ACDC, BAR_MM
from analyze_roictrl import paired_delta

# Vendors present in M&Ms Training AFTER the ed==es drop (see validate_edes_recovery.py).
# Without --recover_edes every model in this project trained on Philips + Siemens ONLY,
# so GE and Canon are both UNSEEN at test time. With recovery, only Canon is unseen.
SEEN_NORECOVER = ("Philips", "Siemens")
SEEN_RECOVERED = ("Philips", "Siemens", "GE")

MATCHED_LABELS = ("NOR", "DCM", "HCM")          # the ACDC-and-M&Ms label intersection

UAD_STREAMS = ["appearance", "flow_SSIM", "mag_SSIM"]
RULES = ["mean", "middle60"]


# ── metadata attachment ──────────────────────────────────────────────────────
def vendor_lut(csv_path):
    """External code -> (VendorName, Centre). The CSV columns no code in this repo reads."""
    df = pd.read_csv(csv_path)
    return {r["External code"]: (str(r["VendorName"]), int(r["Centre"]))
            for _, r in df.iterrows()}


def attach_vendors(a, csv_path):
    """Add per-slice 'vendors'/'centres' to ANY arrays dict, derived from pids + the CSV.

    ACDC rows get the literal "ACDC" / -1 rather than its true "Siemens": ACDC is a
    different centre, protocol and label space, and tagging it Siemens would silently
    pool it into any M&Ms-Siemens groupby.
    """
    lut = vendor_lut(csv_path)
    vend, cent, missing = [], [], set()
    for pid, ds in zip(a["pids"], a["datasets"]):
        if ds != "MM":
            vend.append("ACDC"); cent.append(-1); continue
        sid = pid.split("_", 1)[1]
        if sid not in lut:
            missing.add(sid); vend.append("UNKNOWN"); cent.append(-1); continue
        v, c = lut[sid]
        vend.append(v); cent.append(c)
    if missing:
        print(f"    [warn] {len(missing)} M&Ms pids not in the CSV: {sorted(missing)[:5]}")
    b = dict(a)
    b["vendors"] = np.array(vend)
    b["centres"] = np.array(cent, dtype=np.int64)
    return b


def as_stream(a, src, dst):
    """Shallow copy of `a` exposing a[src] under the name `dst`.

    analyze_roictrl.paired_delta takes ONE stream name for both files, and only ever
    reads a[stream]/pids/slcs/labels/datasets — so a rename view lets us compare
    differently-named streams (sup_both_motion vs mag_SSIM) without touching it.
    """
    b = dict(a)
    b[dst] = a[src]
    return b


def patient_frame(a, stream, rule):
    """patient_reduce + the per-patient vendor/centre, aligned. -> (sc, plab, pds, pids, pvend)."""
    sc, pl, pd_, pids = patient_reduce(a, stream, rule)
    if "vendors" in a:
        lut = {p: v for p, v in zip(a["pids"], a["vendors"])}
        pv = np.array([lut.get(p, "UNKNOWN") for p in pids])
    else:
        pv = np.array(["UNKNOWN"] * len(pids))
    return sc, pl, pd_, pids, pv


# ── AUC cells ────────────────────────────────────────────────────────────────
def _cell(sc, pl, pids, n_boot=2000):
    """(auc, lo, hi, n_nor, n_dis) or NaNs when a class is missing."""
    if len(sc) == 0 or len(np.unique(pl != "NOR")) < 2:
        n = len(sc)
        return (np.nan, np.nan, np.nan, int((pl == "NOR").sum()), int(n - (pl == "NOR").sum()))
    r = _patient_auc(sc, pids, pl, n_boot=n_boot)
    return (r["auc"], r["lo"], r["hi"], r["n_nor"], r["n_dis"])


def _fmt_cell(c):
    a, lo, hi, nn, nd = c
    if not np.isfinite(a):
        return f"{'n/a':>26}"
    return f"{a:.3f} [{lo:.2f},{hi:.2f}] {nn}/{nd}".rjust(26)


def vendor_table(a, streams, rule, matched=False, n_boot=2000):
    """Per-vendor patient-Mean AUC on M&Ms-Testing, one row per vendor."""
    tag = "matched {NOR,DCM,HCM}" if matched else "all disease"
    print(f"\n    M&Ms-Testing by vendor/centre   [rule = {rule}]   [{tag}]")
    print("    " + f"{'vendor':<12}" + "".join(f"{s:>26}" for s in streams))
    print("    " + "-" * (12 + 26 * len(streams)))
    for v in ("Philips", "Siemens", "GE", "Canon"):
        line = f"    {v:<12}"
        for s in streams:
            if s not in a:
                line += f"{'--':>26}"; continue
            sc, pl, pd_, pids, pv = patient_frame(a, s, rule)
            m = (pd_ == "MM") & (pv == v)
            if matched:
                m &= np.isin(pl, MATCHED_LABELS)
            line += _fmt_cell(_cell(sc[m], pl[m], pids[m], n_boot))
        print(line)


def unpaired_delta(a, stream, rule, mask_a, mask_b, n_boot=10000, seed=0):
    """AUC(group A) - AUC(group B) for DISJOINT patient groups.

    NOT analyze_roictrl.paired_delta: that one asserts identical pid vectors and
    resamples them jointly, which is exactly wrong here — seen and unseen vendors are
    different patients, so the two AUCs are independent and must be bootstrapped
    independently within each group.
    """
    sc, pl, pd_, pids, pv = patient_frame(a, stream, rule)
    ya, sa = (pl[mask_a] != "NOR").astype(int), sc[mask_a]
    yb, sb = (pl[mask_b] != "NOR").astype(int), sc[mask_b]
    if len(ya) == 0 or len(yb) == 0 or ya.min() == ya.max() or yb.min() == yb.max():
        return None
    rng = np.random.default_rng(seed)
    pa, na = np.where(ya == 1)[0], np.where(ya == 0)[0]
    pb, nb = np.where(yb == 1)[0], np.where(yb == 0)[0]
    ds = []
    for _ in range(n_boot):
        ia = np.concatenate([rng.choice(pa, len(pa), True), rng.choice(na, len(na), True)])
        ib = np.concatenate([rng.choice(pb, len(pb), True), rng.choice(nb, len(nb), True)])
        if ya[ia].min() == ya[ia].max() or yb[ib].min() == yb[ib].max():
            continue
        ds.append(_fast_auc(ya[ia], sa[ia]) - _fast_auc(yb[ib], sb[ib]))
    ds = np.asarray(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(a=_fast_auc(ya, sa), b=_fast_auc(yb, sb),
                delta=_fast_auc(ya, sa) - _fast_auc(yb, sb), lo=lo, hi=hi,
                p=float(2 * min((ds <= 0).mean(), (ds >= 0).mean())),
                n_a=(int((ya == 0).sum()), int(ya.sum())),
                n_b=(int((yb == 0).sum()), int(yb.sum())))


def seen_unseen_block(a, streams, rules, seen, matched=False, n_boot=10000):
    """The ONE pre-committed vendor contrast: seen-in-training vs unseen, on M&Ms-Test."""
    unseen = tuple(v for v in ("Philips", "Siemens", "GE", "Canon") if v not in seen)
    tag = "matched {NOR,DCM,HCM}" if matched else "all disease"
    print(f"\n    SEEN {seen}  vs  UNSEEN {unseen}   [{tag}]")
    print(f"    {'stream':<26}{'rule':<10}{'seen':>8}{'unseen':>9}{'delta':>9}"
          f"{'95% CI':>18}{'p':>8}   n_seen  n_unseen")
    print("    " + "-" * 102)
    for s in streams:
        if s not in a:
            continue
        for rule in rules:
            sc, pl, pd_, pids, pv = patient_frame(a, s, rule)
            base = pd_ == "MM"
            if matched:
                base &= np.isin(pl, MATCHED_LABELS)
            r = unpaired_delta(a, s, rule, base & np.isin(pv, seen),
                               base & np.isin(pv, unseen), n_boot=n_boot)
            if r is None:
                continue
            sig = " *" if (r["lo"] > 0 or r["hi"] < 0) else ""
            ci = "[{:+.3f},{:+.3f}]".format(r["lo"], r["hi"])
            print(f"    {s:<26}{rule:<10}{r['a']:>8.3f}{r['b']:>9.3f}{r['delta']:>+9.3f}"
                  f"{ci:>18}{r['p']:>8.3f}   {r['n_a'][0]}/{r['n_a'][1]:<6} "
                  f"{r['n_b'][0]}/{r['n_b'][1]}{sig}")


# ── case-mix crosstab (the confound, printed so it can't be forgotten) ───────
def casemix(a):
    pids, seen = [], set()
    rows = []
    for p, d, l, v in zip(a["pids"], a["datasets"], a["labels"], a["vendors"]):
        if d != "MM" or p in seen:
            continue
        seen.add(p); rows.append((v, l))
    df = pd.DataFrame(rows, columns=["vendor", "pathology"])
    print("\n    M&Ms-Testing case mix (patients) — vendor x pathology")
    ct = pd.crosstab(df.vendor, df.pathology, margins=True)
    for line in ct.to_string().splitlines():
        print("      " + line)


# ── regression gate ──────────────────────────────────────────────────────────
def regression_gate(uad):
    """The stored headline numbers must still come out of the reference file."""
    print("\n[GATE] reference UAD arrays must reproduce the stored headline numbers")
    ok = True
    for stream, rule, exp_a, exp_m in (("mag_SSIM", "middle60", 0.860, 0.743),
                                       ("flow_SSIM", "mean", 0.812, 0.714)):
        if stream not in uad:
            print(f"    {stream:<11}{rule:<10} MISSING"); ok = False; continue
        c = auc_cell(uad, stream, rule)
        got_a, got_m = c["ACDC"][0], c["MM"][0]
        good = abs(got_a - exp_a) < 0.005 and abs(got_m - exp_m) < 0.005
        ok &= good
        print(f"    {stream:<11}{rule:<10}got {got_a:.3f}/{got_m:.3f}   "
              f"expected {exp_a:.3f}/{exp_m:.3f}   {'OK' if good else '** MISMATCH **'}")
    if not ok:
        print("    ** the reference file has drifted — nothing below this line is readable **")
    return ok


# ── supervised tables ────────────────────────────────────────────────────────
def sup_streams(a, pools, streams):
    return [f"sup_{p}_{s}" for p in pools for s in streams if f"sup_{p}_{s}" in a]


def ceiling_table(sup, uad, pools, streams, rule, n_boot=2000):
    """The ceiling reference: supervised cells beside the unsupervised bar."""
    print(f"\n    SUPERVISED CEILING   [rule = {rule}]   ACDC / M&Ms patient-Mean AUC")
    print(f"    unsupervised bar: GAN flow-SSIM {BAR_ACDC:.3f}/{BAR_MM:.3f} | "
          f"best QFAE mag_SSIM/middle60 0.860/0.743")
    head = "    " + f"{'train pool':<14}" + "".join(s.rjust(16) for s in streams)
    print(head)
    print("    " + "-" * (len(head) - 4))
    for p in pools:
        line = f"    {p:<14}"
        for s in streams:
            k = f"sup_{p}_{s}"
            line += (fmt(auc_cell(sup, k, rule, n_boot)).rjust(16) if k in sup
                     else f"{'--':>16}")
        print(line)


def vs_uad_block(sup, uad, pairs, rules, n_boot=10000):
    """Paired patient bootstrap, supervised minus unsupervised, on identical patients."""
    print("\n    SUPERVISED minus UNSUPERVISED (paired patient bootstrap, same patients)")
    print(f"    {'supervised stream':<26}{'vs UAD':<12}{'rule':<10}{'ds':<6}"
          f"{'uad':>7}{'sup':>8}{'delta':>8}{'95% CI':>18}{'p':>8}")
    print("    " + "-" * 103)
    for sup_key, uad_key in pairs:
        if sup_key not in sup or uad_key not in uad:
            continue
        for rule in rules:
            for ds in ("ACDC", "MM"):
                r = paired_delta(uad, as_stream(sup, sup_key, uad_key), uad_key, rule, ds,
                                 n_boot=n_boot)
                if r is None:
                    print(f"    {sup_key:<26}{uad_key:<12}{rule:<10}{ds:<6}"
                          f"  ** patient vectors diverged — run align() **")
                    continue
                sig = " *" if (r["lo"] > 0 or r["hi"] < 0) else ""
                ci = "[{:+.3f},{:+.3f}]".format(r["lo"], r["hi"])
                print(f"    {sup_key:<26}{uad_key:<12}{rule:<10}{ds:<6}"
                      f"{r['ref']:>7.3f}{r['new']:>8.3f}{r['delta']:>+8.3f}{ci:>18}"
                      f"{r['p']:>8.3f}{sig}")


def align(a_ref, a_new):
    """Intersect two arrays dicts on pid, loudly. Returns (ref, new) subset to common pids.

    Returns (None, None) when they share no patients — that means the supervised run used a
    different held-out universe than the reference, and every downstream comparison would be
    meaningless rather than merely noisy.
    """
    pr, pn = set(np.unique(a_ref["pids"])), set(np.unique(a_new["pids"]))
    if pr == pn:
        return a_ref, a_new
    common = pr & pn
    if not common:
        print("\n    ** align: the supervised and reference files share NO patients. **")
        print(f"       ref e.g. {sorted(pr)[:3]}   new e.g. {sorted(pn)[:3]}")
        print("       The held-out universe drifted — check GATE 2 in supervised_features.py.")
        return None, None
    print(f"    [align] ref-only: {sorted(pr - pn)[:8]}  new-only: {sorted(pn - pr)[:8]}")
    def sub(a):
        m = np.isin(a["pids"], list(common))
        n = len(a["pids"])
        return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
                for k, v in a.items()}
    return sub(a_ref), sub(a_new)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uad_dir", default="./qfae_dino2d_mae224_out_mmtest")
    ap.add_argument("--sup_dir", default="./sup_probe_mae224_mmtest")
    ap.add_argument("--mm_csv",
                    default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--uad_only", action="store_true",
                    help="Vendor-stratify the existing unsupervised arrays only (no GPU, no probe).")
    ap.add_argument("--recovered", action="store_true",
                    help="Training pool included the recovered GE cases, so GE counts as SEEN.")
    ap.add_argument("--rules", nargs="+", default=RULES)
    ap.add_argument("--pools", nargs="+",
                    default=["acdc", "mm", "both", "acdc_dh", "mm_dh", "both_dh"])
    ap.add_argument("--streams", nargs="+",
                    default=["appe", "motion", "motionstat", "appestat", "motionraw", "both"])
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    seen = SEEN_RECOVERED if args.recovered else SEEN_NORECOVER

    print("=" * 100)
    print("SUPERVISED CEILING PROBE + VENDOR STRATIFICATION — held-out ACDC-50 / M&Ms-Testing")
    print(f"CI ~ +-0.16 ACDC (10 NOR), +-0.09 M&Ms (32 NOR); seen-unseen SE ~ 0.11.")
    print("Read direction, not winners. vendor == centre exactly — say 'vendor/centre'.")
    print("=" * 100)

    uad = load_arm(args.uad_dir)
    if uad is None:
        print(f"** no UAD arrays at {args.uad_dir} — nothing to do **")
        return
    uad = attach_vendors(uad, args.mm_csv)
    regression_gate(uad)
    casemix(uad)

    print("\n" + "#" * 100)
    print("### PART 1 — the EXISTING unsupervised model, stratified by vendor/centre")
    print("### No new experiment: 'vendors' is derived from pids + the diagnosis CSV.")
    print("#" * 100)
    for rule in args.rules:
        vendor_table(uad, UAD_STREAMS, rule, matched=False, n_boot=args.n_boot)
    vendor_table(uad, UAD_STREAMS, args.rules[-1], matched=True, n_boot=args.n_boot)
    seen_unseen_block(uad, UAD_STREAMS, args.rules, seen, matched=False)
    seen_unseen_block(uad, UAD_STREAMS, args.rules, seen, matched=True)

    if args.uad_only:
        print("\n--- Phase 0 only (--uad_only). Run the probe for the ceiling tables. ---")
        return

    sup = load_arm(args.sup_dir)
    if sup is None:
        print(f"\n** no supervised arrays at {args.sup_dir} — run supervised_fit.py **")
        return
    sup = attach_vendors(sup, args.mm_csv)
    uad_a, sup_a = align(uad, sup)
    if uad_a is None:
        return

    print("\n" + "#" * 100)
    print("### PART 2 — the supervised ceiling")
    print("#" * 100)
    for rule in args.rules:
        ceiling_table(sup_a, uad_a, args.pools, args.streams, rule, args.n_boot)

    if "sup_shuffle_both_both" in sup_a:
        print("\n    [CONTROL] patient-level label shuffle (must land ~0.35-0.65):"
              f"  {fmt(auc_cell(sup_a, 'sup_shuffle_both_both', 'mean'))}")

    print("\n" + "#" * 100)
    print("### PART 3 — supervised vs unsupervised, and the vendor read on the probe")
    print("#" * 100)
    vs_uad_block(sup_a, uad_a,
                 [("sup_both_motion", "mag_SSIM"), ("sup_both_appe", "appearance"),
                  ("sup_both_both", "mag_SSIM"), ("sup_both_motionstat", "mag_SSIM")],
                 args.rules)
    for rule in args.rules:
        vendor_table(sup_a, sup_streams(sup_a, ["both"], ["appe", "motion", "motionstat"]),
                     rule, matched=False, n_boot=args.n_boot)
    seen_unseen_block(sup_a, sup_streams(sup_a, ["both"], ["appe", "motion", "motionstat"]),
                      args.rules, seen)


if __name__ == "__main__":
    main()
