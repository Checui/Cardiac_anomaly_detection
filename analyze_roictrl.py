"""analyze_roictrl.py — does the motion-ROI gain survive its controls? (offline, no GPU)

Collates `qfae_roictrl.pbs` (arms: base / roi_motion / roi_pred_motion / roi_center).

THE QUESTION. `--roi motion` lifted MAE@224 M&Ms from 0.743 to 0.800, but it ranks the
pooling mask on the GT flow magnitude and then scores the pred-vs-GT error inside that mask —
the pixels being graded were chosen by looking at the answer key. That can inflate AUC two
ways without the detector improving: SSIM's data_range and local statistics are computed over
the masked region, so the score's normalisation becomes a function of the patient's motion
amplitude; and |GT flow| is itself disease-correlated, so the mask carries diagnostic signal
before the network is consulted. The 2-D models also run single-pass on ED while the GT flow
needs ES, so the mask consumes a frame the network never sees.

THE TESTS, in the order they should be read:

  [1] PAIRED vs whole frame. Same model, same patients, only the pooling mask differs — the
      marginal +-0.09 CI is the wrong yardstick and hides real effects.

  [2] PAIRED pred_motion vs motion. THE DECISIVE ONE. If ranking the mask on the model's own
      predicted field (which depends only on ED, never on the GT) reproduces the GT-ranked
      gain, the effect is about WHERE you pool and the detector earns the credit. If it
      collapses back toward whole-frame, the gain was GT conditioning.

  [3] MODEL-FREE floor. AUC of GT displacement statistics alone, network removed entirely,
      pooled over the same masks. Read |AUC - 0.5| — the direction is not fixed a priori (a
      hypokinetic ventricle moves LESS, so the disease-high direction can flip). If a
      one-line optical-flow statistic matches the detector, the honest claim is "a flow
      statistic separates disease", not "our anomaly detector improved".

A verdict block at the end states which reading the numbers support. It is deliberately
conservative: with 32 M&Ms NOR patients, "inconclusive" is a common and honest outcome.
"""

import os

import numpy as np

from analyze_abcd import GT_ONLY, MOTION, RULES, auc_cell, fmt, load_arm, patient_reduce, table
from qfae_report import _fast_auc

ENCODERS = ("mae224", "dino224")
ARMS = [("base", "whole frame"), ("roi_motion", "ROI from GT flow (headline)"),
        ("roi_pred_motion", "ROI from PREDICTED flow (control)"), ("roi_center", "central disc")]
KEY = [("flow_SSIM", "mean"), ("flow_SSIM", "middle60"), ("mag_SSIM", "mean"), ("mag_SSIM", "middle60")]


def _arm(enc, arm):
    return load_arm(f"./qfae_roictrl_{enc}_{arm}_mmtest")


def paired_delta(a_ref, a_new, stream, rule, ds, n_boot=10000, seed=0):
    """AUC(new) - AUC(ref) on identical patients, with a patient-stratified paired bootstrap."""
    s0, l0, d0, p0 = patient_reduce(a_ref, stream, rule)
    s1, l1, d1, p1 = patient_reduce(a_new, stream, rule)
    if len(p0) != len(p1) or not (p0 == p1).all():
        return None
    m = d0 == ds
    y = (l0[m] != "NOR").astype(int)
    if y.min() == y.max():
        return None
    x0, x1 = s0[m], s1[m]
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    deltas = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[bi]
        if yy.min() != yy.max():
            deltas.append(_fast_auc(yy, x1[bi]) - _fast_auc(yy, x0[bi]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return dict(ref=_fast_auc(y, x0), new=_fast_auc(y, x1), delta=_fast_auc(y, x1) - _fast_auc(y, x0),
                lo=lo, hi=hi, p=p)


def paired_block(title, enc, ref_arm, new_arm, note):
    a0, a1 = _arm(enc, ref_arm), _arm(enc, new_arm)
    if a0 is None or a1 is None:
        print(f"\n{title} — SKIPPED (missing {ref_arm if a0 is None else new_arm})")
        return {}
    print(f"\n{title}")
    print(f"    {note}")
    print(f"    {'stream':<11}{'rule':<10}{'ds':<6}{'ref':>7}{'new':>8}{'delta':>8}{'95% CI':>18}{'p':>8}")
    print("    " + "-" * 74)
    got = {}
    for stream, rule in KEY:
        for ds in ("ACDC", "MM"):
            r = paired_delta(a0, a1, stream, rule, ds)
            if r is None:
                continue
            sig = " *" if (r["lo"] > 0 or r["hi"] < 0) else ""
            ci = "[{:+.3f},{:+.3f}]".format(r["lo"], r["hi"])
            print(f"    {stream:<11}{rule:<10}{ds:<6}{r['ref']:>7.3f}{r['new']:>8.3f}"
                  f"{r['delta']:>+8.3f}{ci:>18}{r['p']:>8.3f}{sig}")
            got[(stream, rule, ds)] = r
    return got


def main():
    print("=" * 92)
    print("ROI CIRCULARITY CONTROL — held-out ACDC-50 / M&Ms-Testing, patient-Mean AUC")
    print("=" * 92)

    for enc in ENCODERS:
        rows = [(lbl, _arm(enc, arm)) for arm, lbl in ARMS]
        if all(a is None for _, a in rows):
            print(f"\n[{enc}] no roictrl outputs found — run `qsub -v ENC={enc} qfae_roictrl.pbs`")
            continue
        print(f"\n{'#' * 92}\n# {enc}\n{'#' * 92}")

        # sanity: base must still reproduce the established pipeline
        base = _arm(enc, "base")
        if base is not None:
            c = auc_cell(base, "flow_SSIM", "mean")
            print(f"\n[GATE] {enc} base flow_SSIM/mean = {fmt(c)}   "
                  f"(mae224 must read 0.812/0.714, dino224 0.775/0.718)")

        for rule in RULES:
            table(f"[0] all masks — {enc}", rows, MOTION, rule)

        d_gt = paired_block(f"[1] PAIRED: GT-flow ROI vs whole frame — {enc}", enc,
                            "base", "roi_motion",
                            "the headline effect, measured on identical patients")
        d_pr = paired_block(f"[1b] PAIRED: PREDICTED-flow ROI vs whole frame — {enc}", enc,
                            "base", "roi_pred_motion",
                            "same rule, mask ranked on the model's own output — no GT involved")
        d_vs = paired_block(f"[2] PAIRED: PREDICTED-flow ROI vs GT-flow ROI — {enc}  <-- DECISIVE",
                            enc, "roi_motion", "roi_pred_motion",
                            "delta ~ 0 => the two masks agree, the gain is not GT conditioning; "
                            "strongly negative => it was")

        # [3] model-free floor
        rows3 = [(lbl, _arm(enc, arm)) for arm, lbl in ARMS if _arm(enc, arm) is not None
                 and any(k in _arm(enc, arm) for k in GT_ONLY)]
        if rows3:
            for rule in RULES:
                table(f"[3] MODEL-FREE floor: GT displacement statistics, no network — {enc}",
                      rows3, GT_ONLY, rule,
                      note="read |AUC - 0.5|, direction is not fixed a priori"
                           if rule == "mean" else "")

        # ── verdict ──
        print(f"\n--- VERDICT ({enc}) ---")
        mm_gt = [v["delta"] for k, v in d_gt.items() if k[2] == "MM"]
        mm_pr = [v["delta"] for k, v in d_pr.items() if k[2] == "MM"]
        mm_vs = [(v["delta"], v["lo"], v["hi"]) for k, v in d_vs.items() if k[2] == "MM"]
        if not (mm_gt and mm_pr and mm_vs):
            print("  incomplete — need base + roi_motion + roi_pred_motion for this encoder")
        else:
            g, p_ = float(np.mean(mm_gt)), float(np.mean(mm_pr))
            worse = sum(1 for d, lo, hi in mm_vs if hi < 0)
            print(f"  mean M&Ms delta vs whole frame:  GT mask {g:+.3f}   predicted mask {p_:+.3f}")
            print(f"  predicted-vs-GT significantly worse in {worse}/{len(mm_vs)} cells")
            if p_ >= 0.6 * g and worse == 0:
                print("  => SURVIVES. The GT-free mask reproduces most of the gain, so the effect is")
                print("     about WHERE the score is pooled and the detector earns the credit.")
            elif p_ <= 0.25 * g or worse >= len(mm_vs) - 1:
                print("  => DOES NOT SURVIVE. The gain largely disappears once the mask stops")
                print("     touching the GT field: it was conditioning on the answer key, not a")
                print("     better detector. Do NOT report 0.800 as a modelling result.")
            else:
                print("  => INCONCLUSIVE at this sample size: the predicted mask keeps part of the")
                print("     gain. Report the GT-masked number only alongside this control, and")
                print("     treat the predicted-mask number as the defensible one.")
        # model-free floor inside the SAME mask as the headline: the strongest GT-only statistic
        gm = _arm(enc, "roi_motion")
        if gm is not None:
            cands = [(abs(auc_cell(gm, s, r)["MM"][0] - 0.5), s, r, auc_cell(gm, s, r)["MM"][0])
                     for s in GT_ONLY if s in gm for r in RULES]
            cands = [c for c in cands if np.isfinite(c[0])]
            if cands:
                dist, s, r, auc = max(cands)
                print(f"  model-free floor inside the motion ROI: {s}/{r} M&Ms AUC={auc:.3f} "
                      f"(|AUC-0.5|={dist:.3f}) — the detector must clear this to earn the credit")


if __name__ == "__main__":
    main()
