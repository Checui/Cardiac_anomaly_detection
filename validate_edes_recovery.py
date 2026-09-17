"""validate_edes_recovery.py — proof that the M&Ms ED/ES recovery rule is sound.

THE PROBLEM
-----------
All 25 GE cases in M&Ms `Training` carry `ED = ES = 0` in the diagnosis CSV (missing
metadata, not a coincidence). `cinema_faithful.load_mm_records_flow` skips any row with
`ed == es`, so **every model in this project has silently trained on Siemens + Philips
only** — 150 of 175 M&Ms training patients, 38 NOR rather than 48. GE and Canon were
therefore both unseen vendors at test time, not Canon alone.

THE RULE
--------
`<sid>_sa_gt.nii.gz` is labelled at exactly two frames — ED and ES. The LV cavity
(label 1 for M&Ms) is by definition largest at end-diastole and smallest at end-systole:

    ED = argmax_t |LV(t)| ,  ES = argmin_t |LV(t)|   over the labelled frames

THE PROOF
---------
Replay the rule on every case whose CSV ED/ES *is* valid and require exact agreement.
If it reproduces the CSV wherever the CSV exists, applying it where the CSV is missing
is interpolation, not invention.

M&Ms `Testing` and `Validation` contain NO `ed == es` rows, so recovery is a provable
no-op there and the held-out universe cannot move — which is what makes the pre-/post-
recovery UAD runs a clean A/B.

    python validate_edes_recovery.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813

LV_LABEL_MM = 1


def recover_edes(gt4d_array, lv_label=LV_LABEL_MM):
    """(T, Z, Y, X) label array -> (ed, es) from LV cavity area, or None if <2 labelled frames.

    ED = the labelled frame with the LARGEST LV cavity, ES = the smallest.
    """
    areas = {t: int((gt4d_array[t] == lv_label).sum())
             for t in range(gt4d_array.shape[0]) if gt4d_array[t].any()}
    areas = {t: v for t, v in areas.items() if v > 0}
    if len(areas) < 2:
        return None
    return max(areas, key=areas.get), min(areas, key=areas.get)


def _gt_array(mm_dir, sid):
    """Labelled 4-D mask as (T, Z, Y, X).

    Goes through cinema_faithful._read_sitk_image, which carries the nibabel fallback for
    the handful of M&Ms files whose direction cosines ITK rejects as non-orthonormal —
    reading them with raw sitk.ReadImage raises.
    """
    from cinema_faithful import _read_sitk_image
    p = os.path.join(mm_dir, f"{sid}_sa_gt.nii.gz")
    if not os.path.exists(p):
        return None
    try:
        arr = sitk.GetArrayFromImage(_read_sitk_image(p, sitk.sitkUInt8))
    except Exception as e:                                    # noqa: BLE001 — report, don't crash
        print(f"      [warn] {sid}: unreadable mask ({type(e).__name__})")
        return None
    return arr if arr.ndim == 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm_dirs", nargs="+",
                    default=["../Dataset_1/Training", "../Dataset_1/Validation",
                             "../Dataset_1/Testing"])
    ap.add_argument("--mm_csv",
                    default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.mm_csv)
    lut = {r["External code"]: (int(r["ED"]), int(r["ES"]), str(r["VendorName"]),
                               str(r["Pathology"])) for _, r in df.iterrows()}

    print("=" * 92)
    print("M&Ms ED/ES RECOVERY — validation")
    print("  rule: ED = argmax LV-area, ES = argmin LV-area over the labelled _gt frames")
    print("=" * 92)

    total_ok = total_bad = 0
    for mm_dir in args.mm_dirs:
        if not os.path.isdir(mm_dir):
            print(f"\n### {mm_dir} — MISSING, skipped")
            continue
        sids = sorted(f[: -len("_sa.nii.gz")] for f in os.listdir(mm_dir)
                      if f.endswith("_sa.nii.gz") and not f.endswith("_sa_gt.nii.gz"))
        valid = [s for s in sids if s in lut and lut[s][0] != lut[s][1]]
        broken = [s for s in sids if s in lut and lut[s][0] == lut[s][1]]

        print(f"\n### {mm_dir}   {len(sids)} patients   "
              f"({len(valid)} with valid CSV ED/ES, {len(broken)} with ED==ES)")
        if broken:
            vend = pd.Series([lut[s][2] for s in broken]).value_counts().to_dict()
            print(f"    ED==ES vendors: {vend}")

        ok = bad = unreadable = 0
        fails = []
        for sid in valid:
            g = _gt_array(mm_dir, sid)
            if g is None:
                unreadable += 1; continue
            r = recover_edes(g)
            if r is None:
                unreadable += 1; continue
            if r == (lut[sid][0], lut[sid][1]):
                ok += 1
            else:
                bad += 1; fails.append((sid, (lut[sid][0], lut[sid][1]), r))
        total_ok += ok; total_bad += bad
        print(f"    VALIDATION: exact match {ok}/{ok + bad}   mismatch {bad}   "
              f"unreadable {unreadable}")
        for f in fails[:10]:
            print(f"      MISMATCH {f[0]}  csv={f[1]}  recovered={f[2]}")

        if broken:
            print(f"    RECOVERY of the {len(broken)} ED==ES cases:")
            rec = 0
            for sid in broken:
                g = _gt_array(mm_dir, sid)
                r = recover_edes(g) if g is not None else None
                if r is None:
                    print(f"      {sid:<8} FAILED (no usable mask)"); continue
                rec += 1
                if rec <= 8:
                    lab = sorted(t for t in range(g.shape[0]) if g[t].any())
                    print(f"      {sid:<8} {lut[sid][2]:<8} {lut[sid][3]:<6} "
                          f"labelled={lab}  ->  ED={r[0]} ES={r[1]}")
            print(f"      recovered {rec}/{len(broken)}")

    print("\n" + "=" * 92)
    if total_bad == 0 and total_ok > 0:
        print(f"VERDICT: rule reproduces the CSV on {total_ok}/{total_ok} validatable cases "
              f"(0 mismatches). Safe to apply where the CSV is missing.")
    else:
        print(f"VERDICT: ** {total_bad} MISMATCHES out of {total_ok + total_bad} — "
              f"DO NOT USE THE RECOVERY **")
    print("=" * 92)


if __name__ == "__main__":
    main()
