"""analyze_regflow.py — registration-flow teacher vs Farneback optical flow, 2-D QFAE (held-out).

For MAE@224 and DINOv2@224 (coupled scoring), compare the registration-net flow teacher against
Farneback optical flow, head-to-head on ACDC-50 + M&Ms-Test. Motion streams (flow_SSIM / mag_SSIM /
flow_L1) at rule=mean and middle-60%, per dataset. Also the appearance+flow equal-weight rank fusion
(the axis where the 3-D CineMA reg-flow won ACDC 0.936).
"""
import numpy as np
from qfae_report import _load, _reduce_slices, _fast_auc

# (encoder, farneback_npz, registration_npz)
PAIRS = [
    ("MAE@224",
     "qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz",
     "qfae_dino2d_mae224_reg_out_mmtest/qfae_dino_arrays.npz"),
    ("DINOv2@224",
     "qfae_dino2d_dino224_out_mmtest/qfae_dino_arrays.npz",
     "qfae_dino2d_dino224_reg_out_mmtest/qfae_dino_arrays.npz"),
]
MOTION = ["flow_SSIM", "mag_SSIM", "flow_L1"]


def per_patient(a, stream, rule):
    pids, slcs, lab, ds = a["pids"], a["slcs"], a["labels"], a["datasets"]
    uniq = np.unique(pids)
    plab = np.array([lab[pids == p][0] for p in uniq])
    pds = np.array([ds[pids == p][0] for p in uniq])
    vals = a[stream]
    sc = np.empty(len(uniq))
    for i, p in enumerate(uniq):
        m = pids == p
        v = vals[m][np.argsort(slcs[m])]
        sc[i] = _reduce_slices(v, len(v), rule)
    return sc, plab, pds


def auc(a, stream, rule, ds):
    sc, plab, pds = per_patient(a, stream, rule)
    m = pds == ds
    y = (plab[m] != "NOR").astype(int)
    return _fast_auc(y, sc[m]) if len(np.unique(y)) > 1 else np.nan


def rank(v):
    return v.argsort().argsort() / max(1, len(v) - 1)


def fuse_auc(a, rule, ds, w=0.5):
    """equal-weight rank fusion of appearance + flow_SSIM."""
    A, plab, pds = per_patient(a, "appearance", rule)
    F, _, _ = per_patient(a, "flow_SSIM", rule)
    m = pds == ds
    y = (plab[m] != "NOR").astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return _fast_auc(y, (1 - w) * rank(F[m]) + w * rank(A[m]))


def main():
    print("=" * 78)
    print("REGISTRATION-FLOW vs FARNEBACK — 2-D QFAE, held-out ACDC-50 / M&Ms-Test (patient AUC)")
    print("GAN bar: ACDC 0.8125 / M&Ms 0.731")
    print("=" * 78)
    for enc, fnpz, rnpz in PAIRS:
        try:
            fa, ra = _load(fnpz), _load(rnpz)
        except FileNotFoundError as e:
            print(f"\n### {enc}: missing ({e})"); continue
        print(f"\n### {enc}   (Farneback -> Registration)")
        print(f"    {'stream':<12}{'rule':<10}{'ACDC farn/reg':>18}{'M&Ms farn/reg':>18}")
        for stream in MOTION:
            for rule in ("mean", "middle60"):
                row = f"    {stream:<12}{rule:<10}"
                for ds in ("ACDC", "MM"):
                    row += f"{auc(fa, stream, rule, ds):.3f}/{auc(ra, stream, rule, ds):.3f}".rjust(18)
                print(row)
        # appearance + flow equal-weight fusion (the ACDC axis for CineMA reg-flow)
        print(f"    {'appe+flow':<12}{'mean':<10}"
              + f"{fuse_auc(fa,'mean','ACDC'):.3f}/{fuse_auc(ra,'mean','ACDC'):.3f}".rjust(18)
              + f"{fuse_auc(fa,'mean','MM'):.3f}/{fuse_auc(ra,'mean','MM'):.3f}".rjust(18))


if __name__ == "__main__":
    main()
