"""summarize_encoders.py — per-frozen-encoder held-out summary (ACDC-50 + M&Ms-Test).

Reports, for each frozen encoder used in the QFAE, the patient-level AUC of the appearance
(reconstruction) stream and the motion (flow) streams, per dataset. Numbers read directly from the
saved held-out arrays so the supervisor table is exact.
"""
import os
import numpy as np
from qfae_report import _load, _reduce_slices, _fast_auc

# (label, format, npz).  'stack' = CineMA single-pass per-stack (+_slices matrices); 'slice' = 2-D.
RUNS = [
    ("CineMA (single-pass, 3-D)", "stack", "qfae_flow_sp_out_mmtest/qfae_arrays.npz"),
    ("DINOv2 @224 (2-D)",         "slice", "qfae_dino2d_dino224_out_mmtest/qfae_dino_arrays.npz"),
    ("DINOv2 @518 (2-D)",         "slice", "qfae_dino2d_dino518_out_mmtest/qfae_dino_arrays.npz"),
    ("MAE @224 (2-D)",            "slice", "qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz"),
    ("DINOv3 @224 (2-D)",         "slice", "qfae_dino2d_dinov3224_out_mmtest/qfae_dino_arrays.npz"),
]
STREAMS = ["appearance", "flow_SSIM", "mag_SSIM"]


def patient_scores(npz, fmt, stream, rule):
    a = _load(npz)
    pids, labels, datasets = a["pids"], a["labels"], a["datasets"]
    if fmt == "stack":
        key = {"appearance": "appe_slices", "flow_SSIM": "flow_SSIM_slices",
               "mag_SSIM": "mag_SSIM_slices"}[stream]
        mat, nreal = a[key], a["n_real_slices"]
        sc = np.array([_reduce_slices(mat[i], nreal[i], rule) for i in range(len(mat))])
        return sc, labels, datasets                                   # 1 stack == 1 patient
    uniq = np.unique(pids)
    plab = np.array([labels[pids == p][0] for p in uniq])
    pds = np.array([datasets[pids == p][0] for p in uniq])
    vals, slcs = a[stream], a["slcs"]
    sc = np.empty(len(uniq))
    for i, p in enumerate(uniq):
        m = pids == p
        v = vals[m][np.argsort(slcs[m])]
        sc[i] = _reduce_slices(v, len(v), rule)
    return sc, plab, pds


def auc(npz, fmt, stream, rule, ds):
    sc, lab, dsarr = patient_scores(npz, fmt, stream, rule)
    m = dsarr == ds
    y = (lab[m] != "NOR").astype(int)
    return _fast_auc(y, sc[m]) if len(np.unique(y)) > 1 else np.nan


def main():
    print("=" * 96)
    print("FROZEN-ENCODER COMPARISON — held-out ACDC-50 + M&Ms-Test (patient-level AUC)")
    print("GAN baseline (Farneback flow-SSIM): ACDC 0.8125 / M&Ms 0.731")
    print("=" * 96)
    hdr = (f"{'encoder':<26}| {'APPEARANCE (mean)':<18}| {'flow_SSIM (mean)':<18}| "
           f"{'flow_SSIM (mid60)':<18}| {'mag_SSIM (mid60)':<18}")
    print(hdr); print("-" * len(hdr))
    for label, fmt, npz in RUNS:
        if not os.path.exists(npz):
            print(f"{label:<26}| (arrays not found: {npz})")
            continue
        cells = []
        for stream, rule in (("appearance", "mean"), ("flow_SSIM", "mean"),
                             ("flow_SSIM", "middle60"), ("mag_SSIM", "middle60")):
            a = auc(npz, fmt, stream, rule, "ACDC"); m = auc(npz, fmt, stream, rule, "MM")
            cells.append(f"{a:.3f} / {m:.3f}")
        print(f"{label:<26}| " + "| ".join(f"{c:<18}" for c in cells))
    print("\n(each cell = ACDC / M&Ms.  'mid60' = middle-60% slice aggregation, the GAN's rule.)")


if __name__ == "__main__":
    main()
