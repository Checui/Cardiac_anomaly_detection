"""analyze_flowabl.py — read out the two flow-head ablations (held-out, ACDC-50 + M&Ms-Test).

Part A: flow-only + lambda_appe sweep — does removing/shrinking the appearance head help the flow
        head?  Report flow_SSIM alone (the pre-committed motion stream) per dataset, rule=mean and
        middle60, across lambda_appe {1.0 base, 0.5, 0.25, 0.0}.
Part B: PIXEL-space appearance vs embedding — is pixel-space (GAN-style) appearance a better stream
        to fuse with flow?  appe-alone / flow-alone / equal-weight rank fuse / oracle / val-select,
        for MAE@224 and DINOv2@224, and pixel-vs-embedding appe-alone side by side.
"""

import numpy as np
from fusion_sweep_2d import per_patient, run
from qfae_report import _fast_auc

GAN = "GAN bar: ACDC 0.8125 / M&Ms-test 0.731. Baseline embedding MAE@224 flow_SSIM: mean 0.714 / middle60 0.732"

FLOWONLY = [
    ("la=1.0 (baseline)",   "qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz"),
    ("la=0.5",              "qfae_flowonly_la0.5_out_mmtest/qfae_dino_arrays.npz"),
    ("la=0.25",             "qfae_flowonly_la0.25_out_mmtest/qfae_dino_arrays.npz"),
    ("la=0.0 (flow-only)",  "qfae_flowonly_la0.0_out_mmtest/qfae_dino_arrays.npz"),
]

PIXEL = [
    ("MAE@224 pixel",    "qfae_pixelappe_mae224_out_mmtest/qfae_dino_arrays.npz",
                         "qfae_pixelappe_mae224_out_mmval/qfae_dino_arrays.npz"),
    ("DINOv2@224 pixel", "qfae_pixelappe_dino224_out_mmtest/qfae_dino_arrays.npz",
                         "qfae_pixelappe_dino224_out_mmval/qfae_dino_arrays.npz"),
]
EMBED = {  # matching embedding-appe runs, for pixel-vs-embedding appe-alone
    "MAE@224 pixel":    "qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz",
    "DINOv2@224 pixel": "qfae_dino2d_dino224_out_mmtest/qfae_dino_arrays.npz",
}


def main():
    print("=" * 78)
    print("FLOW-HEAD ABLATIONS — held-out ACDC-50 + M&Ms-Test (32 NOR/104).", GAN)
    print("=" * 78)

    # ── Part A ──
    print("\n### PART A — flow-only + lambda_appe sweep (flow_SSIM ALONE, patient AUC)")
    print(f"    {'model':<22}{'rule':<10}{'ACDC':>9}{'M&Ms':>9}")
    for name, npz in FLOWONLY:
        for rule in ("mean", "middle60"):
            d = per_patient(npz, "slice", rule)
            a = _fast_auc(d["ACDC"]["y"], d["ACDC"]["flow"])
            m = _fast_auc(d["MM"]["y"], d["MM"]["flow"])
            print(f"    {name:<22}{rule:<10}{a:>9.3f}{m:>9.3f}")

    # ── Part B ──
    print("\n### PART B — PIXEL-space appearance: appe-alone (pixel vs embedding), then fusion")
    print(f"    {'model':<22}{'appe-alone rule':<16}{'ACDC pix/emb':>16}{'M&Ms pix/emb':>16}")
    for name, tnpz, _ in PIXEL:
        for rule in ("mean", "middle60"):
            dp = per_patient(tnpz, "slice", rule)
            de = per_patient(EMBED[name], "slice", rule)
            row = f"    {name:<22}{rule:<16}"
            for ds in ("ACDC", "MM"):
                ap = _fast_auc(dp[ds]["y"], dp[ds]["appe"])
                ae = _fast_auc(de[ds]["y"], de[ds]["appe"])
                row += f"{ap:.3f}/{ae:.3f}".rjust(16)
            print(row)

    print("\n### PART B — fusion (pixel appearance + flow), same machinery as fusion_sweep_2d")
    for name, tnpz, vnpz in PIXEL:
        for rule in ("mean", "middle60"):
            run(name, "slice", tnpz, vnpz, rule, "rank")


if __name__ == "__main__":
    main()
