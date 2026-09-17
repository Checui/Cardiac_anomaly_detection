"""confirm_val.py — gold-standard confirmatory: SELECT recipe on M&Ms-Validation, REPORT on M&Ms-Test.

Removes the "we picked the recipe by looking at M&Ms-Test" optimism. For each model we build the
(stream, rule) grid on the *_mmval arrays (M&Ms-Val, 9 NOR — disjoint from Test) and on the *_mmtest
arrays (M&Ms-Test, 32 NOR). We pick the recipe that maximises M&Ms-VAL AUC, then report its
M&Ms-TEST AUC. Motion pre-committed (only motion crosses the gap); also shown over all streams.
"""

import numpy as np
from confirm_offline import grid_for_run, MOTION

# (name, fmt, val_npz, test_npz)
PAIRS = [
    ("CineMA-SP",  "stack",
     "qfae_flow_sp_out_mmval/qfae_arrays.npz",            "qfae_flow_sp_out_mmtest/qfae_arrays.npz"),
    ("DINOv2@224", "slice",
     "qfae_dino2d_dino224_out_mmval/qfae_dino_arrays.npz", "qfae_dino2d_dino224_out_mmtest/qfae_dino_arrays.npz"),
    ("DINOv2@518", "slice",
     "qfae_dino2d_dino518_out_mmval/qfae_dino_arrays.npz", "qfae_dino2d_dino518_out_mmtest/qfae_dino_arrays.npz"),
    ("MAE@224",    "slice",
     "qfae_dino2d_mae224_out_mmval/qfae_dino_arrays.npz",  "qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz"),
]


def main():
    val_g, test_g = {}, {}
    for name, fmt, vnpz, tnpz in PAIRS:
        try:
            val_g[name] = grid_for_run(name, vnpz, fmt)
            test_g[name] = grid_for_run(name, tnpz, fmt)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
    print("=" * 84)
    print("GOLD-STANDARD CONFIRMATORY — select recipe on M&Ms-VAL (9 NOR), report on M&Ms-TEST (32 NOR)")
    print("GAN held-out M&Ms-Test = 0.731")
    print("=" * 84)

    # per-model: best MOTION recipe on val, its test number
    print(f"\n[per model]  picked-on-val (motion) -> reported-on-test")
    print(f"    {'model':<12}{'stream/rule':<22}{'val MM':>9}{'test MM':>10}{'test ACDC':>11}")
    recs = []
    for name in val_g:
        cand = [(s, r) for (s, r) in val_g[name] if s in MOTION
                and not np.isnan(val_g[name][(s, r)]["MM"])]
        best = max(cand, key=lambda k: val_g[name][k]["MM"])
        vt = val_g[name][best]["MM"]; tt = test_g[name][best]["MM"]; ta = test_g[name][best]["ACDC"]
        recs.append((name, best, vt, tt, ta))
        print(f"    {name:<12}{best[0]+'/'+best[1]:<22}{vt:>9.3f}{tt:>10.3f}{ta:>11.3f}")

    # global: pick (config, stream, rule) maximising val-MM (motion), report test
    allc = [(name, s, r, val_g[name][(s, r)]["MM"], test_g[name][(s, r)]["MM"], test_g[name][(s, r)]["ACDC"])
            for name in val_g for (s, r) in val_g[name]
            if s in MOTION and not np.isnan(val_g[name][(s, r)]["MM"])]
    gb = max(allc, key=lambda r: r[3])
    print(f"\n[global, motion pre-committed] select on val-MM across ALL configs:")
    print(f"    picked: {gb[0]} / {gb[1]} / {gb[2]}   (val-MM={gb[3]:.3f})")
    print(f"    -> held-out M&Ms-TEST = {gb[4]:.3f}   (ACDC {gb[5]:.3f})   vs GAN 0.731")

    # sanity: does val agree with test on mag_SSIM vs flow_L1 for MAE@224 (the 0.743 question)?
    if "MAE@224" in val_g:
        print(f"\n[MAE@224 mag_SSIM vs flow_L1, middle60 — does val pick the same as test?]")
        for s in ("mag_SSIM", "flow_L1", "flow_SSIM"):
            k = (s, "middle60")
            print(f"    {s:<10} middle60:  val-MM={val_g['MAE@224'][k]['MM']:.3f}  "
                  f"test-MM={test_g['MAE@224'][k]['MM']:.3f}")


if __name__ == "__main__":
    main()
