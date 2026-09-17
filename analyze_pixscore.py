"""analyze_pixscore.py — pixel score-way sweep on the pixel-trained QFAE models (held-out).

For each pixel-appe model, report the per-dataset patient AUC of every pixel anomaly measure on the
SAME reconstruction: MSE, MAE(L1), 1-SSIM, MSE+grad (training-loss score). flow_SSIM shown for
reference. rule = mean and middle-60%. (Scoring-only — no retraining.)
"""
import numpy as np
from qfae_report import _load, _reduce_slices, _fast_auc

RUNS = [
    ("MAE@224 pixel",    "qfae_pixelappe_mae224_out_mmtest/qfae_dino_arrays.npz"),
    ("DINOv2@224 pixel", "qfae_pixelappe_dino224_out_mmtest/qfae_dino_arrays.npz"),
]
APPE = ["appe_mse", "appe_mae", "appe_ssim", "appe_msegrad"]


def patient_reduce(a, stream, rule):
    pids, slcs = a["pids"], a["slcs"]
    uniq = np.unique(pids)
    vals = a[stream]
    sc = np.empty(len(uniq))
    for i, p in enumerate(uniq):
        m = pids == p
        v = vals[m][np.argsort(slcs[m])]
        sc[i] = _reduce_slices(v, len(v), rule)
    return sc, uniq


def main():
    print("=" * 60)
    print("PIXEL SCORE-WAY SWEEP — held-out ACDC-50 / M&Ms-Test (patient AUC)")
    print("GAN appearance is dead weight on M&Ms; flow_SSIM ~0.73 is the bar.")
    print("=" * 60)
    for name, npz in RUNS:
        try:
            a = _load(npz)
        except FileNotFoundError:
            print(f"\n### {name}: {npz} MISSING (job not done yet)"); continue
        if "appe_mse" not in a:
            print(f"\n### {name}: no pixel-sweep streams (re-run qfae_pixscore_eval.pbs)"); continue
        pids, labels, datasets = a["pids"], a["labels"], a["datasets"]
        uniq = np.unique(pids)
        plab = np.array([labels[pids == p][0] for p in uniq])
        pds = np.array([datasets[pids == p][0] for p in uniq])
        print(f"\n### {name}")
        print(f"    {'score':<14}{'rule':<10}{'ACDC':>9}{'M&Ms':>9}")
        for stream in APPE + (["flow_SSIM"] if "flow_SSIM" in a else []):
            for rule in ("mean", "middle60"):
                sc, _ = patient_reduce(a, stream, rule)
                row = f"    {stream:<14}{rule:<10}"
                for ds in ("ACDC", "MM"):
                    m = pds == ds
                    y = (plab[m] != "NOR").astype(int)
                    row += f"{_fast_auc(y, sc[m]):>9.3f}" if len(np.unique(y)) > 1 else f"{'n/a':>9}"
                print(row)


if __name__ == "__main__":
    main()
