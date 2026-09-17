"""Score every detector family on BOTH test cohorts: full vs CineMA's in-class protocol.

CineMA's binary-detection protocol excludes test patients whose disease labels are absent
from the corresponding training split (M&Ms: Other/IHD/LVNC, 30 of 136 patients; ACDC: no
exclusions, the two cohorts coincide). The thesis quotes in-class AUCs (released 0.822,
binary FT 0.778, unsupervised 0.72-0.73) that were never persisted; the only saved in-class
file (compare_vs_cinema.json) posterior-ensembles the seeds, which is the wrong convention.
This script recomputes both cohorts through one code path — AUC, Se@90%Sp, native-threshold
sens/spec, and the unsupervised detectors additionally read at a specificity matched to the
binary FT's native P(healthy)<0.5 operating point — under the per-seed-metrics-then-average
convention used everywhere else, and persists supervised_ft_out/two_protocol_metrics.csv.
(The frozen MM-Validation Youden readings live in operating_points.csv from the earlier
analysis; they are deliberately not part of this table.)

Pure re-scoring of saved per-patient arrays (numpy/sklearn, CPU, seconds); reuses the
loaders and metric helpers of supervised_ft_analysis.py. The in-class mask comes from
cinema_ft_probs_{ACDC,MM}.npz (written by cinema_finetuned_eval.py from CineMA's own class
spaces). Sanity gates hard-fail unless the full-cohort AUCs reproduce the published anchors
(GAN 0.8125/0.7310, QFAE 0.8600/0.7428, binary FT 0.7567/0.7404, released 0.9792/0.7960).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import supervised_ft_analysis as A

# full-cohort anchors (label_efficiency_*_baselines.csv / label_efficiency_*.csv)
ANCHORS = {
    ("acdc", "CineMA released FT"): 0.97917,
    ("mnms", "CineMA released FT"): 0.79597,
    ("acdc", "CineMA binary FT (ours, 3 seeds)"): 0.75667,
    ("mnms", "CineMA binary FT (ours, 3 seeds)"): 0.74043,
    ("acdc", "GAN flow-SSIM"): 0.8125,
    ("mnms", "GAN flow-SSIM"): 0.73097,
    ("acdc", "QFAE mag-SSIM mid60"): 0.875,   # qfae_mx_mae224_mae run (tab:qfae_maejudge)
    ("mnms", "QFAE mag-SSIM mid60"): 0.74038,  # qfae_mx_mae224_mae run (tab:qfae_maejudge)
}


def in_class_cohorts() -> dict[str, tuple[set, Counter]]:
    """{ds: (kept pids, excluded-class counts)} from CineMA's own class-space mask."""
    out = {}
    for ds, tag in (("acdc", "ACDC"), ("mnms", "MM")):
        z = np.load(A.REPO / f"cinema_ft_probs_{tag}.npz", allow_pickle=True)
        pids = A.strip_prefix(z["pids"])
        keep = z["in_class"].astype(bool)
        out[ds] = (set(pids[keep]), Counter(z["labels"].astype(str)[~keep]))
    assert not out["acdc"][1], f"ACDC should have no exclusions, got {out['acdc'][1]}"
    return out


def main() -> None:
    cohorts = in_class_cohorts()
    print("[in-class] excluded M&Ms patients by class:", dict(cohorts["mnms"][1]),
          f"(total {sum(cohorts['mnms'][1].values())})")

    gan, qfae, released = A.gan_scores(), A.qfae_scores(), A.released_ft_scores()

    rows = []
    for ds in ("acdc", "mnms"):
        sup = A.sup_seeds(ds, "frac100")
        sup = (A.strip_prefix(sup[0]), sup[1], sup[2])
        models = {"CineMA released FT": released[ds],
                  "CineMA binary FT (ours, 3 seeds)": sup,
                  "GAN flow-SSIM": gan[ds],
                  "QFAE mag-SSIM mid60": qfae[ds]}
        for cohort in ("full", "in-class"):
            ft_native_spec = None  # set by the binary FT before GAN/QFAE iterate
            for name, model in models.items():
                pids, path, S = A.restrict(model, cohorts[ds][0]) if cohort == "in-class" else model
                y = (path != "NOR").astype(int)
                S2 = np.atleast_2d(np.asarray(S, float))
                per_auc = [float(A._fast_auc(y, s)) for s in S2]
                lo, hi = A.boot_ci(y, S)
                row = dict(dataset=A.DS_NAME[ds], cohort=cohort, model=name,
                           n=len(y), n_nor=int((y == 0).sum()),
                           auc=float(np.mean(per_auc)),
                           auc_sd=float(np.std(per_auc, ddof=1)) if len(per_auc) > 1 else np.nan,
                           seed_aucs=";".join(f"{a:.4f}" for a in per_auc) if len(per_auc) > 1 else "",
                           ci_lo=lo, ci_hi=hi,
                           se_at_90sp=float(np.mean([A._sens_at_spec(y, s, 0.90) for s in S2])))
                if name.startswith("CineMA"):  # probabilistic head: native 0.5 threshold
                    per = [A._point_metrics(y, (s > 0.5).astype(int)) for s in S2]
                    row["native_sens"] = float(np.mean([p["sens"] for p in per]))
                    row["native_spec"] = float(np.mean([p["spec"] for p in per]))
                    if name.startswith("CineMA binary"):
                        ft_native_spec = row["native_spec"]
                else:  # anomaly score, read at the binary FT's native operating specificity
                    assert ft_native_spec is not None
                    row["matched_spec"] = ft_native_spec
                    row["matched_sens"] = float(A._sens_at_spec(y, np.asarray(S, float), ft_native_spec))
                rows.append(row)

    df = pd.DataFrame(rows)

    # sanity gates: full cohort must reproduce the published anchors
    for (ds, name), want in ANCHORS.items():
        got = df[(df.dataset == A.DS_NAME[ds]) & (df.cohort == "full") & (df.model == name)].auc.iloc[0]
        assert abs(got - want) < 2e-3, f"anchor mismatch {ds}/{name}: got {got:.5f}, want {want}"
    for name in df.model.unique():  # ACDC: the two protocols must coincide
        a = df[(df.dataset == "ACDC") & (df.model == name)]
        assert abs(a[a.cohort == "full"].auc.iloc[0] - a[a.cohort == "in-class"].auc.iloc[0]) < 1e-12

    out = A.OUT / "two_protocol_metrics.csv"
    df.to_csv(out, index=False)
    print(f"[done] anchors reproduced, ACDC cohorts coincide -> {out}\n")
    print("== AUC ==")
    print(df[["dataset", "cohort", "model", "n", "n_nor", "auc", "auc_sd",
              "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    op = df[df.model != "CineMA released FT"]
    print("\n== operating points (released FT excluded; unsupervised matched to the "
          "binary FT's native-threshold specificity) ==")
    print(op[["dataset", "cohort", "model", "native_sens", "native_spec",
              "se_at_90sp", "matched_sens", "matched_spec"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
