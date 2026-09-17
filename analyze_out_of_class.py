"""Out-of-class AUC on M&Ms-Test: the thirty patients whose disease labels are
absent from the training split (Other 24, IHD 4, LVNC 2) versus the healthy
patients, for every model family in tab:supervised. Complements
two_protocol_metrics.csv: that file gives the full and in-class cohorts; this
gives the third reading, the patients the in-class protocol removes.

Same conventions as supervised_ft_analysis.py (seed-mean AUC, patient bootstrap
averaging per-seed AUCs within each resample). ACDC has no out-of-class cohort
(every test pathology appears in training). GAN rows use its loader's 134-patient
pool (one IHD and one NOR case dropped).

Output: supervised_ft_out/out_of_class_auc.csv
"""

import numpy as np
import pandas as pd

from supervised_ft_analysis import (OUT, auc_of, gan_scores, qfae_scores,
                                    released_ft_scores, sup_seeds)

OOC = {"Other", "IHD", "LVNC"}
IN_CLASS = {"DCM", "HCM", "HHD", "ARV"}

MODELS = {
    "CineMA binary FT (ours, 3 seeds)": sup_seeds("mnms", "frac100"),
    "CineMA released FT (3 seeds)": released_ft_scores()["mnms"],
    "GAN flow-SSIM": gan_scores()["mnms"],
    "QFAE mag-SSIM mid60": qfae_scores()["mnms"],
}

rows = []
for name, model in MODELS.items():
    for cohort, subset in (("full", None),
                           ("in-class", IN_CLASS | {"NOR"}),
                           ("out-of-class", OOC | {"NOR"})):
        auc, (lo, hi), n, per_seed = auc_of(model, subset)
        rows.append(dict(dataset="MM", cohort=cohort, model=name, n=n, auc=auc,
                         ci_lo=lo, ci_hi=hi,
                         seed_aucs=";".join(f"{a:.4f}" for a in per_seed)
                         if len(per_seed) > 1 else ""))

df = pd.DataFrame(rows)
df.to_csv(OUT / "out_of_class_auc.csv", index=False)
print(df.round(4).to_string(index=False))
