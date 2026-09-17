#!/usr/bin/env python
"""Original-repo (Nguyen ICCV2019) discrepancy functions on the baseline GAN motion stream.

Extends the flow-metric comparison of gan_flowmetric_eval.py (job 3740842) with the score
set the original repository itself supports — utils.calc_anomaly_score_one_frame /
calc_measures_single_item: per stream {PSNR_X, PSNR_inv, PSNR, SSIM, MSE, maxSE, stdSE,
channel-summed variants} on the flow (3-channel), angle (wrap-around angular difference)
and magnitude streams. Everything is recomputed OFFLINE from the flows saved by that job
(gan_flowmetric_out/flows.npz, float16, same checkpoint / same patients), so no TF and no
GPU are involved; pids/labels and the flow-SSIM reference come from scores.npz.

Rank-equivalences (measures collapsed, not lost): the repo's fixed-max PSNR is a monotone
transform of MSE -> identical AUC; MSE_1channel = 3 x MSE -> identical AUC. PSNR_X /
PSNR_inv use the per-sample max, so they are genuinely distinct. SSIM(flow) / SSIM(mag)
are rows 1-2 of tab:gan_flowmetric already and serve here as recomputation anchors, with
L1 and EPE as further anchors (hard gates vs summary.json, tolerance for float16 storage).

Scores are oriented higher = more anomalous (PSNR_X and SSIM negated / 1-SSIM).
Patient-mean aggregation, patient-level bootstrap CI and paired delta vs flow-SSIM mirror
gan_flowmetric_eval.py (functions copied — that module imports TF at module level).
Output: gan_flowmetric_out/nguyen_metrics.csv + printed table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent / "gan_flowmetric_out"
DATASETS = (("ACDC", "acdc"), ("MM", "mm"), ("MM_VAL", "mmval"))
REF = "flow_ssim"
N_BOOT = 2000
EPS = 1e-12

# anchors: summary.json AUCs of streams recomputable from the saved float16 flows
ANCHOR_STREAMS = ("flow_ssim_rec", "mag_ssim_rec", "flow_l1_rec", "flow_EPE_rec")
ANCHOR_OF = {"flow_ssim_rec": "flow_ssim", "mag_ssim_rec": "mag_ssim",
             "flow_l1_rec": "flow_l1", "flow_EPE_rec": "flow_EPE"}
ANCHOR_TOL = 0.012  # float16 storage of pred+gt shifts per-slice scores slightly

NEW_STREAMS = ("flow_MSE", "flow_PSNR_X", "flow_PSNR_inv", "flow_maxSE", "flow_stdSE",
               "flow_maxSE_1c", "flow_stdSE_1c",
               "ang_MSE", "ang_maxSE", "ang_stdSE", "ang_PSNR_X", "ang_SSIM",
               "mag_MSE", "mag_PSNR_X", "mag_maxSE", "mag_stdSE")


def _ssim_score(a, b, multichannel):
    dr = float(np.max([a, b]) - np.min([a, b])) or 1.0
    return 1.0 - ssim(a, b, data_range=dr, channel_axis=-1 if multichannel else None)


def measures_one(gt16, pr16) -> dict:
    gt, pr = gt16.astype(np.float64), pr16.astype(np.float64)
    out = {}
    # flow stream: squared error over all 3 channels [dx, dy, mag]
    sq = (gt - pr) ** 2
    mse = float(sq.mean())
    peak2 = float(np.max(pr)) ** 2 + EPS
    out["flow_MSE"] = mse
    out["flow_PSNR_X"] = -10.0 * np.log10(peak2 / (mse + EPS))
    out["flow_PSNR_inv"] = peak2 * mse
    out["flow_maxSE"] = float(sq.max())
    out["flow_stdSE"] = float(sq.std())
    sq1 = sq.sum(axis=-1)
    out["flow_maxSE_1c"] = float(sq1.max())
    out["flow_stdSE_1c"] = float(sq1.std())
    out["flow_ssim_rec"] = _ssim_score(gt, pr, True)
    out["flow_l1_rec"] = float(np.abs(gt - pr).mean())
    out["flow_EPE_rec"] = float(np.sqrt(((gt[..., :2] - pr[..., :2]) ** 2).sum(-1)).mean())
    # angle stream: the repo's wrap-around angular difference (cartToPolar range [0, 2pi))
    tt = np.mod(np.arctan2(gt[..., 1], gt[..., 0]), 2 * np.pi)
    tp = np.mod(np.arctan2(pr[..., 1], pr[..., 0]), 2 * np.pi)
    d = np.abs(tt - tp)
    da = np.minimum(d, 2 * np.pi - d) ** 2
    amse = float(da.mean())
    apeak2 = float(np.max(tp)) ** 2 + EPS
    out["ang_MSE"] = amse
    out["ang_maxSE"] = float(da.max())
    out["ang_stdSE"] = float(da.std())
    out["ang_PSNR_X"] = -10.0 * np.log10(apeak2 / (amse + EPS))
    out["ang_SSIM"] = _ssim_score(tt, tp, False)
    # magnitude stream: squared error on the last channel
    gm, pm = gt[..., -1], pr[..., -1]
    sqm = (gm - pm) ** 2
    mmse = float(sqm.mean())
    mpeak2 = float(np.max(pm)) ** 2 + EPS
    out["mag_MSE"] = mmse
    out["mag_PSNR_X"] = -10.0 * np.log10(mpeak2 / (mmse + EPS))
    out["mag_maxSE"] = float(sqm.max())
    out["mag_stdSE"] = float(sqm.std())
    out["mag_ssim_rec"] = _ssim_score(gm, pm, False)
    return out


# ── copied from gan_flowmetric_eval.py (module imports TF, so not importable) ──
def patient_mean(scores, pids):
    by = {}
    for s, p in zip(scores, pids):
        by.setdefault(p, []).append(s)
    return {p: float(np.mean(v)) for p, v in by.items()}


def patient_auc(score_by_pid, label_by_pid):
    pids = sorted(score_by_pid)
    y = [0 if label_by_pid[p] == "NOR" else 1 for p in pids]
    s = [score_by_pid[p] for p in pids]
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def bootstrap_stats(per_sample, pids, label_by_pid, streams, n_boot=N_BOOT, seed=0):
    ps = sorted(set(pids))
    y = np.array([0 if label_by_pid[p] == "NOR" else 1 for p in ps])
    mat = {k: np.array([patient_mean(per_sample[k], pids)[p] for p in ps]) for k in streams}
    rng = np.random.RandomState(seed)
    n = len(ps)
    aucs = {k: [] for k in streams}
    deltas = {k: [] for k in streams if k != REF}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        ref = roc_auc_score(yy, mat[REF][idx])
        aucs[REF].append(ref)
        for k in deltas:
            a = roc_auc_score(yy, mat[k][idx])
            aucs[k].append(a)
            deltas[k].append(a - ref)
    stats = {}
    for k in streams:
        a = np.array(aucs[k])
        stats[k] = {"ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
        if k != REF:
            d = np.array(deltas[k])
            stats[k]["delta"] = {"mean": float(d.mean()),
                                 "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return stats


def main() -> None:
    flows = np.load(OUT / "flows.npz")
    sc = np.load(OUT / "scores.npz", allow_pickle=True)
    summary = json.load(open(OUT / "summary.json"))

    rows = []
    for tag, pk in DATASETS:
        gt, pr = flows[f"gt__{tag}"], flows[f"pred__{tag}"]
        pids, labels = sc[f"{pk}_pids"].astype(str), sc[f"{pk}_labels"].astype(str)
        label_by_pid = {p: l for p, l in zip(pids, labels)}
        per = {k: np.zeros(len(gt)) for k in NEW_STREAMS + ANCHOR_STREAMS}
        for i in range(len(gt)):
            m = measures_one(gt[i], pr[i])
            for k in per:
                per[k][i] = m[k]
        per[REF] = sc[f"{tag}__{REF}"].astype(float)  # job's float32 originals

        # anchor gates: recomputed streams must reproduce the job's AUCs
        for k in ANCHOR_STREAMS:
            want = summary["auc"][tag][ANCHOR_OF[k]]
            got = patient_auc(patient_mean(per[k], pids), label_by_pid)
            print(f"[anchor] {tag:6s} {ANCHOR_OF[k]:9s} recomputed {got:.4f} vs job {want:.4f} "
                  f"(delta {got - want:+.4f})")
            assert abs(got - want) <= ANCHOR_TOL, f"anchor failed: {tag}/{k}"

        streams = (REF,) + NEW_STREAMS
        boots = bootstrap_stats(per, pids, label_by_pid, streams) if tag != "MM_VAL" else None
        for k in NEW_STREAMS:
            row = dict(dataset=tag, stream=k,
                       auc=patient_auc(patient_mean(per[k], pids), label_by_pid))
            if boots is not None:
                row["ci_lo"], row["ci_hi"] = boots[k]["ci"]
                row["delta_vs_flow_ssim"] = boots[k]["delta"]["mean"]
                row["delta_lo"], row["delta_hi"] = boots[k]["delta"]["ci"]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "nguyen_metrics.csv", index=False)
    print(f"\n[done] -> {OUT / 'nguyen_metrics.csv'}\n")
    wide = df.pivot(index="stream", columns="dataset", values="auc").loc[list(NEW_STREAMS)]
    sig = df[df.dataset.isin(["ACDC", "MM"])].set_index(["stream", "dataset"])
    marks = {}
    for k in NEW_STREAMS:
        a = sig.loc[(k, "ACDC")]
        m = sig.loc[(k, "MM")]
        marks[k] = ("dagger" if a.delta_lo > 0 or a.delta_hi < 0 else "") + \
                   ("*worseMM" if m.delta_hi < 0 else ("*betterMM" if m.delta_lo > 0 else ""))
    wide["signif_vs_flow_ssim"] = [marks[k] for k in wide.index]
    print(wide.round(4).to_string())


if __name__ == "__main__":
    main()
