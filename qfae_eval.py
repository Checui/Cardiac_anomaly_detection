"""qfae_eval.py — anomaly scoring for the cardiac QFAE (Phase 3).

Scores validation stacks (ACDC test + M&Ms val) by the CineMA perceptual reconstruction
error and reports the same AUCs as the frozen probe, reusing derisk_cinema's
aggregate_stacks_to_patient + one_vs_nor_aucs. Writes qfae_out/qfae_results.json and
prints a comparison against the frozen probe (derisk_out) if present.
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim

from cinema import CineMA
import cinema_faithful as cf
from qfae_cinema import QFormerAE
from qfae_perceptual import make_scorer
from derisk_cinema import aggregate_stacks_to_patient, one_vs_nor_aucs
import qfae_masking as qm


def flow_scores(gt_flow, pred_flow, vox_mask=None, per_slice=False):
    """(3,192,192,16) GT vs predicted flow -> per-stack (flow_SSIM, mag_SSIM, flow_L1).

    Per-slice 1-SSIM over [dx,dy,mag] + over the magnitude channel, averaged over slices —
    mirrors utils.compute_flow_ssim_scores (the GAN's strongest stream; reimplemented here to
    avoid importing the TF-dependent utils.py). Higher = more anomalous.

    vox_mask: optional (192,192,16) bool, True on voxels hidden from the encoder. When given,
    every score is restricted to those voxels, so it measures how well the model *inpainted*
    motion it could not see (the MME objective) rather than copied visible motion through.
    SSIM is a windowed statistic and is not defined on a scattered mask, so we take skimage's
    full per-pixel SSIM map and average it over the masked voxels — the standard way to
    localise SSIM. Slices with no masked voxel are skipped.
    """
    fs, ms, l1 = [], [], []
    for z in range(gt_flow.shape[-1]):
        g = np.transpose(gt_flow[:, :, :, z], (1, 2, 0))       # (192,192,3)
        p = np.transpose(pred_flow[:, :, :, z], (1, 2, 0))
        mz = vox_mask[:, :, z] if vox_mask is not None else None
        if mz is not None and not mz.any():
            continue
        dr = float(np.max([g, p]) - np.min([g, p])) or 1.0
        gm, pm = g[..., -1], p[..., -1]
        drm = float(np.max([gm, pm]) - np.min([gm, pm])) or 1.0
        if mz is None:
            fs.append(1.0 - ssim(g, p, data_range=dr, channel_axis=-1))
            ms.append(1.0 - ssim(gm, pm, data_range=drm))
            l1.append(float(np.mean(np.abs(g - p))))
        else:
            _, s_full = ssim(g, p, data_range=dr, channel_axis=-1, full=True)
            fs.append(1.0 - float(s_full.mean(axis=-1)[mz].mean()))
            _, s_mag = ssim(gm, pm, data_range=drm, full=True)
            ms.append(1.0 - float(s_mag[mz].mean()))
            l1.append(float(np.abs(g - p).mean(axis=-1)[mz].mean()))
    if per_slice:                                               # per-z vectors for the reduction sweep
        # unmasked: one value per depth-z (length = gt_flow.shape[-1], incl. padding). mean == the scalar.
        return np.asarray(fs, np.float32), np.asarray(ms, np.float32), np.asarray(l1, np.float32)
    if not fs:                                                  # no masked voxel anywhere
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(fs)), float(np.mean(ms)), float(np.mean(l1))


def report_stream(name, scores, pids, labels, datasets):
    """Stack/patient/per-dataset AUCs for one anomaly stream; prints patient-Mean, returns dict."""
    stack = one_vs_nor_aucs(scores, labels)
    pat, plabels = aggregate_stacks_to_patient(scores, pids, labels)
    patient = {agg: one_vs_nor_aucs(pat[agg], plabels) for agg in ("Mean", "Max")}
    per_ds = {ds: one_vs_nor_aucs(scores[datasets == ds], labels[datasets == ds])
              for ds in sorted(set(datasets))}
    pm = patient["Mean"]
    print(f"[{name:11s}] PAT-Mean overall={pm.get('overall', float('nan')):.4f}   "
          + "   ".join(f"{ds}={per_ds[ds]['overall']:.3f}" for ds in ("ACDC", "MM")
                       if ds in per_ds and "overall" in per_ds[ds]))
    return {"stack": stack, "patient": patient, "per_dataset": per_ds}


def eval_mask(args, x, gt_flow_t, device, generator):
    """Seeded encoder mask for one eval batch, or None when masking is off."""
    if args.mask == "none":
        return None
    n_patches = qm.n_patches_of()
    if args.mask == "random":
        return qm.random_patch_mask(x.shape[0], n_patches, args.mask_ratio, device,
                                    generator=generator)
    return qm.motion_guided_patch_mask(gt_flow_t, args.mask_ratio, alpha=args.mask_alpha,
                                       temperature=args.mask_temperature, generator=generator)


def load_model(cinema, path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = QFormerAE(cinema, **ckpt["arch"]).to(device)
    res = model.load_state_dict(ckpt["model"], strict=False)
    unexpected = getattr(res, "unexpected_keys", [])
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:3]}"
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_val_dir", default="../Dataset_1/Validation")
    ap.add_argument("--mm_csv", default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--val_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--model_path", default="./qfae_out/qfae.pt")
    ap.add_argument("--out_dir", default="./qfae_out")
    ap.add_argument("--perceptual_layers", type=int, nargs="+", default=[5, 8, 11])
    ap.add_argument("--scorer", default="cinema",
                    help="MUST match training: 'cinema' (default) or any timm model name applied "
                         "per depth-slice. Only affects the appearance stream — the flow streams "
                         "compare predicted vs GT flow directly and never touch the scorer.")
    ap.add_argument("--top_frac", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--flow", action="store_true",
                    help="Dual-stream model: also score the optical-flow head (Flow-SSIM / "
                         "mag-SSIM / flow-L1) and a combined appearance+motion score.")
    ap.add_argument("--single_pass", action="store_true",
                    help="ED-only single pass (MUST match training): 1 stack/patient.")
    ap.add_argument("--flow_backend", choices=["farneback", "registration"], default="farneback",
                    help="GT-flow source — MUST match training (the motion score compares the "
                         "predicted flow against this GT).")
    ap.add_argument("--reg_repo", default="../biomechanics-cardiac-motion-hpc")
    # masked scoring (qfae_masking.py). Default none -> the unmasked baseline eval, unchanged.
    ap.add_argument("--mask", choices=["none", "random", "motion"], default="none",
                    help="Mask patches at eval and score the inpainted motion. Should match "
                         "how the model was trained.")
    ap.add_argument("--mask_ratio", type=float, default=0.5)
    ap.add_argument("--mask_alpha", type=float, default=1.0)
    ap.add_argument("--mask_temperature", type=float, default=1.0)
    ap.add_argument("--mask_repeats", type=int, default=4,
                    help="Monte-Carlo mask draws per stack, averaged. A single random draw "
                         "makes the score stochastic and the AUC partly noise.")
    ap.add_argument("--mask_seed", type=int, default=1234,
                    help="Fixes the eval masks so the reported AUC is reproducible.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    if args.mask == "motion" and not args.flow:
        ap.error("--mask motion needs --flow: the mask is sampled from the GT flow magnitude.")
    if args.mask != "none" and args.mask_repeats < 1:
        ap.error("--mask_repeats must be >= 1")
    # ratio 0 masks nothing, so the masked-region scores have no voxels to average and come
    # back NaN — which would poison roc_auc_score rather than fail. Use --mask none instead.
    if args.mask != "none" and not 0.0 < args.mask_ratio < 1.0:
        ap.error(f"--mask_ratio must be in (0, 1) when masking, got {args.mask_ratio}")

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    if args.flow:
        cf.set_flow_backend(args.flow_backend, reg_repo=args.reg_repo)
    cf.set_single_pass(args.single_pass)
    print("=== Building validation records ===")
    val = cf.build_val_records_flow(args) if args.flow else cf.build_val_records(args)
    print(f"[qfae] val label counts: {dict(Counter(r['label'] for r in val))}")

    print("=== Loading CineMA + trained QFAE ===")
    cinema = CineMA.from_pretrained()
    model = load_model(cinema, args.model_path, device)
    perceptual, scorer_desc = make_scorer(args.scorer, "3d", cinema=cinema,
                                          layers=tuple(args.perceptual_layers), device=device)
    print(f"[qfae] encoder = CineMA (3-D depth-16) | perceptual scorer = {scorer_desc}")

    reps = args.mask_repeats if args.mask != "none" else 1
    mask_note = ("" if args.mask == "none" else
                 f" (mask={args.mask} ratio={args.mask_ratio} x{reps} draws, seed={args.mask_seed})")
    print(f"=== Scoring val stacks ==={mask_note}")
    # Seeded so the reported AUC is reproducible rather than a function of mask luck.
    gen = torch.Generator(device=device)
    gen.manual_seed(args.mask_seed)
    appe, fssim, mssim, fl1 = [], [], [], []
    appe_sl, fs_sl, ms_sl, l1_sl, nreal = [], [], [], [], []   # per-slice error maps (mask=="none" only)
    n = len(val)
    for start in range(0, n, args.batch_size):
        chunk = val[start:start + args.batch_size]
        x = torch.from_numpy(np.stack([cf.stack_to_tensor(r["stack"]) for r in chunk]))
        x = x.to(device=device, dtype=dtype)
        # real (non-padded) depth per stack — used by the per-slice 2-D scorers, ignored by CineMA's
        valid_t = torch.tensor([int(r["stack"].shape[2]) for r in chunk], device=device)
        gt_flow_t = None
        if args.mask == "motion":
            gt_flow_t = torch.from_numpy(np.stack([r["gt_flow"] for r in chunk])).to(
                device=device, dtype=torch.float32)
        use_amp = device.type == "cuda" and dtype != torch.float32

        r_appe, r_fs, r_ms, r_l1 = [], [], [], []
        for _ in range(reps):
            mask = eval_mask(args, x, gt_flow_t, device, gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                out = model(x, mask=mask)
            recon, pred_flow = out if args.flow else (out, None)
            r_appe.append(perceptual.score(x.float(), recon.float(), top_frac=args.top_frac,
                                           valid=valid_t).float().cpu().numpy())
            if args.flow:
                pf = pred_flow.float().cpu().numpy()                   # (b,3,192,192,16)
                vox = None
                if mask is not None:
                    vox = qm.upsample_mask_to_voxels(mask)[:, 0].bool().cpu().numpy()
                fs_b, ms_b, l1_b = [], [], []
                for i, r in enumerate(chunk):
                    fs, ms, l1 = flow_scores(r["gt_flow"], pf[i], vox[i] if vox is not None else None)
                    fs_b.append(fs); ms_b.append(ms); l1_b.append(l1)
                r_fs.append(fs_b); r_ms.append(ms_b); r_l1.append(l1_b)
        # Average over mask draws (a single draw would leave the score stochastic).
        appe.append(np.mean(r_appe, axis=0))
        if args.flow:
            fssim.extend(np.mean(r_fs, axis=0)); mssim.extend(np.mean(r_ms, axis=0))
            fl1.extend(np.mean(r_l1, axis=0))
        # Per-slice decomposition for the OFFLINE reduction sweep (unmasked runs only; reps==1).
        # Appearance per depth-slice = mean over the 144 (h,w) tokens of each z (token order h,w,d;
        # d is the last grid axis). Flow per-slice comes straight from flow_scores(per_slice=True).
        if args.mask == "none":
            # (B,16) appearance error by depth-slice — the reshape-and-average that used to live
            # here now sits in the scorer, so a per-slice 2-D scorer can supply it natively.
            amap = perceptual.score_depth(x.float(), recon.float()).float().cpu().numpy()
            for i, r in enumerate(chunk):
                appe_sl.append(amap[i].astype(np.float32))
                nreal.append(int(r["stack"].shape[2]))
                if args.flow:
                    fsv, msv, l1v = flow_scores(r["gt_flow"], pred_flow[i].float().cpu().numpy(),
                                                per_slice=True)
                    fs_sl.append(fsv); ms_sl.append(msv); l1_sl.append(l1v)
        print(f"\r[qfae] scored {min(start + args.batch_size, n)}/{n}", end="", flush=True)
    print()
    pids = np.array([r["pid"] for r in val])
    labels = np.array([r["label"] for r in val])
    datasets = np.array([r["dataset"] for r in val])
    appe = np.concatenate(appe)

    print("\n" + "=" * 62)
    print("QFAE RESULTS (AUC: NOR vs disease; patient-Mean per stream)")
    print("=" * 62)
    streams = {"appearance": report_stream("appearance", appe, pids, labels, datasets)}
    save = {"appearance": appe}
    if args.flow:
        fssim, mssim, fl1 = np.array(fssim), np.array(mssim), np.array(fl1)
        streams["flow_SSIM"] = report_stream("flow_SSIM", fssim, pids, labels, datasets)
        streams["mag_SSIM"] = report_stream("mag_SSIM", mssim, pids, labels, datasets)
        streams["flow_L1"] = report_stream("flow_L1", fl1, pids, labels, datasets)
        def _z(a):
            return (a - a.mean()) / (a.std() + 1e-8)
        combined = _z(appe) + _z(fssim)                               # appearance + motion
        streams["combined"] = report_stream("combined", combined, pids, labels, datasets)
        save.update(flow_SSIM=fssim, mag_SSIM=mssim, flow_L1=fl1, combined=combined)

    # Per-slice error maps + real-slice count → offline reduction sweep (qfae_report.py --reduction_sweep).
    if args.mask == "none" and appe_sl:
        save["appe_slices"] = np.stack(appe_sl)                       # (n_stacks, 16)
        save["n_real_slices"] = np.array(nreal, np.int64)
        if args.flow:
            save["flow_SSIM_slices"] = np.stack(fs_sl)
            save["mag_SSIM_slices"] = np.stack(ms_sl)
            save["flow_L1_slices"] = np.stack(l1_sl)

    results = {"method": "qfae_flow" if args.flow else "qfae", "n_val": int(n),
               "encoder": "cinema", "scorer": args.scorer,
               "perceptual_layers": args.perceptual_layers, "top_frac": args.top_frac,
               "lambda_note": "combined = zscore(appearance)+zscore(flow_SSIM)",
               "masking": {"mask": args.mask, "mask_ratio": args.mask_ratio,
                           "mask_alpha": args.mask_alpha, "mask_temperature": args.mask_temperature,
                           "mask_repeats": reps, "mask_seed": args.mask_seed,
                           "note": ("scores restricted to masked voxels via the full SSIM map"
                                    if args.mask != "none" else "unmasked baseline")},
               "streams": streams}
    with open(os.path.join(args.out_dir, "qfae_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, "qfae_arrays.npz"),
                        pids=pids, labels=labels, datasets=datasets, **save)

    # comparison vs the frozen probe if available
    frozen_p = "derisk_out/derisk_results.json"
    if os.path.exists(frozen_p):
        fr = json.load(open(frozen_p))["scorers"]["mahalanobis"]
        fpd = {k: v.get("overall") for k, v in fr.get("per_dataset", {}).items()}
        print("\n--- vs FROZEN probe (patient-Mean overall / stack per-dataset) ---")
        print(f"frozen : overall={fr['patient']['Mean']['overall']:.3f}  "
              f"ACDC={fpd.get('ACDC', float('nan')):.3f}  MM={fpd.get('MM', float('nan')):.3f}")
        print("(baselines: frozen ACDC 0.76 / M&Ms 0.59; QFAE-appearance 0.845/0.55; "
              "Flow-SSIM GAN 0.73-0.77 — watch flow_SSIM M&Ms)")

    print(f"\n[qfae] wrote {args.out_dir}/qfae_results.json + qfae_arrays.npz")


if __name__ == "__main__":
    main()
