"""qfae_dino_eval.py — anomaly scoring for the 2-D DINOv2 QFAE.

Scores validation frames (ACDC test + M&Ms val, NOR + disease) by the multi-layer DINOv2
perceptual reconstruction error, aggregates to patient level and reports the SAME patient-Mean
per-dataset AUCs used everywhere else (reuses derisk_cinema.aggregate_to_patient +
one_vs_nor_aucs). Compare against CineMA-faithful 0.76/0.59, QFAE-CineMA appearance 0.845/0.55,
and the frozen DINOv2 probe peak (ACDC 0.735 / M&Ms 0.704).
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import torch

import derisk_cinema as D
import cinema_faithful as cf
from derisk_cinema import aggregate_to_patient, one_vs_nor_aucs, _AGGS
from skimage.metrics import structural_similarity as ssim
from qfae_eval import flow_scores
from qfae_dino import QFormerAE2D, load_dinov2
from qfae_perceptual import make_scorer
from qfae_dino_train import frames_to_tensor, pixel_appe_score, pixel_appe_scores_multi, resolve_scorer_spec
from qfae_flowmetrics import (ResidualMahalanobis, _EXTRA_FLOW_STREAMS, _GT_ONLY_STREAMS,
                              build_roi, extra_flow_scores, gt_flow_stats, motion_mask,
                              roi_to_token_mask)


def scorer_n_tokens(perceptual):
    """Token count of the perceptual scorer's own grid, or None if it can't be derived.

    Needed to map a pixel-space ROI onto the scorer's tokens: the scorer may be a different
    model from the encoder (that is the whole point of the encoder x scorer matrix), so its
    grid is not the encoder's.
    """
    pe = getattr(getattr(perceptual, "dino", None), "patch_embed", None)
    n = getattr(pe, "num_patches", None)
    return int(n) if n else None


def fit_residual_mahalanobis(model, args, device, amp_dtype, grid):
    """Fit the per-position residual Gaussian on the NOR TRAINING slices.

    Must be the training set: the statistic being learned is "what does this model's flow
    residual normally look like", and fitting it on val would leak the very distinction the
    AUC measures. Costs one extra data build (Farneback over the NOR train set) + one forward
    pass, which is why it sits behind --maha.
    """
    print("=== [maha] building NOR training slices for the residual fit ===")
    fit_frames, fit_flows = cf.build_fit_slices_2d(args, args.img_size)
    print(f"[maha] {len(fit_frames)} NOR slices; fitting {grid}x{grid} per-position Gaussians")
    mh = ResidualMahalanobis(grid=grid)
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    for start in range(0, len(fit_frames), args.batch_size):
        xb = torch.from_numpy(fit_frames[start:start + args.batch_size]).to(device)
        gb = fit_flows[start:start + args.batch_size]
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            _, pred = model(xb)
        mh.partial_fit(pred.float().cpu().numpy() - gb)
    mh.finalize()
    print(f"[maha] fitted on {mh.state()['n_fit']} slices")
    return mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_val_dir", default="../Dataset_1/Validation")
    ap.add_argument("--mm_csv", default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--val_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--model_path", default="./qfae_dino_out/qfae.pt")
    ap.add_argument("--out_dir", default="./qfae_dino_out")
    ap.add_argument("--dino_model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--flow", action="store_true",
                    help="2-D motion rebuild (MUST match training): CineMA-faithful per-slice, also score "
                         "the predicted 2-D ED->ES flow (flow_SSIM/mag_SSIM/flow_L1).")
    ap.add_argument("--flow_backend", choices=["farneback", "registration"], default="farneback",
                    help="GT source for the 2-D flow (MUST match training, else AUC is measured against "
                         "the wrong field). 'registration' uses the reg-net teacher via --reg_repo.")
    ap.add_argument("--reg_repo", default="../biomechanics-cardiac-motion-hpc",
                    help="Registration teacher repo (only used when --flow_backend registration).")
    ap.add_argument("--recover_edes", action="store_true",
                    help="Accepted for symmetry with training and recorded in the results JSON. "
                         "Provably a NO-OP here: M&Ms Testing/Validation contain no ED==ES rows, "
                         "so the held-out universe is identical either way.")
    ap.add_argument("--perceptual_layers", type=int, nargs="+", default=[5, 8, 11])
    ap.add_argument("--scorer", default=None,
                    help="MUST match training. 'coupled' | 'cinema' | any timm model name. "
                         "Checked against the checkpoint's recorded scorer_name when present.")
    ap.add_argument("--perceptual_backbone", choices=["dino", "mae"], default="mae",
                    help="DEPRECATED, use --scorer. 'dino' == coupled, 'mae' == separate --mae_model.")
    ap.add_argument("--appe_space", choices=["embedding", "pixel"], default="embedding",
                    help="MUST match training. 'pixel' scores appearance as GAN-style MSE + gradient "
                         "error in pixel space (no perceptual scorer); 'embedding' uses the encoder.")
    ap.add_argument("--mae_model", default="vit_base_patch16_224.mae")
    ap.add_argument("--top_frac", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # ── (D) ROI pooling — restrict every score to the heart instead of the whole 192 mm box ──
    ap.add_argument("--roi", choices=["none", "center", "motion", "pred_motion", "lv"],
                    default="none",
                    help="'none' (default) pools over the whole frame. 'center' = fixed central "
                         "disc. 'motion' = pixels with the largest GT displacement. "
                         "'pred_motion' = the same rule on the PREDICTED field (the circularity "
                         "control: the mask no longer conditions on the GT that appears in the "
                         "error term, and needs no second cardiac phase). 'lv' = the "
                         "dataset heart segmentation carried through the same crop as the image "
                         "(no new test-time dependency — the crop already uses it).")
    ap.add_argument("--roi_frac", type=float, default=0.5,
                    help="Radius of the 'center' ROI as a fraction of the half-frame.")
    ap.add_argument("--motion_frac", type=float, default=0.3,
                    help="Fraction of pixels kept by the 'motion' ROI (top |GT flow|).")
    # ── (C) residual Mahalanobis — needs a fit pass over the NOR training set ──
    ap.add_argument("--maha", action="store_true",
                    help="Also score the flow residual with a PaDiM-style per-position Gaussian "
                         "fitted on NOR training residuals (stream 'flow_maha'). Adds one train "
                         "data build + forward pass to the eval.")
    ap.add_argument("--maha_grid", type=int, default=16, help="Spatial grid for --maha.")
    ap.add_argument("--fit_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"],
                    help="Training sets for --maha's residual fit; MUST match the training run.")
    args = ap.parse_args()
    for k, v in dict(orient_normalize=False, spacing_normalize=False, n4_bias_correct=False,
                     target_spacing=1.5, target_size=128, recon_spacing=2.0, n4_shrink=4,
                     n4_iterations=50, n4_levels=4, orient_params=None).items():
        setattr(args, k, v)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    if args.roi != "none" and not args.flow:
        raise SystemExit("--roi is implemented for the 2-D flow path (--flow); the legacy "
                         "appearance-only loader has no per-slice geometry to build an ROI from.")
    if args.maha and not args.flow:
        raise SystemExit("--maha scores the flow residual and needs --flow.")

    print("=== Loading validation data ===")
    lv_rois = None
    if args.flow:                                            # CineMA-faithful 2-D per-slice + 2-D flow
        cf.set_flow_backend(args.flow_backend, reg_repo=args.reg_repo)   # MUST match training's backend
        cf.set_edes_recovery(args.recover_edes)   # no-op on Testing/Validation (no ED==ES rows)
        if args.roi == "lv":
            Xf, Flowf, lv_rois, pids, slcs, labels, datasets = cf.build_val_slices_2d(
                args, args.img_size, want_roi=True)
            have = float(np.mean([r.any() for r in lv_rois]))
            print(f"[qfae-dino] ROI=lv: {have:.1%} of slices have a heart mask "
                  f"(the rest fall back to the central disc)")
        else:
            Xf, Flowf, pids, slcs, labels, datasets = cf.build_val_slices_2d(args, args.img_size)
        X = torch.from_numpy(Xf)                             # (N,3,S,S) already ViT-sized
        gt_flows = Flowf                                     # (N,3,S,S) [dx,dy,mag]
        frames = Xf
    else:
        D.configure_loaders(args)
        frames, labels, pids, slcs, datasets = D.load_val(args)
        X = frames_to_tensor(frames, args.img_size)
        gt_flows = None
    print(f"[qfae-dino] val slices={len(frames)} patients={len(np.unique(pids))} "
          f"labels={dict(Counter(labels))}")

    print(f"=== Loading frozen {args.dino_model} + trained 2-D QFAE ===")
    dino, mean, std = load_dinov2(args.dino_model, args.img_size, device)
    ckpt = torch.load(args.model_path, map_location="cpu")
    scorer_spec = resolve_scorer_spec(args)
    # Guard the matrix: load_state_dict(strict=False) asserts only on UNEXPECTED keys, so a wrong
    # --dino_model would silently pair the trained decoder with a different frozen encoder (and its
    # normalisation) and still "work". Checkpoints written before this guard have no recorded names.
    arch = dict(ckpt["arch"])
    for flag, got, key in (("--dino_model", args.dino_model, "encoder_name"),
                           ("--scorer", "pixel" if args.appe_space == "pixel" else scorer_spec,
                            "scorer_name")):
        want = arch.get(key)
        if want is None:
            print(f"[qfae-dino] NOTE: checkpoint predates the {key} guard — {flag} unverified")
        elif want != got:
            raise SystemExit(f"{flag}={got!r} does not match the checkpoint's {key}={want!r}. "
                             f"Re-run with the config this model was trained under.")
    model = QFormerAE2D(dino, mean, std, **arch).to(device)
    res = model.load_state_dict(ckpt["model"], strict=False)
    assert not getattr(res, "unexpected_keys", []), f"unexpected keys: {res.unexpected_keys[:3]}"
    model.eval()
    if args.appe_space == "pixel":
        perceptual = None
        print("[qfae-dino] appearance scored in PIXEL space (GAN-style MSE + gradient); no perceptual scorer")
    else:
        perceptual, scorer_desc = make_scorer(
            scorer_spec, "2d", coupled=(dino, mean, std), img_size=args.img_size,
            layers=tuple(args.perceptual_layers), device=device)
        print(f"[qfae-dino] encoder = {args.dino_model} @{args.img_size} | "
              f"perceptual scorer = {scorer_desc}")

    maha = (fit_residual_mahalanobis(model, args, device, amp_dtype, args.maha_grid)
            if args.maha else None)

    print("=== Scoring val slices ===")
    scores, fssim, mssim, fl1 = [], [], [], []
    extra = {k: [] for k in _EXTRA_FLOW_STREAMS}
    gt_only = {k: [] for k in _GT_ONLY_STREAMS}
    maha_scores = []
    # pixel score-way sweep (mirrors the GAN notebook): re-measure the SAME reconstruction with MSE,
    # MAE(L1), 1-SSIM and MSE+grad. Scoring-only — no retraining; the decoder output is fixed.
    appe_sweep = {"mse": [], "mae": [], "ssim": [], "msegrad": []}
    n = len(X)
    size = X.shape[-1]
    # ROI (D). 'none'/'center' are the same mask for every slice, so build them once; 'motion'
    # and 'lv' are per-slice. n_tok maps the pixel ROI onto the *scorer's* token grid.
    static_roi = (build_roi(args.roi, size=size, frac=args.roi_frac)
                  if args.roi in ("none", "center") else None)
    n_tok = None if perceptual is None else scorer_n_tokens(perceptual)
    # 'pred_motion' can only be built after the forward pass, so its mask cannot reach the
    # appearance scorer (which is applied inside the same no_grad block). Appearance therefore
    # stays whole-frame in that arm — acceptable, since the control is a claim about motion.
    pred_roi = args.roi == "pred_motion"
    if args.roi != "none":
        if pred_roi:
            print("[qfae-dino] ROI=pred_motion (control) -> ranked on the PREDICTED field, flow "
                  "streams only; appearance stays whole-frame")
        elif n_tok is None:
            print("[qfae-dino] NOTE: scorer token grid unknown — ROI applies to the flow streams "
                  "only, appearance stays whole-frame")
        else:
            print(f"[qfae-dino] ROI={args.roi} -> appearance pooled over the scorer's {n_tok} tokens")

    def slice_roi(i):
        """ROI for val slice i (None = whole frame, or deferred to after the forward pass)."""
        if static_roi is not None or args.roi == "none" or pred_roi:
            return static_roi
        return build_roi(args.roi, gt_flow=gt_flows[i],
                         lv_mask=(lv_rois[i] if lv_rois is not None else None),
                         size=size, frac=args.roi_frac, motion_frac=args.motion_frac)

    for start in range(0, n, args.batch_size):
        x = X[start:start + args.batch_size].to(device)
        b = x.shape[0]
        rois = [slice_roi(start + j) for j in range(b)]
        use_amp = device.type == "cuda" and amp_dtype != torch.float32
        tok_mask = None
        if args.roi != "none" and n_tok is not None and not pred_roi:
            tok_mask = torch.from_numpy(
                np.stack([roi_to_token_mask(r, n_tok) for r in rois])).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            recon, pred_flow = out if args.flow else (out, None)
            if args.appe_space == "pixel":
                d = pixel_appe_scores_multi(recon.float(), x.float())
                s = d["msegrad"]                                    # default appearance = training-loss score
            else:
                s = perceptual.score(x, recon, top_frac=args.top_frac, token_mask=tok_mask)
        scores.append(s.float().cpu().numpy())
        if args.appe_space == "pixel":                             # collect the pixel score-way sweep
            for k in ("mse", "mae", "msegrad"):
                appe_sweep[k].append(d[k].float().cpu().numpy())
            rg = recon.float().cpu().numpy()[:, 0]; xg = x.float().cpu().numpy()[:, 0]   # grayscale ch0
            for j in range(rg.shape[0]):
                dr = float(max(rg[j].max(), xg[j].max()) - min(rg[j].min(), xg[j].min())) or 1.0
                appe_sweep["ssim"].append(1.0 - ssim(rg[j], xg[j], data_range=dr))
        if args.flow:
            pf = pred_flow.float().cpu().numpy()                    # (b,3,S,S)
            for j in range(pf.shape[0]):
                gt = gt_flows[start + j]
                r = rois[j]
                if pred_roi:
                    # Rank the mask on the model's own magnitude channel. Nothing about the GT
                    # field selects the region, and no second cardiac phase is required.
                    r = motion_mask(pf[j], args.motion_frac)
                # reuse the 3-D flow_scores by adding a length-1 depth axis (one 2-D slice)
                fs, ms, l1 = flow_scores(gt[..., None], pf[j][..., None],
                                         vox_mask=(None if r is None else r[..., None]))
                fssim.append(fs); mssim.append(ms); fl1.append(l1)
                for k, v in extra_flow_scores(gt, pf[j], mask=r).items():
                    extra[k].append(v)
                for k, v in gt_flow_stats(gt, mask=r).items():       # model-free control
                    gt_only[k].append(v)
                if maha is not None:
                    maha_scores.append(float(maha.score(pf[j][None] - gt[None], mask=r)[0]))
        print(f"\r[qfae-dino] scored {min(start + args.batch_size, n)}/{n}", end="", flush=True)
    print()
    scores = np.concatenate(scores)

    uniq = np.unique(pids)
    pds = np.array([datasets[pids == p][0] for p in uniq])

    def report(name, vals):
        pat, plabels = aggregate_to_patient(vals, pids, slcs, labels)
        out = {"frame": one_vs_nor_aucs(vals, labels), "patient": {}, "per_dataset": {}}
        for agg in _AGGS:
            out["patient"][agg] = one_vs_nor_aucs(pat[agg], plabels)
        for ds in ("ACDC", "MM"):
            m = pds == ds
            out["per_dataset"][ds] = one_vs_nor_aucs(pat["Mean"][m], plabels[m])
        pm = out["per_dataset"]
        print(f"[{name:10s}] PAT-Mean  ACDC={pm['ACDC'].get('overall', float('nan')):.3f}  "
              f"MM={pm['MM'].get('overall', float('nan')):.3f}")
        return out

    print("\n" + "=" * 60)
    print(f"QFAE-DINO{'2D+FLOW' if args.flow else ''} RESULTS (patient-Mean AUC per stream)")
    print("=" * 60)
    save = {"scores": scores, "appearance": scores, "pids": pids, "labels": labels,
            "datasets": datasets, "slcs": slcs}
    stream_res = {"appearance": report("appearance", scores)}
    if args.appe_space == "pixel":                                 # pixel score-way sweep streams
        print("--- pixel score-way sweep (MSE / MAE / 1-SSIM / MSE+grad), same recon ---")
        for k in ("mse", "mae", "ssim", "msegrad"):
            # mse/mae/msegrad = list of per-BATCH arrays -> concatenate; ssim = flat list -> asarray
            v = (np.asarray(appe_sweep[k], np.float32) if k == "ssim"
                 else np.concatenate(appe_sweep[k]).astype(np.float32))
            save[f"appe_{k}"] = v
            stream_res[f"appe_{k}"] = report(f"appe_{k}", v)
    if args.flow:
        fssim, mssim, fl1 = np.array(fssim), np.array(mssim), np.array(fl1)
        stream_res["flow_SSIM"] = report("flow_SSIM", fssim)
        stream_res["mag_SSIM"] = report("mag_SSIM", mssim)
        stream_res["flow_L1"] = report("flow_L1", fl1)
        save.update(flow_SSIM=fssim, mag_SSIM=mssim, flow_L1=fl1)
        print("--- flow score-way sweep (EPE / normalised EPE / angular / magnitude-ratio) ---")
        for k in _EXTRA_FLOW_STREAMS:
            v = np.asarray(extra[k], np.float32)
            save[k] = v
            stream_res[k] = report(k, v)
        # Model-free control: the GT displacement field alone, pooled over the same ROI. These
        # use no network output at all, so they are the floor the detector has to clear before
        # an ROI gain can be attributed to the model rather than to the mask. Direction is not
        # fixed a priori — read |AUC - 0.5|.
        print("--- model-free control (GT displacement statistics, same ROI, no network) ---")
        for k in _GT_ONLY_STREAMS:
            v = np.asarray(gt_only[k], np.float32)
            save[k] = v
            stream_res[k] = report(k, v)
        if maha is not None:
            v = np.asarray(maha_scores, np.float32)
            save["flow_maha"] = v
            stream_res["flow_maha"] = report("flow_maha", v)

    results = {"method": "qfae_dino2d_flow" if args.flow else "qfae_dino", "n_val": int(n),
               "img_size": args.img_size, "dino_model": args.dino_model,
               "perceptual_layers": args.perceptual_layers,
               # encoder/scorer are the matrix coordinates — record them resolved, not as the
               # legacy two-valued flag (which wrote "dino" even for an MAE encoder scoring itself).
               "encoder": args.dino_model,
               "scorer": "pixel" if args.appe_space == "pixel" else scorer_spec,
               "appe_space": args.appe_space,
               "perceptual_backbone": args.perceptual_backbone, "top_frac": args.top_frac,
               # the A/B architecture of the checkpoint being scored, so a results file is
               # self-describing when the sweep is collated offline
               "arch": {k: v for k, v in arch.items() if k in (
                   "n_queries", "decoder_mode", "qformer_dropout", "attn_dropout", "linear_attn")},
               "roi": args.roi, "roi_frac": args.roi_frac, "motion_frac": args.motion_frac,
               "maha": bool(args.maha), "maha_grid": args.maha_grid,
               "streams": stream_res}
    with open(os.path.join(args.out_dir, "qfae_dino_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, "qfae_dino_arrays.npz"), **save)
    print(f"\n[qfae-dino] wrote {args.out_dir}/qfae_dino_results.json + qfae_dino_arrays.npz")
    print("Aggregation sweep: python qfae_report.py --agg_sweep " + args.out_dir + "/qfae_dino_arrays.npz")


if __name__ == "__main__":
    main()
