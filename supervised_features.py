"""supervised_features.py — frozen-encoder + handcrafted features for the supervised probe.

Stage 1 of the supervised NOR-vs-disease ceiling baseline. Extracts, for every 2-D slice of
every patient in the four official folders (ACDC training/testing, M&Ms Training/Testing),
five parallel feature blocks, and caches them so the fitting stage is free to iterate:

  emb_frame   (768)  frozen encoder GAP over patch tokens of the FRAME        -> "appe"
  emb_flow    (768)  frozen encoder GAP of the GT FLOW rendered as 3-channel   -> "motion"
  stat_flow   ( 44)  handcrafted flow statistics, 3 regions                    -> "motionstat"
  stat_appe   ( 24)  handcrafted intensity statistics                          -> "appestat"
  raw_flow    (768)  16x16 mean-pool of [dx,dy,mag], NO ENCODER                -> "motionraw"

WHY THE LAST TWO EXIST. A good AUC from `emb_flow` alone is uninterpretable: it cannot be
distinguished from "any spatial summary of the flow field would have done it". `raw_flow` has
the SAME dimensionality by construction (16*16*3 = 768), so a gap between them is not a
dimensionality artefact; `stat_flow` is the network-free floor. The likely outcome — all three
comparable — is a stronger result than a marginal win: it says the cross-vendor ceiling is a
property of the displacement field, not of any learned representation.

FLOW NORMALISATION IS THE MOST DANGEROUS KNOB HERE. Do NOT per-slice or per-patient normalise:
amplitude *is* the disease signal (a hypokinetic or dilated ventricle moves less), so any
per-sample rescaling deletes exactly what we are trying to measure. One global constant
`--flow_scale` is used for every split and dataset, and the clip rate is reported.

Feeding a flow field to an RGB-pretrained ViT is defensible as a PROBE, not as a claim that
the encoder understands motion: [dx,dy,mag] has no RGB analogue and channel 2 is a
deterministic function of 0-1, so the tensor is rank-2. What it buys is a fixed high-capacity
feature map with image-appropriate inductive biases. `raw_flow` / `stat_flow` are what make
the comparison interpretable.

    python supervised_features.py --encoder vit_base_patch16_224.mae --cache_dir ./sup_cache_mae224
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

import cinema_faithful as cf
from qfae_dino import load_dinov2

EPS = 1e-8

SPLITS = ("acdc_train", "acdc_test", "mm_train", "mm_test")

# Expected patient counts. mm_train depends on --recover_edes (see validate_edes_recovery.py):
# all 25 GE cases carry ED==ES==0 in the CSV and are skipped without it.
EXPECTED = {"acdc_train": 100, "acdc_test": 50, "mm_test": 136}
EXPECTED_MM_TRAIN = {False: 150, True: 175}

FLOW_REGIONS = ("frame", "centre", "heart")
FLOW_STAT_PER_REGION = ("mag_mean", "mag_std", "mag_cv", "mag_p50", "mag_p90", "mag_p99",
                        "mag_max", "frac_gt1", "frac_gt2", "frac_gt4", "rad_mean", "rad_std",
                        "tan_absmean", "dir_circvar")
FLOW_STAT_NAMES = tuple(f"{r}_{s}" for r in FLOW_REGIONS for s in FLOW_STAT_PER_REGION) + \
                  ("heart_area_frac", "heart_frame_mag_ratio")

APPE_STAT_NAMES = ("mean", "std", "p1", "p10", "p50", "p90", "p99", "skew", "kurt",
                   "sobel_mean", "sobel_p99", "entropy",
                   "heart_mean", "heart_std", "heart_contrast", "heart_area_frac") + \
                  tuple(f"hist{i}" for i in range(8))


# ── flow -> 3-channel image ──────────────────────────────────────────────────
def flow_to_rgb(flow, scale, mode="linear"):
    """(3,S,S) [dx,dy,mag] in pixels -> (3,S,S) in [0,1]. ONE global scale, never per-sample."""
    dx, dy, mag = flow[0], flow[1], flow[2]
    if mode == "linear":
        c0 = np.clip(dx / (2.0 * scale) + 0.5, 0.0, 1.0)
        c1 = np.clip(dy / (2.0 * scale) + 0.5, 0.0, 1.0)
        c2 = np.clip(mag / (2.0 * scale), 0.0, 1.0)
    elif mode == "tanh":                       # soft, no hard clip; keeps the tail ordered
        c0 = 0.5 * (np.tanh(dx / scale) + 1.0)
        c1 = 0.5 * (np.tanh(dy / scale) + 1.0)
        c2 = np.tanh(mag / scale)
    else:
        raise ValueError(mode)
    return np.stack([c0, c1, c2]).astype(np.float32)


def clip_fraction(flows, scale):
    """Fraction of displacement components the linear map would clip. Diagnostic only."""
    d = np.abs(flows[:, :2])
    return float((d > 2.0 * scale).mean())


# ── handcrafted statistics ───────────────────────────────────────────────────
def _centre_mask(size, frac=0.5):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    return ((yy - c) ** 2 + (xx - c) ** 2) <= (frac * size / 2.0) ** 2


def _region_flow_stats(dx, dy, mag, m):
    """14 statistics of the displacement field inside boolean mask `m`."""
    if m.sum() < 8:
        return [0.0] * len(FLOW_STAT_PER_REGION)
    mm, dxm, dym = mag[m], dx[m], dy[m]
    mean, std = float(mm.mean()), float(mm.std())
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    ry, rx = ys - cy, xs - cx
    rn = np.sqrt(ry ** 2 + rx ** 2) + EPS
    ry, rx = ry / rn, rx / rn
    radial = dxm * rx + dym * ry                 # inward (contraction) is negative
    tangential = dxm * (-ry) + dym * rx
    w = mm + EPS
    ang = np.arctan2(dym, dxm)
    r_bar = np.abs((w * np.exp(1j * ang)).sum()) / w.sum()
    return [mean, std, std / (mean + EPS),
            float(np.percentile(mm, 50)), float(np.percentile(mm, 90)),
            float(np.percentile(mm, 99)), float(mm.max()),
            float((mm > 1).mean()), float((mm > 2).mean()), float((mm > 4).mean()),
            float(radial.mean()), float(radial.std()), float(np.abs(tangential).mean()),
            float(1.0 - r_bar)]


def flow_stat_vector(flow, centre, heart):
    dx, dy, mag = flow[0], flow[1], flow[2]
    hm = heart > 0.5
    if hm.sum() < 8:
        hm = centre
    out = []
    for m in (np.ones_like(centre, dtype=bool), centre, hm):
        out += _region_flow_stats(dx, dy, mag, m)
    frame_mean = float(mag.mean()) + EPS
    out += [float(hm.mean()), float(mag[hm].mean()) / frame_mean if hm.sum() else 0.0]
    return np.asarray(out, dtype=np.float32)


def appe_stat_vector(frame, heart):
    """Intensity / gradient / histogram statistics of the (already [0,1]) frame."""
    g = frame[0].astype(np.float32)
    hm = heart > 0.5
    mean, std = float(g.mean()), float(g.std())
    z = (g - mean) / (std + EPS)
    sob = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + \
          np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    hist, _ = np.histogram(g, bins=8, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32) / max(g.size, 1)
    p = hist[hist > 0]
    out = [mean, std,
           float(np.percentile(g, 1)), float(np.percentile(g, 10)),
           float(np.percentile(g, 50)), float(np.percentile(g, 90)),
           float(np.percentile(g, 99)),
           float((z ** 3).mean()), float((z ** 4).mean()),
           float(sob.mean()), float(np.percentile(sob, 99)),
           float(-(p * np.log(p)).sum()),
           float(g[hm].mean()) if hm.sum() else 0.0,
           float(g[hm].std()) if hm.sum() else 0.0,
           float(g[hm].mean() - g[~hm].mean()) if hm.sum() and (~hm).sum() else 0.0,
           float(hm.mean())]
    return np.asarray(out + list(hist), dtype=np.float32)


def pool_flow_raw(flow, grid=16):
    """grid x grid mean-pool of each of [dx,dy,mag] -> 3*grid*grid. No encoder involved."""
    ch = [cv2.resize(flow[i], (grid, grid), interpolation=cv2.INTER_AREA) for i in range(3)]
    return np.concatenate([c.ravel() for c in ch]).astype(np.float32)


# ── frozen encoder embedding ─────────────────────────────────────────────────
def embed(model, mean, std, imgs, device, dtype, batch_size, tag):
    """(N,3,S,S) float32 in [0,1] -> (N,768) GAP over patch tokens (prefix/CLS dropped)."""
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    feats, n, printed = [], len(imgs), False
    for s in range(0, n, batch_size):
        x = torch.from_numpy(imgs[s:s + batch_size]).to(device)
        x = (x - mean_t) / std_t
        use_amp = (device.type == "cuda" and dtype != torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            tok = model.forward_features(x)
        tok = tok.float()
        vec = tok[:, n_prefix:, :].mean(dim=1) if tok.dim() == 3 else tok
        if not printed:
            print(f"    [{tag}] forward_features {tuple(tok.shape)} -> {tuple(vec.shape)}")
            printed = True
        feats.append(vec.cpu().numpy())
        print(f"\r    [{tag}] {min(s + batch_size, n)}/{n}", end="", flush=True)
    print()
    return np.concatenate(feats, axis=0).astype(np.float32)


# ── records ──────────────────────────────────────────────────────────────────
def build_split_records(split, args):
    if split == "acdc_train":
        return cf.load_acdc_records_flow(args.acdc_dir, "training", nor_only=False)
    if split == "acdc_test":
        return cf.load_acdc_records_flow(args.acdc_dir, "testing", nor_only=False)
    if split == "mm_train":
        return cf.load_mm_records_flow(args.mm_train_dir, args.mm_csv, nor_only=False)
    if split == "mm_test":
        return cf.load_mm_records_flow(args.mm_test_dir, args.mm_csv, nor_only=False)
    raise ValueError(split)


def vendor_arrays(pids, datasets, csv_path):
    df = pd.read_csv(csv_path)
    lut = {r["External code"]: (str(r["VendorName"]), int(r["Centre"])) for _, r in df.iterrows()}
    vend, cent = [], []
    for pid, ds in zip(pids, datasets):
        if ds != "MM":
            vend.append("ACDC"); cent.append(-1); continue
        v, c = lut.get(pid.split("_", 1)[1], ("UNKNOWN", -1))
        vend.append(v); cent.append(c)
    return np.array(vend), np.array(cent, dtype=np.int64)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_train_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_test_dir", default="../Dataset_1/Testing")
    ap.add_argument("--mm_csv",
                    default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--encoder", default="vit_base_patch16_224.mae")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--flow_scale", type=float, default=8.0)
    ap.add_argument("--flow_map", default="linear", choices=["linear", "tanh"])
    ap.add_argument("--flow_grid", type=int, default=16)
    ap.add_argument("--flow_backend", default="farneback", choices=["farneback", "registration"])
    ap.add_argument("--recover_edes", action="store_true",
                    help="Recover the 25 GE M&Ms-Training cases whose CSV has ED==ES==0.")
    ap.add_argument("--cache_dir", default="./sup_cache_mae224")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ref_npz", default="./qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz",
                    help="Held-out universe gate: acdc_test+mm_test must match this exactly.")
    ap.add_argument("--limit_patients", type=int, default=0, help="Smoke test only.")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cuda" if args.device == "cuda" else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("=" * 92)
    print(f"SUPERVISED FEATURES | encoder={args.encoder} @{args.img_size} | device={device}")
    print(f"  flow_map={args.flow_map} scale={args.flow_scale} | recover_edes={args.recover_edes}")
    print("=" * 92)

    cf.set_single_pass(True)                    # MUST match the eval universe
    cf.set_flow_backend(args.flow_backend)      # explicit, not defaulted
    cf.set_edes_recovery(args.recover_edes)

    print(f"\n=== Loading frozen {args.encoder} ===")
    model, enc_mean, enc_std = load_dinov2(args.encoder, args.img_size, device)

    centre = _centre_mask(args.img_size, 0.5)
    manifest = {}

    for split in args.splits:
        out_path = os.path.join(args.cache_dir, f"{split}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"\n### {split}: cached at {out_path} (use --overwrite to rebuild)")
            continue
        t0 = time.time()
        print(f"\n### {split}")
        recs = build_split_records(split, args)
        if args.limit_patients:
            recs = recs[:args.limit_patients]

        n_pat = len({r["pid"] for r in recs})
        exp = EXPECTED_MM_TRAIN[args.recover_edes] if split == "mm_train" else EXPECTED.get(split)
        if exp is not None and not args.limit_patients:
            assert n_pat == exp, (f"GATE 1 FAILED: {split} has {n_pat} patients, expected {exp}. "
                                  f"The loader universe changed — stop and investigate.")
        print(f"    patients: {n_pat}  (gate: {exp})")

        frames, flows, rois, pids, slcs, labels, datasets = cf._explode_2d(
            recs, args.img_size, want_meta=True, want_roi=True)
        del recs
        print(f"    slices: {len(frames)}")

        cr = clip_fraction(flows, args.flow_scale)
        mags = flows[:, 2]
        print(f"    flow |d| clip@{args.flow_scale}: {100 * cr:.2f}%  "
              f"mag p50={np.percentile(mags, 50):.2f} p99={np.percentile(mags, 99):.2f} "
              f"max={mags.max():.2f}")
        if cr > 0.02:
            print(f"    ** WARNING: clip rate {100 * cr:.1f}% > 2% — raise --flow_scale **")

        flow_rgb = np.stack([flow_to_rgb(f, args.flow_scale, args.flow_map) for f in flows])
        stat_flow = np.stack([flow_stat_vector(f, centre, r) for f, r in zip(flows, rois)])
        stat_appe = np.stack([appe_stat_vector(f, r) for f, r in zip(frames, rois)])
        raw_flow = np.stack([pool_flow_raw(f, args.flow_grid) for f in flows])
        del flows

        emb_frame = embed(model, enc_mean, enc_std, frames, device, dtype,
                          args.batch_size, f"{split}:frame")
        del frames
        emb_flow = embed(model, enc_mean, enc_std, flow_rgb, device, dtype,
                         args.batch_size, f"{split}:flow")
        del flow_rgb

        # GATE 6: if the flow tensor was never actually swapped in, `motion` is a copy of `appe`.
        a, b = emb_frame.mean(0), emb_flow.mean(0)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
        print(f"    cos(mean emb_frame, mean emb_flow) = {cos:.4f}"
              + ("   ** SUSPICIOUS: ~1.0 means the flow stream is a copy **" if cos > 0.99 else ""))
        for nm, arr in (("emb_frame", emb_frame), ("emb_flow", emb_flow),
                        ("stat_flow", stat_flow), ("stat_appe", stat_appe),
                        ("raw_flow", raw_flow)):
            assert np.isfinite(arr).all(), f"non-finite values in {nm} for {split}"

        vend, cent = vendor_arrays(pids, datasets, args.mm_csv)
        assert not (vend == "UNKNOWN").any(), f"unresolved vendor in {split}"

        np.savez_compressed(out_path, emb_frame=emb_frame, emb_flow=emb_flow,
                            stat_flow=stat_flow, stat_appe=stat_appe, raw_flow=raw_flow,
                            pids=pids, slcs=slcs, labels=labels, datasets=datasets,
                            vendors=vend, centres=cent,
                            flow_stat_names=np.array(FLOW_STAT_NAMES),
                            appe_stat_names=np.array(APPE_STAT_NAMES))
        manifest[split] = dict(patients=int(n_pat), slices=int(len(pids)),
                               clip_frac=cr, cos_frame_flow=cos,
                               nor=int((labels == "NOR").sum()),
                               seconds=round(time.time() - t0, 1))
        print(f"    wrote {out_path}   ({time.time() - t0:.0f}s)")

    # ── GATE 2: the held-out universe must be element-wise identical to the reference ──
    if {"acdc_test", "mm_test"} <= set(args.splits) and os.path.exists(args.ref_npz):
        print("\n[GATE 2] held-out universe vs reference")
        ref = np.load(args.ref_npz, allow_pickle=True)
        a = np.load(os.path.join(args.cache_dir, "acdc_test.npz"), allow_pickle=True)
        m = np.load(os.path.join(args.cache_dir, "mm_test.npz"), allow_pickle=True)
        ok = True
        for k in ("pids", "slcs", "labels", "datasets"):
            got = np.concatenate([a[k], m[k]])
            same = got.shape == ref[k].shape and (got == ref[k]).all()
            ok &= same
            print(f"    {k:<10} {'OK' if same else '** MISMATCH **'}  "
                  f"({got.shape} vs {ref[k].shape})")
        print(f"    n_slices={len(np.concatenate([a['pids'], m['pids']]))} "
              f"(ref {len(ref['pids'])}), patients="
              f"{len(np.unique(np.concatenate([a['pids'], m['pids']])))} (ref "
              f"{len(np.unique(ref['pids']))})")
        if not ok:
            print("    ** held-out universe drifted — paired comparison against the UAD "
                  "would silently fail. STOP. **")
        manifest["gate2_heldout_matches_reference"] = bool(ok)

    # ── GATE 3: train/test patient disjointness ──
    for tr, te in (("acdc_train", "acdc_test"), ("mm_train", "mm_test")):
        pa = os.path.join(args.cache_dir, f"{tr}.npz")
        pb = os.path.join(args.cache_dir, f"{te}.npz")
        if os.path.exists(pa) and os.path.exists(pb):
            s1 = set(np.load(pa, allow_pickle=True)["pids"])
            s2 = set(np.load(pb, allow_pickle=True)["pids"])
            inter = s1 & s2
            print(f"[GATE 3] {tr} n {te}: {len(inter)} shared patients "
                  f"{'OK' if not inter else '** LEAKAGE **'}")
            assert not inter, f"train/test overlap between {tr} and {te}: {sorted(inter)[:5]}"

    with open(os.path.join(args.cache_dir, "manifest.json"), "w") as f:
        json.dump(dict(encoder=args.encoder, img_size=args.img_size,
                       flow_scale=args.flow_scale, flow_map=args.flow_map,
                       flow_grid=args.flow_grid, flow_backend=args.flow_backend,
                       recover_edes=args.recover_edes, splits=manifest), f, indent=2)
    print(f"\n[done] cache -> {args.cache_dir}")


if __name__ == "__main__":
    main()
