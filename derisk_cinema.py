"""
derisk_cinema.py — Minimal de-risking experiment for the "CineMA + reverse
distillation" idea (see MAE_CineMA_feasibility_memo.md).

QUESTION IT ANSWERS
-------------------
Do FROZEN CineMA encoder features *already* separate cardiac disease from NOR,
with NO decoder training at all? If a training-free normal-feature model
(Mahalanobis / kNN memory bank, PatchCore-style) fitted on NOR-only frames
matches or beats the current Flow-SSIM baseline (patient-level `Mean`
AUC ~ 0.73-0.77), the full reverse-distillation build is justified. If it sits
at chance, that is a cheap early signal to reconsider the backbone or add
encoder adaptation — before writing any student decoder.

PIPELINE
--------
  1. Load NOR training ED/ES frames  (the "fit" / normal set)
  2. Load validation ED/ES frames    (NOR + disease, the "score" set)
     -> both via the EXISTING data_loader.py, so preprocessing matches training
  3. Extract frozen CineMA encoder features per frame (GAP over feature maps)
  4. Fit Mahalanobis (Ledoit-Wolf) + kNN memory bank on NOR-train features
  5. Score every val frame, then aggregate to patient level with the SAME five
     aggregators used in GAN_tf.py (Mean / FrameMax / FrameTop20 / SliceMax /
     SliceTop20)
  6. Report ROC-AUC (NOR vs disease): frame-level and patient-level, overall,
     per-disease (one-vs-NOR) and per-dataset.

Nothing here touches the GAN. It reuses the loaders so results are directly
comparable to what run_model.py logs.

INPUT GEOMETRY — TWO PREPROCESSING PATHS (--cinema_preproc)
----------------------------------------------------------
CineMA's SAX pathway expects a (1, 192, 192, 16) single-channel stack.

  * faithful (default): CineMA's OWN canonical SAX pipeline (cinema_faithful.py)
    — resample each 3-D ED/ES volume to 1.0 mm/px, LV-bbox center-crop to 192,
    clip 0.95/99.5 percentiles, feed the REAL depth-16 slice stack. Read straight
    from the raw ACDC/M&Ms files, so the frozen features are ON-distribution. The
    unit is a 3-D stack per (patient, phase); patient scores aggregate Mean/Max
    over ED+ES. The orient/spacing/N4 and --sax_fill flags DO NOT apply here.

  * legacy: single 2-D SAX slice at 128x128 (via data_loader.py) resized to 192
    and placed in a depth stack that is zero-padded to 16 (--sax_fill zero,
    matching CineMA's feature-extraction example) or replicated (--sax_fill
    replicate). Off-distribution for CineMA, but the padding is a *systematic*
    offset that mostly cancels in the rank-based AUC. Only this path honours the
    orient/spacing/N4 flags (its data comes from the ICCV loaders).
"""

import os
import sys
import json
import argparse

import numpy as np
import cv2

# data_loader.py imports only os/cv2/numpy/pandas/SimpleITK (+ loader helpers) —
# NO tensorflow — so it is importable in the torch-only de-risk env.
import data_loader as dl

import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from cinema import CineMA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


# ── SAX preprocessing transforms (mirror CineMA's mae_feature_extraction.py) ──
_TF_PAD   = Compose([ScaleIntensityd(keys="sax"),
                     SpatialPadd(keys="sax", spatial_size=(192, 192, 16), method="end")])
_TF_SCALE = Compose([ScaleIntensityd(keys="sax")])


def frame_to_sax(gray, sax_fill):
    """One 2-D grayscale frame -> CineMA SAX tensor (1, 192, 192, 16), float32."""
    img = cv2.resize(gray.astype(np.float32), (192, 192), interpolation=cv2.INTER_LINEAR)
    if sax_fill == 'replicate':
        vol = np.repeat(img[..., None], 16, axis=-1)          # (192,192,16)
        d = _TF_SCALE({"sax": torch.from_numpy(vol[None].astype(np.float32))})
    else:  # 'zero' — single real slice, SpatialPadd end-pads depth 1 -> 16
        vol = img[..., None]                                  # (192,192,1)
        d = _TF_PAD({"sax": torch.from_numpy(vol[None].astype(np.float32))})
    return np.asarray(d["sax"], dtype=np.float32)             # (1,192,192,16)


def extract_features(model, frames, device, dtype, args):
    """frames: (N, H, W, 3) in [0,1] -> (N, D) frozen-CineMA feature vectors.

    Each feature tensor from feature_forward is (B, n_patches, enc_emb_dim=768)
    — channel-LAST tokens — so it is average-pooled over the patch/token dim,
    keeping the 768-d embedding (cls token -> its 768-d vector; sax patches ->
    their mean 768-d vector). Actual shapes are printed on the first batch.
    """
    feats, printed = [], False
    n = len(frames)
    for start in range(0, n, args.batch_size):
        chunk = frames[start:start + args.batch_size]
        sax = np.stack([frame_to_sax(f[..., 0], args.sax_fill) for f in chunk])  # (b,1,192,192,16)
        t = torch.from_numpy(sax).to(device=device, dtype=dtype)
        use_amp = (device.type == 'cuda' and dtype != torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            fd = model.feature_forward({"sax": t})
        if not printed:
            print("[derisk] CineMA feature_forward() output tensors:")
            for k, v in fd.items():
                print(f"    {k}: {tuple(v.shape)}")
            printed = True
        keys = sorted(fd.keys())
        if args.feature_layers == 'last':
            keys = [keys[-1]]
        vecs = []
        for k in keys:
            v = fd[k].float()
            # feature_forward returns (batch, n_patches, enc_emb_dim=768) — token/
            # patch axis in the MIDDLE, embedding LAST. Pool over the patch dim
            # (keep the 768-d embedding): cls (b,1,768)->(b,768), sax (b,2304,768)->(b,768).
            vecs.append(v.flatten(1, -2).mean(dim=1) if v.dim() > 2 else v)  # -> (b, 768)
        feats.append(torch.cat(vecs, dim=1).cpu().numpy())
        print(f"\r[derisk] features {min(start + args.batch_size, n)}/{n}", end="", flush=True)
    print()
    return np.concatenate(feats, axis=0)


# ── Alternative frozen backbones: 2-D timm ViTs (DINOv2 / vanilla MAE) ────────
def load_timm_backbone(args, device):
    """Load a frozen 2-D timm ViT (DINOv2 or ImageNet-MAE); return (model, data_cfg).

    Both candidates are ViT-B (768-d), matching CineMA's embedding width, so the
    downstream StandardScaler/LedoitWolf/kNN scorers are unchanged. img_size is
    forced to 224 (pos-embed interpolated) to keep token counts + compute modest
    and comparable across encoders — the same resolution the original QFAE uses.
    """
    import timm
    from timm.data import resolve_model_data_config
    name = args.dino_model if args.backbone == 'dino' else args.mae_model
    print(f"\n=== Loading frozen timm backbone: {name} ({args.backbone}) ===")
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = resolve_model_data_config(model)          # mean/std for this checkpoint
    n_prefix = getattr(model, 'num_prefix_tokens', 1)
    print(f"[derisk] {name}: mean={cfg['mean']} std={cfg['std']} "
          f"num_prefix_tokens={n_prefix} (CLS+registers dropped before GAP)")
    return model, cfg


def extract_features_timm(model, frames, device, dtype, args, cfg):
    """frames: (N, H, W, 3) in [0,1] -> (N, 768) frozen 2-D ViT features.

    Per frame: resize to 224, timm-normalise (the checkpoint's own mean/std),
    forward_features -> (b, num_prefix + n_patch, 768); drop the prefix tokens
    (CLS + any DINOv2 registers) and mean-pool the patch tokens -> (b, 768).
    Same GAP-over-patch-tokens reduction the CineMA path uses (extract_features).
    """
    size = 224
    mean = torch.tensor(cfg['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg['std'], device=device).view(1, 3, 1, 1)
    n_prefix = getattr(model, 'num_prefix_tokens', 1)
    feats, printed = [], False
    n = len(frames)
    for start in range(0, n, args.batch_size):
        chunk = frames[start:start + args.batch_size]                      # (b,H,W,3) in [0,1]
        batch = np.stack([cv2.resize(f.astype(np.float32), (size, size),
                                     interpolation=cv2.INTER_LINEAR) for f in chunk])  # (b,224,224,3)
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)        # (b,3,224,224)
        x = (x - mean) / std                                               # normalise in fp32
        use_amp = (device.type == 'cuda' and dtype != torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            tok = model.forward_features(x)                                # (b, prefix+patches, 768)
        tok = tok.float()
        vec = tok[:, n_prefix:, :].mean(dim=1) if tok.dim() == 3 else tok  # GAP over patch tokens
        if not printed:
            print(f"[derisk] {args.backbone} forward_features {tuple(tok.shape)} -> feat {tuple(vec.shape)}")
            printed = True
        feats.append(vec.cpu().numpy())
        print(f"\r[derisk] features {min(start + args.batch_size, n)}/{n}", end="", flush=True)
    print()
    return np.concatenate(feats, axis=0)


# ── Patient-level aggregation (identical to GAN_tf.py) ───────────────────────
def _top20_mean(arr):
    n = len(arr)
    if n == 0:
        return np.nan
    k = max(1, int(np.ceil(0.2 * n)))
    return float(np.mean(np.sort(arr)[-k:]))


_AGGS = ['Mean', 'FrameMax', 'FrameTop20', 'SliceMax', 'SliceTop20']


def aggregate_to_patient(scores, pids, slcs, labels):
    """Per-frame scores -> per-patient scores under all five aggregators."""
    scores = np.asarray(scores, dtype=float)
    pids = np.asarray(pids)
    slcs = np.asarray(slcs)
    labels = np.asarray(labels)
    uniq = np.unique(pids)
    out = {a: np.zeros(len(uniq)) for a in _AGGS}
    plabels = []
    for i, pid in enumerate(uniq):
        m = (pids == pid)
        s = scores[m]
        sl = slcs[m]
        out['Mean'][i] = float(np.mean(s))
        out['FrameMax'][i] = float(np.max(s))
        out['FrameTop20'][i] = _top20_mean(s)
        slice_means = np.array([float(np.mean(s[sl == u])) for u in np.unique(sl)])
        out['SliceMax'][i] = float(np.max(slice_means))
        out['SliceTop20'][i] = _top20_mean(slice_means)
        plabels.append(labels[m][0])
    return out, np.array(plabels)


def aggregate_stacks_to_patient(scores, pids, labels):
    """Per-stack scores -> per-patient (Mean / Max over that patient's stacks).

    The faithful path's unit is a 3-D stack per (patient, phase), so a patient
    has at most two scores (ED, ES). SliceMax / FrameTop20 etc. are meaningless
    at this granularity, so only Mean and Max are reported.
    """
    scores = np.asarray(scores, dtype=float)
    pids = np.asarray(pids)
    labels = np.asarray(labels)
    uniq = np.unique(pids)
    out = {'Mean': np.zeros(len(uniq)), 'Max': np.zeros(len(uniq))}
    plabels = []
    for i, pid in enumerate(uniq):
        m = (pids == pid)
        out['Mean'][i] = float(np.mean(scores[m]))
        out['Max'][i] = float(np.max(scores[m]))
        plabels.append(labels[m][0])
    return out, np.array(plabels)


def one_vs_nor_aucs(scores, labels):
    """AUC (NOR=0, disease=1): overall + each disease one-vs-NOR."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    res = {}
    y = (labels != 'NOR').astype(int)
    if y.min() != y.max():
        res['overall'] = float(roc_auc_score(y, scores))
    for d in sorted(set(labels)):
        if d == 'NOR':
            continue
        m = (labels == 'NOR') | (labels == d)
        yy = (labels[m] != 'NOR').astype(int)
        if yy.min() != yy.max():
            res[d] = float(roc_auc_score(yy, scores[m]))
    return res


# ── Data loading (mirrors run_model.py, ed_es mode) ──────────────────────────
def configure_loaders(args):
    if args.orient_normalize:
        orient_csv = args.orient_params or os.path.join(
            '..', 'reconstructed_sax_images_training_2023',
            'segmentation', 'orientation_params.csv')
        print(f"[derisk] orientation normalisation ON, params={orient_csv}")
        dl.set_orientation_normalization(True, orient_csv)
    else:
        dl.set_orientation_normalization(False)

    if args.spacing_normalize:
        print(f"[derisk] spacing normalisation ON, target={args.target_spacing} mm/px, "
              f"size={args.target_size} px")
        dl.set_spacing_normalization(True, args.target_spacing, args.target_size,
                                     (args.recon_spacing, args.recon_spacing))
    else:
        dl.set_spacing_normalization(False)

    if args.n4_bias_correct:
        print(f"[derisk] N4 bias-field correction ON (shrink={args.n4_shrink})")
        dl.set_n4_bias_correction(True, args.n4_shrink, args.n4_iterations, args.n4_levels)
    else:
        dl.set_n4_bias_correction(False)

    dl.set_edes_direction('es')       # ed_es: ES frame is the model input
    dl.set_flow_backend('farneback')  # flows are computed then discarded here


def load_fit_frames(args):
    """NOR-only training ED/ES frames (the 'normal' fit set)."""
    parts = []
    if 'ACDC' in args.fit_datasets:
        imgs, _ = dl.load_acdc_ed_es_data(args.acdc_dir)
        print(f"[derisk] fit ACDC NOR: {len(imgs)} frames")
        parts.append(imgs)
    if 'MM' in args.fit_datasets:
        imgs, _ = dl.load_mm_ed_es_data(args.mm_dir, args.mm_csv)
        print(f"[derisk] fit MM   NOR: {len(imgs)} frames")
        parts.append(imgs)
    frames = np.concatenate(parts, axis=0)
    if args.max_fit and len(frames) > args.max_fit:
        rng = np.random.RandomState(0)
        frames = frames[rng.permutation(len(frames))[:args.max_fit]]
        print(f"[derisk] fit set capped to {len(frames)} frames (--max_fit)")
    return frames


def load_val(args):
    """Validation ED/ES frames (NOR + disease) with labels / pids / slice idx / dataset."""
    imgs, labels, pids, slcs, dsids = [], [], [], [], []
    if 'ACDC' in args.val_datasets:
        (v1, _v2, vl, vp, vs, t1, _t2, tl, tp, ts) = dl.load_acdc_test_val_ed_es_data(args.acdc_dir)
        if len(t1) > 0:                                    # full ACDC test set (val + test halves)
            v1 = np.concatenate([v1, t1], axis=0)
            vl = list(vl) + list(tl); vp = list(vp) + list(tp); vs = list(vs) + list(ts)
        vp = [f"ACDC_{p}" for p in vp]
        imgs.append(v1); labels += list(vl); pids += vp; slcs += list(vs)
        dsids += ['ACDC'] * len(v1)
        print(f"[derisk] val ACDC: {len(v1)} frames, {len(set(vp))} patients")
    if 'MM' in args.val_datasets:
        (m1, _m2, ml, mp, ms) = dl.load_mm_validation_ed_es_data(args.mm_val_dir, args.mm_csv)
        mp = [f"MM_{p}" for p in mp]
        imgs.append(m1); labels += list(ml); pids += mp; slcs += list(ms)
        dsids += ['MM'] * len(m1)
        print(f"[derisk] val MM:   {len(m1)} frames")
    frames = np.concatenate(imgs, axis=0)
    return frames, np.array(labels), np.array(pids), np.array(slcs), np.array(dsids)


# ── Shared scoring + AUC report (both preprocessing paths) ───────────────────
def _score_and_report(args, fit_feats, val_feats, val_labels, val_pids, val_ds,
                      patient_agg, agg_names, unit, extra_meta, save_extra):
    """Fit Mahalanobis + kNN on NOR features, score val, print/save AUCs.

    `unit` is 'FRAME' (legacy 2-D slices) or 'STACK' (faithful 3-D stacks) and
    only affects labels. `patient_agg(scores) -> (dict_of_agg_arrays, plabels)`
    supplies the path-specific patient aggregation; `agg_names` names the
    aggregators to report. `extra_meta` / `save_extra` add path-specific fields
    to the JSON / npz.
    """
    print(f"[derisk] feature dim = {fit_feats.shape[1]} "
          f"(fit {fit_feats.shape[0]}, val {val_feats.shape[0]})")
    unit_key = unit.lower()

    # fit training-free normal models on NOR features
    scaler = StandardScaler().fit(fit_feats)
    Ftr, Fva = scaler.transform(fit_feats), scaler.transform(val_feats)
    if args.pca > 0:
        n_comp = min(args.pca, Ftr.shape[1], Ftr.shape[0])
        pca = PCA(n_components=n_comp).fit(Ftr)
        Ftr, Fva = pca.transform(Ftr), pca.transform(Fva)
        print(f"[derisk] PCA -> {n_comp} dims")

    lw = LedoitWolf().fit(Ftr)
    maha = lw.mahalanobis(Fva)                        # squared Mahalanobis distance
    k = min(args.knn_k, len(Ftr))
    nn = NearestNeighbors(n_neighbors=k).fit(Ftr)
    dist, _ = nn.kneighbors(Fva)
    knn = dist.mean(axis=1)                           # mean distance to k nearest normals

    results = {'feature_dim': int(fit_feats.shape[1]),
               'n_fit': int(fit_feats.shape[0]), 'n_val': int(val_feats.shape[0]),
               'feature_layers': args.feature_layers,
               'cinema_preproc': args.cinema_preproc, 'unit': unit_key,
               'scorers': {}}
    results.update(extra_meta)
    scorers = {'mahalanobis': maha, 'knn': knn}

    print("\n" + "=" * 68)
    print("RESULTS  (AUC: NOR vs disease; compare patient-level Mean to Flow-SSIM ~0.73-0.77)")
    print("=" * 68)
    for name, sc in scorers.items():
        entry = {unit_key: one_vs_nor_aucs(sc, val_labels), 'patient': {}}
        print(f"\n### scorer = {name}")
        fr = entry[unit_key]
        print(f"  [{unit}]  overall={fr.get('overall', float('nan')):.4f}  "
              + "  ".join(f"{d}={fr[d]:.4f}" for d in sorted(fr) if d != 'overall'))
        pat, plabels = patient_agg(sc)
        for agg in agg_names:
            au = one_vs_nor_aucs(pat[agg], plabels)
            entry['patient'][agg] = au
            print(f"  [PAT-{agg:<10}] overall={au.get('overall', float('nan')):.4f}  "
                  + "  ".join(f"{d}={au[d]:.4f}" for d in sorted(au) if d != 'overall'))
        # per-dataset overall (unit level)
        entry['per_dataset'] = {}
        for ds in sorted(set(val_ds)):
            m = (val_ds == ds)
            au = one_vs_nor_aucs(sc[m], val_labels[m])
            entry['per_dataset'][ds] = au
            if 'overall' in au:
                print(f"  [{unit}-{ds}] overall={au['overall']:.4f}")
        results['scorers'][name] = entry

    # save
    with open(os.path.join(args.out_dir, 'derisk_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, 'derisk_arrays.npz'),
                        fit_feats=fit_feats, val_feats=val_feats,
                        maha=maha, knn=knn,
                        val_labels=val_labels, val_pids=val_pids, val_ds=val_ds,
                        **save_extra)
    print(f"\n[derisk] wrote {args.out_dir}/derisk_results.json and derisk_arrays.npz")
    print("[derisk] DONE. If patient-level Mean AUC >~ 0.73, the reverse-distillation build is justified.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # paths (defaults mirror run_model.py / submit_job.pbs)
    ap.add_argument('--acdc_dir',   default='../Dataset_2')
    ap.add_argument('--mm_dir',     default='../Dataset_1/Training')
    ap.add_argument('--mm_val_dir', default='../Dataset_1/Validation')
    ap.add_argument('--mm_csv',
                    default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    ap.add_argument('--fit_datasets', nargs='+', default=['ACDC', 'MM'], choices=['ACDC', 'MM'])
    ap.add_argument('--val_datasets', nargs='+', default=['ACDC', 'MM'], choices=['ACDC', 'MM'])
    ap.add_argument('--out_dir', default='./derisk_out')

    # preprocessing (mirror run_model.py; default OFF so the job is turnkey)
    ap.add_argument('--orient_normalize', action='store_true')
    ap.add_argument('--orient_params', default=None)
    ap.add_argument('--spacing_normalize', action='store_true')
    ap.add_argument('--target_spacing', type=float, default=1.5)
    ap.add_argument('--target_size', type=int, default=128)
    ap.add_argument('--recon_spacing', type=float, default=2.0)
    ap.add_argument('--n4_bias_correct', action='store_true')
    ap.add_argument('--n4_shrink', type=int, default=4)
    ap.add_argument('--n4_iterations', type=int, default=50)
    ap.add_argument('--n4_levels', type=int, default=4)

    # frozen backbone: CineMA (default) or a generic 2-D timm ViT (DINOv2 / MAE).
    # dino/mae are 2-D single-frame encoders -> only the legacy 2-D frame path applies
    # (faithful is CineMA's 3-D SAX pipeline). Both are ViT-B/768-d so scorers are unchanged.
    ap.add_argument('--backbone', choices=['cinema', 'dino', 'mae'], default='cinema',
                    help='Frozen feature extractor. cinema = CineMA (default, unchanged); '
                         'dino/mae = 2-D timm ViT (forces --cinema_preproc legacy).')
    ap.add_argument('--dino_model', default='vit_base_patch14_dinov2.lvd142m')
    ap.add_argument('--mae_model', default='vit_base_patch16_224.mae')

    # CineMA feature extraction
    ap.add_argument('--sax_fill', choices=['zero', 'replicate'], default='zero',
                    help='How to build the 16-slice SAX depth stack from one 2-D frame. '
                         '"zero" matches CineMA\'s example; "replicate" repeats the slice.')
    ap.add_argument('--feature_layers', choices=['all', 'last'], default='all',
                    help='Use all feature_forward() tensors (concat) or only the deepest.')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')

    # scoring
    ap.add_argument('--knn_k', type=int, default=5)
    ap.add_argument('--pca', type=int, default=0, help='PCA dims before scoring (0 = off).')
    ap.add_argument('--max_fit', type=int, default=0, help='Cap NOR-fit frames (0 = all).')

    ap.add_argument('--cinema_preproc', choices=['faithful', 'legacy'], default='faithful',
                    help='faithful = CineMA-canonical 3-D SAX stacks (resample to 1mm -> '
                         'LV-bbox center-crop 192 -> clip 0.95/99.5 -> depth-16), read straight '
                         'from the raw ACDC/M&Ms files via cinema_faithful.py; legacy = single '
                         '2-D ICCV frame (data_loader.py) resized to 192 with zero/replicate '
                         'depth padding. faithful reads CineMA features on-distribution; the '
                         'orientation/spacing/N4 flags apply to the legacy path only.')
    ap.add_argument('--adapter_path', default=None,
                    help='Path to a cinema_lora.py adapter checkpoint (.pt). If set, LoRA is '
                         'injected into the CineMA encoder and its weights loaded right after '
                         'from_pretrained(), so BOTH preprocessing paths extract features from '
                         'the NOR-adapted encoder. Default None = frozen pretrained encoder.')
    args = ap.parse_args()

    # 2-D timm encoders can't consume CineMA's 3-D faithful stacks -> force legacy frames.
    if args.backbone != 'cinema' and args.cinema_preproc != 'legacy':
        print(f"[derisk] backbone={args.backbone} is a 2-D encoder; forcing --cinema_preproc legacy")
        args.cinema_preproc = 'legacy'

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    dtype = torch.float32
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    print(f"[derisk] device={device}, dtype={dtype}")

    # frozen backbone: CineMA (both paths) or a 2-D timm ViT (legacy path only).
    _timm_cfg = None
    if args.backbone == 'cinema':
        print("\n=== Loading frozen CineMA (from_pretrained) ===")
        model = CineMA.from_pretrained()
        if args.adapter_path:
            import cinema_lora as lora
            meta = lora.apply_adapter(model, args.adapter_path)
            print(f"[derisk] LoRA-adapted encoder loaded from {args.adapter_path} (meta={meta})")
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model, _timm_cfg = load_timm_backbone(args, device)

    def _extract(frames):
        if args.backbone == 'cinema':
            return extract_features(model, frames, device, dtype, args)
        return extract_features_timm(model, frames, device, dtype, args, _timm_cfg)

    from collections import Counter

    if args.cinema_preproc == 'faithful':
        # CineMA-canonical 3-D SAX stacks, read straight from the raw files.
        import cinema_faithful as cf
        print("\n=== CineMA-faithful preprocessing (canonical 3-D SAX stacks) ===")
        print("--- Building NOR fit records ---")
        fit_records = cf.build_fit_records(args)
        if args.max_fit and len(fit_records) > args.max_fit:
            rng = np.random.RandomState(0)
            idx = rng.permutation(len(fit_records))[:args.max_fit]
            fit_records = [fit_records[i] for i in idx]
            print(f"[derisk] fit set capped to {len(fit_records)} stacks (--max_fit)")
        print("--- Building validation records ---")
        val_records = cf.build_val_records(args)
        print(f"[derisk] val label counts: "
              f"{dict(Counter(r['label'] for r in val_records))}")

        print("\n=== Extracting features: NOR fit set ===")
        fit_feats, *_ = cf.extract_records_features(
            model, fit_records, device, dtype, args.batch_size, args.feature_layers)
        print("=== Extracting features: validation set ===")
        val_feats, val_pids, val_labels, val_ds, val_phases = cf.extract_records_features(
            model, val_records, device, dtype, args.batch_size, args.feature_layers)

        _score_and_report(
            args, fit_feats, val_feats, val_labels, val_pids, val_ds,
            patient_agg=lambda sc: aggregate_stacks_to_patient(sc, val_pids, val_labels),
            agg_names=['Mean', 'Max'], unit='STACK',
            extra_meta={}, save_extra={'val_phases': val_phases})
    else:
        # Legacy single-2-D-slice path via the ICCV data_loader.
        configure_loaders(args)
        print("\n=== Loading NOR fit frames ===")
        fit_frames = load_fit_frames(args)
        print("\n=== Loading validation frames ===")
        val_frames, val_labels, val_pids, val_slcs, val_ds = load_val(args)
        print(f"[derisk] val label counts: {dict(Counter(val_labels))}")

        print("\n=== Extracting features: NOR fit set ===")
        fit_feats = _extract(fit_frames)
        print("=== Extracting features: validation set ===")
        val_feats = _extract(val_frames)

        _score_and_report(
            args, fit_feats, val_feats, val_labels, val_pids, val_ds,
            patient_agg=lambda sc: aggregate_to_patient(sc, val_pids, val_slcs, val_labels),
            agg_names=_AGGS, unit='FRAME',
            extra_meta={'sax_fill': args.sax_fill}, save_extra={'val_slcs': val_slcs})


if __name__ == "__main__":
    main()
