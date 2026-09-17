"""derisk_dino_sweep.py — DINOv2 feature-readout sweep for the encoder-swap probe.

The first probe read DINOv2 the crudest way (single last-layer, patch-GAP, 224px) and
still crossed the M&Ms vendor gap (patient-Mean M&Ms 0.667 — the first APPEARANCE encoder
to do so; ACDC collapsed to 0.55). This sweeps the readout to see whether 0.667 is a floor
and whether any config recovers ACDC, BEFORE committing to a full DINOv2-QFAE build:

    resolution : 224 (forced) vs 518 (DINOv2 native)
    layer      : block outputs {2, 5, 8, 11}  (early / mid / late)
    token      : CLS  vs  patch-GAP  vs  multi-layer GAP-concat (last 4)

One forward pass per resolution captures all layers (forward hooks on model.blocks),
then every (layer, token) matrix goes through the SAME scorer as derisk_cinema.py
(StandardScaler -> Ledoit-Wolf Mahalanobis + kNN) and patient-Mean per-dataset AUC.
Reuses derisk_cinema's legacy loaders so the frames/split match the previous probe exactly.

GPU job (login node is contended); weights are pre-cached offline in $HF_HOME.
"""

import os
import json
import argparse

import numpy as np
import cv2
import torch
import timm

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

import derisk_cinema as D   # reuse legacy loaders (configure_loaders / load_fit_frames / load_val)

LAYERS = [2, 5, 8, 11]      # DINOv2 ViT-B has 12 blocks (0..11)


class LayerCapture:
    """Forward hooks on model.blocks[i] -> capture each block's (B, N, C) token output."""
    def __init__(self, model, layers):
        self.store, self.hooks = {}, []
        for li in layers:
            self.hooks.append(model.blocks[li].register_forward_hook(self._mk(li)))

    def _mk(self, li):
        def hook(_m, _inp, out):
            self.store[li] = out
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()


def extract(model, frames, layers, n_prefix, size, mean, std, device, dtype, bs):
    """-> {layer: {'cls': (N,768), 'gap': (N,768)}} from ONE forward pass per batch."""
    cap = LayerCapture(model, layers)
    acc = {li: {'cls': [], 'gap': []} for li in layers}
    n = len(frames)
    try:
        for start in range(0, n, bs):
            chunk = frames[start:start + bs]
            batch = np.stack([cv2.resize(f.astype(np.float32), (size, size),
                                         interpolation=cv2.INTER_LINEAR) for f in chunk])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
            x = (x - mean) / std
            use_amp = (device.type == 'cuda' and dtype != torch.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                model(x)                                   # triggers hooks; output ignored
            for li in layers:
                t = cap.store[li].float()
                acc[li]['cls'].append(t[:, 0, :].cpu().numpy())
                acc[li]['gap'].append(t[:, n_prefix:, :].mean(dim=1).cpu().numpy())
            print(f"\r[sweep] {min(start + bs, n)}/{n}", end="", flush=True)
        print()
    finally:
        cap.remove()
    return {li: {k: np.concatenate(v, 0) for k, v in d.items()} for li, d in acc.items()}


def score(fitF, valF, knn_k=5):
    sc = StandardScaler().fit(fitF)
    a, b = sc.transform(fitF), sc.transform(valF)
    lw = LedoitWolf().fit(a)
    maha = lw.mahalanobis(b)
    nn = NearestNeighbors(n_neighbors=knn_k).fit(a)
    dd, _ = nn.kneighbors(b)
    return {'maha': maha, 'knn': dd.mean(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acdc_dir', default='../Dataset_2')
    ap.add_argument('--mm_dir', default='../Dataset_1/Training')
    ap.add_argument('--mm_val_dir', default='../Dataset_1/Validation')
    ap.add_argument('--mm_csv',
                    default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    ap.add_argument('--dino_model', default='vit_base_patch14_dinov2.lvd142m')
    ap.add_argument('--resolutions', type=int, nargs='+', default=[224, 518])
    ap.add_argument('--knn_k', type=int, default=5)
    ap.add_argument('--out_json', default='derisk_out_dino/dino_sweep.json')
    args = ap.parse_args()
    # fields the legacy loaders expect (all preproc OFF -> matches the previous probe feed)
    for k, v in dict(orient_normalize=False, spacing_normalize=False, n4_bias_correct=False,
                     target_spacing=1.5, target_size=128, recon_spacing=2.0, n4_shrink=4,
                     n4_iterations=50, n4_levels=4, orient_params=None, max_fit=0,
                     fit_datasets=['ACDC', 'MM'], val_datasets=['ACDC', 'MM']).items():
        setattr(args, k, v)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"[sweep] device={device} dtype={dtype}")

    D.configure_loaders(args)
    print("=== loading legacy 2-D frames (same feed as the previous probe) ===")
    fit_frames = D.load_fit_frames(args)
    val_frames, val_labels, val_pids, _val_slcs, val_ds = D.load_val(args)
    print(f"[sweep] fit={len(fit_frames)} val={len(val_frames)} "
          f"patients={len(np.unique(val_pids))}")

    # patient-Mean per-dataset AUC (aligned to val_pids/labels/ds)
    uniq = np.unique(val_pids)
    grp = [val_pids == p for p in uniq]
    plab = np.array([val_labels[m][0] for m in grp])
    pds = np.array([val_ds[m][0] for m in grp])

    def pat_perds(frame_scores):
        pm = np.array([frame_scores[m].mean() for m in grp])
        def auc(mask):
            y = (plab[mask] != 'NOR').astype(int)
            return float(roc_auc_score(y, pm[mask])) if y.min() != y.max() else float('nan')
        allm = np.ones(len(pm), bool)
        return auc(allm), auc(pds == 'ACDC'), auc(pds == 'MM')

    rows = []
    print(f"\n{'config':22s}{'scorer':6s}{'overall':>9s}{'ACDC':>8s}{'MM':>8s}")
    print("-" * 53)

    def emit(tag, fitF, valF):
        s = score(fitF, valF, args.knn_k)
        for scorer in ('maha', 'knn'):
            o, ac, mm = pat_perds(s[scorer])
            rows.append(dict(config=tag, scorer=scorer, overall=o, ACDC=ac, MM=mm))
            print(f"{tag:22s}{scorer:6s}{o:>9.3f}{ac:>8.3f}{mm:>8.3f}")

    for res in args.resolutions:
        print(f"\n=== DINOv2 @ {res}px ===")
        model = timm.create_model(args.dino_model, pretrained=True, num_classes=0, img_size=res)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        cfg = timm.data.resolve_model_data_config(model)
        n_prefix = getattr(model, 'num_prefix_tokens', 1)
        mean = torch.tensor(cfg['mean'], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg['std'], device=device).view(1, 3, 1, 1)
        bs = 32 if res <= 224 else 12
        print(f"[sweep] n_prefix={n_prefix} bs={bs}; extracting layers {LAYERS}")
        fit = extract(model, fit_frames, LAYERS, n_prefix, res, mean, std, device, dtype, bs)
        val = extract(model, val_frames, LAYERS, n_prefix, res, mean, std, device, dtype, bs)
        for li in LAYERS:
            for readout in ('cls', 'gap'):
                emit(f"L{li}-{readout}-{res}", fit[li][readout], val[li][readout])
        fitM = np.concatenate([fit[li]['gap'] for li in LAYERS], axis=1)
        valM = np.concatenate([val[li]['gap'] for li in LAYERS], axis=1)
        emit(f"L{LAYERS}-gapcat-{res}", fitM, valM)
        del model, fit, val
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    best = max((r for r in rows if not np.isnan(r['MM'])), key=lambda r: r['MM'])
    print("\n" + "=" * 53)
    print(f"BEST M&Ms: {best['config']} ({best['scorer']}) "
          f"MM={best['MM']:.3f}  ACDC={best['ACDC']:.3f}  overall={best['overall']:.3f}")
    print("prev crude readout (L-last gap @224): MM 0.667 / ACDC 0.55 ; benchmarks CineMA-faithful "
          "MM 0.59 / ACDC 0.76 ; motion(flow) MM 0.716")
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump({'model': args.dino_model, 'rows': rows, 'best_MM': best}, f, indent=2)
    print(f"[sweep] wrote {args.out_json}")


if __name__ == '__main__':
    main()
