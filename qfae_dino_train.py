"""qfae_dino_train.py — train the 2-D DINOv2 Q-Former Autoencoder (NOR-only).

2-D sibling of qfae_train.py: trains ONLY the Q-Former + 2-D decoder (DINOv2 frozen) to
reconstruct healthy (NOR) 224x224 frames, minimising the multi-layer DINOv2 perceptual loss
(qfae_dino.DINOv2Perceptual). Reuses the exact legacy 2-D frame loaders the encoder-swap
probe used (derisk_cinema.load_fit_frames) + the training utilities from qfae_train.py.
Saves qfae_dino_out/qfae.pt (Q-Former + decoder only; DINOv2 is from_pretrained, not saved).
"""

import os
import json
import time
import argparse

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
import derisk_cinema as D                      # legacy 2-D frame loaders
import cinema_faithful as cf                   # CineMA-faithful 2-D per-slice exploder (--flow)
from qfae_train import adjust_lr, augment      # reuse LR schedule + flip augmentation
from qfae_dino import QFormerAE2D, load_dinov2
from qfae_perceptual import make_scorer        # encoder x scorer matrix (coupled / timm / cinema)


def resolve_scorer_spec(args):
    """Resolve the scorer to build: --scorer wins, else map the legacy --perceptual_backbone.

    Legacy vocabulary was two-valued and misleading: 'dino' meant "reuse the encoder" (whatever
    it is — the mae224 runs used it with an MAE encoder), 'mae' meant "load --mae_model as a
    separate scorer". Both map onto the new spec so old command lines reproduce exactly.
    """
    if args.scorer:
        return args.scorer
    return "coupled" if args.perceptual_backbone == "dino" else args.mae_model


def frames_to_tensor(frames, size=224):
    """(N,H,W,3) in [0,1] -> (N,3,size,size) float32 in [0,1]."""
    out = np.stack([cv2.resize(f.astype(np.float32), (size, size),
                               interpolation=cv2.INTER_LINEAR) for f in frames])
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


# ── GAN-style PIXEL-SPACE appearance (alternative to the embedding/perceptual measure) ──
# The GAN's loss_appe = MSE + intensity-gradient-difference between recon and input frame,
# scored the same way at inference. This lets the 2-D QFAE appearance stream be measured in
# pixel space exactly like the GAN, instead of in the frozen encoder's embedding space.
def _grad_diff_l1(a, b):
    """L1 of the finite-difference spatial gradients (per-sample if reduced later)."""
    dxa = a[:, :, :, 1:] - a[:, :, :, :-1]; dxb = b[:, :, :, 1:] - b[:, :, :, :-1]
    dya = a[:, :, 1:, :] - a[:, :, :-1, :]; dyb = b[:, :, 1:, :] - b[:, :, :-1, :]
    return (dxa - dxb).abs(), (dya - dyb).abs()


def pixel_appe_loss(recon, x):
    """Scalar GAN-style appearance loss: MSE + gradient-difference (mean over batch)."""
    gx, gy = _grad_diff_l1(recon, x)
    return F.mse_loss(recon, x) + gx.mean() + gy.mean()


def pixel_appe_score(recon, x):
    """Per-sample GAN-style appearance anomaly score (N,): MSE + gradient-difference, per slice."""
    se = ((recon - x) ** 2).mean(dim=(1, 2, 3))
    gx, gy = _grad_diff_l1(recon, x)
    return se + gx.mean(dim=(1, 2, 3)) + gy.mean(dim=(1, 2, 3))


def pixel_appe_scores_multi(recon, x):
    """Per-sample pixel appearance scores (higher = more anomalous), for the score-way sweep.

    Returns a dict of (N,) tensors — 'mse', 'mae' (L1), 'msegrad' (MSE + gradient-difference = the
    training loss / pixel_appe_score). SSIM (1-SSIM) is added in eval via skimage. Mirrors the GAN
    notebook's multi-measure appearance sweep, but on the QFAE decoder's pixel reconstruction.
    """
    diff = recon - x
    se = (diff ** 2).mean(dim=(1, 2, 3))
    ae = diff.abs().mean(dim=(1, 2, 3))
    gx, gy = _grad_diff_l1(recon, x)
    return {"mse": se, "mae": ae, "msegrad": se + gx.mean(dim=(1, 2, 3)) + gy.mean(dim=(1, 2, 3))}


def flow_l1_loss(pred, gt, hard_mine_frac=1.0):
    """L1 flow loss, optionally restricted to the hardest fraction of pixels (Dinomaly).

    Same argument as the perceptual hard-mining (qfae_dino.DINOv2Perceptual._reduce): most of
    the frame is background the flow head nails immediately, and continuing to back-prop
    through it drives the head toward reconstructing *any* field handed to it. Note this is
    the loss on the stream that actually carries the cross-vendor signal, so it is the one
    place hard mining can move the M&Ms number.
    """
    e = (pred - gt).abs()
    if hard_mine_frac >= 1.0:
        return e.mean()
    e = e.flatten(1)
    k = max(1, int(round(hard_mine_frac * e.shape[1])))
    return e.topk(k, dim=1).values.mean()


def save_model(model, path):
    """Save only the trainable Q-Former + decoder (DINOv2 is from_pretrained, not saved)."""
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if not (k.startswith("dino.") or k in ("img_mean", "img_std"))}
    torch.save({"model": state, "arch": model.arch_config()}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_csv", default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--fit_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--out_dir", default="./qfae_dino_out")
    # encoder + model
    ap.add_argument("--dino_model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--flow", action="store_true",
                    help="genuinely-2-D motion rebuild: CineMA-faithful per-slice frames, ED single-pass, "
                         "reconstruct the slice + predict its 2-D ED->ES flow. (Off = legacy appearance-only.)")
    ap.add_argument("--lambda_flow", type=float, default=2.0, help="weight of the 2-D L1 flow loss")
    ap.add_argument("--flow_backend", choices=["farneback", "registration"], default="farneback",
                    help="GT source for the 2-D flow head. 'farneback' = optical flow (default); "
                         "'registration' = the biomechanics Registration_Net teacher (needs --reg_repo "
                         "+ torch). Flow mode only. Eval MUST use the same backend.")
    ap.add_argument("--reg_repo", default="../biomechanics-cardiac-motion-hpc",
                    help="Registration teacher repo (network.py + checkpoints/ckpt_best.pth). "
                         "Only used when --flow_backend registration.")
    ap.add_argument("--recover_edes", action="store_true",
                    help="Recover ED/ES from the GT mask for the 25 GE M&Ms-Training cases whose "
                         "CSV has ED==ES==0 (see validate_edes_recovery.py). Off = the historical "
                         "150-patient/38-NOR Philips+Siemens pool. Cannot affect Testing/Validation "
                         "(no ED==ES rows there), so pre/post runs are a strict A/B.")
    ap.add_argument("--lambda_appe", type=float, default=1.0,
                    help="weight of the appearance (perceptual reconstruction) loss. 0.0 = FLOW-ONLY "
                         "(appearance head gets zero gradient; the shared Q-Former serves the flow "
                         "head alone). Flow mode only; ignored for appearance-only runs.")
    ap.add_argument("--perceptual_layers", type=int, nargs="+", default=[5, 8, 11])
    ap.add_argument("--scorer", default=None,
                    help="Scoring/loss network, for the encoder x scorer matrix. 'coupled' = the "
                         "encoder scores itself; 'cinema' = CineMA (2-D slice lifted to a depth-16 "
                         "stack); or ANY timm model name for a separate frozen scorer, e.g. "
                         "vit_base_patch16_224.mae / vit_base_patch14_dinov2.lvd142m. "
                         "Overrides the legacy --perceptual_backbone when given.")
    ap.add_argument("--perceptual_backbone", choices=["dino", "mae"], default="mae",
                    help="DEPRECATED, use --scorer. Kept so existing PBS scripts keep working: "
                         "'dino' == --scorer coupled (the encoder scores itself, whatever it is — "
                         "the name is historical and does NOT mean DINOv2); 'mae' == --scorer "
                         "<--mae_model> (a separate frozen scorer).")
    ap.add_argument("--appe_space", choices=["embedding", "pixel"], default="embedding",
                    help="Appearance loss/score space. 'embedding' = frozen-encoder perceptual loss "
                         "(default). 'pixel' = GAN-style pixel MSE + gradient loss on the recon (no "
                         "perceptual scorer needed); anomaly score then matches the GAN's appearance.")
    ap.add_argument("--mae_model", default="vit_base_patch16_224.mae")
    ap.add_argument("--qformer_blocks", type=int, default=2)
    ap.add_argument("--qformer_heads", type=int, default=8)
    ap.add_argument("--decoder_dim", type=int, default=512)
    ap.add_argument("--decoder_depth", type=int, default=6)
    ap.add_argument("--decoder_heads", type=int, default=8)
    # ── (A) the actual bottleneck ────────────────────────────────────────────────
    ap.add_argument("--decoder_mode", choices=["direct", "perceiver"], default="direct",
                    help="'direct' (default, historical): the latents ARE the decoder grid, so "
                         "n_queries is pinned to grid*grid and the Q-Former compresses NOTHING. "
                         "'perceiver': grid*grid output queries cross-attend to --n_queries "
                         "latents, making the latent count a real bottleneck that can be swept.")
    ap.add_argument("--n_queries", type=int, default=0,
                    help="Q-Former latent count (0 = grid*grid, i.e. no compression). Values "
                         "below grid*grid require --decoder_mode perceiver.")
    # ── (B) Dinomaly tricks (arXiv 2405.14325) ───────────────────────────────────
    ap.add_argument("--qformer_dropout", type=float, default=0.01,
                    help="Dropout inside the Q-Former (self-attention + FFN). Dinomaly's 'noisy "
                         "bottleneck' uses ~0.2; 0.01 is the historical value.")
    ap.add_argument("--attn_dropout", type=float, default=0.0,
                    help="Dropout on the cross-attention weights. 0.0 = the historical behaviour "
                         "(the original port built this dropout but never applied it).")
    ap.add_argument("--linear_attn", action="store_true",
                    help="Use elu+1 linear attention instead of softmax in the Q-Former and the "
                         "Perceiver decoder. Softmax attention can concentrate on one key, which "
                         "is how a reconstruction AE learns the identity shortcut.")
    ap.add_argument("--hard_mine_frac", type=float, default=1.0,
                    help="Back-prop through only this fraction of the hardest points, in BOTH the "
                         "perceptual appearance loss and the flow L1 loss. 1.0 = plain mean.")
    # optim (mirror qfae_train.py)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--max_fit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 short epoch on a tiny subset (sanity).")
    args = ap.parse_args()
    # loader preprocessing OFF (same feed as the probe); fields the legacy loaders expect
    for k, v in dict(orient_normalize=False, spacing_normalize=False, n4_bias_correct=False,
                     target_spacing=1.5, target_size=128, recon_spacing=2.0, n4_shrink=4,
                     n4_iterations=50, n4_levels=4, orient_params=None).items():
        setattr(args, k, v)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[qfae-dino] device={device}, amp={use_amp}")
    os.makedirs(args.out_dir, exist_ok=True)

    # patch/grid derived from the encoder + resolution (patch14 dino / patch16 mae; grid = size/patch)
    patch = 14 if "patch14" in args.dino_model else 16
    grid = args.img_size // patch
    assert grid * patch == args.img_size, f"img_size {args.img_size} not divisible by patch {patch}"

    if args.flow:
        cf.set_flow_backend(args.flow_backend, reg_repo=args.reg_repo)   # farneback (default) or reg-net teacher
        cf.set_edes_recovery(args.recover_edes)   # +25 GE M&Ms-Training cases whose CSV has ED==ES
        print("=== Building NOR training data (CineMA-faithful 2-D per-slice + ED->ES flow) ===")
        frames, flows = cf.build_fit_slices_2d(args, args.img_size)   # (N,3,S,S), (N,3,S,S)
        if args.smoke:
            frames, flows = frames[:16], flows[:16]; args.epochs = 1
        X, Fl = torch.from_numpy(frames), torch.from_numpy(flows)
        print(f"[qfae-dino] {X.shape[0]} NOR slices, frame {tuple(X.shape)} flow {tuple(Fl.shape)}")
        dataset = TensorDataset(X, Fl)
    else:
        print("=== Building NOR training data (legacy 2-D frames, appearance-only) ===")
        D.configure_loaders(args)
        frames = D.load_fit_frames(args)                             # (N,H,W,3) in [0,1]
        if args.smoke:
            frames = frames[:16]; args.epochs = 1
        X = frames_to_tensor(frames, args.img_size)                 # (N,3,224,224)
        print(f"[qfae-dino] {X.shape[0]} NOR frames, tensor {tuple(X.shape)}")
        dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=len(dataset) > args.batch_size, num_workers=2, pin_memory=use_amp)

    print(f"=== Loading frozen {args.dino_model} + building 2-D QFAE ===")
    dino, mean, std = load_dinov2(args.dino_model, args.img_size, device)
    scorer_spec = resolve_scorer_spec(args)
    model = QFormerAE2D(dino, mean, std, grid=grid, patch=patch, flow_head=args.flow,
                        n_queries=(args.n_queries or None),
                        qformer_blocks=args.qformer_blocks,
                        qformer_heads=args.qformer_heads, decoder_dim=args.decoder_dim,
                        decoder_depth=args.decoder_depth, decoder_heads=args.decoder_heads,
                        decoder_mode=args.decoder_mode, qformer_dropout=args.qformer_dropout,
                        attn_dropout=args.attn_dropout, linear_attn=args.linear_attn,
                        encoder_name=args.dino_model,
                        scorer_name=("pixel" if args.appe_space == "pixel" else scorer_spec)).to(device)
    print(f"[qfae-dino] bottleneck: {model.n_queries} latents vs {grid*grid} encoder tokens / "
          f"{grid*grid} decoder patches (decoder_mode={args.decoder_mode}"
          f"{', linear-attn' if args.linear_attn else ''}, dropout={args.qformer_dropout})")
    if args.appe_space == "pixel":
        perceptual = None
        print("[qfae-dino] appearance = PIXEL-space (GAN-style MSE + gradient loss); no perceptual scorer")
    else:
        perceptual, scorer_desc = make_scorer(
            scorer_spec, "2d", coupled=(dino, mean, std), img_size=args.img_size,
            layers=tuple(args.perceptual_layers), device=device,
            hard_mine_frac=args.hard_mine_frac)
        print(f"[qfae-dino] encoder = {args.dino_model} @{args.img_size} | "
              f"perceptual scorer = {scorer_desc}")
    trainable = model.trainable_parameters()
    print(f"[qfae-dino] trainable (Q-Former+decoder): {sum(p.numel() for p in trainable)/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print(f"=== Training {args.epochs} epochs on {len(dataset)} frames ({len(loader)} steps/epoch) ===")
    history, t0 = [], time.time()
    for epoch in range(args.epochs):
        model.train()
        lr = adjust_lr(opt, epoch, args)
        losses, appe_l, flow_l = [], [], []
        for batch in loader:
            if args.flow:
                x, gt_flow = batch
                x = x.to(device, non_blocking=True)             # no augment: flips break flow signs
                gt_flow = gt_flow.to(device, non_blocking=True)
            else:
                (x,) = batch
                x = augment(x).to(device, non_blocking=True)    # flips dims 2,3 = H,W (valid)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                recon, pred_flow = out if args.flow else (out, None)
                la = (pixel_appe_loss(recon.float(), x.float()) if args.appe_space == "pixel"
                      else perceptual.loss(x, recon))
                if args.flow:
                    lf = flow_l1_loss(pred_flow.float(), gt_flow.float(), args.hard_mine_frac)
                    loss = args.lambda_appe * la + args.lambda_flow * lf
                else:
                    lf = torch.zeros((), device=device); loss = args.lambda_appe * la
            if not torch.isfinite(loss):
                print(f"[qfae-dino] non-finite loss at epoch {epoch}; skipping step")
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item()); appe_l.append(la.item()); flow_l.append(lf.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        m_appe = float(np.mean(appe_l)) if appe_l else float("nan")
        m_flow = float(np.mean(flow_l)) if flow_l else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss, "appe": m_appe, "flow": m_flow, "lr": lr})
        print(f"[qfae-dino] epoch {epoch+1}/{args.epochs} loss={mean_loss:.5f} "
              f"(appe={m_appe:.4f} flow={m_flow:.4f}) lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_model(model, os.path.join(args.out_dir, "qfae.pt"))
            with open(os.path.join(args.out_dir, "qfae_loss.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=2)

    print(f"[qfae-dino] DONE. model -> {args.out_dir}/qfae.pt. "
          f"loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}. Next: qsub qfae_dino_eval.pbs.")


if __name__ == "__main__":
    main()
