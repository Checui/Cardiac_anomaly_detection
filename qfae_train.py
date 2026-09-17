"""qfae_train.py — train the cardiac Q-Former Autoencoder (Phase 2).

Trains ONLY the Q-Former + 3-D decoder (CineMA frozen) to reconstruct healthy (NOR)
SAX stacks, minimising the CineMA perceptual loss (qfae_perceptual.py). NOR-only, so it
stays an unsupervised anomaly detector. Loop mirrors adapt_cinema.py. Data reuses
cinema_faithful's canonical-SAX stacks. Saves qfae_out/qfae.pt (Q-Former+decoder only).
"""

import os
import json
import math
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cinema import CineMA
import cinema_faithful as cf
from qfae_cinema import QFormerAE
from qfae_perceptual import make_scorer
import qfae_masking as qm


def real_depth(x):
    """(B,1,H,W,D) -> (B,) count of non-padded depth planes.

    cinema_faithful pads short stacks with zeros at the END (SpatialPadd method="end"), so the
    real planes are the leading ones — which is the convention DINOv2Perceptual3D's `valid`
    expects. Derived from the tensor so no change to the data pipeline is needed.
    """
    return (x.abs().amax(dim=(1, 2, 3)) > 0).sum(dim=1)


def build_nor_data(args):
    """NOR training data -> TensorDataset: (X,) or (X, gt_flow) when --flow."""
    if args.flow:
        records = cf.build_fit_records_flow(args)
        if not records:
            raise RuntimeError("no NOR flow records — check dataset paths / fit_datasets")
        X = np.stack([cf.stack_to_tensor(r["stack"]) for r in records]).astype(np.float32)
        Ff = np.stack([r["gt_flow"] for r in records]).astype(np.float32)       # (N,3,192,192,16)
        print(f"[qfae] built {len(records)} flow records; X{tuple(X.shape)} flow{tuple(Ff.shape)}")
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(Ff))
    records = cf.build_fit_records_allframes(args) if args.all_frames else cf.build_fit_records(args)
    if not records:
        raise RuntimeError("no NOR fit records — check dataset paths / fit_datasets")
    X = np.stack([cf.stack_to_tensor(r["stack"]) for r in records]).astype(np.float32)
    print(f"[qfae] built {X.shape[0]} NOR stacks, tensor {tuple(X.shape)}")
    return TensorDataset(torch.from_numpy(X))


def augment(batch):
    if torch.rand(()) < 0.5:
        batch = torch.flip(batch, dims=[2])
    if torch.rand(()) < 0.5:
        batch = torch.flip(batch, dims=[3])
    return batch


def adjust_lr(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    else:
        prog = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def save_model(model, path):
    """Save only the trainable Q-Former + decoder (CineMA is from_pretrained, not saved)."""
    state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("cinema.")}
    torch.save({"model": state, "arch": model.arch_config()}, path)


def sample_mask(args, x, gt_flow, device):
    """Per-batch encoder mask, or None when masking is off (the unmasked baseline path)."""
    if args.mask == "none":
        return None
    n_patches = qm.n_patches_of()
    if args.mask == "random":
        return qm.random_patch_mask(x.shape[0], n_patches, args.mask_ratio, device)
    return qm.motion_guided_patch_mask(gt_flow, args.mask_ratio, alpha=args.mask_alpha,
                                       temperature=args.mask_temperature)


def masked_flow_loss(pred_flow, gt_flow, mask, args):
    """L1 flow loss, optionally restricted to voxels the encoder could not see (MME).

    Weighting by the masked region is the whole point of --mask_loss_only: it scores the
    head on *inpainting* motion rather than copying visible motion through.
    """
    if not (args.mask_loss_only and mask is not None):
        return F.l1_loss(pred_flow, gt_flow)
    w = qm.upsample_mask_to_voxels(mask).to(pred_flow.dtype)     # (B,1,H,W,D), 1 on masked
    denom = w.sum() * pred_flow.shape[1]
    if denom == 0:                       # mask_ratio 0 -> fall back to the dense loss
        return F.l1_loss(pred_flow, gt_flow)
    return ((pred_flow - gt_flow).abs() * w).sum() / denom


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_csv", default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--fit_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--all_frames", action="store_true")
    ap.add_argument("--flow", action="store_true",
                    help="Dual-stream: add the optical-flow head + L1 flow loss (needs paired "
                         "ED/ES records with GT Farneback flow). Augmentation is disabled in this "
                         "mode (spatial flips would need flow-vector sign handling).")
    ap.add_argument("--lambda_flow", type=float, default=2.0, help="Weight of the L1 flow loss (GAN lw_aux).")
    ap.add_argument("--single_pass", action="store_true",
                    help="ED-only single pass: input ED, reconstruct ED, predict ED->ES flow (1 stack/patient).")
    ap.add_argument("--flow_backend", choices=["farneback", "registration"], default="farneback",
                    help="GT-flow source for the motion head: Farneback, or the fine-tuned "
                         "biomechanics Registration_Net teacher (--reg_repo).")
    ap.add_argument("--reg_repo", default="../biomechanics-cardiac-motion-hpc",
                    help="Reg-net repo (network.py + checkpoints/ckpt_best.pth) for --flow_backend registration.")
    ap.add_argument("--out_dir", default="./qfae_out")
    # model
    ap.add_argument("--qformer_blocks", type=int, default=2)
    ap.add_argument("--qformer_heads", type=int, default=8)
    ap.add_argument("--decoder_dim", type=int, default=512)
    ap.add_argument("--decoder_depth", type=int, default=6)
    ap.add_argument("--decoder_heads", type=int, default=8)
    ap.add_argument("--perceptual_layers", type=int, nargs="+", default=[5, 8, 11])
    ap.add_argument("--scorer", default="cinema",
                    help="Perceptual scoring/loss network, for the encoder x scorer matrix. "
                         "'cinema' (default, = the historical coupled behaviour) or ANY timm model "
                         "name, e.g. vit_base_patch16_224.mae / vit_base_patch14_dinov2.lvd142m — "
                         "applied per depth-slice to the stack, padded planes excluded.")
    # masking (qfae_masking.py). Off by default -> reproduces the unmasked baseline exactly.
    ap.add_argument("--mask", choices=["none", "random", "motion"], default="none",
                    help="Mask patches in the frozen encoder. 'random' = uniform tubes "
                         "(control); 'motion' = MGMAE-style, sampled from the GT flow "
                         "magnitude so masks land on moving tissue (needs --flow).")
    ap.add_argument("--mask_ratio", type=float, default=0.5,
                    help="Fraction of the 2304 SAX patches hidden from the encoder.")
    ap.add_argument("--mask_alpha", type=float, default=1.0,
                    help="--mask motion only: 0 = uniform, 1 = fully motion-driven.")
    ap.add_argument("--mask_temperature", type=float, default=1.0,
                    help="--mask motion only: <1 sharpens the magnitude weighting, >1 softens.")
    ap.add_argument("--mask_loss_only", action="store_true",
                    help="Restrict the flow loss to masked voxels (the MME objective: score "
                         "the head on inpainting motion it could not see).")
    # optim
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=25)
    args = ap.parse_args()

    # Fail before loading CineMA / building data, not with a NoneType deref 10 minutes in.
    if args.mask == "motion" and not args.flow:
        ap.error("--mask motion needs --flow: the mask is sampled from the GT flow magnitude.")
    if args.mask_loss_only and not args.flow:
        ap.error("--mask_loss_only weights the flow loss, which only exists under --flow.")
    if args.mask != "none" and not 0.0 < args.mask_ratio < 1.0:
        ap.error(f"--mask_ratio must be in (0, 1) when masking, got {args.mask_ratio}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[qfae] device={device}, amp={use_amp}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.flow:
        cf.set_flow_backend(args.flow_backend, reg_repo=args.reg_repo)
    cf.set_single_pass(args.single_pass)
    print("=== Building NOR training data ===")
    dataset = build_nor_data(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=len(dataset) > args.batch_size, num_workers=2, pin_memory=use_amp)

    print("=== Loading CineMA + building QFAE ===")
    cinema = CineMA.from_pretrained()
    model = QFormerAE(cinema, qformer_blocks=args.qformer_blocks, qformer_heads=args.qformer_heads,
                      decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
                      decoder_heads=args.decoder_heads, flow_head=args.flow).to(device)
    perceptual, scorer_desc = make_scorer(args.scorer, "3d", cinema=cinema,
                                          layers=tuple(args.perceptual_layers), device=device)
    print(f"[qfae] encoder = CineMA (3-D depth-16) | perceptual scorer = {scorer_desc}")
    trainable = model.trainable_parameters()
    n_tr = sum(p.numel() for p in trainable)
    print(f"[qfae] trainable (Q-Former+decoder): {n_tr/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print(f"=== Training {args.epochs} epochs on {len(dataset)} samples ({len(loader)} steps/epoch) ===")
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        lr = adjust_lr(opt, epoch, args)
        losses, appe_l, flow_l = [], [], []
        for batch in loader:
            if args.flow:
                x, gt_flow = batch
                x = x.to(device, non_blocking=True); gt_flow = gt_flow.to(device, non_blocking=True)
            else:
                (x,) = batch
                x = augment(x).to(device, non_blocking=True)
            mask = sample_mask(args, x, gt_flow if args.flow else None, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x, mask=mask)
                valid = real_depth(x)          # ignored by the CineMA scorer, used by 2-D scorers
                if args.flow:
                    recon, pred_flow = out
                    la = perceptual.loss(x, recon, valid=valid)
                    lf = masked_flow_loss(pred_flow, gt_flow, mask, args)
                    loss = la + args.lambda_flow * lf
                else:
                    recon = out
                    la = perceptual.loss(x, recon, valid=valid)
                    lf = torch.zeros((), device=device)
                    loss = la
            if not torch.isfinite(loss):
                print(f"[qfae] non-finite loss at epoch {epoch}; skipping step")
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item()); appe_l.append(la.item()); flow_l.append(lf.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        m_appe = float(np.mean(appe_l)) if appe_l else float("nan")
        m_flow = float(np.mean(flow_l)) if flow_l else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss, "appe": m_appe, "flow": m_flow, "lr": lr})
        print(f"[qfae] epoch {epoch+1}/{args.epochs} loss={mean_loss:.5f} "
              f"(appe={m_appe:.4f} flow={m_flow:.4f}) lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_model(model, os.path.join(args.out_dir, "qfae.pt"))
            with open(os.path.join(args.out_dir, "qfae_loss.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=2)

    print(f"[qfae] DONE. model -> {os.path.join(args.out_dir, 'qfae.pt')}. "
          f"loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}. Next: qsub qfae_eval.pbs.")


if __name__ == "__main__":
    main()
