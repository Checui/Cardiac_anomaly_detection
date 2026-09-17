"""qfae_hybrid_train.py — train a hybrid QFAE (encoder x scorer matrix), NOR-only.

    --encoder dino   --scorer cinema   = Variant A (is DINOv2 a better ENCODER?)
    --encoder cinema --scorer dino     = Variant B (does a vendor-robust JUDGE lift M&Ms?)
    --encoder cinema --scorer cinema   = reproduces the QFAE-CineMA baseline (0.845/0.549)

Reconstructs the 3-D SAX stack either way, so only one slot differs from the baseline.
Reuses cinema_faithful data + qfae_train's LR schedule/augmentation. Saves qfae.pt (trainable only).
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cinema import CineMA
import cinema_faithful as cf
from qfae_train import adjust_lr, augment
from qfae_perceptual import CineMAPerceptual
from qfae_dino import load_dinov2
from qfae_hybrid import (QFormerAE3DHybrid, DinoSliceEncoder, DINOv2Perceptual3D,
                         select_slices, _gather_slices, DEPTH)


def build_nor_data(args):
    """NOR stacks -> TensorDataset(X, valid[, gt_flow]).

    valid = REAL slice count (pre-pad), for the DINOv2 scorer's padded-plane masking.
    With --flow, also carries the GT optical flow (N,3,192,192,16) for the motion head.
    """
    records = cf.build_fit_records_flow(args) if args.flow else cf.build_fit_records(args)
    if not records:
        raise RuntimeError("no NOR fit records — check dataset paths / fit_datasets")
    X = np.stack([cf.stack_to_tensor(r["stack"]) for r in records]).astype(np.float32)
    valid = np.array([min(DEPTH, r["stack"].shape[2]) for r in records], dtype=np.int64)
    print(f"[hybrid] {X.shape[0]} NOR stacks {tuple(X.shape)}; real slices "
          f"min={valid.min()} max={valid.max()} mean={valid.mean():.1f}"
          + ("  (+flow)" if args.flow else ""))
    tensors = [torch.from_numpy(X), torch.from_numpy(valid)]
    if args.flow:
        Ff = np.stack([r["gt_flow"] for r in records]).astype(np.float32)
        tensors.append(torch.from_numpy(Ff))
    return TensorDataset(*tensors)


def save_model(model, path):
    """Save trainable weights only — strip the frozen CineMA and the frozen DINOv2 inside dino_enc."""
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if not (k.startswith("cinema.") or k.startswith("dino_enc.dino."))}
    torch.save({"model": state, "arch": model.arch_config()}, path)


def build_models(args, device):
    """Load whichever frozen nets this (encoder, scorer) combination needs."""
    need_cinema = (args.encoder == "cinema") or (args.scorer == "cinema")
    need_dino = (args.encoder == "dino") or (args.scorer == "dino")
    cinema = dino = None
    dmean = dstd = None
    if need_cinema:
        print("[hybrid] loading frozen CineMA")
        cinema = CineMA.from_pretrained().eval().to(device)
        for p in cinema.parameters():
            p.requires_grad_(False)
    if need_dino:
        print(f"[hybrid] loading frozen {args.dino_model}")
        dino, dmean, dstd = load_dinov2(args.dino_model, args.img_size, device)

    dino_enc = DinoSliceEncoder(dino, dmean, dstd).to(device) if args.encoder == "dino" else None
    model = QFormerAE3DHybrid(cinema=cinema, dino_enc=dino_enc, encoder_kind=args.encoder,
                              qformer_blocks=args.qformer_blocks, qformer_heads=args.qformer_heads,
                              decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
                              decoder_heads=args.decoder_heads, flow_head=args.flow).to(device)
    layers = tuple(args.perceptual_layers)
    if args.scorer == "cinema":
        perceptual = CineMAPerceptual(cinema, layers=layers)
    else:
        perceptual = DINOv2Perceptual3D(dino, dmean, dstd, layers=layers)
    print(f"[hybrid] encoder={args.encoder}  scorer={args.scorer}  perceptual_layers={layers}")
    return model, perceptual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_csv", default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--fit_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--out_dir", default="./qfae_hybrid_out")
    # the matrix
    ap.add_argument("--encoder", choices=["cinema", "dino"], default="cinema")
    ap.add_argument("--scorer", choices=["cinema", "dino"], default="cinema")
    ap.add_argument("--dino_model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--flow", action="store_true",
                    help="add the optical-flow motion head (encoder-agnostic; GT = Farneback by default)")
    ap.add_argument("--lambda_flow", type=float, default=2.0, help="weight of the L1 flow loss")
    ap.add_argument("--perceptual_layers", type=int, nargs="+", default=[5, 8, 11])
    ap.add_argument("--perceptual_slices", type=int, default=6,
                    help="DINOv2 scorer only: random VALID slices scored per step (memory bound). "
                         "0 or >=16 = all slices. Eval always uses all valid slices.")
    # model
    ap.add_argument("--qformer_blocks", type=int, default=2)
    ap.add_argument("--qformer_heads", type=int, default=8)
    ap.add_argument("--decoder_dim", type=int, default=512)
    ap.add_argument("--decoder_depth", type=int, default=6)
    ap.add_argument("--decoder_heads", type=int, default=8)
    # optim (mirrors qfae_train.py)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[hybrid] device={device} amp={use_amp}")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Building NOR training data ===")
    dataset = build_nor_data(args)
    if args.smoke:
        dataset = torch.utils.data.Subset(dataset, range(min(4, len(dataset))))
        args.epochs = 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=len(dataset) > args.batch_size, num_workers=2, pin_memory=use_amp)

    print("=== Building models ===")
    model, perceptual = build_models(args, device)
    trainable = model.trainable_parameters()
    print(f"[hybrid] trainable: {sum(p.numel() for p in trainable)/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    k = args.perceptual_slices

    print(f"=== Training {args.epochs} epochs on {len(dataset)} stacks ({len(loader)} steps/ep) ===")
    history, t0 = [], time.time()
    for epoch in range(args.epochs):
        model.train()
        lr = adjust_lr(opt, epoch, args)
        losses, appe_l, flow_l = [], [], []
        for batch in loader:
            if args.flow:
                x, valid, gt_flow = batch
                x = x.to(device, non_blocking=True)             # no augment in flow mode (flips break flow)
                gt_flow = gt_flow.to(device, non_blocking=True)
            else:
                x, valid = batch
                x = augment(x).to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                recon, pred_flow = out if args.flow else (out, None)
                if args.scorer == "dino":
                    if 0 < k < x.shape[-1]:                     # subsample slices to bound memory
                        xs, idx = select_slices(x, valid, k)
                        rs = _gather_slices(recon, idx)
                        vv = torch.full((x.shape[0],), k, device=device)
                        la = perceptual.loss(xs, rs, valid=vv)
                    else:
                        la = perceptual.loss(x, recon, valid=valid)
                else:
                    la = perceptual.loss(x, recon)              # CineMA scorer (baseline behaviour)
                if args.flow:
                    lf = F.l1_loss(pred_flow.float(), gt_flow.float())
                    loss = la + args.lambda_flow * lf
                else:
                    lf = torch.zeros((), device=device)
                    loss = la
            if not torch.isfinite(loss):
                print(f"[hybrid] non-finite loss at epoch {epoch}; skipping step")
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item()); appe_l.append(la.item()); flow_l.append(lf.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        m_appe = float(np.mean(appe_l)) if appe_l else float("nan")
        m_flow = float(np.mean(flow_l)) if flow_l else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss, "appe": m_appe, "flow": m_flow, "lr": lr})
        print(f"[hybrid] epoch {epoch+1}/{args.epochs} loss={mean_loss:.5f} "
              f"(appe={m_appe:.4f} flow={m_flow:.4f}) lr={lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_model(model, os.path.join(args.out_dir, "qfae.pt"))
            with open(os.path.join(args.out_dir, "qfae_loss.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=2)

    print(f"[hybrid] DONE -> {args.out_dir}/qfae.pt | loss {history[0]['loss']:.4f} -> "
          f"{history[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
