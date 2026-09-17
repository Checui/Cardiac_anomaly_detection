"""adapt_cinema.py — Phase 1 of Option A: train a NOR-only LoRA adapter for CineMA.

WHAT / WHY
----------
The frozen-CineMA de-risk showed the encoder separates ACDC disease (AUC 0.76) but not
M&Ms (0.59): a cross-dataset domain gap that score-tuning can't close. This adapts the
encoder to our cardiac domain *without disease labels* by continuing CineMA's OWN
self-supervised MAE pretraining (masked-patch reconstruction) on HEALTHY (NOR) SAX
stacks only, while training just a light LoRA adapter (cinema_lora.py) on the ViT encoder
plus the MAE decoder (discarded at inference). Base encoder weights stay frozen.

Then re-run the frozen-feature probe with `derisk_cinema.py --adapter_path` and check
whether M&Ms AUC rises (the Phase-1 gate).

Loop mirrors cinema/mae/pretrain.py (AdamW, bf16 autocast, cosine LR, mask ratio 0.75).
Data reuses cinema_faithful's canonical-SAX preprocessing so the adapter sees exactly the
distribution the probe scores.
"""

import os
import json
import math
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cinema import CineMA
import cinema_faithful as cf
import cinema_lora as lora


def build_nor_tensors(args):
    """NOR training stacks (ACDC + M&Ms) -> (N, 1, 192, 192, 16) float32 tensor."""
    records = cf.build_fit_records_allframes(args) if args.all_frames else cf.build_fit_records(args)
    if not records:
        raise RuntimeError("no NOR fit records built — check dataset paths / fit_datasets")
    arr = np.stack([cf.stack_to_tensor(r["stack"]) for r in records])  # (N,1,192,192,16)
    print(f"[adapt] built {arr.shape[0]} NOR stacks, tensor {tuple(arr.shape)}")
    return torch.from_numpy(arr.astype(np.float32))


def augment(batch):
    """Random horizontal/vertical flips over the two in-plane spatial dims (2,3)."""
    if torch.rand(()) < 0.5:
        batch = torch.flip(batch, dims=[2])
    if torch.rand(()) < 0.5:
        batch = torch.flip(batch, dims=[3])
    return batch


def adjust_lr(optimizer, epoch, args):
    """Warmup then cosine decay (per-epoch), matching CineMA's schedule shape."""
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    else:
        prog = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def build_model(args, device):
    model = CineMA.from_pretrained()
    n_lora = lora.inject_lora(model, rank=args.rank, alpha=args.alpha,
                              prefix=args.lora_prefix, dropout=args.lora_dropout)
    trainable = lora.set_trainable(model, train_decoder=args.train_decoder)
    model.to(device)
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[adapt] injected {n_lora} LoRA linears (rank={args.rank}, alpha={args.alpha}, "
          f"prefix='{args.lora_prefix}'); train_decoder={args.train_decoder}")
    print(f"[adapt] trainable params: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.2f}%)")
    return model, trainable


def smoke_test(args, device):
    """One forward+backward step on random input; assert grads land only on LoRA+decoder."""
    print("=== SMOKE: LoRA injection + one train step ===")
    model, trainable = build_model(args, device)
    model.train()
    x = torch.randn(2, 1, 192, 192, 16, device=device)
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    loss, _, _, _ = model({"sax": x}, args.mask_ratio)
    loss.backward()
    # every param with a grad must be a LoRA param (+ decoder base only if --train_decoder);
    # base encoder (and base decoder when frozen) must have no grads.
    def allowed(n):
        return n.endswith(("lora_A", "lora_B")) or (args.train_decoder and n.startswith(lora.DECODER_PREFIXES))
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not allowed(n)]
    base_grads = [n for n, p in model.named_parameters()
                  if ".base." in n and p.grad is not None]
    assert not bad, f"grad leaked to non-trainable params: {bad[:5]}"
    assert not base_grads, f"a frozen base weight got grads: {base_grads[:5]}"
    opt.step()
    # save/load round-trip
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "smoke_adapter.pt")
    lora.save_adapter(model, p, args.rank, args.alpha, args.lora_prefix)
    model2 = CineMA.from_pretrained().to(device)
    lora.apply_adapter(model2, p)
    print(f"[smoke] loss={loss.item():.4f}; grads only on trainable ✓; adapter save/load ✓")
    print("=== SMOKE PASSED ===")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # data (same conventions as derisk_cinema.py / cinema_faithful.py)
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_csv",
                    default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--fit_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--all_frames", action="store_true",
                    help="Train the adapter on ALL cardiac frames (whole cine), not just ED/ES "
                         "(~13x more NOR stacks; ED LV bbox reused for every frame's crop). "
                         "Adapter training only; the probe still scores ED/ES.")
    ap.add_argument("--out_dir", default="./adapter_out")
    # LoRA
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lora_prefix", nargs="+", default=["encoder.", "decoder."],
                    help="Module-name prefixes to inject LoRA into. Default targets BOTH the "
                         "encoder and decoder ViT stacks (low-rank), so no big trainable module "
                         "absorbs the loss — the adaptation is forced into the encoder LoRA.")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    # Default now False: the 28M full-trainable decoder absorbed the MAE loss in gate #1
    # (encoder features barely moved, cosine 0.994). Keep the base decoder FROZEN and let a
    # low-rank decoder LoRA follow the encoder instead. Pass --train_decoder to restore the
    # old full-decoder-training behaviour.
    ap.add_argument("--train_decoder", action="store_true", default=False)
    ap.add_argument("--no_train_decoder", dest="train_decoder", action="store_false")
    # optimisation
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true",
                    help="1 forward+backward step on random input, then exit (no data load).")
    args = ap.parse_args()
    args.lora_prefix = tuple(args.lora_prefix)  # str.startswith needs a tuple, not a list

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[adapt] device={device}, amp={use_amp} ({amp_dtype if use_amp else 'off'})")

    if args.smoke:
        smoke_test(args, device)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print("=== Building NOR training stacks ===")
    X = build_nor_tensors(args)
    loader = DataLoader(TensorDataset(X), batch_size=args.batch_size, shuffle=True,
                        drop_last=len(X) > args.batch_size, num_workers=2, pin_memory=use_amp)

    print("=== Loading CineMA + injecting LoRA ===")
    model, trainable = build_model(args, device)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    # fp16 needs a grad scaler; bf16 does not
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    print(f"=== Training {args.epochs} epochs on {len(X)} stacks "
          f"({len(loader)} steps/epoch) ===")
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        lr = adjust_lr(opt, epoch, args)
        losses = []
        for (batch,) in loader:
            batch = augment(batch).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss, _, _, _ = model({"sax": batch}, args.mask_ratio)
            if not torch.isfinite(loss):
                print(f"[adapt] non-finite loss at epoch {epoch}; skipping step")
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss, "lr": lr})
        print(f"[adapt] epoch {epoch+1}/{args.epochs}  loss={mean_loss:.5f}  lr={lr:.2e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.out_dir, "cinema_lora.pt")
            lora.save_adapter(model, ckpt, args.rank, args.alpha, args.lora_prefix)
            with open(os.path.join(args.out_dir, "adapt_loss.json"), "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=2)

    print(f"[adapt] DONE. adapter -> {os.path.join(args.out_dir, 'cinema_lora.pt')}")
    print(f"[adapt] loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f} over "
          f"{args.epochs} epochs. Next: qsub derisk with --adapter_path this file.")


if __name__ == "__main__":
    main()
