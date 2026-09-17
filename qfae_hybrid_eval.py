"""qfae_hybrid_eval.py — score + report a hybrid QFAE (encoder x scorer matrix).

Reuses qfae_eval.report_stream -> aggregate_stacks_to_patient + one_vs_nor_aucs + per-dataset, so
the patient-Mean ACDC / M&Ms numbers are directly comparable to every prior run. Saves per-stack
scores to qfae_arrays.npz and the AUC tree to qfae_results.json.

--encoder / --scorer MUST match training (encoder is also checked against the checkpoint's arch).
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import torch

import cinema_faithful as cf
from qfae_eval import report_stream, flow_scores
from qfae_hybrid import DEPTH
from qfae_hybrid_train import build_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_dir", default="../Dataset_1/Training")
    ap.add_argument("--mm_val_dir", default="../Dataset_1/Validation")
    ap.add_argument("--mm_csv", default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--val_datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--model_path", default="./qfae_hybrid_out/qfae.pt")
    ap.add_argument("--out_dir", default="./qfae_hybrid_out")
    ap.add_argument("--encoder", choices=["cinema", "dino"], default="cinema")
    ap.add_argument("--scorer", choices=["cinema", "dino"], default="cinema")
    ap.add_argument("--dino_model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--flow", action="store_true", help="score the motion head (must match training)")
    ap.add_argument("--perceptual_layers", type=int, nargs="+", default=[5, 8, 11])
    ap.add_argument("--top_frac", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # model dims (must match training; also come from the checkpoint arch)
    ap.add_argument("--qformer_blocks", type=int, default=2)
    ap.add_argument("--qformer_heads", type=int, default=8)
    ap.add_argument("--decoder_dim", type=int, default=512)
    ap.add_argument("--decoder_depth", type=int, default=6)
    ap.add_argument("--decoder_heads", type=int, default=8)
    ap.add_argument("--perceptual_slices", type=int, default=0)   # eval scores ALL valid slices
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Building validation records ===")
    val = cf.build_val_records_flow(args) if args.flow else cf.build_val_records(args)
    print(f"[hybrid] val stacks={len(val)} labels={dict(Counter(r['label'] for r in val))}"
          + ("  (+flow)" if args.flow else ""))

    print("=== Building models ===")
    model, perceptual = build_models(args, device)
    ckpt = torch.load(args.model_path, map_location="cpu")
    assert ckpt["arch"]["encoder_kind"] == args.encoder, (
        f"--encoder {args.encoder} != checkpoint encoder_kind {ckpt['arch']['encoder_kind']}")
    res = model.load_state_dict(ckpt["model"], strict=False)
    assert not getattr(res, "unexpected_keys", []), f"unexpected keys: {res.unexpected_keys[:3]}"
    model.eval()

    print("=== Scoring val stacks ===")
    scores, fssim, mssim, fl1 = [], [], [], []
    n = len(val)
    for start in range(0, n, args.batch_size):
        chunk = val[start:start + args.batch_size]
        x = torch.from_numpy(np.stack([cf.stack_to_tensor(r["stack"]) for r in chunk]))
        valid = torch.tensor([min(DEPTH, r["stack"].shape[2]) for r in chunk])
        x = x.to(device=device, dtype=amp_dtype)
        use_amp = device.type == "cuda" and amp_dtype != torch.float32
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            recon, pred_flow = out if args.flow else (out, None)
            if args.scorer == "dino":
                s = perceptual.score(x.float(), recon.float(), top_frac=args.top_frac, valid=valid)
            else:
                s = perceptual.score(x.float(), recon.float(), top_frac=args.top_frac)
        scores.append(s.float().cpu().numpy())
        if args.flow:
            pf = pred_flow.float().cpu().numpy()                        # (b,3,192,192,16)
            for i, r in enumerate(chunk):
                fs, ms, l1 = flow_scores(r["gt_flow"], pf[i])
                fssim.append(fs); mssim.append(ms); fl1.append(l1)
        print(f"\r[hybrid] scored {min(start + args.batch_size, n)}/{n}", end="", flush=True)
    print()
    scores = np.concatenate(scores)
    pids = np.array([r["pid"] for r in val])
    labels = np.array([r["label"] for r in val])
    datasets = np.array([r["dataset"] for r in val])

    print("\n" + "=" * 64)
    print(f"HYBRID QFAE RESULTS — encoder={args.encoder}  scorer={args.scorer}"
          + ("  +FLOW" if args.flow else ""))
    print("=" * 64)
    tag = f"{args.encoder}->{args.scorer}"
    # appearance stream saved under BOTH `scores` (hybrid convention) and `appearance`
    # (so qfae_report.py treats a flow-hybrid run exactly like qfae_flow_out).
    streams = {tag: report_stream(tag, scores, pids, labels, datasets)}
    save = {"scores": scores, "pids": pids, "labels": labels, "datasets": datasets}
    if args.flow:
        fssim, mssim, fl1 = np.array(fssim), np.array(mssim), np.array(fl1)
        streams["flow_SSIM"] = report_stream("flow_SSIM", fssim, pids, labels, datasets)
        streams["mag_SSIM"] = report_stream("mag_SSIM", mssim, pids, labels, datasets)
        streams["flow_L1"] = report_stream("flow_L1", fl1, pids, labels, datasets)
        combined = ((scores - scores.mean()) / (scores.std() + 1e-8)
                    + (fssim - fssim.mean()) / (fssim.std() + 1e-8))
        streams["combined"] = report_stream("combined", combined, pids, labels, datasets)
        save.update(appearance=scores, flow_SSIM=fssim, mag_SSIM=mssim, flow_L1=fl1, combined=combined)

    results = {"method": "qfae_hybrid", "encoder": args.encoder, "scorer": args.scorer,
               "flow": bool(args.flow), "perceptual_layers": args.perceptual_layers,
               "top_frac": args.top_frac, "n_val": int(n), "streams": streams}
    with open(os.path.join(args.out_dir, "qfae_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, "qfae_arrays.npz"), **save)
    print(f"\n[hybrid] wrote {args.out_dir}/qfae_results.json + qfae_arrays.npz")


if __name__ == "__main__":
    main()
