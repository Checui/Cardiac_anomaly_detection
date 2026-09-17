"""cinema_finetuned_eval.py — OPTION 3: CineMA's own fine-tuned CVD classifiers, on OUR held-out set.

This is the only genuinely apples-to-apples comparison available. Options 1 and 2 match CineMA's
metric and task but not their split; this one removes that ambiguity entirely by running THEIR
released checkpoints on OUR exact held-out patients (ACDC database/testing 50, M&Ms Testing 136),
scored with OUR code.

NO LEAKAGE — verified from their preprocessing source, not assumed:
  cinema/data/acdc/preprocess.py :  data_dir/"training" -> train,  data_dir/"testing" -> test
  cinema/data/mnms/preprocess.py :  Training/Labeled -> train, Validation -> val, Testing -> test
Both use the official challenge folders, which is exactly our protocol. Their classifiers were
therefore fitted on ACDC-training / M&Ms-Training and never saw the patients we evaluate on.

WHAT IT DOES. Rebuilds each held-out patient as CineMA's own classifier input — a 2-channel
(ED, ES) stack of shape (2, 192, 192, 16) — using cinema_faithful.preprocess_sax_stack, which is
a reimplementation of their preprocess.py built on their own cinema.data.sitk helpers. Runs all
three released seeds per dataset, scores each separately, and reports the per-seed and mean
one-vs-rest macro AUC (they also average metrics over three seeds).

CLASS SPACE is taken from the downloaded config.yaml, so it is theirs by construction:
  acdc_sax : DCM, HCM, MINF, NOR, RV        (our full ACDC label space)
  mnms_sax : DCM, HCM, NOR, ARV, HHD        (drops AHS / IHD / LVNC / Other)
Test patients outside that set cannot be predicted and are excluded; the count is printed.

    python cinema_finetuned_eval.py --datasets ACDC MM
"""

import argparse
import json
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from omegaconf import OmegaConf

import cinema_faithful as cf
from qfae_report import _fast_auc, _middle_slice_range

SPATIAL = (192, 192, 16)
_TF = Compose([ScaleIntensityd(keys="sax"),
               SpatialPadd(keys="sax", spatial_size=SPATIAL, method="end")])

CINEMA_PAPER = {"ACDC": 0.9798, "MM": 0.7741}     # Table 5, CineMAFineTune "Classification"
REPO = "mathpluscode/CineMA"


def build_patient_inputs(dsn, args):
    """-> (X (N,2,192,192,16) float32, labels, pids). Channel 0 = ED, channel 1 = ES."""
    if dsn == "ACDC":
        recs = cf.load_acdc_records(args.acdc_dir, "testing", nor_only=False)
    else:
        recs = cf.load_mm_records(args.mm_test_dir, args.mm_csv, nor_only=False)
    by_pid = {}
    for r in recs:
        by_pid.setdefault(r["pid"], {})[r["phase"]] = r
    X, labels, pids = [], [], []
    for pid in sorted(by_pid):
        d = by_pid[pid]
        if "ED" not in d or "ES" not in d:
            continue
        ed, es = cf._pad_stack(d["ED"]["stack"]), cf._pad_stack(d["ES"]["stack"])
        X.append(np.stack([ed, es], axis=0))          # (2, 192, 192, 16)
        labels.append(d["ED"]["label"]); pids.append(pid)
    return np.asarray(X, np.float32), np.array(labels), np.array(pids)


def macro_ovr_auc(P, y, classes):
    per, sup = {}, {}
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.min() == yy.max():
            continue
        per[c] = float(_fast_auc(yy, P[:, i])); sup[c] = int(yy.sum())
    if not per:
        return np.nan, np.nan, {}
    macro = float(np.mean(list(per.values())))
    weighted = float(sum(per[c] * sup[c] for c in per) / sum(sup.values()))
    return macro, weighted, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_dir", default="../Dataset_2")
    ap.add_argument("--mm_test_dir", default="../Dataset_1/Testing")
    ap.add_argument("--mm_csv",
                    default="../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
    ap.add_argument("--datasets", nargs="+", default=["ACDC", "MM"], choices=["ACDC", "MM"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_json", default="./cinema_finetuned_heldout.json")
    args = ap.parse_args()

    from cinema import ConvViT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) \
        else torch.float32

    print("=" * 100)
    print("OPTION 3 — CineMA's OWN fine-tuned CVD classifiers, evaluated on OUR held-out patients")
    print("  No leakage: their preprocess.py uses the official ACDC training/testing and M&Ms")
    print("  Training/Validation/Testing folders — the same protocol we use.")
    print(f"  device={device} dtype={dtype}")
    print("=" * 100)

    results = {}
    for dsn in args.datasets:
        tag = "acdc" if dsn == "ACDC" else "mnms"
        cfg = OmegaConf.load(hf_hub_download(
            repo_id=REPO, filename=f"finetuned/classification_cvd/{tag}_sax/config.yaml"))
        classes = list(cfg.data[cfg.data.class_column])

        X, labels, pids = build_patient_inputs(dsn, args)
        keep = np.isin(labels, classes)
        n_drop = int((~keep).sum())
        dropped = sorted(set(labels[~keep])) if n_drop else []
        # Run inference on EVERY held-out patient, including those outside CineMA's label space.
        # Their 5-way softmax is still defined there, and 1 - P(NOR) remains a valid binary
        # "disease present" score for an LVNC/Other/IHD patient — which is what the threshold
        # metrics need. The multi-class AUC below is still computed on the in-class subset only.
        Xk, yk, pk = X, labels, pids
        print(f"\n### {dsn}   CineMA classes {classes}")
        print(f"    held-out patients {len(labels)} -> evaluated {len(yk)}"
              + (f"   (excluded {n_drop}: {dropped} — not in their label space)" if n_drop else ""))
        print(f"    class counts: {dict(zip(*np.unique(yk, return_counts=True)))}")

        macros, all_probs = [], {}
        for seed in args.seeds:
            model = ConvViT.from_finetuned(
                repo_id=REPO,
                model_filename=f"finetuned/classification_cvd/{tag}_sax/{tag}_sax_{seed}.safetensors",
                config_filename=f"finetuned/classification_cvd/{tag}_sax/config.yaml")
            model.eval().to(device)
            probs = []
            for s in range(0, len(Xk), args.batch_size):
                chunk = Xk[s:s + args.batch_size]
                batch = torch.stack([_TF({"sax": torch.from_numpy(x)})["sax"] for x in chunk])
                batch = {"sax": batch.to(device=device, dtype=dtype)}
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype,
                                                     enabled=(device.type == "cuda")):
                    logits = model(batch)
                probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                print(f"\r    seed {seed}: {min(s + args.batch_size, len(Xk))}/{len(Xk)}",
                      end="", flush=True)
            print()
            P = np.concatenate(probs, axis=0)
            all_probs[seed] = P
            macro, weighted, per = macro_ovr_auc(P[keep], yk[keep], classes)
            macros.append(macro)
            print(f"    seed {seed}: macro AUC {macro:.3f}  weighted {weighted:.3f}   "
                  f"per-class " + " ".join(f"{c}:{v:.2f}" for c, v in per.items()))
            results[f"{dsn}_seed{seed}"] = dict(macro_auc=macro, weighted_auc=weighted,
                                                per_class=per, n_patients=int(keep.sum()))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean, sd = float(np.mean(macros)), float(np.std(macros))
        print(f"    --> {dsn} mean over {len(macros)} seeds: {mean:.3f} (sd {sd:.3f})")
        print(f"        CineMA's own reported Table 5 number: {CINEMA_PAPER[dsn]:.3f}")
        results[f"{dsn}_mean"] = dict(mean_macro_auc=mean, sd=sd, seeds=macros,
                                      paper=CINEMA_PAPER[dsn], n_patients=int(keep.sum()),
                                      excluded=n_drop, classes=classes)

        # Persist per-patient probabilities so threshold metrics (sens/spec/F1) can be computed
        # against OUR detectors on identical patients, rather than against the paper's numbers.
        np.savez_compressed(f"cinema_ft_probs_{dsn}.npz",
                            pids=pk, labels=yk, classes=np.array(classes),
                            in_class=keep,
                            **{f"probs_seed{s}": all_probs[s] for s in all_probs})
        print(f"    wrote cinema_ft_probs_{dsn}.npz "
              f"({len(pk)} patients x {len(classes)} classes x {len(all_probs)} seeds)")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
