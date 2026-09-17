"""Evaluate a locally fine-tuned CineMA classification checkpoint on the held-out cohort.

Companion to ``cinema_ft_cell.pbs``: after ``{acdc,mnms}_clf_train`` finishes, this
script loads the single best checkpoint from ``<run_dir>/ckpt`` and scores the
official held-out split (ACDC ``database/testing`` 50 pts / M&Ms ``Testing`` 136 pts,
preprocessed by CineMA's own pipeline) through the trainer's own val transform and
``classification_eval`` path — so train and test preprocessing agree by construction.

Cohort gate (``supervised_features.py`` pattern): evaluated pids must be a subset of
the reference cohort (hard fail otherwise); binary cells must additionally cover it
fully. Reference pids come from the existing artifacts (``cinema_ft_probs_ACDC.npz``
/ ``qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz``) with their ``ACDC_``/``MM_``
prefixes stripped.

Output npz (one per experiment cell): pids, pathology (true disease), binary_label,
probs (N, n_classes), classes, cell, seed, ckpt_epoch, n_train, n_val — consumed by
``supervised_ft_analysis.py``.

    python supervised_ft_eval.py --run_dir $EPHEMERAL/cinema_ft_runs/acdc_frac25_s1 \
        --dataset acdc --processed_dir $EPHEMERAL/cinema_processed/acdc \
        --out_npz supervised_ft_out/acdc_frac25_s1.npz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler

REPO = Path(__file__).resolve().parent
DEFAULT_REF = {
    "acdc": REPO / "cinema_ft_probs_ACDC.npz",
    "mnms": REPO / "qfae_dino2d_mae224_out_mmtest" / "qfae_dino_arrays.npz",
}


def find_ckpt(ckpt_dir: Path) -> tuple[Path, int]:
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: int(re.search(r"ckpt_(\d+)", p.name).group(1)))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {ckpt_dir}")
    best = ckpts[-1]
    return best, int(re.search(r"ckpt_(\d+)", best.name).group(1))


def ref_pids(ref_npz: Path, dataset: str) -> set[str]:
    """Unique patient codes from a reference artifact, dataset prefix stripped.

    Mixed-cohort archives (the QFAE arrays hold ACDC + MM rows) are filtered to the
    requested dataset via their ``datasets`` column before collecting pids.
    """
    z = np.load(ref_npz, allow_pickle=True)
    arr = z["pids"]
    if "datasets" in z.files:
        tag = {"acdc": "ACDC", "mnms": "MM"}[dataset]
        arr = arr[z["datasets"].astype(str) == tag]
    return {re.sub(r"^(ACDC_|MM_|MMVAL_)", "", str(p)) for p in np.unique(arr)}


def replicate_train_counts(cfg, processed_dir: Path) -> tuple[int, int]:
    """Recompute the training/val set sizes the trainer saw (independent audit)."""
    train_df = pd.read_csv(processed_dir / "train_metadata.csv", dtype={"pid": str})
    if cfg.data.name == "mnms":
        val_df = pd.read_csv(processed_dir / "val_metadata.csv", dtype={"pid": str})
    excl_csv = cfg.data.get("exclude_pids_csv", None)
    if excl_csv:
        excl = set(pd.read_csv(excl_csv)["pid"].astype(str))
        train_df = train_df[~train_df["pid"].isin(excl)].reset_index(drop=True)
        if cfg.data.name == "mnms":
            val_df = val_df[~val_df["pid"].isin(excl)].reset_index(drop=True)
    if cfg.data.name == "acdc":
        val_pids = train_df.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()
        val_df = train_df[train_df["pid"].isin(val_pids)]
        train_df = train_df[~train_df["pid"].isin(val_pids)].reset_index(drop=True)
    else:
        class_col = cfg.data.class_column
        classes = list(cfg.data[class_col])
        val_df = val_df[val_df[class_col].isin(classes)]
    if cfg.data.proportion < 1:
        strat = cfg.data.get("stratify_column", "")
        if strat:
            train_df = train_df.groupby(strat, group_keys=False).sample(
                frac=cfg.data.proportion, random_state=cfg.seed
            )
        else:
            train_df = train_df.sample(n=int(cfg.data.proportion * len(train_df)), random_state=cfg.seed)
    return len(train_df), len(val_df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=["acdc", "mnms"])
    ap.add_argument("--processed_dir", required=True, type=Path)
    ap.add_argument("--out_npz", required=True, type=Path)
    ap.add_argument("--ref_npz", type=Path, default=None)
    args = ap.parse_args()

    from cinema.classification.dataset import EndDiastoleEndSystoleDataset, get_image_transforms
    from cinema.classification.train import classification_eval, get_classification_or_regression_model
    from cinema.device import get_amp_dtype_and_device

    ckpt_dir = args.run_dir / "ckpt"
    cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    cfg.data.dir = str(args.processed_dir)
    cfg.data.max_n_samples = -1  # never subsample the held-out split (smoke configs set it > 0)
    ckpt_path, ckpt_epoch = find_ckpt(ckpt_dir)
    class_col = cfg.data.class_column
    classes = list(cfg.data[class_col])
    cell = args.run_dir.name
    print(f"[eval] {cell}: ckpt epoch {ckpt_epoch}, class_column={class_col}, classes={classes}")

    meta_df = pd.read_csv(args.processed_dir / "test_metadata.csv", dtype={"pid": str})
    n_all = len(meta_df)
    meta_df = meta_df[meta_df[class_col].isin(classes)].reset_index(drop=True)
    if len(meta_df) < n_all:
        dropped = n_all - len(meta_df)
        print(f"[eval] {dropped} test patients outside the class space (5-way M&Ms expected)")

    _, val_transform = get_image_transforms(cfg)
    dataset = EndDiastoleEndSystoleDataset(
        data_dir=args.processed_dir / "test",
        meta_df=meta_df,
        class_col=class_col,
        classes=classes,
        views=cfg.model.views,
        transform=val_transform,
    )
    dataloader = DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=1, drop_last=False, pin_memory=True, num_workers=4
    )

    amp_dtype, device = get_amp_dtype_and_device()
    model = get_classification_or_regression_model(cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    views = [cfg.model.views] if isinstance(cfg.model.views, str) else cfg.model.views
    patch_size_dict = {v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size for v in views}
    logits_lst, pids = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits, _ = classification_eval(
                model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device
            )
            logits_lst.append(logits.cpu().to(torch.float32))
            pids += list(batch["pid"])
    probs = F.softmax(torch.cat(logits_lst, dim=0), dim=1).numpy()
    pids = np.array([str(p) for p in pids])

    ref_npz = args.ref_npz or DEFAULT_REF[args.dataset]
    ref = ref_pids(ref_npz, args.dataset)
    extra = sorted(set(pids) - ref)
    missing = sorted(ref - set(pids))
    if extra:
        raise AssertionError(f"cohort gate: {len(extra)} evaluated pids not in reference {ref_npz}: {extra[:5]}")
    if missing:
        msg = f"cohort gate: {len(missing)} reference pids not evaluated: {missing[:5]}"
        if class_col == "binary":
            raise AssertionError(msg)
        print(f"[eval] {msg} (allowed for {class_col} class space)")

    order = np.argsort(pids)
    pids, probs = pids[order], probs[order]
    pathology = meta_df.set_index("pid").loc[pids, "pathology"].to_numpy(dtype=str)
    binary_label = np.where(pathology == "NOR", "NOR", "ABNORMAL")

    n_train, n_val = replicate_train_counts(cfg, args.processed_dir)
    m = re.match(r"(acdc|mnms)_(.+)_s(\d+)$", cell)
    seed = int(m.group(3)) if m else int(cfg.seed)

    # quick log-line AUC: P(abnormal) = 1 - P(NOR)
    nor_idx = classes.index("NOR")
    score = 1.0 - probs[:, nor_idx]
    y = (binary_label == "ABNORMAL").astype(int)
    from qfae_report import _fast_auc

    print(f"[eval] {cell}: n={len(pids)} n_train={n_train} n_val={n_val} binary AUC={_fast_auc(y, score):.4f}")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        pids=pids,
        pathology=pathology,
        binary_label=binary_label,
        probs=probs,
        classes=np.array(classes),
        cell=cell,
        seed=seed,
        ckpt_epoch=ckpt_epoch,
        n_train=n_train,
        n_val=n_val,
    )
    print(f"[eval] wrote {args.out_npz}")


if __name__ == "__main__":
    main()
