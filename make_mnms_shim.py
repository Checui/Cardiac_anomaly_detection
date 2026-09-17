"""Build a symlink tree presenting the flat local M&Ms layout in CineMA's expected structure.

The local copy is flat (``Dataset_1/Training/<pid>_sa.nii.gz``) but CineMA's
``cinema/data/mnms/preprocess.py`` expects the official per-patient layout::

    <root>/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
    <root>/Training/Labeled/<pid>/<pid>_sa{,_gt}.nii.gz
    <root>/Validation/<pid>/...
    <root>/Testing/<pid>/...

Training/Labeled includes only the 150 pids with ED != ES in the diagnosis CSV —
the 25 GE cases with ED == ES == 0 are the official *unlabeled* pool, which CineMA's
own training pool also excludes. GT masks are searched across all three flat split
dirs to repair single-file misplacements (J9L6N9's GT sits in Testing while its
image is in Training).

Idempotent: the destination tree is deleted and rebuilt on every run (symlinks only).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

CSV_NAME = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
DEFAULT_SRC = "/rds/general/user/cc4525/home/Dataset_1"
DEFAULT_EPHEMERAL = "/rds/general/ephemeral/user/cc4525/ephemeral"


def find_gt(pid: str, src: Path, splits: tuple[str, ...]) -> Path | None:
    """Locate <pid>_sa_gt.nii.gz across the flat split dirs."""
    for split in splits:
        p = src / split / f"{pid}_sa_gt.nii.gz"
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path(DEFAULT_SRC))
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(os.environ.get("EPHEMERAL", DEFAULT_EPHEMERAL)) / "mnms_shim",
    )
    args = parser.parse_args()

    src, dst = args.src, args.dst
    meta = pd.read_csv(src / CSV_NAME)
    meta = meta.set_index(meta["External code"].astype(str))
    labeled = set(meta.index[meta["ED"] != meta["ES"]])

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    os.symlink(src / CSV_NAME, dst / CSV_NAME)

    flat_splits = ("Training", "Validation", "Testing")
    layout = {
        "Training": dst / "Training" / "Labeled",
        "Validation": dst / "Validation",
        "Testing": dst / "Testing",
    }
    for split in flat_splits:
        out_root = layout[split]
        out_root.mkdir(parents=True, exist_ok=True)
        pids = sorted(p.name[: -len("_sa.nii.gz")] for p in (src / split).glob("*_sa.nii.gz"))
        kept, skipped_unlabeled, skipped_nogt = [], [], []
        for pid in pids:
            if split == "Training" and pid not in labeled:
                skipped_unlabeled.append(pid)
                continue
            gt = find_gt(pid, src, flat_splits)
            if gt is None:
                skipped_nogt.append(pid)
                continue
            pid_dir = out_root / pid
            pid_dir.mkdir()
            os.symlink(src / split / f"{pid}_sa.nii.gz", pid_dir / f"{pid}_sa.nii.gz")
            os.symlink(gt, pid_dir / f"{pid}_sa_gt.nii.gz")
            kept.append(pid)

        table = meta.loc[kept, "Pathology"].value_counts().to_dict()
        print(f"{split}: linked {len(kept)} pids  pathology={table}")
        if skipped_unlabeled:
            print(f"  skipped {len(skipped_unlabeled)} unlabeled (ED==ES): {skipped_unlabeled}")
        if skipped_nogt:
            print(f"  WARNING skipped (no GT found anywhere): {skipped_nogt}")

    n_train = len(list(layout["Training"].iterdir()))
    assert n_train == 150, f"expected 150 labeled Training pids, got {n_train}"
    print(f"shim ready at {dst}")


if __name__ == "__main__":
    main()
