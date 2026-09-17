"""Prepare CineMA fine-tune experiment cells: binary labels + LODO/control exclusion CSVs.

Run AFTER CineMA's preprocess scripts have written the metadata CSVs under
``$EPHEMERAL/cinema_processed/{acdc,mnms}``. Idempotent.

1. Adds a ``binary`` column (NOR / ABNORMAL) to every metadata CSV in place —
   used via hydra overrides ``data.class_column=binary '+data.binary=[NOR,ABNORMAL]'``.
2. Writes exclusion CSVs (one ``pid`` column) to ``supervised_cells/``:
   - ``acdc_lodo<D>.csv``   D in {DCM, HCM, MINF, RV}: all training pids of D.
   - ``mnms_lodo<D>.csv``   D in {DCM, HCM}: training AND validation pids of D
     (keeps the unseen disease out of early-stopping model selection too).
   - ``<ds>_ctrl<D>_s<seed>.csv``: the same NUMBER of abnormal *training* pids
     drawn at random from the remaining abnormal classes (NOR and val untouched)
     — separates "unseen pathology" from "smaller training set".
3. Prints an audit table per cell (resulting train/val class counts, recomputing
   the ACDC loader's exact val split) and snapshots the metadata CSVs into
   ``supervised_cells/csv_snapshot/`` (ephemeral is purged after 30 days).

``--print_fraction <ds> <frac> <seed>`` replicates the (patched) loader's
stratified subsample and prints the selected pids — for verifying seed
reproducibility against the trainer log.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EPHEMERAL = "/rds/general/ephemeral/user/cc4525/ephemeral"
REPO = Path(__file__).resolve().parent
CELLS_DIR = REPO / "supervised_cells"

ACDC_LODO = ("DCM", "HCM", "MINF", "RV")
MNMS_LODO = ("DCM", "HCM")
SEEDS = (0, 1, 2)

META_FILES = {
    "acdc": ("train_metadata.csv", "test_metadata.csv"),
    "mnms": ("train_metadata.csv", "val_metadata.csv", "test_metadata.csv"),
}


def processed_root() -> Path:
    return Path(os.environ.get("EPHEMERAL", DEFAULT_EPHEMERAL)) / "cinema_processed"


def load_meta(ds: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(processed_root() / ds / name)
    df["pid"] = df["pid"].astype(str)
    return df


def acdc_val_split(meta_df: pd.DataFrame) -> list[str]:
    """Replicate load_acdc_dataset's val split on a (possibly filtered) frame."""
    return meta_df.groupby("pathology").sample(n=2, random_state=0)["pid"].tolist()


def add_binary_column() -> None:
    for ds, names in META_FILES.items():
        for name in names:
            path = processed_root() / ds / name
            df = pd.read_csv(path)
            df["binary"] = np.where(df["pathology"].astype(str) == "NOR", "NOR", "ABNORMAL")
            df.to_csv(path, index=False)
            counts = df["binary"].value_counts().to_dict()
            print(f"{ds}/{name}: n={len(df)} binary={counts}")


def write_cell(path: Path, pids: list[str]) -> None:
    pd.DataFrame({"pid": sorted(pids)}).to_csv(path, index=False)


def audit(ds: str, cell: str, excl: set[str], train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> None:
    if ds == "acdc":
        meta = train_df[~train_df["pid"].isin(excl)].reset_index(drop=True)
        val_pids = set(acdc_val_split(meta))
        tr = meta[~meta["pid"].isin(val_pids)]
        va = meta[meta["pid"].isin(val_pids)]
    else:
        tr = train_df[~train_df["pid"].isin(excl)]
        va = val_df[~val_df["pid"].isin(excl)]
    print(
        f"  {ds}_{cell}: excl={len(excl)}  "
        f"train n={len(tr)} {tr['pathology'].value_counts().to_dict()}  "
        f"val n={len(va)} {va['pathology'].value_counts().to_dict()}"
    )


def make_cells() -> None:
    CELLS_DIR.mkdir(exist_ok=True)
    for ds, lodo_classes in (("acdc", ACDC_LODO), ("mnms", MNMS_LODO)):
        train_df = load_meta(ds, "train_metadata.csv")
        val_df = load_meta(ds, "val_metadata.csv") if ds == "mnms" else None
        for disease in lodo_classes:
            lodo_pids = train_df.loc[train_df["pathology"] == disease, "pid"].tolist()
            if ds == "mnms":
                lodo_pids += val_df.loc[val_df["pathology"] == disease, "pid"].tolist()
            write_cell(CELLS_DIR / f"{ds}_lodo{disease}.csv", lodo_pids)
            audit(ds, f"lodo{disease}", set(lodo_pids), train_df, val_df)

            n_remove = int((train_df["pathology"] == disease).sum())
            other_abn = train_df.loc[
                (train_df["pathology"] != disease) & (train_df["pathology"] != "NOR"), "pid"
            ].tolist()
            for seed in SEEDS:
                rng = np.random.RandomState(1000 + seed)
                if n_remove >= len(other_abn):
                    raise ValueError(f"{ds} ctrl{disease}: cannot remove {n_remove} of {len(other_abn)}")
                ctrl_pids = list(rng.choice(other_abn, size=n_remove, replace=False))
                write_cell(CELLS_DIR / f"{ds}_ctrl{disease}_s{seed}.csv", ctrl_pids)
                audit(ds, f"ctrl{disease}_s{seed}", set(ctrl_pids), train_df, val_df)


def snapshot_csvs() -> None:
    snap = CELLS_DIR / "csv_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for ds, names in META_FILES.items():
        for name in names:
            shutil.copy2(processed_root() / ds / name, snap / f"{ds}_{name}")
    print(f"metadata CSVs snapshotted to {snap}")


def print_fraction(ds: str, frac: float, seed: int) -> None:
    """Replicate the patched loader path: exclusion=none -> (acdc val split) -> stratified sample."""
    train_df = load_meta(ds, "train_metadata.csv")
    if ds == "acdc":
        val_pids = set(acdc_val_split(train_df))
        train_df = train_df[~train_df["pid"].isin(val_pids)].reset_index(drop=True)
    sampled = (
        train_df.groupby("pathology", group_keys=False)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
    )
    print(f"{ds} frac={frac} seed={seed}: n={len(sampled)} {sampled['pathology'].value_counts().to_dict()}")
    print(sorted(sampled["pid"].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print_fraction", nargs=3, metavar=("DS", "FRAC", "SEED"), default=None)
    args = parser.parse_args()
    if args.print_fraction:
        ds, frac, seed = args.print_fraction
        print_fraction(ds, float(frac), int(seed))
        return
    add_binary_column()
    make_cells()
    snapshot_csvs()


if __name__ == "__main__":
    main()
