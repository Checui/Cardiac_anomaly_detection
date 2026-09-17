"""qfae_matrix_report.py — the 3x3 encoder x scorer table (offline, no GPU).

Reports, per cell, the two streams asked for:
  * MOTION      flow_SSIM with the 'mean' reduction — the recipe that actually survived
                val-selection in confirm_val.py, so it is PRE-COMMITTED here, not chosen post hoc.
  * RECONSTRUCTION  the appearance stream, same reduction.
on held-out ACDC-50 and M&Ms-Test, plus the honest select-on-M&Ms-Val / report-on-M&Ms-Test
number per cell. 'middle60' is shown as a secondary row (the robust rule on the powered test set).

Reuses confirm_offline.grid_for_run for the reduction + per-dataset AUC machinery, so these
numbers are computed identically to every other table in the project.

READ THE MATRIX BY ROW. Within a row only the scorer changes (same encoder, same container, same
batch size) — that is the question this experiment answers. Across rows the container changes too
(CineMA is 3-D depth-16, DINOv2/MAE are 2-D per-slice), so row-to-row gaps are confounded.
"""

import numpy as np

from confirm_offline import grid_for_run, MOTION

GAN = {"ACDC": 0.8125, "MM": 0.731}
ENCODERS = ["cinema", "dino224", "mae224"]
SCORERS = ["cinema", "dino", "mae"]
PRECOMMITTED = ("flow_SSIM", "mean")


def _paths(enc, sco):
    """(test_npz, val_npz, fmt) for one matrix cell, or None if the cell has no run."""
    if enc == "cinema":
        fmt, arr = "stack", "qfae_arrays.npz"
        # the diagonal cinema x cinema cell predates the matrix and lives under its own name
        base = "qfae_flow_sp_out" if sco == "cinema" else f"qfae_mx_cinema_{sco}_out"
    else:
        fmt, arr = "slice", "qfae_dino_arrays.npz"
        base = f"qfae_mx_{enc}_{sco}_out"
    return f"{base}_mmtest/{arr}", f"{base}_mmval/{arr}", fmt


# BS=32 runs from the original 2-D sweep — same cell as the coupled diagonal but a different
# batch size, kept as an external check on how much the BS=8 matrix block moved.
EXTERNAL = {
    "dino224 x dino (BS=32)": ("qfae_dino2d_dino224_out_mmtest/qfae_dino_arrays.npz",
                               "qfae_dino2d_dino224_out_mmval/qfae_dino_arrays.npz", "slice"),
    "mae224 x mae (BS=32)":   ("qfae_dino2d_mae224_out_mmtest/qfae_dino_arrays.npz",
                               "qfae_dino2d_mae224_out_mmval/qfae_dino_arrays.npz", "slice"),
}


def load_all():
    """{(enc, sco): {'test': grid, 'val': grid or None}} for every cell that exists on disk."""
    cells, missing = {}, []
    for enc in ENCODERS:
        for sco in SCORERS:
            tnpz, vnpz, fmt = _paths(enc, sco)
            try:
                test = grid_for_run(f"{enc}x{sco}", tnpz, fmt)
            except (FileNotFoundError, OSError):
                missing.append(f"{enc} x {sco}  ({tnpz})")
                continue
            try:
                val = grid_for_run(f"{enc}x{sco}", vnpz, fmt)
            except (FileNotFoundError, OSError):
                val = None
            cells[(enc, sco)] = {"test": test, "val": val}
    return cells, missing


def _cell(grid, stream, rule):
    g = grid.get((stream, rule)) if grid else None
    return f"{g['ACDC']:.3f}/{g['MM']:.3f}" if g else "--"


def table(cells, stream, rule, title):
    print(f"\n{title}   [stream={stream}, reduction={rule}]   cells are ACDC / M&Ms-Test")
    header = "encoder \\ scorer"
    print(f"    {header:<20}" + "".join(f"{s:>16}" for s in SCORERS))
    for enc in ENCODERS:
        row = f"    {enc:<20}"
        for sco in SCORERS:
            c = cells.get((enc, sco))
            row += f"{_cell(c['test'] if c else None, stream, rule):>16}"
        print(row)


def honest(cells):
    """Per cell: pick the best MOTION (stream, rule) on M&Ms-Val, report its M&Ms-Test AUC."""
    print("\nHONEST per cell — select motion recipe on M&Ms-Val (9 NOR), report on M&Ms-Test (32 NOR)")
    print(f"    {'cell':<22}{'picked':<22}{'val MM':>9}{'test MM':>10}{'test ACDC':>11}")
    rows = []
    for (enc, sco), c in cells.items():
        if not c["val"]:
            print(f"    {enc + ' x ' + sco:<22}{'(no M&Ms-Val eval)':<22}")
            continue
        cand = [k for k in c["val"] if k[0] in MOTION and not np.isnan(c["val"][k]["MM"])]
        if not cand:
            continue
        best = max(cand, key=lambda k: c["val"][k]["MM"])
        v, t, a = c["val"][best]["MM"], c["test"][best]["MM"], c["test"][best]["ACDC"]
        rows.append((f"{enc} x {sco}", best, v, t, a))
        print(f"    {enc + ' x ' + sco:<22}{best[0] + '/' + best[1]:<22}{v:>9.3f}{t:>10.3f}{a:>11.3f}")
    if rows:
        gb = max(rows, key=lambda r: r[2])
        print(f"\n    Global pick (highest val-MM across the matrix): {gb[0]} / {gb[1][0]}/{gb[1][1]}")
        print(f"      -> held-out M&Ms-Test = {gb[3]:.3f}   ACDC = {gb[4]:.3f}"
              f"   vs GAN {GAN['MM']:.3f} / {GAN['ACDC']:.3f}")


# Re-scores of the two existing diagonal checkpoints through the refactored scorer/score_depth
# code path. These must reproduce their published flow_SSIM/mean numbers, else the matrix is suspect.
REGRESSION = {
    "mae224 x mae (BS=32)": ("qfae_dino2d_mae224_out_recheck/qfae_dino_arrays.npz", "slice",
                             {"ACDC": 0.812, "MM": 0.714}),
    "cinema x cinema":      ("qfae_flow_sp_out_recheck/qfae_arrays.npz", "stack",
                             {"ACDC": 0.795, "MM": 0.689}),
}


def regression(tol=0.005):
    """Did the refactor preserve behaviour? Compares re-scored diagonals to published values."""
    print("\n[0] REGRESSION — refactored code re-scoring the existing checkpoints (flow_SSIM/mean)")
    any_run = False
    for name, (npz, fmt, want) in REGRESSION.items():
        try:
            g = grid_for_run(name, npz, fmt)
        except (FileNotFoundError, OSError):
            print(f"    {name:<24} -- not run yet (qfae_matrix.pbs runs this first)")
            continue
        any_run = True
        got = g[("flow_SSIM", "mean")]
        bad = [ds for ds in ("ACDC", "MM") if abs(got[ds] - want[ds]) > tol]
        flag = "MISMATCH -> DO NOT TRUST THE MATRIX" if bad else "reproduced ✓"
        print(f"    {name:<24} got {got['ACDC']:.3f}/{got['MM']:.3f}  "
              f"expected {want['ACDC']:.3f}/{want['MM']:.3f}   {flag}")
    if not any_run:
        print("    (no regression re-scores on disk — the matrix numbers are unverified)")


def external(rule):
    print(f"\nExternal BS=32 reference (same cell, different batch size), {rule}:")
    for name, (tnpz, _vnpz, fmt) in EXTERNAL.items():
        try:
            g = grid_for_run(name, tnpz, fmt)
        except (FileNotFoundError, OSError):
            print(f"    {name:<26} -- missing")
            continue
        print(f"    {name:<26} flow_SSIM {_cell(g, 'flow_SSIM', rule):>14}"
              f"   appearance {_cell(g, 'appearance', rule):>14}")


def main():
    cells, missing = load_all()
    print("=" * 86)
    print("ENCODER x SCORER MATRIX — held-out ACDC-50 (10 NOR/40) + M&Ms-Test (32 NOR/104)")
    print(f"Bar: GAN flow-SSIM = ACDC {GAN['ACDC']:.4f} / M&Ms-Test {GAN['MM']:.4f}")
    print(f"Pre-committed comparison: {PRECOMMITTED[0]} / {PRECOMMITTED[1]}")
    print("=" * 86)
    if missing:
        print(f"\n{len(missing)} cell(s) not on disk yet — run qfae_matrix.pbs:")
        for m in missing:
            print(f"    {m}")
    if not cells:
        print("\nNothing to report yet.")
        return

    regression()
    table(cells, *PRECOMMITTED, title="[1] MOTION — the headline stream")
    table(cells, "appearance", "mean", title="[2] RECONSTRUCTION — the appearance stream")
    table(cells, "flow_SSIM", "middle60", title="[3] MOTION, secondary rule (middle-60%)")
    table(cells, "appearance", "middle60", title="[4] RECONSTRUCTION, secondary rule (middle-60%)")
    honest(cells)
    external(PRECOMMITTED[1])

    print("\nCaveats to carry into any write-up:")
    print("  * 32 NOR on M&Ms-Test => 95% CI ~ +/-0.09. Differences below that are not measurable;")
    print("    read the matrix for direction and consistency, not for a single winning cell.")
    print("  * Rows use different containers (CineMA 3-D vs 2-D per-slice) AND different batch")
    print("    sizes (2 vs 8), so compare WITHIN a row; across rows is confounded.")
    print("  * top_frac=0.2 is a FRACTION of the scorer's own token count, so the absolute number")
    print("    of tokens pooled differs by scorer (CineMA 2304 -> 461, p14 256 -> 51, p16 196 -> 39).")
    print("  * The CineMA scorer on a 2-D reconstruction sees 16 replicated identical slices —")
    print("    off-distribution for its depth-wise attention. Documented in qfae_perceptual.py.")


if __name__ == "__main__":
    main()
