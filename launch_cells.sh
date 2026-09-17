#!/bin/bash

# =============================================================================
# Launcher for the CineMA supervised fine-tune experiment grid.
# Each line is one independent PBS GPU job — trim the experiment by commenting
# lines out. Collects job IDs and chains the CPU analysis job with afterok.
#   ./launch_cells.sh            # submit everything + chained analysis
#   ./launch_cells.sh --dry_run  # print qsub commands only
# Prerequisites: preprocess_cinema.pbs finished OK (metadata CSVs + cells exist).
# =============================================================================

set -u
DRY=${1:-}
REPO=/rds/general/user/cc4525/home/ICCV2019_github
cd $REPO
JOBS=()

submit() {  # submit DS CELL SEED WALLTIME
    local ds=$1 cell=$2 seed=$3 wt=$4
    local cmd="qsub -N cf_${ds}_${cell}_s${seed} -l walltime=$wt -v DS=$ds,CELL=$cell,SEED=$seed cinema_ft_cell.pbs"
    if [ "$DRY" = "--dry_run" ]; then echo "$cmd"; return; fi
    local jid=$($cmd)
    echo "$jid  cf_${ds}_${cell}_s${seed}"
    JOBS+=("$jid")
}

# Walltimes calibrated from the 2026-08-25 grid run: real cells finished in
# 7-28 min (early stopping); 1-2 h caps give >4x margin while keeping queue
# priority. If a cell ever hits the wall, rerun just it with -l walltime=...
# --- G1: label-efficiency curve (binary head), 4 fractions x 3 seeds x 2 datasets ---
for ds in acdc mnms; do
    for pct in 10 25 50; do
        for seed in 0 1 2; do submit $ds frac$pct $seed 01:00:00; done
    done
    for seed in 0 1 2; do submit $ds frac100 $seed 02:00:00; done
done

# --- G1 anchor: 5-way typing at 100% labels (validates the loop vs the authors' numbers) ---
submit acdc 5way 0 02:00:00
submit mnms 5way 0 02:00:00

# --- G2: leave-one-disease-out (binary head) + matched-N controls ---
for seed in 0 1 2; do
    for d in DCM HCM MINF RV; do
        submit acdc lodo$d $seed 02:00:00
        submit acdc ctrl$d $seed 02:00:00
    done
    for d in DCM HCM; do
        submit mnms lodo$d $seed 02:00:00
        submit mnms ctrl$d $seed 02:00:00
    done
done

if [ "$DRY" = "--dry_run" ]; then exit 0; fi

DEPEND=$(IFS=:; echo "${JOBS[*]}")
AJID=$(qsub -W depend=afterok:$DEPEND supervised_ft_analysis.pbs)
echo "$AJID  cf_analysis (afterok on ${#JOBS[@]} cells)"
