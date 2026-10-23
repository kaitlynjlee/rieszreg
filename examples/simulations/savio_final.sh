#!/usr/bin/env bash
# Savio job array for the final run of pilot.py (results go to final_cf5, not
# pilot_cf5): 5-fold cross-fitting with 5-fold CV tuning inside each fold, on
# the blocks and DGPs of the paper (AIMS.md sections 1-3):
#   ate, att    the 13 DGPs outside the alignment ladder, n = 1000;
#   tsm         additive, additive_g2, interaction, n = 250, 1000, 4000, with
#               riesznet, krrr and forestriesz added for the rate across learners;
#   shift, localshift   the four shift DGPs, n = 1000.
# The alignment DGPs (hard_*, mix*) and the tsm_align block belong to the
# later paper (AIMS.md section 8) and are left out. Each array task runs a
# fixed shard of the task list on 32 cores, one single-threaded process per
# core. Tasks resume: a task whose result file exists is skipped, so
# resubmitting fills only the gaps, and a later REPS range extends the run.
#
# Setup and smoke run as in savio.sh. Pull first; after any pull that touches
# packages/, check the compiled riesztree is current:
#   uv run python -c "from riesztree.fast._grow_c import GrowableFlatTree as G; print('n_aug' in dir(G))"
#
# Cost, from the per-task Savio times of pilot_cf5, scaled for the grids moved
# for this run (rieszboost 30 settings instead of 25, riesznet 8 instead of
# 6), plus the TSM block's three new learners (timed on the laptop, where a
# whole krrr task at n = 4000 takes under a minute): about 24 core-hours per
# replicate, three quarters of it rieszboost.
#   200 replicates (the default): about 4,700 core-hours. 96 array tasks of
#   about 49 core-hours, so about 1.5 hours each on 32 cores (3-hour limit),
#   10 at a time: about 15 hours in all.
#   500 replicates: about 11,800 core-hours. After the 200, extend with
#     mkdir -p logs && REPS=200:500 N_SHARDS=144 sbatch --array=0-143%10 savio_final.sh
#   (about 22 hours more), or run 0:500 at once with N_SHARDS=240, --array=0-239%10.
# Coverage near 0.95 has a Monte Carlo SE of about 0.015 at 200 replicates and
# 0.010 at 500.
#
# Submit (from examples/simulations, so logs/ exists before Slurm opens it):
#   mkdir -p logs && sbatch savio_final.sh
# Then, on a login node:
#   RIESZ_SIM_OUT=/global/scratch/users/$USER/rieszreg_sim RIESZ_SIM_RUN=final_cf5 uv run python pilot.py summarize
#
#SBATCH --job-name=riesz_final_cf5
#SBATCH --account=co_biostat
#SBATCH --partition=savio3
#SBATCH --qos=biostat_savio3_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --array=0-95%10
#SBATCH --output=logs/final_cf5_%A_%a.log

set -euo pipefail
N_SHARDS=${N_SHARDS:-96}         # must match the --array range (0 to N_SHARDS - 1)
REPS=${REPS:-0:200}
SHARD="$SLURM_ARRAY_TASK_ID/$N_SHARDS"
BLOCKS="ate att tsm shift localshift"
DGPS="overlap_g05 overlap_g1 overlap_g2 overlap_g3 tail_feature bulk_feature logistic smooth step
      sparse50 additive interaction steps10 additive_g2
      shift_gauss shift_hetero shift_skew shift_bimodal"

export RIESZ_SIM_OUT=${RIESZ_SIM_OUT:-/global/scratch/users/$USER/rieszreg_sim}
export RIESZ_SIM_RUN=final_cf5
export UV_CACHE_DIR=${UV_CACHE_DIR:-/global/scratch/users/$USER/.uv-cache}
export PYTHONUNBUFFERED=1

cd "$HOME/rieszreg/examples/simulations"
echo "shard $SHARD on $SLURM_CPUS_PER_TASK cores; reps $REPS; blocks $BLOCKS"
echo "code: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ../../packages || echo ' with uncommitted changes')"

# shellcheck disable=SC2086  # BLOCKS and DGPS are word lists
uv run python -u pilot.py run --blocks $BLOCKS --dgps $DGPS --reps "$REPS" --shard "$SHARD" --jobs "$SLURM_CPUS_PER_TASK"
