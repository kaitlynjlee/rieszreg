#!/usr/bin/env bash
# Savio job array for the final run of pilot.py (results go to final_split,
# not pilot_cf5): sample splitting, as in pilot_cv5. Every learner is tuned by
# 5-fold CV and fit on a training sample of n, and psi-hat is computed on an
# independent estimation sample of n (see pilot.py's docstring). On the blocks
# and DGPs of the paper (AIMS.md sections 1-3):
#   ate, att    the 13 DGPs outside the alignment ladder for replicates 0-199,
#               the eight core DGPs (ATE_DGPS) from 200 on, n = 1000;
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
# Cost. With pilot_cf5's design (cross-fitting with nested tuning) this run
# would take about 24 core-hours per replicate, from pilot_cf5's per-task
# Savio times scaled for the moved grids and the TSM block's new learners.
# Sample splitting tunes once and fits once, about 3.8 times fewer fits
# (rieszboost: 5 x 30 tuning fits on 4n/5 plus one fit on n, against 25 x 30
# on 16n/25 plus 5 on 4n/5), so about 6.3 core-hours per replicate, three
# quarters of it rieszboost; the smoke run suggests nearer 5 times.
#   200 replicates (the default): about 1,250 core-hours. 32 array tasks of
#   about 39 core-hours, so about 1.2 hours each on 32 cores (3-hour limit),
#   8 at a time: about 5 hours in all, plus queue time.
#   The 200 replicates measured 5.3 core-hours per replicate (2026-10-23).
#   500 replicates, on the core DGPs only (2026-10-23): the ATE and ATT DGPs
#   tail_feature, bulk_feature, smooth, step and additive go to the paper's
#   appendix at 200 replicates (AIMS.md section 2), so the extension runs
#   ate and att on the eight core DGPs (ATE_DGPS below) and the other blocks
#   in full, about 4.3 core-hours per replicate, so about 1,300 core-hours:
#     mkdir -p logs && REPS=200:500 N_SHARDS=40 sbatch --array=0-39%8 savio_final.sh
#   (40 array tasks of about 1 hour, about 5 hours in all). To run ate and att
#   on all 13 DGPs, as the first 200 replicates did, also pass
#   ATE_DGPS="overlap_g05 ... steps10" (BLOCKS["ate"]["dgps"] in pilot.py).
# Coverage near 0.95 has a Monte Carlo SE of about 0.015 at 200 replicates and
# 0.010 at 500.
#
# Account: the co_biostat condo on savio3 (condo priority, no service-unit
# charge), 8 array tasks at a time; at most as many run as the condo has free
# nodes. To use the faculty allowance instead, set --account=fc_rwmg
# --qos=savio_normal; it has more nodes but charges service units (roughly
# the core-hours above).
#
# Submit (from examples/simulations, so logs/ exists before Slurm opens it):
#   mkdir -p logs && sbatch savio_final.sh
# Then, on a login node:
#   RIESZ_SIM_OUT=/global/scratch/users/$USER/rieszreg_sim RIESZ_SIM_RUN=final_split uv run python pilot.py summarize
#
#SBATCH --job-name=riesz_final_split
#SBATCH --account=co_biostat
#SBATCH --partition=savio3
#SBATCH --qos=biostat_savio3_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --array=0-31%8
#SBATCH --output=logs/final_split_%A_%a.log

set -euo pipefail
N_SHARDS=${N_SHARDS:-32}         # must match the --array range (0 to N_SHARDS - 1)
REPS=${REPS:-0:200}
SHARD="$SLURM_ARRAY_TASK_ID/$N_SHARDS"
# The ATE and ATT run on ATE_DGPS; the TSM, shift and LASE run on all their DGPs.
ATE_DGPS=${ATE_DGPS:-"overlap_g05 overlap_g1 overlap_g2 overlap_g3 logistic sparse50 interaction steps10"}

export RIESZ_SIM_OUT=${RIESZ_SIM_OUT:-/global/scratch/users/$USER/rieszreg_sim}
export RIESZ_SIM_RUN=final_split
export UV_CACHE_DIR=${UV_CACHE_DIR:-/global/scratch/users/$USER/.uv-cache}
export PYTHONUNBUFFERED=1

cd "$HOME/rieszreg/examples/simulations"
echo "shard $SHARD on $SLURM_CPUS_PER_TASK cores; reps $REPS; ate and att on: $ATE_DGPS"
echo "code: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ../../packages || echo ' with uncommitted changes')"

# Each call runs shard K of N of its own task list. --dgps filters every block
# it is given, so the ATE and ATT get their own call (additive is a TSM DGP too).
# shellcheck disable=SC2086  # ATE_DGPS is a word list
uv run python -u pilot.py run --blocks ate att --dgps $ATE_DGPS --reps "$REPS" --shard "$SHARD" --jobs "$SLURM_CPUS_PER_TASK"
uv run python -u pilot.py run --blocks tsm shift localshift --reps "$REPS" --shard "$SHARD" --jobs "$SLURM_CPUS_PER_TASK"
