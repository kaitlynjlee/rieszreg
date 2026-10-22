#!/usr/bin/env bash
# Savio job array for pilot.py: 5-fold cross-fitting, with 5-fold CV tuning
# inside each fold (results go to pilot_cf5). Each array task runs a
# fixed shard of the task list (pilot.py run --shard K/N) on 32 cores, one
# single-threaded process per core:
#   1. 50 replicates of every DGP in every block (ate, tsm, shift);
#   2. replicates 50-199 of the four middle rungs of the alignment ladder.
# Tasks resume: a task whose result file exists is skipped, so resubmitting
# the same array fills only the gaps (e.g. after a timeout).
#
# One-time setup, on a Savio login node (as in ../lee_schuler_10cov_shift/savio.sh):
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   echo 'export UV_CACHE_DIR=/global/scratch/users/$USER/.uv-cache' >> ~/.bashrc
#   git clone https://github.com/kaitlynjlee/rieszreg.git ~/rieszreg
#   cd ~/rieszreg && git checkout simulations && uv sync --all-packages --all-extras
# If the clone exists already: cd ~/rieszreg && git checkout simulations && git pull && uv sync --all-packages --all-extras
#
# Cost: cross-fitting repeats every task's tuning and fit once per fold, on
# 4/5 of the sample. From the per-task times of the 5-fold CV run
# (results/pilot_cv5), that is about 470 laptop core-hours, mostly rieszboost
# (ATE, TSM). Savio has run these studies about 3x slower per core: roughly
# 1,400 core-hours: 32 array tasks of 32 cores, about 1.4 hours each (2-hour
# limit), at most 10 running at once (the %10 in --array), so about 5.5 hours
# in all. Each shard starts its slowest tasks first. A task that times out is
# finished by resubmitting. To change the split, change the --array range and
# N_SHARDS together.
#
# Submit (from examples/simulations, so logs/ is created before Slurm opens it):
#   mkdir -p logs && sbatch savio.sh
# Check a smoke run first (one shard, 2 replicates, a few minutes):
#   mkdir -p logs && REPS=0:2 MID_REPS=50:52 sbatch --array=0 --time=1:00:00 savio.sh
# Then, on a login node (the results are on scratch, not in this folder):
#   RIESZ_SIM_OUT=/global/scratch/users/$USER/rieszreg_sim uv run python pilot.py summarize
# and copy them back for the comparison with the sample-splitting pilot:
#   rsync -av savio:/global/scratch/users/<user>/rieszreg_sim/pilot_cf5 results/
#   python pilot.py summarize --compare results/pilot_cv5
#
#SBATCH --job-name=riesz_pilot_cf5
#SBATCH --account=co_biostat
#SBATCH --partition=savio3
#SBATCH --qos=biostat_savio3_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --array=0-31%10
#SBATCH --output=logs/pilot_cf5_%A_%a.log

set -euo pipefail
N_SHARDS=${N_SHARDS:-32}         # must match the --array range (0 to N_SHARDS - 1)
REPS=${REPS:-0:50}
MID_REPS=${MID_REPS:-50:200}
SHARD="$SLURM_ARRAY_TASK_ID/$N_SHARDS"

export RIESZ_SIM_OUT=${RIESZ_SIM_OUT:-/global/scratch/users/$USER/rieszreg_sim}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/global/scratch/users/$USER/.uv-cache}
export PYTHONUNBUFFERED=1

cd "$HOME/rieszreg/examples/simulations"
echo "shard $SHARD on $SLURM_CPUS_PER_TASK cores; reps $REPS, middle rungs $MID_REPS"
echo "code: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ../../packages || echo ' with uncommitted changes')"

uv run python -u pilot.py run --reps "$REPS" --shard "$SHARD" --jobs "$SLURM_CPUS_PER_TASK"
uv run python -u pilot.py run --blocks ate --dgps mix33_1 mix67_1 mix33_2 mix67_2 \
    --reps "$MID_REPS" --shard "$SHARD" --jobs "$SLURM_CPUS_PER_TASK"
