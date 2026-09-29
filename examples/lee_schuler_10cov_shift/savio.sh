#!/usr/bin/env bash
# Savio job array for run.py. Each array task takes one block of
# REPS_PER_TASK replicates and runs every (replicate, learner) pair in it, one
# single-threaded process per core. Tasks resume: a pair whose cache file
# exists is skipped, so resubmitting the same array fills only the gaps.
# With OVERWRITE=1, every pair is refit and its cache file replaced instead.
#
# One-time setup, on a Savio login node (as in examples/simulation/savio_run.ipynb):
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   echo 'export UV_CACHE_DIR=/global/scratch/users/$USER/.uv-cache' >> ~/.bashrc
#   git clone https://github.com/kaitlynjlee/rieszreg.git ~/rieszreg
#   cd ~/rieszreg && git checkout simulations-shift && uv sync --all-packages --all-extras
#
# Cost: on a laptop one replicate is about 19 CPU-minutes. The outcome
# regression (10 min) is the longest task, then the classifier (4-5 min) and
# rieszboost_l2 (1-3 min per estimand). Savio ran ../lee_schuler_10cov about 3x
# slower than the laptop, so expect about 1 CPU-hour per replicate, and about
# 1.5 hours for a block of 40 replicates on 32 cores.
#
# Pilot (reps 0-119):  sbatch --array=0-2  savio.sh
# Full (reps 0-999):   sbatch --array=0-24 savio.sh
# Redo:                OVERWRITE=1 sbatch --array=0-2 savio.sh   (refits and replaces those reps)
# Some learners only:  COMPONENTS="rieszboost_l2_ASE rieszboost_l2_LASE" sbatch --array=0-2 savio.sh
# Then, on a login node (the results are on scratch, not in this folder):
#   RIESZ_SIM_OUT=/global/scratch/users/$USER/rieszreg_sim uv run python run.py summarize
#
#SBATCH --job-name=ls_shift
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --output=logs/ls_shift_%A_%a.log

set -euo pipefail
REPS_PER_TASK=${REPS_PER_TASK:-40}
LO=$(( SLURM_ARRAY_TASK_ID * REPS_PER_TASK ))
HI=$(( LO + REPS_PER_TASK ))

export RIESZ_SIM_OUT=${RIESZ_SIM_OUT:-/global/scratch/users/$USER/rieszreg_sim}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/global/scratch/users/$USER/.uv-cache}
export PYTHONUNBUFFERED=1

cd "$HOME/rieszreg/examples/lee_schuler_10cov_shift"
mkdir -p logs
echo "array task $SLURM_ARRAY_TASK_ID: reps $LO to $((HI - 1)) on $SLURM_CPUS_PER_TASK cores"
echo "code: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ../../packages || echo ' with uncommitted changes')"
uv run python -u run.py run --reps "$LO:$HI" --jobs "$SLURM_CPUS_PER_TASK" ${OVERWRITE:+--overwrite} ${COMPONENTS:+--components $COMPONENTS}
