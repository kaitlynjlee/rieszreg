#!/usr/bin/env bash
# Savio job array for lee_schuler_icml.py. Each array task takes one block of
# REPS_PER_TASK replicates and runs every (replicate, component) pair in it,
# one single-threaded process per core. Tasks resume: a pair whose cache file
# exists is skipped, so resubmitting the same array fills only the gaps.
#
# One-time setup, on a Savio login node:
#   git clone https://github.com/kaitlynjlee/icml_rieszboost.git ~/icml_rieszboost
#   git -C ~/icml_rieszboost checkout e8eef99      # the commit the replication was built against
#   (and the rieszreg repo + `uv sync --all-packages --all-extras`, as in savio_run.ipynb)
#
# Cost: one replicate is about 95 CPU-minutes (all 15 components), and the
# longest single task (the authors' ATE RieszBoost) about 20 minutes, so a
# 20-replicate block takes about an hour on 32 cores.
#
# Pilot (reps 0-99):   sbatch --array=0-4  savio_lee_schuler_icml.sh
# Full (reps 0-999):   sbatch --array=0-49 savio_lee_schuler_icml.sh
# Then, anywhere with the cache:  python lee_schuler_icml.py summarize
#
#SBATCH --job-name=ls_icml
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=logs/ls_icml_%A_%a.log

set -euo pipefail
REPS_PER_TASK=${REPS_PER_TASK:-20}
LO=$(( SLURM_ARRAY_TASK_ID * REPS_PER_TASK ))
HI=$(( LO + REPS_PER_TASK ))

export RIESZ_SIM_OUT=/global/scratch/users/$USER/rieszreg_sim
export ICML_RIESZBOOST=$HOME/icml_rieszboost
export UV_CACHE_DIR=/global/scratch/users/$USER/.uv-cache
export PYTHONUNBUFFERED=1

cd "$HOME/rieszreg/examples/simulation"
mkdir -p logs
echo "array task $SLURM_ARRAY_TASK_ID: reps $LO to $((HI - 1)) on $SLURM_CPUS_PER_TASK cores"

# The authors' RieszNet needs JAX + equinox; they go on top of the project env
# without changing it. (Their pinned equinox 0.11.10 does not import under
# jax 0.9; 0.13.8 does.)
uv run --with equinox==0.13.8 --with optax==0.2.6 --with polars --with rich \
       --with chex --with jaxtyping --with tqdm \
    python -u lee_schuler_icml.py run --reps "$LO:$HI" --jobs "$SLURM_CPUS_PER_TASK"
