#!/usr/bin/env bash
# Savio job array that adds classif_aug, the generic classification method
# (pilot.py, fit_classif_aug), to the final run in final_split. It runs only
# classif_aug tasks, on exactly the cells and replicates the final run has:
#   ate, att    the eight core DGPs (ATE_DGPS), n = 1000, 0-499;
#   tsm         additive, additive_g2, interaction, n = 250, 1000, 4000, 0-499;
#   shift, localshift   the four shift DGPs, n = 1000, 0-499.
# The five appendix DGPs of the ATE and ATT (tail_feature, bulk_feature,
# smooth, step, additive) are left out: classif_aug is not run on them, so
# their tables have no classif_aug column. additive stays in the TSM block.
# The alignment DGPs (hard_*, mix*) and tsm_align belong to the later paper
# and are left out too.
# The outcome tasks and every other method's results are already in
# final_split and are not touched: --methods classif_aug skips them, and the
# summary pairs classif_aug with them replicate by replicate (same seeds,
# same training and estimation samples).
#
# Before submitting: commit and push the pilot.py that defines classif_aug,
# then pull on Savio. The log prints the commit it ran.
#
# Cost. A 2-replicate local smoke run (8 processes) took, per task: ate 15 s,
# att 9 s, shift 9 s, localshift 8 s, tsm 2 / 3 / 8 s at n = 250 / 1000 /
# 4000. Over the 16,500 tasks that is about 42 core-hours; allow up to 1.5x
# on Savio3 cores. 8 array tasks on 32 cores: about 5-8 core-hours, so 10-15
# minutes each.
#
# Account and partition as in savio_final.sh (the co_biostat condo on savio3).
#
# Submit (from examples/simulations, so logs/ exists before Slurm opens it):
#   mkdir -p logs && sbatch savio_classif_aug.sh
# Tasks resume, so resubmitting fills only the gaps. Then, on a login node:
#   RIESZ_SIM_OUT=/global/scratch/users/$USER/rieszreg_sim RIESZ_SIM_RUN=final_split uv run python pilot.py summarize
#
#SBATCH --job-name=riesz_classif_aug
#SBATCH --account=co_biostat
#SBATCH --partition=savio3
#SBATCH --qos=biostat_savio3_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:30:00
#SBATCH --array=0-7
#SBATCH --output=logs/classif_aug_%A_%a.log

set -euo pipefail
N_SHARDS=${N_SHARDS:-8}          # must match the --array range (0 to N_SHARDS - 1)
SHARD="$SLURM_ARRAY_TASK_ID/$N_SHARDS"
ATE_DGPS="overlap_g05 overlap_g1 overlap_g2 overlap_g3 logistic sparse50 interaction steps10"

export RIESZ_SIM_OUT=${RIESZ_SIM_OUT:-/global/scratch/users/$USER/rieszreg_sim}
export RIESZ_SIM_RUN=final_split
export UV_CACHE_DIR=${UV_CACHE_DIR:-/global/scratch/users/$USER/.uv-cache}
export PYTHONUNBUFFERED=1

cd "$HOME/rieszreg/examples/simulations"
echo "shard $SHARD on $SLURM_CPUS_PER_TASK cores; classif_aug only"
echo "code: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ../../packages || echo ' with uncommitted changes')"
grep -q "def fit_classif_aug" pilot.py || { echo "pilot.py has no classif_aug: pull first"; exit 1; }

RUN=(uv run python -u pilot.py run --methods classif_aug --shard "$SHARD" --jobs "$SLURM_CPUS_PER_TASK")
# --dgps filters every block it is given, so the ATE and ATT get their own
# calls (additive is a TSM DGP too).
# shellcheck disable=SC2086  # the DGP lists are word lists
"${RUN[@]}" --blocks ate att --dgps $ATE_DGPS --reps 0:500
"${RUN[@]}" --blocks tsm shift localshift --reps 0:500
