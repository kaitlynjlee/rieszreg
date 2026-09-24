#!/bin/bash
# Savio SLURM job for run_backend_comparison.py. One node, one task, many
# cores -- reps are embarrassingly parallel, so --n_jobs=$SLURM_CPUS_PER_TASK
# runs that many reps concurrently inside a single Python process (see
# `Parallel(n_jobs=...)` in run_backend_comparison.py), rather than
# using a SLURM array.
#
# FILL IN before submitting: --account, --partition, --qos, --time, the
# module/conda lines, and BACKEND_COMPARISON_DIR below.
#SBATCH --job-name=rieszboost-backend-comparison
#SBATCH --account=<FILL_IN: e.g. fc_yourgroup or co_yourgroup>
#SBATCH --partition=<FILL_IN: e.g. savio3>
#SBATCH --qos=<FILL_IN: e.g. savio_normal>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=<FILL_IN: e.g. 04:00:00>
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# Pin every worker process to one BLAS/OMP thread. run_backend_comparison.py
# spawns $SLURM_CPUS_PER_TASK worker processes (one per concurrent rep); if
# each of those also tries to multi-thread its own xgboost/BLAS calls, you
# oversubscribe the node (24 workers x 24 threads each on 24 cores). These
# vars are read once at library-load time and ignored after, so they must be
# set here, before python starts -- see
# packages/rieszreg/python/rieszreg/_omp.py for the same caveat applied to
# torch+xgboost coexistence.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module load <FILL_IN: e.g. python/3.11 or anaconda3>
source activate <FILL_IN: your conda env name>
# One-time setup of that env (run once, not per job):
#   conda create -n rieszreg python=3.11
#   conda activate rieszreg
#   pip install -e packages/rieszreg/python -e packages/rieszboost/python \
#       -e packages/krrr/python -e packages/forestriesz/python \
#       -e packages/riesznet/python -e packages/riesztree/python \
#       xgboost scikit-learn pandas numpy joblib

BACKEND_COMPARISON_DIR=<FILL_IN: /path/to/rieszreg/examples/backend_comparison>
cd "$BACKEND_COMPARISON_DIR"
mkdir -p logs results

python run_backend_comparison.py \
    --n_reps 20 \
    --n_jobs "${SLURM_CPUS_PER_TASK}" \
    --out_csv "results/run_${SLURM_JOB_ID}.csv"
