"""Laptop-sized version of run_backend_comparison.py.

Same DGP, learners, and estimators, with a smaller sample size, fewer
replicates, and fewer tuning folds so it finishes in a few minutes.
Edit the settings below to scale it up.

Run from anywhere:
    .venv/bin/python examples/backend_comparison/run_local.py
"""

import os
import sys
from glob import glob
from pathlib import Path

# One thread per process: the reps already run in parallel, and torch
# (riesznet) and xgboost (rieszboost) can deadlock on macOS when both use
# multithreaded OpenMP. These must be set before either library is imported.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"

# Put the workspace packages and this folder on the path explicitly, so the
# imports work even when the .venv's editable-install .pth files are skipped.
# PYTHONPATH is set too so the parallel worker processes inherit it.
HERE = Path(__file__).resolve().parent
PATHS = [str(HERE)] + sorted(glob(str(HERE.parents[1] / "packages" / "*" / "python")))
sys.path[:0] = PATHS
os.environ["PYTHONPATH"] = os.pathsep.join(PATHS + [os.environ.get("PYTHONPATH", "")])

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from run_backend_comparison import ESTIMANDS, run_rep, summarize

""" Settings """
N_REPS = 10
N_DATA = 1000
CV_FOLDS = 3
N_MC_TRUTH = 500_000
N_JOBS = max(1, (os.cpu_count() or 2) - 2)   # leave a couple of cores free
OUT_CSV = HERE / "results_local.csv"

""" Main execution """
if __name__ == "__main__":
    true_psi = {
        name: dgp_cls.true_psi(np.random.default_rng(12345), N_MC_TRUTH)
        for name, (dgp_cls, _, _) in ESTIMANDS.items()
    }
    print(f"True psi: {true_psi}", flush=True)

    rows = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_rep)(rep, n=N_DATA, cv_folds=CV_FOLDS) for rep in range(N_REPS)
    )
    results = pd.DataFrame([row for rep_rows in rows for row in rep_rows])
    results["truth"] = results.estimand.map(true_psi)
    results.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    print(summarize(results).round(3).to_string())
