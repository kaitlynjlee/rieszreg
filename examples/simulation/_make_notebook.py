import json, sys

cells = []
def md(s): cells.append({"id": f"c{len(cells):02d}", "cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})
def code(s): cells.append({"id": f"c{len(cells):02d}", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""
# rieszboost and riesznet simulation: interactive runs

This notebook runs the simulation pipeline in `run.py` for a chosen DGP, set of
sample sizes and replicates, then prints the tuning checks, the performance
tables, and the pilot resolvability check. It calls the same code as the
command line, so results from here, from a laptop, and from a later SLURM job
are interchangeable.

Runs resume. Each (DGP, n, rep) cell writes its own results file, and a cell
whose file exists is skipped, so an interrupted session picks up where it
stopped. Fits are cached per setting, so changing one grid refits only the new
settings.

## One-time setup on Savio

In a Savio terminal (login node, or a terminal in Open OnDemand):

```bash
# 1. uv, with its package cache on scratch (home quota is small)
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export UV_CACHE_DIR=/global/scratch/users/$USER/.uv-cache' >> ~/.bashrc
source ~/.bashrc

# 2. the repository and its environment
git clone https://github.com/kaitlynjlee/rieszreg.git ~/rieszreg
cd ~/rieszreg
git checkout simulations
uv sync --all-packages --all-extras
uv pip install ipykernel

# 3. a Jupyter kernel that uses this environment
.venv/bin/python -m ipykernel install --user --name rieszreg --display-name "Python (rieszreg)"
```

If `examples/simulation/` is not yet committed and pushed, copy it from your laptop instead:

```bash
rsync -av --exclude cache --exclude results \
    ~/Documents/research/rieszreg_sims/rieszreg/examples/simulation/ \
    YOUR_SAVIO_USERNAME@dtn.brc.berkeley.edu:~/rieszreg/examples/simulation/
```

Then start a Jupyter server in Open OnDemand (https://ood.brc.berkeley.edu),
ask for as many cores as the run needs (a full node is simplest), open this
notebook from `~/rieszreg/examples/simulation/`, and pick the kernel
**Python (rieszreg)**.

## Rough cost

With the current grids, one cell takes about 5 core-minutes at n = 500 and
several times that at n = 2000. Most of it is rieszboost at learning rate
0.0003, which runs to about 20,000 trees. One replicate at all three sample
sizes is roughly 30 to 45 core-minutes, so 100 replicates are about 50 to 75
core-hours, or 1 to 1.5 hours on a 56-core node. Section 5 reports the actual
minutes per cell, so check it after a first small run.
""")

md("## 1. Settings\n\nEdit this cell, then run the notebook top to bottom.")
code(r"""
import os
from pathlib import Path

DGP = "five_confounders"        # "five_confounders", "shared_terms" or "lee_schuler"
N_OBS = [500, 1000, 2000]       # sample sizes
REPS = range(0, 20)             # replicates, half-open: range(0, 20) is reps 0..19
N_SIM_PLANNED = 500             # replicate count the pilot check sizes against

# Where the repository lives, and where fits and results go. On Savio, keep
# the output on scratch; the cache can reach several GB.
REPO = Path.home() / "rieszreg"
OUT_ROOT = Path(os.environ.get("RIESZ_SIM_OUT",
                               f"/global/scratch/users/{os.environ.get('USER', 'me')}/rieszreg_sim"))

# One worker per core, less one for the notebook itself. Each fit uses a single thread.
try:
    CORES = len(os.sched_getaffinity(0))
except AttributeError:          # macOS
    CORES = os.cpu_count()
N_JOBS = max(1, CORES - 1)
""")

md("## 2. Environment\n\n`_env` must be imported before numpy, xgboost or torch: it pins every fit to one thread and puts the workspace packages on the path.")
code(r"""
import sys
SIM_DIR = REPO / "examples" / "simulation"
os.environ["RIESZ_SIM_OUT"] = str(OUT_ROOT)     # read by run.py at import
OUT_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SIM_DIR))
os.chdir(SIM_DIR)

import _env  # noqa: F401  (threads and import paths; must precede numpy)

import time
import numpy as np
import pandas as pd
import sklearn, torch, xgboost

import dgp
import learners as L
import run
import summaries as S

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(f"python {sys.version.split()[0]}, numpy {np.__version__}, xgboost {xgboost.__version__}, "
      f"torch {torch.__version__}, scikit-learn {sklearn.__version__}")
print(f"cores {CORES}, workers {N_JOBS}, OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
print(f"results -> {run.CELL_DIR / DGP}")
print(f"cache   -> {run.CACHE_DIR / DGP}")
""")

md("## 3. The DGP and the learners\n\nThe DGP diagnostics on one draw of 10^6 observations (overlap, variance explained, how nonlinear the truth is, true estimands), and the tuning grid of every learner type.")
code(r"""
G = dgp.DGPS[DGP]
pd.Series(G.diagnose()).round(4).to_frame(DGP)
""")
code(r"""
pd.DataFrame([dict(learner=name, fixed=str(fixed), settings=len(grid),
                   grid={k: sorted({s[k] for s in grid}) for k in grid[0]})
              for name, (func, fixed, grid) in L.LEARNERS.items()])
""")

md("## 4. Run\n\nCells run largest n first. A cell whose results are on disk and were made by the current learners and grids is skipped, so rerunning this is safe. After a grid change, those cells rerun (fits for unchanged settings still come from the cache).")
code(r"""
from joblib import Parallel, delayed

cells = [(DGP, n, r) for n in sorted(N_OBS, reverse=True) for r in REPS]
todo = [c for c in cells if not run.is_done(*c)]   # done = results exist AND match the current learners and grids
print(f"{len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run on {N_JOBS} workers")

t0 = time.perf_counter()
if todo:
    jobs = Parallel(n_jobs=N_JOBS, batch_size=1, return_as="generator_unordered")(
        delayed(run.run_cell)(*c) for c in todo)
    for i, _ in enumerate(jobs, 1):
        if i % max(1, len(todo) // 20) == 0 or i == len(todo):
            el = time.perf_counter() - t0
            print(f"{i}/{len(todo)} cells, {el / 60:.1f} min elapsed, "
                  f"about {el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
""")

md("## 5. What was run\n\nReplicates per n, failed fold fits (a failed fit is recorded as missing and the replicate continues), and the time per cell.")
code(r"""
res = run.load_results(DGP, n_obs=N_OBS, reps=REPS)
est, tuning, nuis = res["estimates"], res["tuning"], res["nuisance"]
secs = pd.Series({(n, r): pd.read_pickle(run.CELL_DIR / DGP / f"n{n}_rep{r}.pkl")["seconds"]
                  for (_, n, r) in cells if run.is_done(DGP, n, r)})
pd.DataFrame({
    "replicates": est.groupby("n")["rep"].nunique(),
    "failed fold fits": tuning.groupby("n")["failed"].sum(),
    "minutes per cell": secs.groupby(level=0).mean() / 60,
}).round(2)
""")

md(r"""
## 6. Tuning checks

A comparison of learners is fair only if each is tuned close to its best.

**Edge rule.** For each learner type and n, this is the setting with the smallest median cross-validated regret over replicates. Regret is a setting's risk minus the smallest risk in its grid on the same dataset. If a tuned hyperparameter of that setting sits at the smallest or largest grid value, the grid should move that way. A depth of 1 and a weight decay of 0 are hard limits and exempt.

**Cap check.** The share of selected fits where early stopping kept at least 90% of the tree or epoch cap in some fold.
""")
code(r"""
S.best_setting_edges(tuning)
""")
code(r"""
S.cap_check(tuning).round(2)
""")
code(r"""
# share of replicates selecting each setting, top three per learner and n
st = S.selection_table(tuning)
st.sort_values("share", ascending=False).groupby(["learner", "n"]).head(3).sort_values(["learner", "n"]).round(2)
""")

md("## 7. Nuisance and representer accuracy\n\nRMSE against the truth over each dataset's out-of-fold predictions, averaged over replicates, with Monte Carlo SEs. The oracle has zero error and is left out.")
code(r"""
S.nuisance_accuracy(nuis).round(3)
""")
code(r"""
acc = S.alpha_accuracy(est)
acc.pivot_table(index=["estimand", "n"], columns="method", values=["alpha_rmse", "mcse"]).round(3)
""")

md(r"""
## 8. The one-step estimator

Bias, empirical SE, the ratio of the mean model SE to the empirical SE, RMSE, RMSE relative to the oracle on the same replicates, and 95% coverage, each with its Monte Carlo SE. Every method uses the same cross-fitted $\hat\mu$, so the rows differ only in $\hat\alpha$.
""")
code(r"""
perf = S.psi_performance(est, DGP)
cols = ["n", "method", "reps", "bias", "bias_mcse", "emp_se", "se_ratio", "rmse", "rel_rmse",
        "coverage", "coverage_mcse"]
print(f"true values: {S.true_values(DGP)}")
perf[perf.estimand == "ATE"][cols].round(3)
""")
code(r"""
perf[perf.estimand == "ATT"][cols].round(3)
""")

md(r"""
## 9. Pilot check: how many replicates does the full run need?

Every contrast becomes a per-replicate difference $d$. At $n_{sim}$ replicates its Monte Carlo SE is $\mathrm{sd}(d)/\sqrt{n_{sim}}$. `n_sim_safe` is the number of replicates that resolves the gap at 5 MCSEs, computed from a lower 95% bound on the gap so a lucky pilot does not undersize the run. A pilot $|z|$ below 2 means this pilot cannot size that contrast: the gap may be zero, or the pilot too small. "riesznet − oracle" contrasts that stay unresolved are what equivalence looks like, and need a stated margin instead.
""")
code(r"""
pc = S.pilot_contrasts(est, k=5, n_sim_planned=N_SIM_PLANNED, dgp_name=DGP)
pc = pc[["display", "estimand", "n", "contrast", "gap", "sd_d", "pilot_z", "n_sim_safe", "verdict"]]
pc["verdict"] = pc["verdict"].str.split(":").str[0].str.split("(").str[0]
pc.sort_values(["display", "estimand", "n"]).round(4)
""")

md("## 10. Figure (optional)\n\nMean RMSE of $\\hat\\alpha$ against $n$, with ±2 MCSE bars. Needs matplotlib in the kernel (`uv pip install matplotlib`).")
code(r"""
try:
    import outputs
    fig_path = OUT_ROOT / "results" / f"alpha_rmse_{DGP}.pdf"
    outputs.alpha_figure(acc, fig_path)
    print(f"saved {fig_path}")
except ImportError as e:
    print(f"skipped: {e}")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (rieszreg)", "language": "python", "name": "rieszreg"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
