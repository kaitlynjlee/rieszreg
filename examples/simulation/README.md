# Simulation: do rieszboost and riesznet work?

This study compares Riesz representers fit by rieszboost and by riesznet with
two other ways of getting a representer. One is the usual propensity plug-in,
an XGBoost classifier that is then inverted. The other is the oracle, the true
α₀. Every representer goes into a cross-fitted one-step estimator of the ATE
and the ATT. The design is in `PLAN.md`, and the write-up is in the Overleaf
paper.

| File | Contents |
|---|---|
| `dgp.py` | The DGP, the true α₀, the true ψ₀, and the DGP diagnostics |
| `learners.py` | Learners, fixed settings, tuning grids, and held-out losses |
| `run.py` | One cell (n, rep) end to end, the cache, and the command line |
| `summaries.py` | Tuning checks, performance tables with MCSEs, and pilot contrasts |
| `outputs.py` | The paper's `.tex` tables and `.pdf` figures |
| `pilot_check.py` | Resolvability check (from the `design-and-report-simulations` skill) |
| `savio_run.ipynb` | Interactive runs on Savio (or anywhere): run, tuning checks, tables, pilot check |
| `_make_notebook.py` | Writes `savio_run.ipynb`; edit this, then `python3 _make_notebook.py savio_run.ipynb` |
| `lee_schuler.py` | Lee & Schuler (2025) Section 3.1 with their protocol (one 500/500 split, their grid) |
| `diagnose_boosting.py`, `diagnose_nuisance.py` | Diagnostics: boosting learning curves, and the mu-hat x alpha-hat crossing |

Run from this directory:

```sh
../../.venv/bin/python dgp.py                                  # DGP diagnostics
../../.venv/bin/python run.py --dgp five_confounders --n 500 1000 2000 --reps 0:100 --jobs 9
../../.venv/bin/python summaries.py                            # print every summary
uv run --with matplotlib python outputs.py                     # tables and figures
```

`run.py` writes one results file per cell to `results/cells/<dgp>/`. That
directory is the record. Fits are cached in `cache/<dgp>/`, one file for each
cell, learner and setting. A cache key covers the data, the folds, the
learner's source code, its setting, and the package source and versions.
Deleting `cache/` changes only the runtime. Both directories are git-ignored.
Set `RIESZ_SIM_OUT` to put both somewhere else, such as cluster scratch
storage.

Each cell's data come from `numpy.random.default_rng([20260924, crc32(dgp), n,
rep])`, so any cell reruns alone and gets the same data. `shared_terms` keeps
its original seed, `[20260924, n, rep]`, so it stays paired with the archived
runs. `--reps` is half-open: `0:100` runs reps 0 to 99.
