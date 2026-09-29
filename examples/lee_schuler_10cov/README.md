# Lee & Schuler 10-confounder binary DGP: rieszboost, riesznet, ForestRiesz

This folder runs three of our Riesz learners on the binary-treatment DGP of the RieszBoost ICML manuscript (Lee & Schuler, Section 3.3 and Appendix C.2.1), for the ATE and the ATT:

- `rieszboost` (gradient boosting on the augmented data)
- `riesznet` (a PyTorch neural network)
- `forestriesz.ForestRieszRegressor`, the non-augmented ForestRiesz of Chernozhukov et al. (2022) on EconML's GRF

The oracle, which uses the true representer, is included as a reference. It shares the outcome regression with every learner, so a learner's gap to the oracle is the cost of estimating α alone.

| File | Contents |
|---|---|
| `dgp.py` | The DGP, the true α₀, the true ψ₀, and DGP diagnostics |
| `run.py` | Learners, tuning grids, the per-replicate pipeline and cache, and the summary tables |
| `savio.sh` | Slurm job array for Savio |

## Protocol

This follows the manuscript's Section 3.3 and Appendix C.2.3.

- **Data.** Each replicate draws an estimation set of 500 and then an independent training set of 500.
- **Tuning.** Every learner is tuned by 5-fold cross-validation on the training set: squared error for the outcome regression, and the squared Riesz loss for the representers. The selected setting is refit on the whole training set.
- **Estimation.** The one-step (DML) estimator is evaluated on the estimation set.
- **Shared outcome regression.** Each replicate fits one XGBoost outcome regression, and every representer uses it.

| Learner | Fixed | Tuned |
|---|---|---|
| outcome (XGBoost) | patience 200, cap 20000 trees, subsample 0.9, 20% early-stopping split | learning rate {1e-5, 1e-4, 1e-3, 1e-2} × depth {3, 5, 7} |
| rieszboost | the same, with no L2 penalty on leaves | the same grid |
| riesznet | 3 hidden layers of 200, ELU, Adam, weight decay 1e-3, batch 64, patience 10, cap 1000 epochs | learning rate {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} |
| ForestRiesz | the published ForestRiesz settings (package defaults) | minimum leaf size {2, 5, 10, 20, 50} |

ForestRiesz is not in the manuscript.

On the first replicate, the outcome regression and rieszboost both picked learning rate 0.01, the largest value in the grid. The `summarize` tuning check reports how often each learner lands on an edge of its grid. If one lands there consistently, its grid should move.

The seeds and draw order match `examples/simulation/lee_schuler_icml.py`, which also runs the authors' own code (`icml_rieszboost`) on the same datasets. The two studies therefore pair replicate by replicate.

## Running

```sh
../../.venv/bin/python dgp.py                          # true values and diagnostics
../../.venv/bin/python run.py run --reps 0:2 --jobs 8  # smoke test
../../.venv/bin/python run.py summarize
```

On Savio, see the header of `savio.sh`:

- Pilot (100+ replicates): `sbatch --array=0-2 savio.sh`
- Full run (1000 replicates): `sbatch --array=0-24 savio.sh`

`run.py` writes one cache file per (replicate, learner) to `cache/lee_schuler_10cov/<learner>/`. Each file holds the learner's estimation-set predictions and the cross-validated risk of every setting. The cache key covers:

- the learner's source and settings
- the data and CV code
- the package versions
- the source of the rieszreg packages it uses

Changing one learner refits only that learner. `summarize` builds the tables from the cache and writes them to `results/lee_schuler_10cov/`. Set `RIESZ_SIM_OUT` to put both directories somewhere else, such as scratch.
