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
| ForestRiesz | the published ForestRiesz settings (package defaults) | minimum leaf size {5, 10, 20, 50, 100} |

ForestRiesz is not in the manuscript.

`rieszboost_l2` is a second rieszboost learner, not the manuscript's protocol, fit on the same datasets so each replicate gives a paired comparison with `rieszboost`:

- **Grid:** learning rate {3e-3, 1e-2, 3e-2, 1e-1} × depth {1, 2, 3, 5} × L2 penalty on leaf values `reg_lambda` {10, 30, 100, 300}, with early-stopping patience fixed at 50 and a subsample of 0.8 of individuals per round (rieszboost uses 0.9). Otherwise its fixed settings are rieszboost's.
- **Why the learning rate moved up:** in the pilot the regret surface was flat along the learning rate, 1e-3 took most of the compute and produced every refit at the tree cap, and the ATE chose 3e-2, the top, 43% of the time.
- **Why patience is fixed:** riesznet's patience isn't tuned, so this one isn't either. In the pilot's patience grid {10, 50, 200}, 50 had the smallest median regret for both estimands.
- **Why this grid:** the pilot of the manuscript grid chose the largest learning rate and the smallest depth often, never chose 1e-5, and chose 1e-4 mostly in fits that hit the tree cap.
- **Why λ moved up:** with λ ∈ {0, 1, 10}, λ = 10 was chosen in 73% (ATE) and 67% (ATT) of replicates and had the smallest median regret for both. With {3, 10, 30, 100}, the smallest median regret was again at the top, λ = 100, for both estimands, so λ moved up once more.
- **What the L2 penalty does:** it shrinks a leaf of n augmented rows by 2n / (2n + λ), which damps the few-row leaves that produce extreme α̂ where overlap is weak.
- **Patience grids cost no extra fits:** the code still accepts a patience grid. Each fit runs at the largest patience, and early stopping at the smaller ones is replayed on its per-tree validation-loss path, which equals a fresh fit at that patience. `replay_matches_booster` in each record checks the replay at the largest patience against the booster's own stopping point.

To add it to replicates that are already cached, fit only this learner:

```sh
RIESZ_SIM_OUT=/global/scratch/users/$USER/rieszreg_sim uv run python run.py run --reps 0:120 \
    --components rieszboost_l2_ATE rieszboost_l2_ATT --jobs 32
```

Run this inside a Slurm job. A plain `sbatch savio.sh` array fits it as well, since every uncached pair gets fit.

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
