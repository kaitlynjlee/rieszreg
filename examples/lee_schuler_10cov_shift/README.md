# Lee & Schuler 10-confounder continuous-exposure DGP: ASE and LASE

This folder is the continuous-exposure companion to `../lee_schuler_10cov`. It runs our Riesz learners on the DGP of the RieszBoost ICML manuscript (Lee & Schuler, Section 3.4 and Appendix C.2.2), for two estimands of the shift A → A + δ with δ = 1:

- the average shift effect, ψ_ASE = E[μ(A + δ, X) − μ(A, X)] = 4.786
- the local average shift effect, ψ_LASE = E[μ(A + δ, X) − μ(A, X) | A < 0] = 4.058

The learners:

- `rieszboost_l2`: gradient boosting on the augmented data, with the grid `../lee_schuler_10cov` settled on
- `riesznet`: a PyTorch neural network
- `forestriesz`: `forestriesz.ForestRieszRegressor`, the non-augmented ForestRiesz of Chernozhukov et al. (2022) on EconML's GRF, with a basis in A (below)
- `classification`, for the ASE only: the manuscript's indirect comparator (its "Indirect" row, Appendix B.2.2). An XGBoost classifier separates observed rows (A, X) from shifted rows (A + δ, X), and its odds estimate the density ratio. The manuscript says no comparable method exists for the LASE.

Two oracles are included as references. `oracle` uses the true representer with the estimated outcome regression, so a learner's gap to it is the cost of estimating α alone. `oracle_mu0` uses the true representer and the true outcome regression, which checks the pipeline and the Monte Carlo noise.

| File | Contents |
|---|---|
| `dgp.py` | The DGP, the true α₀, the true ψ₀, and DGP diagnostics |
| `run.py` | Learners, tuning grids, the per-replicate pipeline and cache, and the summary tables |
| `savio.sh` | Slurm job array for Savio |

## The DGP

X₁, …, X₁₀ are Uniform(0, 1). A | X is Normal with mean m(X) and standard deviation 3, and Y | A, X is N(μ(A, X), 1). `dgp.py` gives m and μ.

- **Exposure scale.** The manuscript prints the variance of A | X as 3. The authors' code uses standard deviation 3, and only that reproduces the manuscript's ψ values: variance 3 gives ψ_ASE = 5.098 and ψ_LASE = 4.289. This study uses standard deviation 3.
- **Diagnostics** (`python dgp.py`, one draw of 10⁶):

| Diagnostic | Value |
|---|---|
| P(A < 0), the share the LASE shifts | 0.354 |
| P(0 ≤ A < 1) | 0.123 |
| Density ratio r = p(a − 1 \| x) / p(a \| x): 1%, 50%, 99% quantiles, max | 0.44, 0.95, 2.05, 5.1 |
| sd of α₀: ASE, LASE | 0.34, 0.37 |
| Var(μ) / Var(Y) | 0.996 |

The shift keeps overlap good: r stays near 1. μ explains almost all of Var(Y), because the 3A term dominates.

The representers are those of the manuscript's Appendix B. The LASE's is that of the partial parameter E[1(A < 0)(μ(A + 1, X) − μ(A, X))]. The one-step estimator divides by the estimated P(A < 0) and uses the delta-method influence function, as for the ATT in `../lee_schuler_10cov`.

## Protocol

This follows the manuscript's Section 3.4 and Appendix C.2.3, as `../lee_schuler_10cov` does.

- **Data.** Each replicate draws an estimation set of 500 and then an independent training set of 500. The ASE and the LASE use the same datasets, so the two estimands pair replicate by replicate. The authors ran them in separate scripts, on separate draws.
- **Tuning.** Every learner is tuned by 5-fold cross-validation on the training set: squared error for the outcome regression, log loss for the classifier, and the squared Riesz loss for the representers. The selected setting is refit on the whole training set.
- **Estimation.** The one-step (DML) estimator is evaluated on the estimation set.
- **Shared outcome regression.** Each replicate fits one XGBoost outcome regression, and every representer and both estimands use it.

Every grid is the one `../lee_schuler_10cov` uses:

| Learner | Fixed | Tuned |
|---|---|---|
| outcome (XGBoost) | patience 200, cap 20000 trees, subsample 0.9, 20% early-stopping split | learning rate {1e-5, 1e-4, 1e-3, 1e-2} × depth {3, 5, 7} |
| classification (XGBoost) | the same, with the early-stopping split by individual | the same grid |
| rieszboost_l2 | patience 50, cap 20000 trees, subsample 0.8 of individuals per round | learning rate {3e-3, 1e-2, 3e-2, 1e-1} × depth {1, 2, 3, 5} × `reg_lambda` {10, 30, 100, 300} |
| riesznet | 3 hidden layers of 200, ELU, Adam, weight decay 1e-3, batch 64, patience 10, cap 1000 epochs | learning rate {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} |
| ForestRiesz | the published ForestRiesz settings (package defaults), the basis below, splits on X only | minimum leaf size {5, 10, 20, 50, 100} |

The rieszboost_l2 grid was tuned on the ATE and the ATT (see `../lee_schuler_10cov/README.md`). It is only a starting point here. The pilot's tuning check says whether it has to move for the shift estimands. The manuscript-grid rieszboost is not run here, so the manuscript's RieszBoost rows are a reference, not a paired comparison.

### ForestRiesz's basis

The published ForestRiesz fits α(a, x) = θ(x)ᵀφ(a). Its forest splits on x only, and φ is a basis in the treatment. `../lee_schuler_10cov` uses [1 − a, a]. A continuous exposure needs a basis we choose, and this one uses only the estimand (δ and the threshold t = 0), never the DGP:

- **ASE:** 1, u, u², with u = (a − t) / 3. Each leaf's θ is the L2 projection of α₀ onto these polynomials. At degree 1 that is δ(a − Ā_leaf) / Var_leaf(A), the first-order expansion of any location-family density ratio. Degree 2 adds the curvature a density ratio has.
- **LASE:** α₀ is 0 for a ≥ t + δ, because α enters the loss there only through α². It can jump at t and at t + δ, because m(α) evaluates α only below t + δ and subtracts α(a) only below t. So the basis is 1, u, u² on a < t, plus a constant on t ≤ a < t + δ.
- **How the degrees were chosen:** on replicates 0–7, by α̂ RMSE against the truth, before the pilot. The interval t ≤ a < t + δ holds about 12% of rows, a few per leaf. Polynomials there made the leaf solves unstable. At leaf size 20, α̂ RMSE was 0.22 with a constant there, 0.29 with degree 1, and 0.41 with degree 2 (up to 7.5 at leaf size 10). For the ASE at leaf size 50, degrees 1, 2 and 3 gave 0.126, 0.105 and 0.147.
- **Where to look first:** larger leaves were better throughout that check, so ForestRiesz's leaf size is the first thing to check against the top edge in the pilot.

## Running

```sh
../../.venv/bin/python dgp.py                          # true values and diagnostics
../../.venv/bin/python run.py run --reps 0:2 --jobs 8  # smoke test
../../.venv/bin/python run.py summarize
```

On Savio, see the header of `savio.sh`:

- Pilot (120 replicates): `sbatch --array=0-2 savio.sh`
- Full run (1000 replicates): `sbatch --array=0-24 savio.sh`

Each (replicate, learner) fit runs in its own Python process, as in `../lee_schuler_10cov`, so a crash in native code loses only that fit. On Savio, xgboost once aborted there with "stack smashing detected", and under a shared worker pool that stopped the whole array task. A crashed fit is retried once. If it crashes again, it's recorded as a failed fit, which `summarize` counts in its `failed` column.

`run.py` writes one cache file per (replicate, learner) to `cache/lee_schuler_10cov_shift/<learner>/`. Each file holds the learner's estimation-set predictions (α at A and at A + δ) and the cross-validated risk of every setting. The cache key covers:

- the learner's source and settings
- the data and CV code
- the package versions
- the source of the rieszreg packages it uses

Changing one learner refits only that learner. `summarize` builds the tables from the cache and writes them to `results/lee_schuler_10cov_shift/`. It also reports how often each learner lands on an edge of its grid, and the median regret of each setting. Set `RIESZ_SIM_OUT` to put both directories somewhere else, such as scratch. `status` and `rekey` work as in `../lee_schuler_10cov`.
