# Simulation plan: do rieszboost and riesznet work?

Backwards design, as in the `design-and-report-simulations` skill. The claims
decide the mockups, and the mockups decide the DGP and settings.

## 1. Claims

| # | Claim | Evidence |
|---|---|---|
| 1 | Riesz representers fit directly by rieszboost or riesznet, with hyperparameters chosen by cross-validated Riesz loss, give one-step ATE and ATT estimates with small bias and close-to-nominal 95% coverage for n from 500 to 2000, when overlap is good. | simulation |
| 2 | Their error is close to the oracle that knows the true representer, and the gap closes as n grows. | simulation |
| 3 | They are at least as accurate as the usual route of fitting a propensity score and inverting it, using the same tuned boosting learner. | simulation |

The boundary for now is good overlap. The planned next factor, overlap
strength, is where we expect the claims to fail, and that failure is the
"and not when" half of the claims.

## 2. Sharpened goals

**Goal A (representer accuracy).** For the ATE and the ATT, at n ∈ {500, 1000,
2000}, under a 5-confounder DGP with a nonlinear propensity and good overlap:
is the average over datasets of the RMSE of the cross-fitted α̂ against α₀
smaller for rieszboost and for riesznet than for the propensity plug-in
(XGBoost classifier, tuned on log loss, π̂ clipped to [0.01, 0.99])? Does it
fall as n grows? Resolved by paired per-dataset differences at 5 MCSEs.

**Goal B (estimator performance).** Same settings. For the cross-fitted
one-step estimator with a shared tuned XGBoost outcome regression μ̂: is its
|bias| small next to its empirical SE, is ModSE/EmpSE near 1, is its 95% CI
coverage within a few points of 0.95, and how does its RMSE compare with the
oracle's (true α₀, same μ̂) and with the propensity plug-in's?

The oracle serves claim 2. It shares μ̂ with every other method, so any gap in
ψ̂ between a method and the oracle comes from estimating α alone. It is not a
strict ceiling. In finite samples an estimated α̂ can partly correct errors in
μ̂, so a method can beat the oracle in a given cell.

## 3. Mockups

- **Table 1, DGP diagnostics.** Overlap, variance explained, how nonlinear μ
  and ρ are, and the true ψ₀. Shows what the DGP spans.
- **Table 2, learners and grids.** Fixed settings and tuned grids. Needed to
  reimplement the study.
- **Table 3, tuning checks.** For each learner type and each tuned
  hyperparameter, the share of fits that select the smallest or the largest
  grid value, and the share of fits whose early stopping reaches 90% of the
  iteration cap. This is the premise of every comparison: no learner is held
  back by its grid.
- **Figure 1, α̂ accuracy (Goal A).** Mean RMSE(α̂, α₀) against n on log axes,
  one line per method, one panel per estimand, with ±2 MCSE bars.
- **Table 4, ψ̂ performance for the ATE (Goal B).** Rows are n × method
  (oracle, propensity, rieszboost, riesznet). Columns are bias, EmpSE,
  ModSE/EmpSE, RMSE, RMSE relative to the oracle, and coverage, each with its
  MCSE.
- **Table A1, the same for the ATT (appendix).**

## 4. DGP and settings

See `dgp.py`. X ~ Unif(−1, 1)^5, and all five covariates are confounders.
ρ = 0.8X1 − 0.8X2 + 1.5(X3² − 1/3) + 1.2X4X5, and
μ = X1 − 0.8X2 + sin(πX3) + 0.8X4X5 + A(1 + 0.8X1 + X3² − 1/3), with noise
SD 1. The true ATE is 1.

- Sample sizes are n ∈ {500, 1000, 2000}, as asked. Overlap strength comes later.
- The estimands are the ATE (main text) and the ATT (appendix).
- The α sources are oracle α₀, the propensity plug-in, rieszboost, and riesznet.
- The learners are in `learners.py`. There is one grid per learner type. The
  selected setting has the smallest pooled out-of-fold risk on the same 5
  cross-fitting folds, and its out-of-fold predictions are used directly.
- Repetitions: 2 to check the code runs, then 20 to check the tuning grids,
  then 100 for the pilot, then the final count, sized by the pilot.

**Grid check (20 reps per n, first grids).** The first grids were depth
{2, 3, 4} × learning rate {0.01, 0.03, 0.1} for every tree learner, and
learning rate {1e-4, 1e-3, 1e-2} × weight decay {0, 1e-3, 1e-2} for riesznet.
Selection frequencies alone were misleading. The risk surfaces are flat, so
each edge of a three-value grid wins about a third of the time by chance.
Mean regret was misleading too, because rieszboost at learning rate 0.1 and
riesznet at 1e-2 sometimes diverge, and CV rejects those fits. So the edge
rule uses the setting with the smallest *median* regret over replicates. It
moved the grids as follows:
- outcome and propensity: depth {1, 2, 3}, learning rate {0.03, 0.1, 0.3}.
  The median was best at depth 2 and learning rate 0.1, both edges.
- rieszboost: depth {1, 2, 3, 4}, learning rate {0.003, 0.01, 0.03, 0.1},
  cap 5000 trees. The ATT was best at 0.01, the smallest value, at every n.
- riesznet: learning rate {3e-4, 1e-3, 3e-3, 1e-2}, weight decay
  {0, 1e-3, 1e-2}. 1e-4 ran into the 500-epoch cap.

A rieszboost fit that reaches its tree cap is usually one that diverges, with
a median regret of 0.3–1.0, against 0.04 for fits that stop early. Its
validation Riesz loss keeps falling while its held-out fold loss is bad. So a
cap hit here does not mean the learner wants more trees.

## 5. Build

- **Grid.** n × rep, with every learner, estimand and method in every cell.
- **Shared computation.** Figure 1 and Tables 4 and A1 read the same
  cross-fitted predictions. The tuning table reads the risk of every setting
  in every cell.
- **Cache.** `cache/<learner>/`, one file per (cell, setting), holding
  out-of-fold predictions and early-stopping iterations. The key covers the
  data, the folds, the learner source, the setting, and the package source
  and versions. The directory is git-ignored.
- **Seeds.** Each cell's generator is `default_rng([20260924, n, rep])`
  (PCG64). Every learner uses `random_state = 0`.
- **Parallelism.** joblib across cells, with one thread per fit.

## 6. Pilot

To be filled from `summaries.pilot_contrasts` on 100 repetitions.
