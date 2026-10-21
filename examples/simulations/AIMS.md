# New simulation scenarios: aims

Fill this in top to bottom. Each section feeds the next: the claims decide the
goals, the goals decide the tables and figures, and those decide the DGPs and
settings. Nothing below section 3 needs to be settled before the code starts,
but sections 1 to 3 do.

## What already exists

For reference, so new scenarios add something rather than repeat it.

| Folder | DGP | Estimands | Methods | Status |
|---|---|---|---|---|
| `../simulation/` | 5 confounders, Unif(−1, 1), nonlinear propensity, good overlap | ATE, ATT | oracle, propensity plug-in, rieszboost, riesznet | plan written, pilot pending |
| `../lee_schuler_10cov/` | Lee & Schuler 10-confounder binary treatment | ATE, ATT | oracle, indirect, rieszboost, rieszboost_l2, riesznet, ForestRiesz | piloted on Savio |
| `../lee_schuler_10cov_shift/` | Lee & Schuler continuous exposure | ASE, LASE | oracle, rieszboost_l2, riesznet, ForestRiesz, density ratio (ASE) | on `simulations-shift` |

Known open issues these scenarios might target: α̂ blowing up where one arm is
missing from a leaf (rieszboost and riesznet), riesznet ATT instability, and
the tuning-grid edge hits seen in the 10-covariate pilot.

## 1. Claims

What should a reader believe after these simulations that they did not before?
Aim for the shape *[method] works [well] when [condition] and not when [other
condition]*. A claim without the "and not when" half is usually overstated.

| # | Claim | Who cares, and what would they do differently |
|---|-------------------------------------------------------|------------------------------------------------------|
| 1 | Riesz regression does not work worse than plug-in for most scenarios, and maybe works better than plug-in in settings with practical positivity violations | Researchers who want a package to use for empirical work|
| 2 | There are different DGPs where some learners work better than others for Riesz regression | |
| 3 | There are different DGPs for which different losses work better than others (maybe low n?) | |

## 2. Sharpened goals

One per claim. Pin down every word that could mean more than one thing, until
two people reading the goal would agree on whether a given table meets it.

- **Better / works by what measure?** (bias, RMSE, coverage, RMSE of α̂ vs α₀, runtime, ...)
- **At what sample sizes?**
- **Under which DGPs, and which factor is varied?** (overlap, dimension, smoothness, sparsity, estimand, ...)
- **Compared with what?** (oracle α₀, semi-oracle, propensity plug-in, other Riesz learners, ...)
- **With which learners and tuning?**

**Goal A.**

**Goal B.**

**Goal C.**

## 3. Tables and figures

Sketch each display with empty cells and write its caption, including which
claim it serves. Methods go in adjacent rows or columns, since that is the
comparison the reader makes.

- **Table / Figure 1.** Serves claim __.
- **Table / Figure 2.** Serves claim __.

What result would make you change the claim? Write it down now, while it is
cheap to be honest about.

## 4. DGPs and settings

The varied factor and its levels, and a reason for each functional form and
parameter value (shows a phenomenon, prevents a problem, or is a neutral
default).

| Scenario | Covariates | Treatment / exposure | Outcome | What it is meant to show |
|---|---|---|---|---|
| | | | | |

- **Sample sizes:**
- **Estimands:**
- **Methods and comparators:**
- **Learners and grids:** reuse `../lee_schuler_10cov/` grids unless there is a reason not to.
- **Repetitions:** sized by the pilot.

### Candidate DGPs (draft, edit freely)

A method wins where its built-in assumptions match α₀, or where a rival's
known failure is switched on. So each DGP below starts from a mechanism, says
which method that mechanism should favour, and is the smallest change to a
shared baseline that switches the mechanism on. The baseline could be the
5-covariate DGP in `../simulation/dgp.py`. Change one factor at a time from it,
so that any difference can be traced to that factor.

**Claim 1: Riesz regression vs propensity plug-in**

| Lever | Mechanism | Prediction |
|---|---|---|
| Overlap strength: scale the propensity logit by γ ∈ {0.5, 1, 2, 3}, and report min π and the share of units with π < 0.05 at each level | The plug-in α̂ = 1/π̂ multiplies propensity error by about 1/π², and a classifier tuned on log loss spends its capacity where π is near 0.5, not in the tails where α is large. The Riesz loss measures error on the scale of α. | Tie at good overlap. Riesz pulls ahead as overlap weakens. At the most extreme level, rieszboost and riesznet may drift where one arm is missing (see the known issues above), which would give the "and not when" half of the claim. ForestRiesz should hold up best there. |
| Where the propensity is complex: a smooth logit plus a sharp feature only in the tail (e.g. π falls steeply for X₁ > 0.8), vs the same feature in the bulk | Same as above: log loss rewards fitting the bulk, the Riesz loss rewards fitting where α is large. | Riesz wins when the complexity is in the tail, and ties when it is in the bulk. |
| A correctly specified parametric propensity: logit linear in X, with a logistic-regression plug-in | Maximum likelihood with the right model is hard to beat. | Plug-in ties or wins. This belongs in the study as the honest "not when". |
| Continuous exposure (shift estimands): the conditional law of A given X is Gaussian with constant variance, then with variance depending on X, then skewed or bimodal | The plug-in is the classification density ratio: a classifier separates observed rows (A, X) from shifted rows (A + δ, X), and its odds estimate p(a − δ \| x) / p(a \| x). It is tuned on log loss, which rewards the bulk, where the ratio is near 1. The Riesz loss weights errors by the size of α. | A Riesz advantage that is largest where the ratio has heavy tails (bimodal). Extends `../lee_schuler_10cov_shift/`. |

Comparators: the plug-in unclipped and clipped at [0.01, 0.99] and at [0.05,
0.95], since practitioners clip. Without the clipped versions the plug-in is a
straw man.

The cancellation trap: the one-step estimator's error depends on the product
of the α̂ error and the μ̂ error. If μ is easy to fit, ψ̂ barely depends on α̂,
and every method looks the same. To make the α̂ differences show up in ψ̂, put
something hard to fit in μ where π is small. Either way, report the RMSE of α̂
against α₀ as the main measure and the performance of ψ̂ second.

**Claim 2: learners**

| Lever | Should favour | Why |
|---|---|---|
| Smooth α₀ (a logit of smooth functions of 2–3 covariates) | krrr, riesznet | Trees approximate smooth surfaces with steps, and forests are known to be weaker on smooth functions. |
| Step-function propensity (π depends on 1{X₁ > 0}, 1{X₂ > 0.5}) | riesztree, rieszboost, ForestRiesz | Trees put splits at the jumps, while kernels and networks smear the edges. |
| Sparse and high-dimensional (p = 50, with 3 covariates that matter) | rieszboost | Splitting selects variables. An RBF kernel measures distance in all 50 dimensions, so the noise covariates swamp the signal. |
| Additive logit vs high-order interactions | depth-1 boosting vs deep trees or riesznet | The depth of the trees caps the order of interaction they can fit. |
| Weak overlap, with regions where one arm is missing | ForestRiesz (non-augmented) | It splits on X only and solves each leaf in closed form. In the 10-covariate pilot its advantage sat in the weak-overlap rows. |
| n from 250 to 8000 | krrr and simple learners at small n, riesznet at large n | krrr costs O(n³), so it is also a runtime display. |

A sharper version of claim 2 that tells a practitioner what to do: *no Riesz
learner is best in every DGP, and choosing the learner by cross-validated Riesz
loss comes within [x] of the best learner in each one.* The display would be a
DGP × learner matrix of α̂ RMSE, plus a row for the learner chosen by
cross-validation.

**Claim 3: losses**

A constraint first. `KLLoss` needs α > 0, and `BernoulliLoss` needs α in (0,
1). Of the built-in estimands, only `TSM` has a positive α. `ATE`, `ATT` and
`AdditiveShift` are differences, and their α takes negative values. So the
comparison of losses runs on TSMs, or on the ATE fitted as TSM(1) − TSM(0),
with one KL fit for each. `BoundedSquaredLoss` with α ∈ (1, 1/ε) for a TSM is
a soft-trimming analogue, and pairs naturally with the clipped plug-in.

| Lever | Mechanism | Prediction |
|---|---|---|
| Additive propensity logit f | KL's exp link makes the learner additive in log α. For a TSM, log α = log(1 + e^{−f}), which is close to −f where π is small. So with an additive f, KL's model class holds the large weights almost exactly, while the squared loss has to build 1/π out of additive pieces. | KL wins with an additive logit. The gap shrinks when f has interactions that neither loss's learner can fit. |
| Weak overlap | The squared loss can return negative or near-zero α̂ for a TSM, and KL cannot. | KL is better as overlap weakens. |
| Small n (250, 500) vs large n (4000) | Positivity acts as a constraint, which matters most when there are few data. | A KL advantage at small n that disappears at large n. This is your low-n hunch, with a mechanism behind it. |
| Good overlap and large n | Both losses are consistent, and their difference is second-order. | Tie. This is the "not when". |

Before any of these goes into a full run, draw one large sample from each DGP
and check that it does what the table says: max α₀, the share of units with
π < 0.05, and whether the predicted ordering appears on α̂ RMSE in a pilot of
about 100 repetitions.

## 5. Notes and open questions

## 6. Pilot results (2026-10-20, 50 replicates)

The code is in `dgp.py` and `pilot.py`, and the full output is in
`results/pilot_summary_0_50.txt`. Run `python dgp.py` for the diagnostics and
`python pilot.py summarize` for the tables.

**How it was run.** The measure is the RMSE of α̂ against α₀ on a fixed test
set of 20,000 draws. For binary treatment, A is integrated out exactly given
X. It is also reported on the weak-overlap rows alone. Every learner is tuned
on the same 20% held-out split (squared Riesz loss for the Riesz learners),
then refit on the whole sample. The ATE and the shift use n = 1000. The TSM
uses n ∈ {250, 1000, 4000}. The "clipped plug-in" below is XGBoost tuned on
log loss, with π̂ clipped to [0.01, 0.99]. The one-step estimator ψ̂ has not
been run.

**DGP diagnostics (one draw of 1M each).** The overlap ladder spreads well.
The share of units with min(π, 1 − π) < 0.05 is 0%, 0.007%, 9.6% and 28% for
γ = 0.5, 1, 2 and 3, and max |α₀| is 6, 27, 209 and 1034. The tail-feature
DGP also has worse overlap than the bulk-feature one (8.5% vs 1.9%), so the
two differ in more than where the feature sits. In the skewed shift DGP α₀
is bounded (max |α₀| ≈ 1). The bimodal one is the hard case (max |α₀| = 145).

### Claim 1: Riesz vs plug-in

| Prediction | Result |
|---|---|
| Tie at good overlap, Riesz ahead as overlap weakens | **Not supported on α̂ RMSE.** At γ = 0.5 and 1, krrr, riesznet and rieszboost beat the clipped plug-in by 0.04–0.10 (z from −2 to −7). At γ = 2, ForestRiesz is ahead by 0.25 (z = −2.3) but level with the plug-in clipped at 0.05. At γ = 3, the clipped plug-in is best (5.42), and every Riesz learner is worse by 0.34 to 0.74. On the weak-overlap rows, the clipped plug-in beats every Riesz learner at γ = 2 and 3. |
| Riesz wins when the complexity is in the tail | **Unresolved** (z ≈ −1; it would take about 500 replicates). Both DGPs are dominated by their linear 3X₁ term, so the main-terms logistic plug-in beats every method on both. On the weak-overlap rows of the tail DGP, ForestRiesz beats the clipped plug-in by 0.36 (z = −8). |
| Correct logistic model → plug-in wins | **Supported.** The logistic plug-in has RMSE 0.22, against at least 0.49 for every other method. |
| Shift: Riesz beats the classification density ratio, most of all where the ratio has heavy tails | **Supported for krrr, not for rieszboost.** krrr beats the unclipped classifier on all four DGPs: by 0.14 on gauss, 0.07 on hetero, 0.16 on skew and 0.21 on bimodal (z from −5 to −15). It beats the classifier clipped at 0.05 too, by 0.07 to 0.16. riesznet wins on gauss and skew, is unresolved on hetero, and blows up on bimodal. rieszboost, the same tree learner as the classifier, is mixed: it wins on skew (−0.04, z = −4.2), loses on hetero (+0.03, z = 3.0), and blows up on gauss and bimodal. On the weak-overlap rows, krrr beats the classifier by 0.19 to 1.53. Clipping at 0.01 never binds. Clipping at 0.05 helps only on bimodal. The classifier's learning rate picks the top of its grid in 62% of fits. |

At γ = 3, every method's RMSE (5.4 to 6.2) is close to sd(α₀) = 6.2. In L2,
none of them learns α₀ in the weak region. They differ in how they shrink the
extreme weights, and that matters for the bias and variance of ψ̂, which α̂
RMSE does not capture. **So the weak-overlap half of claim 1 has to be judged
on ψ̂** (bias, RMSE and coverage of the one-step estimator), not on α̂.

### Claim 2: learners

Mean α̂ RMSE. The winner is compared with the runner-up by paired z.

| DGP | Predicted | Best Riesz learner | Verdict |
|---|---|---|---|
| overlap γ = 0.5, 1 and logistic | — | krrr ≈ riesznet (z ≈ −1.3 to −2) | smooth, good overlap |
| smooth | krrr, riesznet | ForestRiesz ≈ rieszboost (1.02), krrr 1.15 | **not supported** |
| step | trees | riesztree 0.93 vs rieszboost 1.01 (z = −2.5) | supported |
| sparse50 | rieszboost | ForestRiesz 0.81 vs rieszboost 0.87 (z = −7) | partly: tree ensembles win, krrr and riesznet are worse |
| interaction | deep trees, riesznet | krrr 0.65 vs riesznet 0.68 (z = −5) | partly: kernel and network win, trees don't |
| weak overlap (γ = 2, 3) | ForestRiesz | ForestRiesz (z = −2.9 and −4.4) | supported |

The ordering of learners does change across DGPs, which is the raw material
for claim 2:
- krrr and riesznet do best on smooth surfaces with good overlap.
- ForestRiesz does best with weak overlap, many covariates or additive
  structure, and on the weak-overlap rows of almost every DGP.
- riesztree does best on step functions.

riesznet blows up in some replicates of the weak-overlap, step and additive
DGPs (mean RMSE 23 at γ = 3). This is the counterfactual-drift failure.

### Claim 3: losses (TSM)

| Prediction | Result |
|---|---|
| KL beats squared with an additive logit | **Supported.** At n = 1000 and 4000, KL is ahead by 0.08 and 0.11 on additive, and by 0.30 and 0.40 on additive_g2 (z from −6 to −18). |
| The gap shrinks with interactions | **Supported.** On interaction the two losses tie at every n. |
| The KL advantage is largest at small n | **Not supported.** At n = 250 the two tie on all three DGPs. The advantage grows with n, which fits the link-function mechanism: the difference is approximation error, and it shows once variance is small. On the weak-overlap rows KL is ahead at every n, including n = 250 (by 0.42 to 0.48). |

Against the plug-in, both rieszboost losses lose to the clipped XGBoost
plug-in at n ≥ 1000 on the additive DGPs. They beat it only at n = 250 on
additive (by 0.33 to 0.36, z ≈ −2.5).

### Tuning grids that must move before a full run

| Learner | Edge hit |
|---|---|
| riesztree | max_leaf_nodes chose 4, the smallest value, in 100% of fits |
| krrr | length scale chose 2× median, the top, in 43–49% of fits |
| XGBoost plug-in | learning rate chose 0.3, the top, in 67–73% of fits |
| rieszboost | L2 penalty chose 100, the top, in 39–50% of fits |

The plug-in's grid is itself held back, so widening it can only help the
plug-in. Widening rieszboost's penalty and krrr's length scale may help them.
Every comparison above is provisional until the grids move.

### One-step estimator (50 replicates)

**How it was run.** Each nuisance is fit on the training sample. ψ̂ = mean[m(μ̂) + α̂ (Y − μ̂)] is evaluated on
an independent estimation sample of the same size, with an influence-function
SE and a 95% Wald interval. Each replicate has one XGBoost outcome model,
shared by every α̂ and by the oracle, which plugs in the true α₀. The true ψ₀
is exact: 1 for the ATE and the TSM, 0.5 for the shift. The full output is in
`results/pilot_summary_psi_0_50.txt`.

**The cancellation trap is real here.** |bias| is under 0.05 for every method
in every DGP, including the oracle, so the RMSE of ψ̂ is almost all variance.
Of 120 paired contrasts in squared error, 4 reach |z| > 2.5 at 50 replicates.
Most of the rest would need hundreds to tens of thousands of replicates. The
large α̂ differences mostly vanish in ψ̂:
- **KL vs squared:** z = −18 on α̂ RMSE, but z between −1.1 and 1.0 on ψ̂.
- **krrr vs the classifier on the shifts:** z from −5 to −15 on α̂, z ≈ −2
  on ψ̂.

The reason is in the design. μ is the same in every DGP and has nothing hard
where π is small. μ̂ is about equally good everywhere (its RMSE is 0.37–0.44
at n = 1000), so the product of errors (α̂ − α₀)(μ̂ − μ₀) stays small.

**What ψ̂ does show**
- **At weak overlap, the oracle is not a ceiling.** At γ = 3 the oracle's
  RMSE is 0.160. Every estimated method except riesznet does better: the
  plug-in clipped at 0.05 gets 0.092, riesztree 0.097, ForestRiesz 0.100,
  rieszboost 0.124 and krrr 0.131. With bias near zero, shrinking the extreme
  weights only lowers the variance. At γ = 2, rieszboost is best (0.098 vs
  0.148 for the plug-in clipped at 0.01, z = −3.1).
- **Claim 1, "not worse than plug-in":** it holds for ψ̂ almost everywhere.
  With good overlap, every method is within about 0.01 of the oracle. The
  exception is riesznet, which blows up (RMSE 1.9 at γ = 3, SE/empSE 0.34).
- **The bimodal shift:** krrr has RMSE 0.030, against 0.066 for the
  classifier, 0.045 for the classifier clipped at 0.05, and 0.046 for the
  oracle (z ≈ −2.2).
- **TSM at n = 250, additive:** rieszboost has RMSE 0.181 and the XGBoost
  plug-in 0.266 (z = −1.8). The plug-in's SE/empSE is 0.70 there.
- **Coverage:** every method is between 0.86 and 1.00. With 50 replicates
  the Monte Carlo SE is about 0.03, so only a few cells stand out:
  - rieszboost on shift_hetero (0.80) and shift_gauss (0.86)
  - the classifier on shift_gauss (0.88)
  - riesztree on sparse50 (0.86)

**What this means for the design.** As built, claims 2 and 3 can only be
argued on α̂. To make the quality of α̂ show up in ψ̂, the outcome has to be
hard to fit where α is large. That makes μ̂ wrong in the same place as α̂,
so the product-of-errors term matters. One option: add to μ a term in the
covariates that drive overlap, shaped so the outcome learner can't fit it
(e.g. a high-frequency or interacted term in X₁ and X₂ that is active only
where min(π, 1 − π) is small). Without such a term, a ψ̂ table will show
every method tied.

### One-step estimator with a harder outcome (50 replicates)

μ was replaced by an outcome that is nonlinear and hard to fit everywhere
(`dgp._mu`; the old one is `_mu_easy`):
- Both f(X) and τ(X) now have two periods of a sine, products of covariates,
  a cosine of a sum and a step.
- A linear fit in (A, X) now explains 13–22% of Var(μ), down from 85–89%,
  and 8% of Var(τ), down from 49%.
- ψ₀ is still exactly 1 for the ATE and the TSM. For the shift, a term
  nonlinear in A makes ψ₀ 0.20–0.33, integrated by Monte Carlo.

α̂ is unchanged, since it never uses Y. The full output is in
`results/pilot_summary_hardmu_0_50.txt`.

**μ̂ is now about twice as wrong,** with RMSE 0.73 at n = 1000 against 0.38
before. **But ψ̂ still barely separates the methods.** 15 of 123 contrasts
reach |z| > 2, against 13 with the easy outcome. The term that carries α̂'s
quality into ψ̂ is the bias E[(α̂ − α₀)(μ̂ − μ₀)]. That is an inner product,
and the two errors come from different fits and are not aligned, so it stays
small next to the variance of ψ̂ even when both errors are large. This is
double robustness at work. It is also why "Riesz is not worse than plug-in
on ψ̂" holds so easily: ψ̂ is insensitive to α̂ when μ̂ is reasonable.

**What did change: weak overlap now shows the cost of shrinking the extreme
weights.**
- At γ = 3, the methods that shrink the extreme weights now have visible
  bias: about 0.12 for rieszboost and riesztree, and 0.05–0.07 for the
  clipped plug-ins. Their coverage falls: rieszboost 0.82, riesztree 0.84,
  ForestRiesz and the plug-in clipped at 0.05 0.86. The oracle stays at 0.96.
  With 50 replicates the Monte Carlo SE of these rates is about 0.05.
- The oracle still has the largest RMSE, 0.211, from variance.
- With an easy outcome, shrinking the extreme weights only lowered variance.
  With a hard one, it trades variance for bias and undercoverage.
- This is the trade-off a practical-positivity claim is about. It is the
  first result that speaks to the second half of claim 1.

**What would make ψ̂ depend on α̂.** Make the outcome model systematically
wrong, so that ψ̂ leans on α̂ to correct it. Concretely, add a second outcome
model, main-terms linear regression, and compute ψ̂ for every α̂ with both
outcome models:
- **Tuned XGBoost:** what a careful analyst does.
- **Misspecified linear:** the case where the claim that double robustness
  rescues a bad outcome model is tested, and where the quality of α̂ should
  decide ψ̂.

This costs one extra fit per replicate, and no α̂ refits. The other option,
more replicates (hundreds) on the current design, would resolve differences
that are real but small.

### Hard representers: shared vs unshared features (50 replicates)

**Design (`dgp.py`).** Four new ATE DGPs, with the hard outcome above.
- **hard_shared:** the propensity logit is c·f(X), where f is the outcome's
  own baseline (two periods of a sine, a product, a cosine of a sum, a step).
- **hard_unshared:** the logit is the same f on the covariates permuted as
  (X₃, X₁, X₄, X₅, X₂). None of its terms appears in μ or τ, or correlates
  with one.
- **Matched by construction.** The covariates are iid, so the two logits have
  exactly the same distribution: the same overlap and the same difficulty.
  They differ only in whether α₀'s hard features are μ's.
- **Two overlap levels.** c = 0.78 and 1.56 match sd(logit) of overlap_g1
  and overlap_g2.

The large draw confirms the match: share with min(π, 1 − π) < 0.05 is 9.29%
vs 9.30% at c = 1.56, and sd(α₀) is 3.15 for both. The logits are much
harder than the ladder's (linear R² 0.10 vs 0.69). corr(logit, μ) is 0.91
for shared and 0.07 for unshared. The full output is in
`results/pilot_summary_hardalpha_0_50.txt`.

**α̂ is equally hard to estimate in both.** Mean α̂ RMSE at c = 1.56:
rieszboost 2.31 vs 2.27, krrr 2.38 vs 2.35, clipped plug-in 2.29 vs 2.32.

**ψ̂ is not.** One-step estimator, n = 1000:

| | oracle | best estimated method | worst | |bias|/empSE | coverage |
|---|---|---|---|---|---|
| hard_unshared_1 | RMSE 0.089, cover 0.94 | 0.079 (riesznet) | 0.092 | ≤ 0.12 | 0.94–1.00 |
| hard_shared_1 | 0.093, 0.94 | 0.249 (krrr) | 0.302 (logit) | 2.0–3.0 | 0.08–0.32 |
| hard_unshared_2 | 0.122, 0.98 | 0.111 (krrr) | 0.158 (XGB unclipped) | 0.06–1.0 | 0.80–0.92 |
| hard_shared_2 | 0.165, 0.84 | 0.425 (XGB plug-in) | 0.641 (logit) | 2.8–5.0 | 0.00–0.16 |

With shared features, every estimated method is biased by 0.22–0.29
(c = 0.78) and 0.40–0.63 (c = 1.56), and coverage collapses. The oracle,
which uses the same μ̂, stays nearly unbiased. So the bias is the product of
the errors in α̂ and μ̂, and it shows up only when those errors fall on the
same features. In the unshared version, every method sits next to the
oracle. **The difficulty of α₀ on its own did not matter. What mattered is
whether α₀ is hard in the same way μ is.** Here, 15 and 20 of 22 contrasts
reach |z| > 2 in the shared DGPs, against 0 and 2 in the unshared ones.

**How the methods compare once ψ̂ depends on α̂ (hard_shared_2):**

| method | ψ̂ RMSE |
|---|---|
| XGBoost plug-in (unclipped or clipped at 0.01) | 0.425 |
| XGBoost plug-in, clipped at 0.05 | 0.430 |
| krrr | 0.458 (z = 1.7 against the clipped plug-in) |
| rieszboost | 0.486 (z = 4.0) |
| riesznet | 0.490 |
| ForestRiesz | 0.555 (z = 9.5) |
| riesztree | 0.568 |
| main-terms logistic plug-in | 0.641 |

So the Riesz learners are not better here, and the tree-based ones are
clearly worse. The ordering by α̂ RMSE (rieszboost best) does not predict the
ordering by ψ̂. What matters for ψ̂ is the part of α̂'s error that lies along
μ̂'s error, not its overall size.

**What this means for the design.**
- The shared-feature DGP is where the claims about α̂ actually show up in ψ̂.
  Every claim that is meant to be judged on ψ̂ should be tested there.
- At n = 1000 every method fails in it. That is useful as an "and not when"
  boundary, but no method covers, so it says little about which one to use.
- An intermediate regime is needed, where methods differ and some still
  cover. Two ways to get it: a ladder of the share of hard features that the
  propensity and the outcome have in common (0, 1/3, 2/3, 1 of the terms of
  f), or the same DGP at n = 4000 and 16000.

### The alignment ladder (50 replicates)

**Design.** The logit is c·[w f(X) + √(1 − w²) f(X permuted)] / k_w, for
w = 0, 1/3, 2/3, 1. The endpoints are hard_unshared and hard_shared. k_w
corrects for the small correlation (−0.0625) between the two parts, so
sd(logit) is the same at every w.
- **Overlap is matched along the ladder.** At c = 1.56, 9.3–9.5% of units
  have min(π, 1 − π) < 0.05 and sd(α₀) is 3.15–3.19 at every rung. The
  middle rungs have slightly heavier extreme tails (0.5–0.65% of units with
  min(π, 1 − π) < 0.01, against 0.25%).
- **Alignment steps up.** corr(logit, μ) is 0.14, 0.41, 0.68 and 0.93.

The DGPs are `mix33_*` and `mix67_*`, and the tables are in
`results/pilot_mix_ladder_report.txt`.

**α̂'s RMSE is flat along the ladder, but ψ̂'s bias rises with w for every
estimated method.** The oracle stays unbiased throughout. So the bias of the
one-step estimator is set by how much α̂'s error lines up with μ̂'s error,
not by how large α̂'s error is.

| w | overlap level 1: bias of estimated methods | |bias|/empSE | coverage | level 2: bias | coverage |
|---|---|---|---|---|---|
| 0 | 0.00–0.01 | ≤ 0.12 | 0.94–1.00 | −0.01 to −0.09 | 0.80–0.92 |
| 1/3 | 0.04–0.05 | 0.45–0.66 | 0.88–0.92 | 0.05–0.11 | 0.76–0.94 |
| 2/3 | 0.11–0.17 | 1.0–1.6 | 0.50–0.76 | 0.18–0.30 | 0.06–0.50 |
| 1 | 0.22–0.29 | 2.0–3.0 | 0.08–0.32 | 0.40–0.63 | 0.00–0.16 |

At every rung, the oracle's coverage is 0.84–0.98.

**Which method does best changes along the ladder.** Paired z of the
squared-error difference against the plug-in clipped at 0.01; negative means
the method is better.
- **w = 0:** the Riesz learners are slightly ahead. At level 2, krrr,
  rieszboost and riesznet have z from −1.9 to −2.3.
- **w = 1/3:** ForestRiesz is best. At level 2 its RMSE is 0.101 against
  0.128 for the clipped plug-in (z = −2.4), and its coverage is 0.94. At
  level 1, ForestRiesz and rieszboost are slightly ahead (z = −1.2).
- **w = 2/3:** at level 1, ForestRiesz is narrowly best (0.154 against 0.161
  for the plug-in, z = −1.5). At level 2, the XGBoost plug-in is best (0.219).
  riesznet (z = 7.5) and riesztree (z = 4.8) are clearly worse.
- **w = 1:** the XGBoost plug-in is best.
- **Throughout:** the main-terms logistic plug-in is worst once w ≥ 2/3.

Read with care: with 50 replicates most of these z values are between 1 and
2.5.

**The regime the ladder was built to find is w = 1/3.** The methods differ
there, and several still cover (0.88–0.94). From w = 2/3 on, every method
undercovers at n = 1000. The open question is whether that is a
finite-sample effect, which the n ladder would show by bias/SE shrinking with
n, or a rate problem, which would show as bias/SE growing with n.

### The middle rungs at 200 replicates

Same design, replicates 0–199 on the w = 1/3 and 2/3 rungs at both overlap
levels. The tables are in `results/pilot_mid_rungs_200.txt`. A contrast is
treated as resolved at |z| ≥ 3, since about 36 are being compared. The z
values are paired, on squared error, against the plug-in clipped at 0.01.

| rung | best estimated method (ψ̂ RMSE) | clipped plug-in | resolved | coverage |
|---|---|---|---|---|
| w = 1/3, level 1 | ForestRiesz 0.096 | 0.105 | none. All methods are within 0.096–0.105; ForestRiesz z = −2.5, rieszboost z = −2.2 | 0.89–0.93 (oracle 0.96) |
| w = 2/3, level 1 | ForestRiesz 0.153 | 0.158 (z = −2.7) | krrr (+4.4), riesznet (+7.1) and the logistic plug-in (+10.1) are worse than the clipped plug-in | 0.48–0.73 (oracle 0.92) |
| w = 1/3, level 2 | **ForestRiesz 0.108** | 0.141 | **ForestRiesz beats the clipped plug-in (z = −5.8) and every other method (z ≥ 4.6).** riesztree (−3.0) and the plug-in clipped at 0.05 (−3.6) also beat the plug-in clipped at 0.01 | ForestRiesz 0.885; others 0.77–0.90 (oracle 0.965) |
| w = 2/3, level 2 | **XGBoost plug-in 0.220** | — | **Every Riesz learner is worse:** ForestRiesz +3.6, rieszboost +4.1, krrr +5.0, riesztree +6.4, riesznet +12 | plug-in 0.505; Riesz 0.15–0.38 (oracle 0.925) |

The Monte Carlo SE of RMSE is 0.005–0.008, and of coverage 0.02–0.035.

**What is resolved**
- **The two weak-overlap rungs (level 2) cross over.** At w = 1/3,
  ForestRiesz is the best method by a clear margin. Its RMSE, 0.108, is
  even below the oracle's 0.126 (z = 1.5, unresolved): it shrinks the
  extreme weights, and its bias (0.054) costs less than the variance it
  saves. At w = 2/3, every Riesz learner is worse than the XGBoost plug-in.
- **The XGBoost plug-in has the lowest bias at strong alignment** (0.183 at
  w = 2/3, level 2, against 0.215–0.280 for the Riesz learners). A
  hypothesis to test, not a finding: the Riesz loss's shrinkage of extreme
  weights adds bias exactly where μ̂ is wrong. A semi-oracle that uses the
  true μ with each α̂ would check it, since its bias would then be zero for
  every method.
- **With good overlap (level 1), methods are nearly tied at w = 1/3.** At
  w = 2/3, krrr, riesznet and the logistic plug-in are resolved as worse.
- **Coverage.** No estimated method reaches 0.95 on any middle rung. At
  w = 1/3 the best are around 0.89–0.93.

**What this means for claim 1.** "Riesz regression is not worse than
plug-in" holds where the propensity and the outcome share little, and fails
where they share a lot and overlap is weak. "Riesz is better under practical
positivity violations" holds in one resolved case, ForestRiesz at w = 1/3
with weak overlap. The ladder gives the claim its "when" and its "not when".

### Rerun with the moved grids (v2)

Every grid that hit an edge was moved; the reasons are recorded above
`GRIDS` in `pilot.py`. The whole pilot was rerun on the same datasets: 50
replicates everywhere, and 200 on the middle rungs. The v1 results are in
`results/pilot_grids_v1/`, and the side-by-side comparison is in
`results/compare_grids_v1_v2.txt`.

**Edges after the move.** Almost every learner now picks an edge in at most
about 25% of fits. The exceptions:
- **riesztree "picked" the fewest leaves in 100% of fits. That was a bug in
  the pilot, not a preference.** Under riesztree's default depthwise growth,
  `max_leaf_nodes` is ignored, as its docstring says. With `max_depth=None`,
  every setting grew the same full tree, every setting tied, and `argmax`
  returned the first one. So in v1 riesztree was effectively untuned (a full
  tree with minimum leaf 20). In v2 only its minimum leaf size was tuned.
  - **Fix:** riesztree now grows leafwise, with a grid of 3–64 leaves ×
    minimum leaf size {5, 20, 50}. Only riesztree's tasks were rerun. It
    picks 3 leaves in 9% of fits.
  - **The fix made riesztree's α̂ slightly worse** in most DGPs (e.g. step:
    0.967 to 1.159). The ψ̂ results barely move, and riesztree is still the
    weakest or second-weakest Riesz learner almost everywhere. The before
    and after are in `results/riesztree_before_after.txt`.
  - **The reason is tuning noise.** Choosing riesztree's setting by one 20%
    split costs 0.03–0.16 in α̂ RMSE relative to 5-fold CV (z ≈ 3 on step,
    overlap_g1 and mix33_2, 20 replicates). For example, overlap_g1 gives
    0.966 by the split, 0.873 by 5-fold CV, and 0.814 for the best setting
    in hindsight. This is the same problem that hurt the XGBoost plug-in
    below.
- **The XGBoost plug-in's learning rate picks 1.0, the top, in 24–26%.**
- **rieszboost_kl's L2 penalty picks 1000, the top, in 37%** (TSM).
- Depth 1 for the plug-in is a hard limit, so it is exempt.

**The wider grid made the XGBoost plug-in worse at its job.** Its held-out
log loss can only improve with more settings to choose from, but its α̂
RMSE got worse in 7 of 9 ATE DGPs. For example, it went from 0.742 to 0.838
at γ = 1, and from 0.758 to 0.945 on additive. In the TSM at n = 250 it went
from 1.11 to 1.63. Picking the best of 20 settings by log loss on one 20%
split selects for log loss, not for an accurate 1/π̂. This is the mechanism
behind claim 1, seen directly in the tuning. It also means the plug-in's
results depend on its tuning protocol.

**What held and what moved**

| Result | v1 | v2 |
|---|---|---|
| ForestRiesz best at w = 1/3, weak overlap | 0.108 vs 0.141, z = −5.8 | 0.108 vs 0.153, z = −6.2. **Holds.** It also beats the plug-in clipped at 0.05 (0.138). |
| w = 1/3, good overlap | no resolved contrast | rieszboost (z = −3.2) and ForestRiesz (z = −3.4) now beat the clipped plug-in, partly because the plug-in got worse (0.105 → 0.112) |
| Every Riesz learner worse at w = 2/3, weak overlap | z from +3.6 to +12 | **Weakened.** Only riesznet (+4.4) and riesztree (+4.4) are resolved worse. ForestRiesz ties (−0.2), and rieszboost (+0.4) and krrr (+1.2) are unresolved. The plug-in clipped at 0.05 is best (0.219 vs 0.232 for ForestRiesz). |
| Riesz learners worse at w = 1, weak overlap (50 reps) | — | **Holds** for rieszboost (+3.7), ForestRiesz (+8.3) and riesztree (+10.3); krrr +1.7 |
| KL beats squared on α̂ (TSM) | z = −6.6 to −18 | z = −8.5 to −23. **Holds.** It still ties at n = 250 and on interaction. |
| krrr has the best α̂ in the shift block | yes | **yes**; riesznet improved on bimodal (1.59 → 1.12) |
| Best Riesz learner by α̂ | smooth: ForestRiesz; additive: ForestRiesz | smooth: rieszboost; additive: rieszboost; otherwise unchanged |
| riesznet blows up | at γ = 2 and 3 | at γ = 3 only (RMSE 1.5) |

### Suggested next steps

See section 7, "Still to investigate".

## 7. Evidence map (2026-10-20)

Everything here is pilot-grade:
- It is all at n = 1000, with 50 replicates per cell, except the four
  middle rungs of the alignment ladder, which have 200.
- ψ̂ uses sample splitting, not cross-fitting.
- **Updated for the v2 grids** (section 6). The grids were moved by the edge
  rule. riesztree still always picks a stump, and the XGBoost plug-in got
  worse on α̂ with its wider grid.

Read these as strong leads, not final results. "Resolved" means |z| ≥ 3 on
a paired contrast.

### Claims with evidence

**Claim 1: Riesz regression is not worse than plug-in, and may be better
under practical positivity violations.**
- *Not worse:* **supported where the propensity and the outcome share no
  hard structure.** On ψ̂, every method except riesznet sits near the oracle
  in almost every DGP, with few resolved differences.
- *riesznet is the exception.* It blows up under weak overlap and in some
  other DGPs (ψ̂ RMSE 1.6 at γ = 3), so "not worse" does not hold for it as
  configured.
- *Better:* **supported in one resolved case, robust to the grid move.**
  ForestRiesz at w = 1/3 with weak overlap: ψ̂ RMSE 0.108 against 0.153 for
  the clipped plug-in (z = −6.2, 200 replicates), and against 0.138 for the
  plug-in clipped at 0.05. At w = 1/3 with good overlap, rieszboost and
  ForestRiesz are now also resolved better (z ≈ −3.3), partly because the
  plug-in got worse with its wider grid.
- *Not when:* **resolved at full sharing (w = 1), weakened at w = 2/3.**
  - At w = 1 with weak overlap, the XGBoost plug-in beats rieszboost
    (z = +3.7), ForestRiesz (+8.3) and riesztree (+10.3). This is at 50
    replicates.
  - At w = 2/3, after the grid move, only riesznet and riesztree are
    resolved worse. ForestRiesz ties.
- *Tuning matters to the comparison.* With a wider grid, the XGBoost plug-in
  got lower log loss but worse α̂ in 7 of 9 DGPs. Tuning on log loss does not
  tune for the weights. That is the mechanism behind claim 1, and also a
  reason to fix the plug-in's tuning protocol before final runs.
- *Correct logistic model:* the logistic plug-in clearly has the best α̂.
  On ψ̂, everything ties.
- *Shift:* krrr has a better α̂ than the classification density ratio
  (z from −5 to −15). On ψ̂ the gaps are small (z ≈ −2.7 at most).

**Claim 2: different learners win in different DGPs.**
- **Supported on α̂ (v2 grids):**
  - krrr does best with good overlap (γ = 0.5, 1) and on interaction.
  - ForestRiesz does best with weak overlap and many covariates.
  - rieszboost does best on smooth and additive.
  - riesztree does best on steps, even as a stump.
- One prediction failed: kernels and nets were expected to win on smooth,
  but rieszboost did.
- **On ψ̂, learners differ only where the propensity and the outcome share
  structure.** There, ForestRiesz is best at w = 1/3 with weak overlap, and
  riesznet is unstable.

**Claim 3: different losses win in different DGPs.**
- **Supported on α̂:** KL beats squared with an additive logit, and the two
  tie with interactions.
- **The low-n version is not supported.** The two tie at n = 250, and KL's
  lead grows with n.
- **No evidence on ψ̂ yet.** In the DGPs run so far, the two losses give the
  same ψ̂ (z ≈ 1).

**New claims the pilot turned up**
- **The one-step estimator's bias depends on whether the propensity and the
  outcome share hard structure, not on how hard α₀ is.** The evidence is
  strong:
  - hard_shared and hard_unshared, with identical difficulty and overlap,
    behave completely differently.
  - The alignment ladder shows a dose-response.
  - The oracle stays unbiased throughout.
- **Under weak overlap, estimated weights can beat the true ones.** With the
  easy outcome at γ = 3, every estimated method except riesznet beat the
  oracle. ForestRiesz also edged it at w = 1/3, though that is unresolved.

### Still to investigate

1. **The mechanism of the crossover.** The hypothesis is that shrinking the
   extreme weights adds bias exactly where μ̂ is wrong. Test it with a
   semi-oracle: each α̂ paired with the true μ.
2. **Sample size.** Run the middle rungs at n = 4000, to see whether
   bias/SE shrinks or grows.
3. **Coverage.** No estimated method reaches 0.95 on the middle rungs.
   Questions: does cross-fitting, or a larger n, fix it, and does any method
   give valid intervals there?
4. **Tuning.** ~~Move the edge grids.~~ Done (v2). ~~riesztree stump.~~
   That was a pilot bug, now fixed (section 6). One problem remains, and it
   affects every learner:
   - Every learner is tuned on one 20% held-out split (200 rows at
     n = 1000). With large grids, that picks noisy winners. It made the
     plug-in's α̂ worse when its grid widened, and it costs riesztree
     0.03–0.16 in α̂ RMSE against 5-fold CV (z ≈ 3).
   - **Switch to 5-fold CV for the final runs.** It costs about 5 times as
     much per learner. It will probably move the Riesz-vs-plug-in
     comparisons, since both sides are tuned this way.
5. **Claim 3 on ψ̂.** Losses separate only on α̂ so far. They need a TSM DGP
   where the propensity and the outcome share structure, which is where α̂'s
   quality reaches ψ̂.
6. **The shift block.** Two questions:
   - Does krrr's α̂ advantage reach ψ̂ when the shift DGP's exposure and
     outcome share structure?
   - Does krrr's edge come from the loss or from the kernel? A
     kernel-family classifier would separate the two.
7. **How far the ForestRiesz win extends.** Other levels of sharing
   (w = 1/6, 1/2), other overlap levels, other n.
8. **Leftover design issues.**
   - Redesign tail vs bulk on a base that is not nearly logistic, and with
     matched overlap.
   - Fix or drop riesznet.
9. **Replicates.** Many contrasts in the 50-replicate cells sit at |z|
   between 1 and 2.5.
10. **Many covariates (requested 2026-10-20, to revisit).** At least one
    scenario with a lot of covariates, judged on ψ̂. sparse50 already has
    50 covariates, of which 3 drive treatment, but it was judged mainly on
    α̂. Two options, not yet chosen:
    - **Noise covariates:** the key alignment rungs (w = 1/3 and 2/3, weak
      overlap) with 45 irrelevant covariates added, p = 50. The truth and
      ψ₀ are unchanged. It tests whether each learner can ignore noise; krrr's
      isotropic kernel is the likely casualty. Cheap, since the outcome does
      not change.
    - **Many weak confounders:** p = 50, with 45 extra covariates that each
      nudge both treatment and outcome a little, and overlap re-matched.
      Closer to real empirical data. It needs a per-DGP outcome, so more
      work.

