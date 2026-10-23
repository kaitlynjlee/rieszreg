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

**Rescoped (2026-10-22), then widened the same day.** This paper shows that
the package works and justifies its parts: why it has several learners,
several losses and several estimands. Where and why each method is strong
or weak, including the alignment results, stays in a later paper
(section 8).

| # | Claim | Who cares, and what would they do differently |
|---|-------------------------------------------------------|------------------------------------------------------|
| 1 | **The package works.** For the average treatment effect (ATE), the average treatment effect on the treated (ATT), the treatment-specific mean (TSM), the additive shift effect and the local additive shift effect (LASE), the package's Riesz learners (rieszboost, riesznet, krrr, riesztree and the forests) give one-step estimates whose 95% intervals cover close to 95%, whose root mean squared error (RMSE) is comparable to a well-tuned plug-in, and whose error shrinks at the √n rate as n grows. Coverage and accuracy degrade as overlap weakens. | Researchers who want a package for empirical work. They can use it for these estimands without first checking it against a plug-in, and they know weak overlap is where to be careful. |
| 2 | **Several learners earn their place.** No Riesz learner has the most accurate representer in every data-generating process (DGP): krrr wins with good overlap, forestriesz with weak overlap or many covariates, and rieszboost on the LASE. The choice reaches ψ̂ mainly where the representer is hard to estimate (weak overlap, continuous exposures); in ordinary binary-treatment DGPs its value for ψ̂ is avoiding a poor learner. | Users choosing a learner: which one to try first, and when the choice matters for the final estimate. |
| 3 | **A second loss earns its place.** For a TSM, the KL loss gives a more accurate representer than the squared loss when the propensity logit is additive, and its lead grows with n. The two tie at n = 250 and when the logit has interactions, and the gain does not reach ψ̂ at these sample sizes. | Users who need positive weights or who use the representer itself (for example as weights), and anyone choosing a loss. |
| 4 | **The estimands are useful.** One interface fits five estimands, including two ratio estimands (the ATT and the LASE), and each gives a working one-step estimator. | Users whose question is not the ATE. |

The "and not when" halves: weak overlap for claim 1; ψ̂ in ordinary
binary-treatment DGPs for claim 2; small n, interactions and ψ̂ for claim 3.

**What moved to the later paper (section 8), and what came back:**
- old claims 2 and 3, as questions of *when and why* a learner or a loss
  wins. Claims 2 and 3 above keep only their existence form: each learner
  or loss wins somewhere;
- everything about the propensity and the outcome sharing hard structure:
  the hard-representer DGPs, whose propensity is as hard to fit as the
  outcome, and the alignment ladder, which varies how much of that hard
  structure the two share.

Old claim 1 was "Riesz regression is not worse than plug-in, and may be
better under practical positivity violations". Its "not worse" half is now
the RMSE part of claim 1. Its "better" half rested on the alignment ladder,
so it moved too.

## 2. Sharpened goals

Pin down every word that could mean more than one thing, until two people
reading the goal would agree on whether a given table meets it.

Notation for this section and the next: X is the covariates and A the
treatment, binary for the ATE, ATT and TSM and a continuous exposure for the
shifts, with lowercase a and x their values. μ₀(a, x) = E[Y | A = a, X = x]
is the true outcome regression and μ̂ its fit; π(x) = P(A = 1 | X = x) is
the propensity score; ψ̂ is the one-step estimate; α̂ is the fitted Riesz
representer and α₀ the true one; γ is the factor that scales the
propensity logit on the overlap ladder (larger γ, weaker overlap); and
1(·) is the indicator. For the shifts, δ = 0.5 is the shift, t = 0 the
LASE's threshold (only units with A < t are shifted), p(a | x) the
conditional density of A given X, and r(a, x) = p(a − δ | x) / p(a | x)
the density ratio. Good overlap means γ ≤ 1, moderate γ = 2 and weak
γ = 3; the "ordinary" ATE DGPs are the ATE DGPs other than the alignment
ones of section 8. Two
different SEs appear: the SE of ψ̂ that the interval uses, and the Monte
Carlo SE of a summary over replicates, which says how precisely the
simulation pins that summary down. A contrast between two methods is
*resolved* when its paired z (the mean per-replicate difference over its
Monte Carlo SE) has |z| ≥ 3.
Section 6 has the full notation.

**Claim 1, the package works.**
- **By what measure?** Four parts, each judged at the final number of
  replicates:
  1. **Coverage.** The 95% one-step interval covers within about 2 Monte
     Carlo SEs of 0.95 in every DGP with good or moderate overlap.
  2. **Accuracy.** The RMSE of the one-step estimate ψ̂ is at most 1.25
     times the reference plug-in's, judged by a paired contrast. The 1.25 is
     a proposal, still to confirm.
  3. **Rate.** In the TSM block, RMSE × √n is flat or falling from n = 250
     to 4000, and coverage stays near 0.95.
  4. **Boundary.** Coverage and RMSE are reported along the overlap ladder,
     γ ∈ {0.5, 1, 2, 3}, so the reader sees where the degradation starts.

**Claim 2, several learners.**
- **By what measure?** The RMSE of the fitted representer α̂ against the
  true α₀ on a fixed test set, by DGP. A learner earns its place if it is resolved better than every other
  Riesz learner in at least one DGP. For the one-step estimate ψ̂: the spread of its RMSE
  across learners, and the share of its mean squared error that comes from
  α̂
  (the method-specific part of the error, beyond the oracle's).
- **Over which learners?** All of them, in the ATE, ATT, shift and LASE
  blocks.

**Claim 3, a second loss.**
- **By what measure?** α̂'s RMSE, KL against squared, paired, for
  rieszboost on the TSM at n = 250, 1000 and 4000; ψ̂'s RMSE second; the
  share of treated units whose α̂ is not positive (KL's is positive by
  construction).

**Claim 4, the estimands.** The same measures as claim 1, for each of the
five estimands.

**Shared settings.**
- **Sample sizes.** The ATE, ATT, shift and LASE at n = 1000; the TSM at
  n = 250, 1000 and 4000, which carries the rate. n is the size of each of
  two independent samples per replicate.
- **Sample splitting (since 2026-10-22).** Every nuisance is tuned and fit
  on a training sample of n, and ψ̂ is computed on an independent
  estimation sample of n, the design of `results/pilot_cv5` (same samples,
  so its replicates pair with the final run's). The pilot `pilot_cf5` used
  5-fold cross-fitting with the tuning nested in each fold instead; that
  costs 4–5 times as much, which is why the final run splits.
- **Replicates.** 500, sized from the first 200 of the final run (see
  "Replicates" below). Coverage then has a Monte Carlo SE of about 0.010.
- **DGPs.** All are in `dgp.py`.
  - **ATE, 8 core DGPs (main text, 500 replicates):**
    - the overlap ladder, `overlap_g05` to `overlap_g3`, for the boundary;
    - `logistic`, where the logistic plug-in is correctly specified;
    - `sparse50`, with 50 covariates (forestriesz's clearest win);
    - `interaction`, interactions in the logit (krrr's win off the ladder,
      and a TSM DGP);
    - `steps10`, built for rieszboost: 20 covariates, a logit that adds ten
      centred steps, overlap matched to `overlap_g1`.
  - **ATE, 5 appendix DGPs (200 replicates; user's decision, 2026-10-23):**
    `tail_feature` and `bulk_feature` (a sharp propensity feature where the
    true representer α₀ is large and where it is near 2), `smooth`, `step`
    and `additive`. They repeat what the core DGPs show for the claims: their
    resolved α̂ winner is forestriesz, which already wins on the ladder and
    `sparse50`, or there is none, and the tail-versus-bulk question is a
    "when and why" question for the later paper. They stay in the paper
    because three of them hold failures the claims must state: riesztree
    and augforestriesz resolved worse than the plug-in on `tail_feature`
    and `step` (ATE and ATT), and on the ATT rieszboost and forestriesz at
    1.33 and 1.40 times the plug-in on `bulk_feature`.
  - **ATT:** the same 8 core and 5 appendix DGPs.
  - **TSM:** `additive`, `additive_g2` and `interaction`.
  - **Shift and LASE:** `shift_gauss`, `shift_hetero`, `shift_skew` and
    `shift_bimodal`.
  - **Moved to section 8:** `hard_shared_1`, `hard_shared_2`,
    `hard_unshared_1`, `hard_unshared_2`, `mix33_1`, `mix33_2`, `mix67_1`,
    `mix67_2`, and the whole `tsm_align` block (the TSM on the alignment
    ladder).
- **Ratio estimands.** The package fits the representer of the ATT's and
  the LASE's numerator, E[A (μ₀(1, X) − μ₀(0, X))] and E[1(A < t)
  (μ₀(A + δ, X) − μ₀(A, X))]. The one-step estimate divides by the mean of A or of
  1(A < t), with the matching influence function (`pilot.py`, `load_psi`).
- **Compared with what?**
  - The oracle: the one-step estimator with the true α₀ and the same fitted
    outcome regression μ̂.
  - The reference plug-in: the XGBoost propensity clipped to [0.01, 0.99]
    for the ATE; clipped to [0.05, 0.95] for the ATT (see the cap below).
    Also reported: the other clipping levels and the main-terms logistic
    plug-in.
  - For the shift and the LASE: the classification plug-in, which
    estimates the density ratio r by a classifier, unclipped and clipped.
    For the LASE it applies the indicators exactly: with r̂ its estimate of
    r, α̂ = 1(a < t + δ) r̂ − 1(a < t).
- **Learners and tuning.**
  - **ATE and ATT:** rieszboost, riesznet, krrr (kernel ridge Riesz
    regression), riesztree, augforestriesz and forestriesz.
  - **TSM:** rieszboost with the squared loss and with the KL
    (Kullback–Leibler) loss; riesznet, krrr and forestriesz with the
    squared loss, so the rate covers more than one learner.
  - **Shift and LASE:** rieszboost, riesznet and krrr. forestriesz handles
    binary treatments only.
  - **Tuning:** every learner is tuned by 5-fold cross-validation (CV) on
    the training sample, over the grids in `pilot.py`'s `GRIDS`, and the
    chosen setting is refit on the whole training sample. The grids were
    moved a fourth time before the final run (rieszboost's L2 penalty up to
    10⁵; riesznet's learning rate 3·10⁻⁵ to 10⁻¹; riesztree's leaves
    {2, …, 24} and minimum leaf adding 2). The outcome regression is XGBoost
    tuned on squared error, shared by every method in a replicate.
  - **Edge check, final run (200 replicates).** No tuned hyperparameter
    sits on a grid edge that is not a hard limit in 15% or more of fits.
    The edges chosen that often are all hard limits: depth 1 (rieszboost
    24% in the ATE, rieszboost_kl 32% in the TSM, the XGBoost propensity
    56%), L2 penalty 0 (rieszboost 22% on the shift), forestriesz's leaf
    size 1 (16% in the TSM), XGBoost's learning rate 1 (20%; XGBoost caps
    it at 1). The largest share at a movable edge is forestriesz's leaf
    size 100, at 13% (ATE) and 12% (ATT).
  - **The cap on |α̂|.** Every Riesz learner also has capped variants: the
    size of its fitted weights |α̂| capped at the largest weight a plug-in
    whose propensity is clipped to [c, 1 − c] can reach (1/c for the ATE and
    TSM, (1 − c)/c for the ATT and the LASE, (1 − c)/c − 1 for the shift),
    at clipping levels c = 0.01 and 0.05. Only the predictions are capped;
    the fit and its tuning are not. riesznet is reported capped: at c = 0.01
    (|α̂| ≤ 100), and for the ATT at c = 0.05 (|α̂| ≤ 19), against the plug-in
    clipped at 0.05 (see gap 1). The other learners are reported as fitted.

## 3. Tables and figures

Methods go in adjacent rows, since that is the comparison the reader makes.

- **Table 1 (ATE, n = 1000).** Serves claim 1, parts 1 and 2.
  - Rows: the oracle, the reference plug-in, the other plug-ins, then the
    Riesz learners.
  - Columns: the 8 core DGPs, the overlap ladder first.
  - Two panels: coverage, and the RMSE of the one-step estimate as a ratio
    to the reference plug-in's, each with its Monte Carlo SE.
- **Figure 1 (ATE overlap ladder).** Serves claim 1, part 4. The logit scale
  γ on the x-axis; coverage on top, with a dashed line at 0.95, and the RMSE
  of the one-step estimate below; one line per method.
- **Table 2 (TSM, n = 250, 1000, 4000).** Serves claim 1, part 3. Rows:
  methods (the oracle, the plug-in, rieszboost with both losses, riesznet,
  krrr and forestriesz). Columns: DGP × n. Cells: coverage, RMSE × √n, and the bias
  divided by the empirical SE (the standard deviation of ψ̂ over
  replicates). A flat RMSE × √n is the √n rate.
- **Table 3 (the other estimands).** Serves claims 1 and 4. As Table 1, in
  three panels: the ATT (8 core DGPs), the additive shift and the LASE (4 DGPs
  each), each against its plug-in.
- **Table 4 (which learner has the best α̂).** Serves claim 2. Rows: DGPs,
  grouped by block. Columns: the Riesz learners. Cells: α̂'s RMSE as a ratio
  to the best learner in that row, with the winner marked where it is
  resolved.
- **Figure 2 (why learner differences shrink in ψ̂).** Serves claim 2. For
  each DGP, the share of ψ̂'s mean squared error that comes from α̂, against
  the spread of ψ̂'s RMSE across learners.
- **Table 5 (KL vs squared).** Serves claim 3. Rows: DGP × n. Columns: α̂'s
  RMSE for each loss and the paired z; ψ̂'s RMSE for each loss; the share of
  non-positive weights.
- **Appendix table (the 5 appendix DGPs, 200 replicates).** Serves claims
  1 and 2. For the ATE and the ATT: each method's coverage and RMSE ratio
  to the reference plug-in, bold where resolved worse; and α̂'s RMSE by
  learner as in Table 4. The main text says what it shows, failures
  included.
- **Appendix.** The outcome regression's fit, seconds per task, and the
  tuning-grid edges.

**What result would change the claims:**
- Claim 1: a Riesz learner that covers below about 0.90 (at the final
  replicate count) in a good-overlap DGP, or is resolved worse than the
  reference plug-in by more than the accuracy threshold. Then that learner
  is either left out of "the package works", with the reason stated, or the
  claim narrows. RMSE × √n that grows with n in the TSM block.
- Claim 2: a learner that is never resolved best has no justification from
  α̂ alone; the paper then has to justify it some other way or drop it.
- Claim 3: KL resolved worse than squared on α̂ anywhere.

**Where the evidence stands (`final_split`, sample splitting, first 200
of 500 replicates, 2026-10-23).** 67,400 tasks, 0 errors; the pairing
check against `pilot_cv5` passes (125 replicate cells, none differ). Coverage
has a Monte Carlo SE of about 0.015, so a cell below 0.92 is more than 2 SEs
under 0.95. Numbers will move at 500. The ranges below are over all 13 ATE/ATT DGPs;
the main text will use the 8 core ones (`paper_tables.py`, CORE). The 50-replicate `pilot_cf5` evidence
(cross-fitting) is in this file's git history (commit 6b655b5).
- **Claim 1, ATE.**
  - **Coverage.** With good overlap (γ ≤ 1) and in the non-ladder DGPs,
    the Riesz learners cover 0.90–0.98. The exception is augforestriesz on
    `sparse50`, at 0.875. At moderate overlap, γ = 2, coverage is already
    down: rieszboost and augforestriesz 0.855, riesztree 0.870, riesznet
    0.890, forestriesz 0.900, while krrr (≥ 0.92) and the reference
    plug-in (≥ 0.91) hold. At γ = 3: riesztree 0.795, augforestriesz 0.800,
    rieszboost 0.825, forestriesz 0.850, the plug-in 0.885 and the oracle
    0.92. **So the boundary starts at γ = 2, not γ = 3**, and part 1 of
    the claim ("good or moderate overlap") holds only for krrr. The claim
    should narrow to good overlap.
  - **Accuracy.** krrr and forestriesz are never resolved worse than the
    reference plug-in (RMSE ratio 0.88–1.04 and 0.96–1.32). rieszboost is
    0.90–1.21, resolved worse once (`sparse50`, 1.12). Capped riesznet is
    0.97–1.31, never resolved worse. riesztree is resolved worse on 3 DGPs
    (`tail_feature` 1.18, `step` 1.12, `sparse50` 1.15) and augforestriesz
    on 5 (`tail_feature` 1.16, `logistic` 1.10, `step` 1.17, `sparse50`
    1.28, `additive` 1.09). Only augforestriesz on `sparse50` is above the
    proposed 1.25 threshold.
  - **Rate.** In the TSM block, RMSE × √n is flat or falling from n = 250
    to 4000 for the plug-in, rieszboost (both losses), krrr and
    forestriesz, with coverage 0.91–0.98. riesznet capped at |α̂| ≤ 100 is
    not: on `additive_g2` its RMSE × √n goes 4.9, 7.8, 3.6, from three
    blow-up replicates at n = 1000 (errors 1.9, 1.4, 1.2 against at most
    0.35 for the plug-in). Capped at |α̂| ≤ 20 (c = 0.05) they go away
    (ratio to the plug-in 0.57, 1.00, 0.96).
- **Claims 1 and 4, the other estimands.**
  - **ATT.** Coverage 0.87–0.97 in the 12 DGPs other than γ = 3, the
    lowest at γ = 2 (rieszboost 0.870, augforestriesz 0.875) and on
    `additive` (riesztree 0.890, riesznet 0.895). rieszboost's RMSE ratio
    reaches 2.42 on `overlap_g1` from one replicate (error 3.33; the next
    largest is 0.35). Capped at |α̂| ≤ 19, like the reference plug-in, that
    replicate's error is 0.53 and the ratio 1.01. Resolved worse than the
    reference: rieszboost on `sparse50` (1.27), forestriesz on
    `tail_feature` (1.23), riesztree on `step` (1.23), augforestriesz on
    `tail_feature`, `bulk_feature`, `step` (1.18–1.27).
  - **Additive shift.** Every learner covers 0.92–0.96, including
    rieszboost (0.84 on `shift_bimodal` in the pilot). krrr's RMSE is
    0.80–1.00 times the classification plug-in's and capped riesznet's
    0.96–1.68. rieszboost's is 0.92–2.43, the large values (`shift_gauss`
    1.98, `shift_bimodal` 2.43) from a few replicates and unresolved (z 1.6,
    2.6). Capped at c = 0.01 (|α̂| ≤ 98) it is 1.61 and 1.83; at c = 0.05
    (|α̂| ≤ 18), 1.06 and 0.80.
  - **LASE.** Coverage 0.90–0.96, RMSE 0.93–1.18 times the plug-in's,
    nothing resolved worse.
- **Claim 2, learners (α̂ RMSE; Table 4).**
  - **ATE:** krrr resolved best on `overlap_g05`, `overlap_g1` and
    `interaction` (z −7.8 to −10.4); forestriesz on `overlap_g2`,
    `overlap_g3`, `smooth`, `sparse50` and `additive` (z −3.9 to −18);
    rieszboost on `steps10` (z −7.2, 2.6% ahead of forestriesz). krrr on
    `logistic` is best but unresolved (z −2.4).
  - **ATT:** forestriesz resolved best on `smooth`, `sparse50`, `additive`
    and `steps10`; krrr on `interaction`. krrr leads at γ = 1 and 2 and
    forestriesz at γ = 3, unresolved (z −2.4 to −2.7).
  - **Additive shift:** krrr resolved best on `shift_gauss`, riesznet on
    `shift_skew`. **LASE:** rieszboost on three of four DGPs (z −6.3 to
    −29).
  - **Never resolved best:** riesztree and augforestriesz. augforestriesz
    is the unresolved best on `tail_feature` and `bulk_feature` (ATE and
    ATT) and `step` (ATT), by 2–4% (|z| ≤ 1.9).
  - **Why the differences shrink in ψ̂.** In the 12 ATE DGPs, the α̂ part
    of ψ̂'s error is 0.32–0.86 times the shared part, so a 10–68% spread in
    α̂'s RMSE becomes a 5–49% spread in ψ̂'s. In the additive shift the
    ratio is 0.40–1.37 and ψ̂'s RMSE spreads up to 3.0 times.
- **Claim 3, losses.** KL's α̂ RMSE is 13–23% below the squared loss's on
  `additive` and `additive_g2` at n = 1000 and 4000 (z −12 to −42), and the
  gap grows with n. The two tie at n = 250. **On `interaction`, KL is
  resolved worse at n = 1000** (3.6%, z = +5.4) and slightly worse,
  unresolved, at n = 4000 (z = +1.9). This is the result section 3 said
  would change the claim: it now needs "and slightly worse when the logit
  has interactions". On ψ̂ the losses tie everywhere (|z| ≤ 2.6).

**Gaps.**
1. **Caps, for every learner, not only riesznet.** A single replicate in
   which |α̂| drifts can dominate an RMSE over 200 replicates, and more
   replicates do not fix that. In the final run it happens to rieszboost
   (ATT `overlap_g1`; shift `shift_gauss`, `shift_bimodal`) and to riesznet
   capped at 100 (TSM `additive_g2`), besides riesznet uncapped. Capping at
   the reference plug-in's level (c = 0.05 for the ATT) removes each case.
   Proposed rule, for the user to decide: report every Riesz learner capped
   at the clipping level of the reference plug-in it is compared with
   (c = 0.01 for the ATE and TSM, 0.05 for the ATT), and pick one level for
   the shift and the LASE, whose reference is unclipped.
2. **Coverage at γ = 2.** Claim 1 promised coverage near 0.95 with good or
   moderate overlap. At γ = 2 four of six learners are resolved below
   (0.855–0.890). Either narrow to γ ≤ 1 or report γ = 2 as the start of
   the boundary.
3. **KL is worse on `interaction`** (claim 3, above).
4. **riesztree and augforestriesz never win, and are resolved worse than
   the plug-in in 3–5 ATE DGPs.** augforestriesz's 2–4% leads on
   `tail_feature`, `bulk_feature` and `step` are its only candidate wins;
   at |z| ≤ 1.9 they cannot be sized from 200 replicates, and those DGPs
   stay at 200 in the appendix, so they will not be tested. In the core
   DGPs augforestriesz is resolved worse than the plug-in on `logistic`
   and `sparse50`, riesztree on `sparse50`. Since the
   package recommends augforestriesz, the paper should say why or show
   forestriesz as the package's forest.
5. **The classification plug-in has the best α̂ on the LASE.** The edge is
   structural: the LASE's representer, 1(a < t + δ) r − 1(a < t), jumps at
   a = t and a = t + δ; the plug-in applies those indicators exactly, while
   a Riesz learner must learn the jumps. ψ̂ ties. A structured Riesz
   estimate built from an additive-shift fit, 1(a < t + δ)(α̂_shift + 1) −
   1(a < t), would apply them too; it was not tried.
6. **Replicates.** The user set 500 as the minimum. Sizing from these 200
   (n_sim for |z| = 3 is (3 sd(d) / mean(d))², d the per-replicate
   difference):
   - coverage gaps of 0.045 or more from 0.95 resolve by about 380
     replicates, and at 500 coverage has a Monte Carlo SE of 0.010;
   - unresolved best-vs-runner-up α̂ contrasts that 500 should resolve:
     ATE `logistic` (krrr, about 300), ATT γ = 1, 2, 3 (250–330),
     `shift_bimodal` (krrr, about 220);
   - ones 500 will not resolve: `shift_hetero` (about 820), and
     augforestriesz's leads (530 to 3,000, and too noisy to size; their
     DGPs stay at 200 in the appendix);
   - claim 3: KL vs squared on `interaction`, n = 4000, about 500;
   - RMSE-ratio contrasts driven by blow-ups (gap 1) are heavy-tailed, so
     extra replicates help them little; the cap does.
   More than 500 would buy only near-ties that no claim depends on.
   Decision (2026-10-23): extend the 8 core ATE/ATT DGPs and the TSM,
   shift and LASE blocks to 500 (`savio_final.sh`, about 1,300
   core-hours); the 5 appendix DGPs stay at 200.

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

*Since the rescope of 2026-10-22 (section 1), the claim 2 and claim 3
rows below belong to the later paper (section 8). Several DGPs built from
them (`smooth`, `step`, `sparse50`, `additive`, `interaction`) stay in this
paper as ordinary DGPs.*

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

## 6. Pilot results (2026-10-20 to 21)

**Notation used in sections 6 to 8.**

The data and the estimator:
- **Data.** X is the covariates, A the treatment or exposure, Y the outcome.
- **Outcome regression.** μ₀(A, X) = E[Y | A, X] is the true outcome
  regression, and μ̂ the fitted one (the "outcome model").
- **Estimand.** m is the estimand's linear functional, and ψ₀ = E[m(μ₀)]. For
  the ATE (average treatment effect), m(μ)(X) = μ(1, X) − μ(0, X). For the
  TSM (treatment-specific mean at A = 1), m(μ)(X) = μ(1, X).
- **Riesz representer.** α₀ is the Riesz representer, the weights with
  E[α₀ f] = E[m(f)] for every function f of (A, X). For the ATE it is
  A/π − (1 − A)/(1 − π), with π the propensity score. α̂ is an estimate of
  α₀, from a Riesz learner or from a plug-in.
- **One-step estimator.** ψ̂ = mean[m(μ̂) + α̂ (Y − μ̂)], where mean[·] is the
  average over the estimation sample. SE is its influence-function standard
  error, and empSE its standard deviation over replicates.

Comparators and measures:
- **"The plug-in"** with no qualifier is the XGBoost propensity plug-in with
  π̂ clipped to [0.01, 0.99].
- **RMSE** is root mean squared error: of α̂ against α₀ on a test set, or of
  ψ̂ against ψ₀ over replicates.
- **Paired z.** z is the mean paired difference over replicates divided by
  its Monte Carlo SE. "Resolved" means |z| ≥ 3, and each table says what the
  reference is.

DGP names:
- **overlap_gγ** scales the propensity logit by γ. A larger γ means weaker
  overlap.
- **additive_g2** is additive with its logit doubled.
- **The alignment ladder** has w, the share of the propensity's hard features
  that the outcome also has:
  - hard_unshared is w = 0;
  - mix33 is w = 1/3;
  - mix67 is w = 2/3;
  - hard_shared is w = 1.
- **The suffix _1 or _2** is the ladder's overlap level: good (logit scale
  0.78) or weak (1.56).

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
- A linear fit in (A, X) now explains 10–22% of Var(μ), down from 85–89%,
  and 8% of Var(τ), down from 49%.
- ψ₀ is still exactly 1 for the ATE and the TSM. For the shift, a term
  nonlinear in A makes ψ₀ 0.20–0.33, integrated by Monte Carlo.

α̂ is unchanged, since it never uses Y. The full output is in
`results/pilot_summary_hardmu_0_50.txt`.

**μ̂ is now about twice as wrong,** with RMSE 0.73 at n = 1000 against 0.38
before. **But ψ̂ still barely separates the methods.** 15 of 123 contrasts
reach |z| > 2, against 13 with the easy outcome. The term that carries α̂'s
quality into ψ̂ is the bias E[(α̂ − α₀)(μ₀ − μ̂)]. That is an inner product,
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

### Hard representers and the alignment ladder

Moved to section 8 (2026-10-22), with the middle rungs at 200 replicates.

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

### 5-fold CV tuning (2026-10-21)

**How it was run.** The same datasets, grids (v2) and replicates as the
single-split run above. The difference is that every learner, including the
outcome model, is now tuned by 5-fold cross-validation on the training
sample. Each setting is scored by its mean held-out loss, and the best one is
refit on the whole sample. XGBoost early-stops on the held-out fold, and the
refit uses the mean stopping iteration. The run also adds:
- the remainder and the semi-oracle;
- `plugin_xgb_rl`;
- the `tsm_align` block.

It ran on Savio as an 8-task array, with 18,650 task files and no errors.

The output is `results/pilot_summary_cv5.txt`, from
`python pilot.py summarize --compare results/pilot`. Since the clipping
refit (below), it also has the clipped variants. Besides the usual tables,
it has:
- the decomposition of the remainder below (s, s·B, mean[r e] and B), in
  each block's "remainder and semi-oracle" section;
- the single-split and CV results paired replicate by replicate, in the
  final "comparison" section.

**The replicates pair.** The estimation-sample outcomes and the untuned
logistic plug-in's α̂ agree to within 10⁻¹² in all 165 cells checked. Savio
and the laptop differ only in the last bits. μ̂ differs, because the outcome
model was retuned, so a change in ψ̂ mixes changes in α̂ and in μ̂. μ̂ is
about as good as before: the oracle's ψ̂ RMSE barely moves (0.125 to 0.125
on mix33_2, 0.147 to 0.148 on mix67_2).

**CV helped the XGBoost plug-in most.**
- **Its α̂ RMSE fell in all 20 ATE DGPs,** by 0.04 to 0.58. The fall is
  resolved in 8 DGPs, and three more sit at z ≈ −3.0.
  Examples: γ = 2 went from 2.57 to 1.98, hard_unshared_2 from 2.93 to 2.35,
  and mix33_2 from 2.78 to 2.40. This undoes the damage the wider grid did
  under single-split tuning.
- **The Riesz learners gained less.** Their improvements were resolved in
  this many of the 20 cells: riesztree 11, ForestRiesz 10, rieszboost 6,
  riesznet 3, krrr 2. The resolved gains were 0.01 to 0.27.
- **The shift classifier also improved,** e.g. from 0.330 to 0.235 on
  shift_gauss.
- **In ψ̂, the plug-in's gain shows at weak overlap.** Its RMSE went from
  0.153 to 0.139 on mix33_2, 0.236 to 0.216 on mix67_2, and 0.189 to 0.148 at
  γ = 2.

So every comparison against the plug-in moved in the plug-in's favour.

**What held and what moved.** ψ̂ unless marked α̂. The z values are paired,
on squared error, against the plug-in clipped at 0.01.

| Result | Single split (v2) | 5-fold CV | Verdict |
|---|---|---|---|
| ForestRiesz best at w = 1/3, weak overlap | 0.108 vs 0.153, z = −6.2 | 0.102 vs 0.139, z = −4.4. The oracle has 0.125 | **Holds** |
| w = 1/3, good overlap: rieszboost and ForestRiesz beat the plug-in | z = −3.2 and −3.4 | z = −0.3 and −0.7 | **Gone.** It came from the plug-in's noisy tuning, as suspected |
| w = 2/3, good overlap | ForestRiesz z = −2.9 | ForestRiesz z = −3.5 (0.148 vs 0.155). krrr, riesznet, riesztree and logistic are worse (+6.7 to +9.5) | Now resolved, but small |
| w = 2/3, weak overlap: Riesz learners worse | only riesznet and riesztree resolved (+4.4) | **every Riesz learner resolved worse:** ForestRiesz +3.2, rieszboost +6.3, krrr +6.4, riesztree +14.8, riesznet +15.2 | **Back, and stronger** |
| w = 1, weak overlap (50 reps) | rieszboost +3.7, ForestRiesz +8.3, riesztree +10.3 | ForestRiesz +9.1, riesztree +10.0; rieszboost +2.2 and krrr −1.0 unresolved | Holds for ForestRiesz and riesztree |
| w = 1, good overlap (50 reps) | riesztree +3.8 | krrr +5.9, riesznet +5.2, riesztree +6.2; ForestRiesz +2.7 | More resolved worse |
| Outside the alignment ladder, everything ties | yes | yes. The one exception is the logistic plug-in on sparse50 (+3.1) | Holds |
| At γ = 3, estimated weights beat the oracle | yes | yes. The oracle has 0.213, ForestRiesz 0.147, the plug-in clipped at 0.05 0.151, riesztree 0.164 and rieszboost 0.178 | Holds |
| α̂: KL beats squared (TSM) | z = −8.5 to −23 | z = −9.8 to −26. They still tie at n = 250 and on interaction | Holds |
| α̂: the XGBoost plug-in beats rieszboost on the TSM at n ≥ 1000, additive logits | yes | yes, by more. KL vs plug-in: +13.9 on additive at n = 4000, +9.5 and +20.7 on additive_g2 at n = 1000 and 4000 | Holds |
| α̂: krrr beats the shift classifier | z = −5 to −15 | z = −6.4 to −23, on all four DGPs | Holds |
| krrr's ψ̂ edge on the bimodal shift | 0.056 vs 0.087 | 0.052 vs 0.056 (z = −0.5) | **Gone.** The classifier improved |

**Best Riesz learner by α̂ under CV.** Paired z against the runner-up.
- **Resolved (|z| ≥ 3):**
  - krrr at γ = 1 (z = −3.6) and on interaction (−7.1);
  - ForestRiesz at γ = 3 (−6.8), on sparse50 (−7.6), and on mix33_1, mix67_1
    and mix67_2 (−3.0, −10, −38);
  - rieszboost on hard_unshared_2 and hard_shared_2 (−6.5 and −6.4).
- **Unresolved:** smooth, step, additive, tail, bulk, logistic, γ = 0.5,
  γ = 2, mix33_2, and the level-1 endpoints. ForestRiesz is nominally best on
  smooth, step and additive, where single-split tuning had rieszboost.
- **riesztree is best nowhere.** On step it is third (0.980, against 0.899
  for ForestRiesz).
- **The XGBoost plug-in now has a better α̂ than every Riesz learner** on
  smooth (0.891), step (0.717) and additive (0.768).

**Remainder and semi-oracle: the crossover is bias, and the bias comes from
shrunk weights.** The notation is as at the top of section 6. In particular,
the Riesz representation is E[α₀ f] = E[m(f)] for any function f of (A, X).
Three findings:
- **The remainder is ψ̂'s bias.** The remainder of a fit is
  mean[(α̂ − α₀)(μ₀ − μ̂)], computed on each estimation sample. Its average
  over replicates matches the bias of ψ̂ (also an average over replicates)
  for every method. On mix67_2, for example, ForestRiesz has remainder 0.212
  and bias 0.206, and the plug-in 0.183 and 0.179.

  The match is expected. Write Y − μ̂ = (Y − μ₀) + (μ₀ − μ̂) in ψ̂. Then
  ψ̂ − ψ₀ = remainder + mean[α̂ (Y − μ₀)] + mean[m(μ̂) − α₀ μ̂] +
  (mean[α₀ μ₀] − ψ₀). The fits come from the training sample, so on the
  estimation sample they are fixed functions. Each of the three extra terms
  then has mean zero: the first because E[Y − μ₀ | A, X] = 0, and the other
  two by the Riesz representation, applied to f = μ̂ and to f = μ₀.
- **Variance is not what makes the Riesz learners lose.** With the true μ₀ (the
  semi-oracle), every Riesz learner beats the plug-in at both weak-overlap
  middle rungs (z from −3.1 to −5.0). Every Riesz learner's α̂ also beats
  the oracle there (0.070 to 0.075 against 0.105 on mix67_2).
- **The remainder decides the comparison.** On mix67_2, every Riesz learner's
  remainder is larger than the plug-in's (z from +7.2 to +19.7), and 74–85% of
  it comes from the 10% of units with the largest |α₀|. On mix33_2,
  ForestRiesz's remainder is smaller (z = −7.3) and the other Riesz learners'
  remainders are larger (+2.1 to +6.8). So ForestRiesz wins there on both
  bias and variance.

**Why the remainder grows with sharing.** On each estimation sample, define
three quantities:
- **The outcome error** e = μ₀ − μ̂, so the remainder is mean[(α̂ − α₀) e].
- **The shrinkage** s, the least-squares coefficient in a fit through the
  origin: s minimizes mean[(α̂ − α₀ + s α₀)²]. The solution is
  s = −mean[(α̂ − α₀) α₀] / mean[α₀²]. Equivalently, 1 − s is the slope of
  α̂ on α₀ through the origin. With 0 < s < 1 (true in every cell here), α̂
  is on average a shrunk copy of α₀.
- **The residual** r = α̂ − α₀ + s α₀, the part of α̂'s error that is not a
  rescaling of α₀. The least-squares condition makes mean[r α₀] = 0.

Substitute α̂ − α₀ = −s α₀ + r into the remainder, and pull s out of the
mean (it is one number per sample):

  remainder = mean[(−s α₀ + r) e] = −s·mean[α₀ e] + mean[r e] = s·B + mean[r e],

with **B** := −mean[α₀ e] = mean[α₀ (μ̂ − μ₀)].

This is an identity: it holds for any s once r is defined as above. Least
squares only buys the reading of r as "not a rescaling". Three checks:
- the oracle has s = 0 and r = 0, so its remainder is 0;
- α̂ ≡ 0 has s = 1 and r = 0, so ψ̂ is the outcome-only plug-in and the
  remainder is B;
- the semi-oracle has e = 0, so its remainder is 0.

**What B is.** μ̂ is a fixed function on the estimation sample, so B
averages to E[α₀ (μ̂ − μ₀)]. By the Riesz representation with f = μ̂ − μ₀
and the linearity of m, that equals E[m(μ̂)] − E[m(μ₀)] = E[m(μ̂)] − ψ₀.
That is the bias of the outcome-only plug-in mean[m(μ̂)]. So, holding
mean[r e] fixed, a shrunk α̂ leaves a share s of the outcome model's own
bias uncorrected.

**B is the same for every method in a replicate,** because every α̂ shares
one μ̂. Differences between methods come only from s and mean[r e]. The
values below are replicate averages.
- **B is what the alignment ladder moves.** At overlap level 2, B is −0.14,
  0.08, 0.35 and 0.61 for w = 0, 1/3, 2/3 and 1. At level 1 it is −0.11,
  0.03, 0.18 and 0.33. When the propensity and the outcome share hard
  features, μ̂ is wrong in the direction α₀ weights: it overestimates where
  α₀ > 0 and underestimates where α₀ < 0. At w = 0 the reverse holds
  slightly, so B < 0.
- **Every estimated α̂ is shrunk, and the XGBoost plug-in least.** At
  level 2 the shrinkage s is:
  - 0.35–0.39 for the XGBoost plug-in;
  - 0.41–0.48 for rieszboost and ForestRiesz;
  - 0.46–0.53 for krrr, riesztree and riesznet;
  - 0.53–0.57 for the logistic plug-in.

  The same fit run on the 10% of units with the largest |α₀| alone gives
  0.52–0.80, so the extreme weights are shrunk most.
- **At mix67_2 the shrinkage term s·B decides.** Its replicate average is
  0.143 for the plug-in and 0.163–0.193 for the Riesz learners, which is
  65–81% of each remainder.
- **The correlation of s with the remainder follows the sign of B.** B is
  shared, so a method with a larger s has a remainder larger by about B per
  unit of s, as long as its mean[r e] does not move with s. So across
  methods, s and the remainder should correlate with the sign of B. The
  methods here are the five Riesz learners and the three plug-ins clipped at
  0.01. On the alignment ladder they do:
  - positively in every rung with B > 0 (0.55 to 0.94, and 0.72 to 0.94
    from w = 2/3 on);
  - negatively (−0.69 and −0.75) at the unshared endpoints, where B < 0.

  Off the ladder, the sign follows B in 2 of the 3 DGPs with |B| ≥ 0.1:
  - tail −0.86 and bulk −0.82, both with B ≈ −0.12;
  - but sparse50 has +0.13 with B = −0.23.

  Where |B| is small, the correlation says little, because s·B is small
  next to mean[r e].
- **ForestRiesz's win at w = 1/3 is in the residual r, not the shrinkage
  s.** Its residual term mean[r e] is 0.005, against 0.039–0.063 for every
  other method. Its shrinkage term (0.036) is
  slightly larger than the plug-in's (0.029). So, in the sample inner
  product mean[r e], the part of its error that is not a rescaling is nearly
  orthogonal to μ̂'s error there. At w = 2/3 its residual term is the
  plug-in's (0.041), and the shrinkage s decides.

**So the hypothesis holds, in a sharper form.** The identity is only
accounting. The evidence for the mechanism is three findings:
- s·B is most of the remainder;
- the s–remainder correlation flips with the sign of B;
- the Riesz learners win with the true μ₀.

The Riesz learners shrink the extreme weights more than the XGBoost plug-in
does. The cost is s·B, so it decides the comparison once the outcome model's
bias B is large, and sharing hard features is what makes it large. When B is
small, the Riesz learners' lower variance wins.

Why the Riesz learners shrink: the expected squared Riesz loss is
E[α² − 2 m(α)] = E[(α − α₀)²] − E[α₀²], by the Riesz representation with
f = α. So cross-validation on it picks the setting with the smallest mean
squared error of α̂, not the smallest bias of ψ̂. This points to
undersmoothing (section 7).

s is descriptive: it is a regression on each estimation sample, and how much
a method shrinks depends on its tuning.

**plugin_xgb_rl does not help.** This is the XGBoost plug-in with its setting
chosen by the held-out Riesz loss of its weights.
- **α̂: worse or tied everywhere, and never resolved better.** It is resolved
  worse in:
  - five ATE DGPs: bulk +0.32, γ = 2 +0.38, tail +0.24, and mix33_2 and
    mix67_2 +0.11 each;
  - all four tsm_align rungs, +0.18 to +0.46;
  - the TSM on additive_g2 at every n.
- **ψ̂: ties in the ATE and tsm_align blocks** (|z| < 2.5). It is worse on the
  TSM on additive_g2 at n = 4000 (z = 3.4).
- **It shrinks least.** Its shrinkage s (defined above) is 0.24–0.30 in
  tsm_align, against 0.32–0.36 for the log-loss-tuned plug-in, and it has
  the smallest remainder there. But its variance is larger: empSE 0.141
  against 0.097 on mix67_2. A held-out Riesz loss is a
  noisy criterion when the weights have heavy tails.
- **Caveat.** Only depth and learning rate are chosen by the Riesz loss. The
  number of trees still comes from early stopping on log loss, which is
  XGBoost's default evaluation metric. So this is a partial test. Its
  learning rate picks 1.0, the top, in 19–36% of fits.
- **Reading.** Changing the plug-in's selection loss does not make it behave
  like a Riesz learner. Where the Riesz learners' α̂ is better, the
  advantage comes from the loss they are fit with, not the loss they are
  selected by.

**tsm_align: claim 3 judged on ψ̂.** This is the TSM on the weak-overlap
alignment ladder, with 50 replicates.
- **On α̂, KL beats squared at every rung** (z from −3.1 to −8.9, and −8.1 to
  −10.5 on the weak-overlap rows).
- **ψ̂ depends on α̂ here.** The bias rises along the ladder: −0.04, 0.05,
  0.11 and 0.23 for squared. It is smaller than the ATE's at the same rung.
  The outcome-only plug-in's bias B is smaller (0.20 against 0.35 on
  mix67_2), while the shrinkage s is similar (0.38–0.44 for squared, against
  0.41–0.48 for the ATE's rieszboost).
- **On ψ̂, KL and squared tie** (z from −2.3 to +2.0). On mix67_2, KL has
  RMSE 0.135 against 0.141 (z = −2.3; about 90 replicates would reach
  |z| = 3).
- **The mechanism points the right way, but the effect is small.** KL shrinks
  less than squared (s = 0.33–0.41 against 0.38–0.44), and its remainder is
  smaller by 0.005–0.007. That is about 5% of ψ̂'s RMSE.
- **Neither loss beats the plug-in on ψ̂** (|z| ≤ 1.8).
- **Coverage fails from w = 2/3 on, for every method:** 0.70–0.76 on mix67_2
  and 0.20–0.34 on hard_shared_2.

**Grid edges under CV** (share of fits at an edge):
- **riesztree:** minimum leaf size picks 50, the top, in 66% of ATE fits. The
  grid has to extend (e.g. to {20, 50, 100, 200}).
- **rieszboost:** the L2 penalty picks 1000, the top, in 28% of ATE fits, 42%
  of TSM fits and 44% of shift fits. rieszboost_kl picks it in 40% of TSM
  fits. The penalty should extend to 10⁴. In the shift block the penalty
  also picks 0 in 21%, but 0 is a hard limit.
- **plugin_xgb_rl:** the learning rate picks 1.0 in 19–36% of fits.
- **Near the 25% mark:** rieszboost picks depth 8 in 21–25% of tsm_align fits,
  riesznet picks the smallest learning rate in 17% (ATE) and 30% (shift), and
  krrr picks the smallest length scale in 18%.
- **Exempt (hard limits):** the XGBoost plug-in at depth 1 (22% of ATE fits,
  60% of TSM fits), and ForestRiesz at leaf size 1 (15%).

### Clipping the Riesz learners' α̂ (2026-10-21)

**Why.** The plug-ins are reported clipped (π̂ in [c, 1 − c]), while the Riesz
learners were not. That is an asymmetry in the comparison, and riesznet's
blow-ups are exactly the unbounded weights that clipping removes.

**How.** Each Riesz learner gets two variants, `<method>_clip01` and
`<method>_clip05`. They cap |α̂| at the largest |α̂| a plug-in can reach with
its probability clipped at c:
- 1/c for the ATE and TSM (100 and 20);
- (1 − c)/c − 1 for the shift (98 and 18).

Only the predictions are capped. The fit and its tuning are unchanged.

A full clip to the plug-in's range would also raise α̂(1, x) to at least
1/(1 − c), and set the TSM's α̂(0, x) to 0. That was tried on the stored
estimation-sample α̂. It made ψ̂ resolved worse in 9 of 138 cells, all on the
shared-feature rungs, and better in none. So only the cap is used.

**Rerun.** To get the clipped variants' test-set α̂ RMSE, every finished
Riesz task was refit at the setting its 5-fold CV chose
(`python pilot.py run --refit`). The refit skips the tuning, so the 9,900
tasks took 9 minutes on the laptop. It reproduces Savio's fits:
- exactly for rieszboost (both losses), krrr and riesztree;
- to within 1.5 × 10⁻³ in relative α̂ RMSE for ForestRiesz (the platforms
  differ; the laptop is deterministic);
- for riesznet, in all but 21 of 1,800 tasks.

The 21 riesznet tasks are its blow-up replicates (Savio α̂ RMSE 6 to 190),
where training is chaotic and a 10⁻⁷ difference grows. Their α̂ RMSE moved by
up to 6×. So riesznet's unclipped numbers in the tables now differ a little
from the Savio run: at γ = 3, ψ̂ RMSE is 0.63 against 0.90. Using the refit
for every variant keeps each clipped variant paired with its own unclipped
fit. The Savio results are backed up outside the repo.

**The cap at c = 0.01 (100) almost never binds.**
- It changes α̂ in 16 of 138 cells, and no change is resolved, on any
  measure.
- It binds somewhere on the test set in at most 14% of fits (krrr at γ = 3).
  The exception is riesznet, at 10% and 32% of fits at γ = 2 and 3.
- The Riesz learners already shrink the large weights more than the
  plug-in, so a cap at 1/c has little to cut.

**The cap at c = 0.05 (20) binds under weak overlap.** It binds in 88–100% of
ForestRiesz fits at γ = 2 and 3, 94% of krrr fits at γ = 3, and 60–100% of
TSM fits on additive_g2 at n = 4000.
- **α̂ gets worse where it binds.** It is never resolved better. It is
  resolved worse in 5 cells:
  - ForestRiesz at γ = 3 (+0.06, z = 13);
  - krrr at γ = 3 (+0.05);
  - the TSM on additive_g2 (rieszboost_kl at n = 4000, +0.22, z = 15).

  It is resolved worse on the weak-overlap rows in 7 cells. Capping at 20
  where α₀ reaches 1,000 adds error.
- **ψ̂: no resolved change in any cell.** The effect follows the s·B
  decomposition: the cap raises the shrinkage s, which raises the bias and
  lowers the variance. For krrr at γ = 3, s goes from 0.32 to 0.39, the bias
  from 0.021 to 0.043, and the RMSE from 0.192 to 0.161.

**The cap fixes the blow-ups, but z does not show it.** Its large effects are
on the two learners that blow up:

| DGP | riesznet ψ̂ RMSE: unclipped / c = 0.01 / c = 0.05 | rieszboost ψ̂ RMSE: unclipped / c = 0.01 / c = 0.05 |
|---|---|---|
| γ = 3 | 0.632 / 0.211 / 0.157 | 0.178 / 0.178 / 0.170 |
| additive | 0.195 / 0.127 / 0.094 | unchanged (0.088) |
| step | 0.133 / 0.124 / 0.106 | unchanged (0.097) |
| shift_bimodal | 0.278 / 0.129 / 0.089 | 0.128 / 0.099 / 0.052 |

For comparison, at γ = 3 the plug-in clipped at 0.05 has 0.151, and on
shift_bimodal krrr has 0.052. Every one of these paired z values is between
−0.8 and −1.8. Each gain comes from a few blow-up replicates, and a paired
z of about −1 is the signature of one outlier. So the cap's main effect is
real but unresolvable by a mean-based z. Reporting the number of blow-ups,
or a median, would show it.

**Against the plug-in at equal bounds, nothing changes.** On ψ̂, the clipped
Riesz learners are resolved better than the plug-in clipped at the same c in
the same cells as before:
- ForestRiesz on mix33_2 (z = −4.4 at c = 0.01, −5.4 at c = 0.05);
- ForestRiesz on mix67_1 (−3.5).

They are resolved worse in the same shared-feature cells as before: mix67_2,
mix67_1, hard_shared_1 and hard_shared_2.

One contrast is new: on the TSM on additive_g2 at n = 4000, rieszboost capped
at 20 is worse than the plug-in clipped at 0.05 (z = +3.1). Unclipped
rieszboost was never compared with that plug-in.

**What this means.**
- **Clipping is not why the plug-in wins at strong sharing.** Capping the
  Riesz learners' weights does not change any resolved comparison with the
  plug-in.
- **Under weak overlap, the remainder analysis predicts the cap cannot help
  the bias.** At strong sharing the Riesz learners lose because they shrink
  too much. A cap shrinks more.
- **For riesznet, a cap is a cheap fix for its instability.** With c = 0.05
  it is level with the other methods at γ = 3 and on additive. That
  addresses "fix or drop riesznet" (section 7). A cap at the Riesz learner's
  own bound should be a default when riesznet is reported.

### Suggested next steps

See section 7, "Still to investigate".

## 7. Evidence map (2026-10-21)

Everything here is pilot-grade:
- It is all at n = 1000 (the TSM block also has n = 250 and 4000), with 50
  replicates per cell. The exception is the four middle rungs of the
  alignment ladder, which have 200.
- ψ̂ uses sample splitting, not cross-fitting.
- **Updated for 5-fold CV tuning** (section 6), which replaced single-split
  tuning for every learner. CV helped the XGBoost plug-in most, so several
  Riesz-learner wins from the single-split runs shrank or vanished.

Read these as strong leads, not final results. "Resolved" means |z| ≥ 3 on
a paired contrast.

### Claims with evidence

**Claim 1: Riesz regression is not worse than plug-in, and may be better
under practical positivity violations.**
- *Not worse:* **supported where the propensity and the outcome share no
  hard structure.** Outside the alignment ladder, no Riesz learner is
  resolved worse than the plug-in on ψ̂, in any DGP and at any overlap
  level. riesznet is the one weak spot: it blows up at γ = 3 (ψ̂ RMSE 0.63,
  SE/empSE 0.63), though the blow-up is not resolved at 50 replicates.
  Capping its |α̂| at 1/c removes most of the blow-up (ψ̂ RMSE 0.21 at
  c = 0.01, 0.16 at c = 0.05). See "Clipping the Riesz learners' α̂" in
  section 6.
- *Clipping does not change claim 1.* With every Riesz learner's |α̂|
  capped as the plug-in's is, the resolved ψ̂ contrasts against the plug-in
  clipped at the same c are the same as without the cap.
- *Better, not when, and mechanism:* these rested on the alignment ladder
  and moved to section 8.
- *Tuning loss vs model class:* **the loss used for selection is not what
  matters.** Tuning the XGBoost plug-in on the Riesz loss (`plugin_xgb_rl`)
  makes its α̂ worse or no better everywhere, and leaves ψ̂ unchanged. This
  is a partial test, since its number of trees is still chosen by log loss.
- *Correct logistic model:* the logistic plug-in clearly has the best α̂.
  On ψ̂, everything ties.
- *Shift:* krrr has a better α̂ than the classification density ratio on all
  four DGPs (z from −6 to −23). On ψ̂ everything ties under CV. The
  single-split edge on the bimodal shift came from the classifier's
  tuning.

**Claims 2 and 3** moved to section 8.

**New claims the pilot turned up**
- **Under weak overlap, estimated weights can beat the true ones.**
  - At γ = 3, every Riesz learner except riesznet, and every plug-in clipped
    at 0.05, beats the oracle (0.131–0.192 against 0.213).
  - With the true μ₀, every Riesz learner's α̂ also beats the oracle on the
    weak-overlap middle rungs.

  Shrinking the weights lowers the variance. Its cost in bias is s·B: the
  shrinkage s times the outcome-only plug-in's bias B. That cost is small
  when B is small.
- **How the plug-in is tuned changes the Riesz-vs-plug-in comparison.**
  Moving from single-split tuning to 5-fold CV improved the XGBoost plug-in's
  α̂ in all 20 ATE DGPs, by up to 0.58. That is more than any Riesz learner
  gained. Any claim against a plug-in has to state its tuning protocol.

### Still to investigate

Items 1, 2, 5 and 8 moved to section 8 (2026-10-22).

3. **Coverage.** No estimated method reaches 0.95 on the middle rungs. On
   w = 1/3 with weak overlap, ForestRiesz reaches 0.925 and the others
   0.79–0.89. The questions are whether cross-fitting or a larger n fixes
   this, and whether any method gives valid intervals there.

   **Decision (2026-10-21): the final runs use 5-fold cross-fitting.** It is
   what a practitioner does with one sample of n, and the coverage claims
   should describe that.
   - **The design.** For each fold, every nuisance is tuned (5-fold CV
     inside the other four folds) and fit on those four folds, then predicts
     on the held-out fold. ψ̂ averages the out-of-fold predictions over all
     n.
   - **Sample size.** n stays the total sample (1000; for the TSM, 250 to
     4000), so each nuisance is fit on 4n/5.
   - **Pairing.** The cross-fitting sample is the old training sample, so
     the replicates pair with `pilot_cv5`.
   - **Code.** It is in `pilot.py`, with results in `results/pilot_cf5`.
   - **What to expect.** Averaging over all n shrinks the SE, while the
     bias (about s·B) barely moves. So coverage could get worse than under
     sample splitting.
4. ~~**Tuning grids that must move before the final runs.**~~ **Done
   (2026-10-21), in `pilot.py`'s grids; the reasons are recorded above
   `GRIDS`.** The moves:
   - riesztree's minimum leaf size: {5, 20, 50} → {5, 20, 50, 100, 200};
   - rieszboost's L2 penalty: {0, 1, 10, 100, 1000} → {0, 10, 100, 1000,
     10⁴};
   - krrr's length scale: one step down added, to {0.125, …, 8} × median;
   - riesznet's learning rate: 10⁻⁴ added.

   The existing results used the old grids. The next run uses the new ones,
   and its edge table checks them. Single-split tuning is retired, and
   5-fold CV is now the protocol.
6. ~~**The plug-in tuned on the Riesz loss.**~~ **Dropped (2026-10-21).**
   This is for a later paper. `plugin_xgb_rl` is out of every block, and
   its results in section 6 stay as a record.
6b. **The forests (2026-10-21).** The ATE block now has two forests:
   - `augforestriesz` (AugForestRieszRegressor), the package's recommended
     forest;
   - `forestriesz` (ForestRieszRegressor), the ForestRiesz of Chernozhukov
     et al. (2022), kept for the comparison with the published method.

   Every ForestRiesz result above is the published one.

   **AugForest's first results** come from the sample-splitting design
   (`pilot_cv5`), run on the laptop with the pilot's replicate counts. It
   used its defaults with the minimum leaf tuned; CV picked 20 or 50 in 92%
   of fits, with no grid edge.
   - **α̂: AugForest is worse than ForestRiesz almost everywhere.** It is
     resolved worse in 13 of 20 DGPs, and resolved better in none. Examples:
     sparse50 0.930 against 0.803 (z = 19), γ = 3 6.06 against 5.72, and
     mix67_2 2.46 against 2.31. It ties only on tail, bulk, step, smooth and
     the level-1 endpoints.
   - **ψ̂: it is resolved worse in 7 DGPs,** all on sparse50 or the
     alignment ladder. On mix67_2 it has 0.266 against 0.228 (z = 12).
   - **It loses ForestRiesz's win.** On mix33_2, AugForest has ψ̂ RMSE
     0.124, against 0.102 for ForestRiesz and 0.139 for the plug-in
     (z = −1.9 against the plug-in, unresolved). Its coverage there is
     0.865, against 0.925 for ForestRiesz.
   - **Not from shrinkage.** Its shrinkage s is about ForestRiesz's (0.43
     on mix33_2). The gap is in the residual term and the variance.
   - **Possible causes,** not yet tested. The two forests differ in several
     ways:
     - full bootstrap samples against half-samples (45%);
     - trees over the augmented (A, X) rows against a per-leaf solve with a
       treatment-indicator basis;
     - minimum split size 2 against 10.

     So the claim-1 win at w = 1/3 belongs to the published ForestRiesz,
     not to forests in general.
7. **The shift block.** Two questions:
   - Does krrr's α̂ advantage reach ψ̂ when the shift DGP's exposure and
     outcome share structure? Under CV it does not reach ψ̂ on the current
     DGPs.
   - Does krrr's edge come from the loss or from the kernel? A
     kernel-family classifier would separate the two.
9. **Leftover design issues.**
   - Redesign tail vs bulk on a base that is not nearly logistic, and with
     matched overlap.
   - ~~Fix or drop riesznet.~~ **Kept, with a cap (2026-10-21).** riesznet
     is now reported as `riesznet_clip01`, with |α̂| capped at 1/0.01, the
     bound of the reference plug-in. The contrasts against the plug-ins use
     that variant. The other Riesz learners barely change under that cap,
     so they are reported as fitted.
   - **Blow-ups are now counted.** The ψ̂ tables have two extra measures:
     the median |error|, and the number of blow-ups. A blow-up is a
     replicate whose error is further than 5 times the oracle's empirical SE
     from that method's own median error, so a steady bias does not count.
     - At γ = 3, unclipped riesznet has 4 blow-ups in 50 replicates, and
       riesznet capped at 100 has none.
     - The other blow-ups are single replicates: riesznet on additive,
       smooth, step, tail and hard_shared_2; rieszboost on the gauss and
       bimodal shifts; and the XGBoost plug-ins on the TSM at n = 250.
10. **Many covariates (requested 2026-10-20, deferred).** At least one
    scenario with a lot of covariates, judged on ψ̂. sparse50 already has
    50 covariates, of which 3 drive treatment, but it was judged mainly on
    α̂. Two options, not yet chosen:
    - **Noise covariates:** the key alignment rungs (w = 1/3 and 2/3, weak
      overlap) with 45 irrelevant covariates added, p = 50. The truth and
      ψ₀ are unchanged. It tests whether each learner can ignore noise;
      krrr's isotropic kernel is the likely casualty. Cheap, since the
      outcome does not change.
    - **Many weak confounders:** p = 50, with 45 extra covariates that each
      nudge both treatment and outcome a little, and overlap re-matched.
      Closer to real empirical data. It needs a per-DGP outcome, so more
      work.

## 8. Later paper: where methods are strong or weak

Moved here on 2026-10-22, when this paper was rescoped to "the package works"
(section 1). It is the record of what the pilots found, and the starting
point for a later paper on where methods are strong or weak. Unless a part
says otherwise, its results come from the sample-splitting runs
(`results/pilot`, `results/pilot_cv5`), not from cross-fitting.

### Claims

| # | Claim | Who cares, and what would they do differently |
|---|-------------------------------------------------------|------------------------------------------------------|
| 2 | There are different DGPs where some learners work better than others for Riesz regression | |
| 3 | There are different DGPs for which different losses work better than others (maybe low n?) | |

Old claim 1's "better under practical positivity violations" half also
belongs here, since its evidence is the alignment ladder.

### Cross-fitting on the ladder (`pilot_cf5`, 2026-10-22)

ATE coverage, from 50 replicates (200 on the four middle rungs, `mix33_*`
and `mix67_*`):

| DGP | oracle | reference plug-in (XGBoost, clipped at 0.01) | rieszboost | forestriesz |
|---|---|---|---|---|
| mix33_1 | 0.96 | 0.90 | 0.89 | 0.92 |
| mix33_2 | 0.94 | 0.88 | 0.83 | 0.87 |
| mix67_1 | 0.94 | 0.68 | 0.60 | 0.64 |
| mix67_2 | 0.95 | 0.48 | 0.26 | 0.26 |
| hard_shared_1 | 0.94 | 0.12 | 0.14 | 0.12 |
| hard_shared_2 | 0.92 | 0.06 | 0.02 | 0.00 |

- **The intervals have the right width and the wrong centre.** In these
  cells the mean SE of ψ̂ is 0.82–1.05 times the empirical SE (the standard
  deviation of ψ̂ over replicates), but the bias is 1.5–6 times the mean
  SE. The bias equals the remainder mean[(α̂ − α₀)(μ₀ − μ̂)], and the
  semi-oracle, which uses the true outcome regression μ₀ with the same α̂,
  covers 0.90–0.96.
- **Cross-fitting made coverage slightly worse,** as section 7 item 3
  predicted. forestriesz on mix33_2 (w = 1/3, weak overlap) fell from
  0.925 to 0.870, so its
  coverage lead there is gone.
- **Why the outcome regression leaves a large B,** the outcome-only
  plug-in's bias. Let τ(X) = μ₀(1, X) − μ₀(0, X) denote the true treatment
  effect, which is what the ATE averages. The fitted outcome regression μ̂
  explains about 73% of the variance of the true μ₀ in every DGP, but its
  implied effect μ̂(1, X) − μ̂(0, X) explains only about 20% of the variance
  of τ (R² −0.68 on hard_shared_2). In the shared DGPs, μ̂'s error
  concentrates where the true weights |α₀| are large. On hard_shared_2, the
  RMSE of μ̂ is 1.11 in the top 10% of |α₀| against 0.79 overall,
  corr(α₀, μ₀ − μ̂) = −0.31, and the outcome-only bias B = 0.77. On
  hard_unshared_2 the overall RMSE is the same (0.78), but the correlation
  is +0.07 and B = −0.16.
- **What would shrink the outcome-only bias B:** a stronger outcome
  library, a model that learns the effect τ directly (fits by arm, or
  A × X features), or a larger n.

### Pilot results moved from section 6

#### Hard representers: shared vs unshared features (50 replicates)

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

#### The alignment ladder (50 replicates)

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

#### The middle rungs at 200 replicates

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

### Evidence moved from section 7

**Old claim 1: the parts that rested on the ladder.**
- *Better:* **supported in one resolved case, robust to the change of tuning.**
  ForestRiesz at w = 1/3 with weak overlap has ψ̂ RMSE 0.102, against 0.139
  for the clipped plug-in (z = −4.4, 200 replicates) and 0.125 for the
  oracle. Its coverage is 0.925. A second resolved win, ForestRiesz at
  w = 2/3 with good overlap (z = −3.5), is small (0.148 against 0.155).
  The single-split wins at w = 1/3 with good overlap are gone under CV. They
  came from the plug-in's noisy tuning.
- *Not when:* **resolved, and stronger under CV.** At w = 2/3 with weak
  overlap, every Riesz learner is worse than the XGBoost plug-in (z from
  +3.2 for ForestRiesz to +15 for riesznet). At w = 1:
  - riesztree is resolved worse at both overlap levels;
  - ForestRiesz is resolved worse at weak overlap;
  - krrr and riesznet are resolved worse at good overlap.
- *Mechanism:* **well supported.** The crossover is bias, not variance.
  - With the true outcome regression μ₀ (the semi-oracle), every Riesz
    learner beats the plug-in on both weak-overlap middle rungs.
  - With μ̂, ψ̂'s bias is the remainder. The remainder is exactly s·B plus
    mean[r e]:
    - the shrinkage s says how far α̂ is shrunk toward zero;
    - B is approximately the bias of the outcome-only plug-in;
    - mean[r e] is the part that does not rescale α₀.
  - Sharing hard features makes B large (−0.14 to 0.61 along the ladder at
    weak overlap).
  - The Riesz learners shrink more than the XGBoost plug-in (s = 0.41–0.53
    against 0.35–0.39), so they leave more of B uncorrected.
  - ForestRiesz's win at w = 1/3 comes from the other part of the remainder,
    which is nearly zero for it.
  - The decomposition is only accounting. The evidence is three findings:
    - s·B is 65–81% of the remainder;
    - across methods, the s–remainder correlation flips sign with B;
    - the semi-oracle result.

**Claim 2: different learners win in different DGPs.**
- **Supported on α̂, in fewer DGPs than before.** Under CV, the resolved
  winners among Riesz learners are:
  - krrr with good overlap (γ = 1) and on interaction;
  - ForestRiesz with weak overlap (γ = 3), many covariates (sparse50), and on
    three of the four middle rungs of the alignment ladder;
  - rieszboost on the two endpoints of the ladder at weak overlap.
- **Unresolved:** smooth, step, additive, tail, bulk and logistic. The
  single-split picks for smooth and additive (rieszboost) and step
  (riesztree) do not hold. riesztree is best nowhere.
- **Two predictions failed:**
  - kernels and nets were expected to win on smooth, but tree ensembles do
    (unresolved);
  - riesztree was expected to win on step.
- **The plug-in is a real rival on α̂.** The XGBoost plug-in now beats every
  Riesz learner on smooth, step and additive.
- **On ψ̂, learners differ only where the propensity and the outcome share
  structure.** There, ForestRiesz is best at w = 1/3, and riesznet and
  riesztree are worst from w = 2/3 on.

**Claim 3: different losses win in different DGPs.**
- **Supported on α̂:** KL beats squared with an additive logit (z down to
  −26) and on all four rungs of the TSM alignment ladder (z from −3.1 to
  −8.9). The two tie with interactions.
- **The low-n version is not supported.** The two tie at n = 250, and KL's
  lead grows with n.
- **Not supported on ψ̂ at 50 replicates,** even where ψ̂ depends on α̂
  (`tsm_align`). The difference is at most z = −2.3, on mix67_2.
  - The mechanism points the right way: KL shrinks less, so its remainder is
    smaller by 0.005–0.007.
  - But that is about 5% of ψ̂'s RMSE. About 90 replicates would resolve it
    on mix67_2.

**A claim the pilot turned up**
- **The one-step estimator's bias depends on whether the propensity and the
  outcome share hard structure, not on how hard α₀ is.** The evidence is
  strong:
  - hard_shared and hard_unshared, with identical difficulty and overlap,
    behave completely differently.
  - The alignment ladder shows a dose-response.
  - The oracle stays unbiased throughout.
  - The decomposition says why. Sharing raises the outcome-only plug-in's
    bias B, and every estimated α̂, being shrunk, leaves a share s of B in
    ψ̂.

### Still to investigate, moved from section 7

- **Undersmoothing.** The remainder analysis says the Riesz learners lose
   at strong sharing because they shrink the extreme weights. The squared
   Riesz loss on held-out data picks the shrinkage that is best for α̂, not
   for ψ̂.
   - Test: on mix67_2 and mix33_2, take the Riesz learners' settings one
     step less regularized than CV picks (a smaller L2 penalty for
     rieszboost, a smaller minimum leaf for ForestRiesz).
   - Measure whether the shrinkage s and the remainder fall by more than the
     variance rises.
   - The CV scores of every setting are already stored, but the refits with
     other settings are not, so this needs new fits.
- **Sample size.** Run the middle rungs at n = 4000. The question is
   whether bias/SE shrinks or grows with n.
   - The bias is the expected remainder, E[s·B + mean[r e]]. The SE of ψ̂
     is about σ/√n, where σ is the standard deviation of that method's
     influence function m(μ̂) + α̂ (Y − μ̂) − ψ̂, and n is the size of the
     estimation sample.
   - So bias/SE ≈ E[s·B + mean[r e]]·√n / σ. It falls if and only if the
     expected remainder falls faster than 1/√n.
   - The decomposition says where to look. The outcome-only plug-in's bias B
     falls as μ̂ improves. The shrinkage s may fall as CV picks less
     regularization at larger n. Because s·B is a product of the two, both
     help.
- **Claim 3 on ψ̂.** Run tsm_align at 200 replicates on mix67_2 and
   hard_shared_2. That is enough to resolve the KL-vs-squared difference on
   ψ̂ if it is as large as at 50 replicates. If it is still unresolved, the
   honest version of claim 3 is "KL gives a better α̂, and the gain does not
   reach ψ̂ at n = 1000".
- **How far the ForestRiesz win extends.** Other levels of sharing
   (w = 1/6, 1/2), other overlap levels, other n. The decomposition says to
   look where the outcome-only plug-in's bias B is small but nonzero.
