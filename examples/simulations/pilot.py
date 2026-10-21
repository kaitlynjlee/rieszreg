"""Pilot: does each candidate DGP order the methods the way AIMS.md predicts?

The measure is the RMSE of alpha-hat against alpha0 on a fixed test set of
20,000 fresh draws from the DGP, so it estimates the L2(P) error of alpha-hat,
which is what the Riesz loss targets. For a binary treatment, A is integrated
out exactly given X, using alpha-hat(1, x), alpha-hat(0, x) and pi(x), so the
only Monte Carlo noise in the measure comes from X. It is also reported on the
weak-overlap rows of the test set alone: min(pi, 1 - pi) < 0.05 for the ATE,
pi < 0.05 for the TSM, |alpha0| > 3 for the shift.

The one-step estimator uses 5-fold cross-fitting on one sample of n, as an
analyst would: for each fold, every nuisance is tuned and fit on the other
four folds and predicts on the held-out one, and psi-hat = mean[m(mu-hat) +
alpha-hat (Y - mu-hat)] averages the out-of-fold predictions over all n,
with an influence-function SE and a 95% Wald interval. The alpha-hat RMSE
above is the mean over the five fold fits. One XGBoost outcome regression
per replicate and fold (the task "outcome") is shared by every alpha-hat, so
a method's gap to the oracle, which plugs in alpha0, is the cost of
estimating alpha alone. The true psi0 is exact for a binary treatment and a
4M-draw Monte Carlo integral for the shift (dgp.true_psi).

Earlier results (results/pilot, results/pilot_cv5) fit on one sample of n
and evaluated psi-hat on a second, independent sample of n. Their training
sample is this design's sample, so the replicates still pair.

Blocks (see dgp.py for the DGPs):
  ate        ATE at n = 1000, claims 1 and 2. Plug-ins (XGBoost and main-terms
             logistic regression, each unclipped and clipped at 0.01 and 0.05)
             and five Riesz learners.
  tsm        TSM at A = 1, n in {250, 1000, 4000}, claim 3. rieszboost with the
             squared and the KL loss, and the XGBoost plug-in.
  tsm_align  TSM at A = 1, n = 1000, on the weak-overlap alignment ladder: claim 3
             where psi-hat depends on alpha-hat.
  shift      Additive shift by 0.5, n = 1000, claim 1. The classification density
             ratio (unclipped and clipped at 0.01 and 0.05) and three Riesz
             learners.
The ATE block has two forests: AugForestRieszRegressor (augforestriesz), the
package's recommended forest, and ForestRieszRegressor (forestriesz), the
ForestRiesz of Chernozhukov et al. (2022). Results up to 2026-10-21 also had
a plug-in tuned on the Riesz loss (plugin_xgb_rl); its result files are
still on disk but no block lists it, so the summary ignores them.

Every Riesz learner also gets clipped variants, <method>_clip01 and
<method>_clip05: |alpha-hat| capped at the largest |alpha-hat| a plug-in
can reach when its probability is clipped to [c, 1 - c], 1 / c for a binary
treatment (see _clip_alpha). The fit and its tuning are unchanged; only the
predictions are clipped. For Riesz tasks run before these variants existed,
the summary derives them for psi-hat from the stored alpha-hat; their
alpha-hat RMSE needs a rerun.

For every alpha-hat the summary also reports, on the estimation sample:
  * the remainder mean[(alpha-hat - alpha0)(mu0 - mu-hat)], which is the bias
    of the one-step estimator given the fits, overall and the part from the 10%
    of units with the largest |alpha0|;
  * the semi-oracle: the one-step estimator with the true mu and that
    alpha-hat. Its bias is zero, so it shows each alpha-hat's variance cost
    alone.

Tuning, the same for every learner: 5-fold cross-validation inside each
cross-fitting fold's training part (nested, so the held-out fold never
informs the tuning). Each setting is fit on four tuning folds and scored on
the fifth, and the setting with the smallest mean held-out loss is refit on
the whole training part.
The loss is the squared Riesz loss for Riesz learners (`score`), log loss
for the classifiers, and squared error for the outcome and exposure models.
XGBoost early-stops on the held-out fold, and the refit uses the mean
stopping iteration. The earlier results (results/pilot) tuned on a single
20% split, which picked noisy winners from large grids. The datasets are the
same, so the two sets of results pair replicate by replicate.

Each (block, dgp, n, rep, method) is one task, covering all five folds, and
writes one JSON file, plus one .npz of its out-of-fold predictions on the
sample, so a rerun fills only the gaps. Results go to $RIESZ_SIM_OUT/pilot_cf5
(default: results/pilot_cf5).

Usage:
  python pilot.py run --blocks ate tsm shift --reps 0:20 --jobs 9
  python pilot.py run --reps 0:50 --shard 3/8 --jobs 32   # one of 8 array tasks
  python pilot.py run --refit --reps 0:200 --jobs 9    # add new variants to finished Riesz tasks
  python pilot.py summarize
  python pilot.py summarize --compare results/pilot_cv5   # also pair with the sample-splitting run
"""

from __future__ import annotations

import _env  # noqa: F401  (threads and import paths; must precede numpy)

import argparse
import itertools
import json
import os
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import dgp as D

HERE = Path(__file__).resolve().parent
# Set RIESZ_SIM_OUT to put results elsewhere, e.g. cluster scratch storage, and
# RIESZ_SIM_RUN to summarize an earlier run (e.g. pilot_cv5).
OUT = Path(os.environ.get("RIESZ_SIM_OUT", HERE / "results")) / os.environ.get("RIESZ_SIM_RUN", "pilot_cf5")
BASE_SEED = 20261020
SEED = 0
N_TEST = 20_000
VALID_FRAC = 0.2   # the learners' own early-stopping split, inside each fit
K = 5              # tuning folds

BLOCKS = {
    "ate": dict(
        estimand="ATE", ns=[1000],
        dgps=["overlap_g05", "overlap_g1", "overlap_g2", "overlap_g3", "tail_feature",
              "bulk_feature", "logistic", "smooth", "step", "sparse50", "additive", "interaction",
              "hard_unshared_1", "mix33_1", "mix67_1", "hard_shared_1",
              "hard_unshared_2", "mix33_2", "mix67_2", "hard_shared_2"],
        methods=["plugin_xgb", "plugin_logit", "rieszboost", "riesznet", "krrr", "riesztree",
                 "augforestriesz", "forestriesz"]),
    "tsm": dict(
        estimand="TSM", ns=[250, 1000, 4000],
        dgps=["additive", "additive_g2", "interaction"],
        methods=["plugin_xgb", "rieszboost", "rieszboost_kl"]),
    "tsm_align": dict(
        estimand="TSM", ns=[1000],
        dgps=["hard_unshared_2", "mix33_2", "mix67_2", "hard_shared_2"],
        methods=["plugin_xgb", "rieszboost", "rieszboost_kl"]),
    "shift": dict(
        estimand="SHIFT", ns=[1000],
        dgps=list(D.SHIFT),
        methods=["plugin_classif", "rieszboost", "riesznet", "krrr"]),
}


def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


# Grids. The first two replicates put riesznet's learning rate, riesztree's
# leaf count and XGBoost's learning rate on an edge, and rieszboost split
# between the edges of depth and the L2 penalty, so those grids were widened
# before the pilot. Each Riesz learner's fixed settings follow ../lee_schuler_10cov: the
# L2 leaf penalty and subsample of rieszboost_l2, which damp the drift of
# alpha-hat where one arm is missing. Learning rates are fixed where early
# stopping already trades them against the number of trees or epochs.
#
# Second move, after the 50-replicate pilot (shares of fits at the low / high
# edge, pooled over blocks):
#   * XGBoost, every use (outcome, propensity, classifier): learning rate 0.3,
#     the top, in 35-68%; outcome depth 5, the top, in 27-64%. Learning rate
#     {0.03, 0.1, 0.3} -> {0.03, 0.1, 0.3, 1.0}; depth {1, 2, 3, 5} -> {1, 2, 3, 5, 8}.
#   * rieszboost: L2 penalty 1 in 23-36% and 100 in 44-51%, both edges;
#     depth 7, the top, in 13-23%. Penalty -> {0, 1, 10, 100, 1000} (0 is a
#     hard limit); depth -> {1, 2, 3, 5, 8}.
#   * riesznet: learning rate 1e-3 in 27-32% and 1e-2 in 34-45%, both edges.
#     -> {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}.
#   * krrr: length scale 2x median, the top, in 48%. -> {0.25, ..., 8}.
#   * riesztree: 4 leaves, the bottom, in 100%. That was a tie, not an edge:
#     max_leaf_nodes did nothing under depthwise growth (see _riesztree).
#     Now leafwise, {3, ..., 64} leaves x minimum leaf size {5, 20, 50}.
#   * ForestRiesz: leaf size 5, the bottom, in 37%. -> {1, 2, 5, ..., 100}
#     (1 is a hard limit).
# Third move, after the 5-fold CV run (results/pilot_cv5; share of fits at each
# edge, by block):
#   * riesztree: minimum leaf 50, the top, in 66% (ATE). -> {5, 20, 50, 100, 200};
#     64 leaves chosen in 1%, dropped.
#   * rieszboost: L2 penalty 1000, the top, in 28-43% (rieszboost_kl 27-40%);
#     1 chosen in 6-11%. -> {0, 10, 100, 1000, 10000}, same size (0 is a hard limit).
#   * krrr: length scale 0.25x median, the bottom, in 18% (ATE); 8x in 0-1% (ATE)
#     but 2x in 52% (shift). -> {0.125, ..., 8}: one step down, 8 kept.
#   * riesznet: learning rate 3e-4, the bottom, in 17% (ATE) and 29% (shift).
#     -> {1e-4, ..., 3e-2}.
#   * AugForestRieszRegressor is new to the pilot. Its default minimum leaf
#     is 50 ("leaves of a few rows give very noisy alpha-hat"); the grid
#     spans 5 to 200 around it, the range ForestRiesz and riesztree used.
#     100 trees, bootstrap and every feature at each split (max_features =
#     1.0), its defaults, are fixed. ForestRiesz keeps its grid: it picked
#     1 (a hard limit) in 15% and 100 in 5%.
GRIDS = {
    "xgb": _grid(max_depth=[1, 2, 3, 5, 8], learning_rate=[0.03, 0.1, 0.3, 1.0]),
    "rieszboost": _grid(max_depth=[1, 2, 3, 5, 8], reg_lambda=[0.0, 10.0, 100.0, 1000.0, 10000.0]),
    "riesznet": _grid(learning_rate=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]),
    "krrr": _grid(length_scale_mult=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
    "riesztree": _grid(max_leaf_nodes=[3, 4, 6, 8, 12, 16, 24, 32], min_samples_leaf=[5, 20, 50, 100, 200]),
    "augforestriesz": _grid(min_samples_leaf=[5, 10, 20, 50, 100, 200]),
    "forestriesz": _grid(min_samples_leaf=[1, 2, 5, 10, 20, 50, 100]),
}
GRIDS["rieszboost_kl"] = GRIDS["rieszboost"]


# ------------------------------------------------------------- data ---

def sample_data(dgp_name, n, rep):
    """The sample, its cross-fitting folds, and the tuning folds used inside
    each fold's training part. The sample and the cross-fitting folds are
    the training sample and tuning folds of results/pilot_cv5 (the 20% split
    of results/pilot is still drawn, to keep the random stream in place)."""
    rng = np.random.default_rng([BASE_SEED, zlib.crc32(dgp_name.encode()), n, rep])
    d = D.DGPS[dgp_name].draw(n, rng)
    rng.random(n)
    cf = rng.permutation(np.arange(n) % K)
    inner = rng.permutation(np.arange(n) % K)
    return d, cf, inner


def _tune(grid, fit_score):
    """K-fold CV over a grid. fit_score(setting, fit_rows, held_rows) returns
    (held-out score, higher is better; anything to keep, e.g. an early-stopping
    iteration). Returns the best setting, every setting's mean score, and the
    best setting's per-fold extras."""
    scores, extras = [], []
    for s in grid:
        res = [fit_score(s, k) for k in range(K)]
        scores.append(float(np.mean([r[0] for r in res])))
        extras.append([r[1] for r in res])
    i = int(np.argmax(scores))
    return grid[i], scores, extras[i]


def test_data(dgp_name, estimand):
    """Fixed per DGP, so every replicate and method is scored on the same rows."""
    g = D.DGPS[dgp_name]
    d = g.draw(N_TEST, np.random.default_rng([BASE_SEED, zlib.crc32(dgp_name.encode()), 0, 10**6]))
    X, a = d[g.covariates].to_numpy(), d["A"].to_numpy()
    alpha0 = g.true_alpha(estimand, a, X)
    if estimand == "SHIFT":
        return d, alpha0, np.abs(alpha0) > 3, None
    pi = g.propensity(X)
    weak = (np.minimum(pi, 1 - pi) if estimand == "ATE" else pi) < 0.05
    return d, alpha0, weak, pi


def _estimand(estimand, covs):
    from rieszreg import ATE, TSM, AdditiveShift
    if estimand == "ATE":
        return ATE(treatment="A", covariates=tuple(covs))
    if estimand == "TSM":
        return TSM(level=1.0, treatment="A", covariates=tuple(covs))
    return AdditiveShift(delta=D.DELTA, treatment="A", covariates=tuple(covs))


# --------------------------------------------------------- learners ---
# A Riesz learner is make(estimand, covs, n_train, train, **setting) -> unfitted estimator.

def _rieszboost(loss):
    def make(est, covs, train, max_depth, reg_lambda):
        from rieszboost import RieszBooster
        from rieszreg import KLLoss, SquaredLoss
        return RieszBooster(estimand=est, loss=KLLoss() if loss == "kl" else SquaredLoss(),
                            n_estimators=3000, learning_rate=0.05, max_depth=max_depth,
                            reg_lambda=reg_lambda, subsample=0.8, early_stopping_rounds=50,
                            validation_fraction=VALID_FRAC, n_jobs=1, random_state=SEED)
    return make


def _riesznet(est, covs, train, learning_rate):
    from riesznet import RieszNet
    return RieszNet(estimand=est, hidden_sizes=(64, 64), activation="elu", learning_rate=learning_rate,
                    weight_decay=1e-3, epochs=300, batch_size=64, early_stopping_rounds=10,
                    validation_fraction=VALID_FRAC, snapshot_epochs=[], n_jobs=1, random_state=SEED)


def _median_distance(train, covs):
    Z = train[["A"] + covs].to_numpy()[:500]
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    return float(np.median(d[np.triu_indices(len(Z), 1)]))


def _krrr(est, covs, train, length_scale_mult):
    from krrr import Gaussian, KernelRieszRegressor
    return KernelRieszRegressor(estimand=est, kernel=Gaussian(length_scale=length_scale_mult * _median_distance(train, covs)),
                                validation_fraction=VALID_FRAC, random_state=SEED)


def _riesztree(est, covs, train, max_leaf_nodes, min_samples_leaf):
    # Leafwise growth: under the default depthwise policy riesztree ignores
    # max_leaf_nodes (as documented), so with max_depth=None every setting of
    # the v1 and v2 grids grew the same full tree, and the "edge" CV kept
    # choosing was the first of a tie.
    from riesztree import RieszTreeRegressor
    return RieszTreeRegressor(estimand=est, growth_policy="leafwise", max_depth=None,
                              max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                              random_state=SEED)


def _augforestriesz(est, covs, train, min_samples_leaf):
    from forestriesz import AugForestRieszRegressor
    return AugForestRieszRegressor(estimand=est, min_samples_leaf=min_samples_leaf, n_jobs=1, random_state=SEED)


def _forestriesz(est, covs, train, min_samples_leaf):
    # The published ForestRiesz (EconML GRF on the original rows, treatment-
    # indicator sieve, splits on covariates), with its published defaults.
    from forestriesz import ForestRieszRegressor
    return ForestRieszRegressor(estimand=est, min_samples_leaf=min_samples_leaf, n_jobs=1, random_state=SEED)


RIESZ = {"rieszboost": _rieszboost("squared"), "rieszboost_kl": _rieszboost("kl"), "riesznet": _riesznet,
         "krrr": _krrr, "riesztree": _riesztree, "augforestriesz": _augforestriesz,
         "forestriesz": _forestriesz}


# A fit function takes (method, estimand, train, folds, covs) and returns
# (predict, selected setting, CV score of every setting). predict(df)
# maps each variant to alpha-hat at (A, X) for a shift, and to
# (alpha-hat(1, x), alpha-hat(0, x)) for a binary treatment.

def fit_riesz(method, estimand, train, folds, covs, best=None, scores=None):
    """With best given (a setting this task's CV already chose), skip the
    tuning and only refit it on the whole sample."""
    cols = ["A"] + covs
    est = _estimand(estimand, covs)

    def fit_score(s, k):
        part = train[folds != k]
        m = RIESZ[method](est, covs, part, **s).fit(part[cols])
        return float(m.score(train[folds == k][cols])), None
    if best is None:
        best, scores, _ = _tune(GRIDS[method], fit_score)
    m = RIESZ[method](est, covs, train, **best).fit(train[cols])

    def predict(df):
        Z = df[cols]
        if estimand == "SHIFT":
            alpha = m.predict(Z)
        else:
            alpha = (m.predict(Z.assign(A=1.0)), m.predict(Z.assign(A=0.0)))
        out = {method: alpha}
        for c in CLIPS:
            out[f"{method}_clip{str(c)[2:]}"] = _clip_alpha(estimand, alpha, c)
        return out
    return predict, best, scores


CLIPS = (0.01, 0.05)


def _alpha_cap(estimand, c):
    """The largest |alpha-hat| of a plug-in whose probability is clipped to
    [c, 1 - c]: 1 / c for the ATE and TSM, (1 - c) / c - 1 for the shift
    classifier (100 and 98 at c = 0.01)."""
    return (1 - c) / c - 1.0 if estimand == "SHIFT" else 1.0 / c


def _clip_alpha(estimand, alpha, c):
    """A Riesz learner's alpha-hat with |alpha-hat| capped at the plug-in's cap
    at c. The cap catches the drift where an arm is missing (alpha-hat(1, x)
    to +inf, alpha-hat(0, x) to -inf). Unlike a full clip to the plug-in's
    range, it does not raise alpha-hat(1, x) to 1 / (1 - c) or set the TSM's
    alpha-hat(0, x) to 0. On results/pilot_cv5 that inner bound was active on
    up to 55% of rows, and it made psi-hat resolved worse in 9 of 138 cells
    (the shared-feature rungs) and better in none. The cap alone was never
    resolved worse.
    The cap is the same for both arms, so it applies to alpha-hat at the
    observed (A, X) as it is."""
    cap = _alpha_cap(estimand, c)
    if estimand == "SHIFT":
        return np.clip(alpha, -cap, cap)
    return tuple(np.clip(v, -cap, cap) for v in alpha)


def _xgb_tuned(make, X, y, folds, score):
    """Tune an XGBoost model by K-fold CV, early-stopping on the held-out fold;
    refit on everything with the mean stopping iteration of the best setting.
    score(y_true, model, X_held) is higher-is-better."""
    def fit_score(s, k):
        f, h = folds != k, folds == k
        m = make(n_estimators=2000, early_stopping_rounds=20, **s)
        m.fit(X[f], y[f], eval_set=[(X[h], y[h])], verbose=False)
        return score(y[h], m, X[h]), m.best_iteration + 1
    best, scores, iters = _tune(GRIDS["xgb"], fit_score)
    m = make(n_estimators=int(round(np.mean(iters))), **best).fit(X, y)
    return m, best, scores


def _neg_log_loss(y, m, X):
    from sklearn.metrics import log_loss
    return -log_loss(y, m.predict_proba(X)[:, 1], labels=[0, 1])


def _xgb_propensity(train, folds, covs):
    import xgboost as xgb

    def make(**kw):
        return xgb.XGBClassifier(n_jobs=1, random_state=SEED, **kw)
    return _xgb_tuned(make, train[covs], train["A"], folds, _neg_log_loss)


def _plugin_alpha(estimand, pi):
    """(alpha(1, x), alpha(0, x)) implied by a propensity score."""
    if estimand == "ATE":
        return 1 / pi, -1 / (1 - pi)
    return 1 / pi, np.zeros_like(pi)


def fit_plugin_binary(method, estimand, train, folds, covs):
    if method == "plugin_xgb":
        m, best, scores = _xgb_propensity(train, folds, covs)
    else:
        from sklearn.linear_model import LogisticRegression
        m, best, scores = LogisticRegression(C=1e6, max_iter=2000).fit(train[covs], train["A"]), {}, []

    def predict(df):
        pi = m.predict_proba(df[covs])[:, 1]
        out = {method: _plugin_alpha(estimand, np.clip(pi, 1e-12, 1 - 1e-12))}
        for c in CLIPS:
            out[f"{method}_clip{str(c)[2:]}"] = _plugin_alpha(estimand, np.clip(pi, c, 1 - c))
        return out
    return predict, best, scores


def fit_plugin_classif(method, estimand, train, folds, covs):
    """Density ratio by classification, as in ../lee_schuler_10cov_shift: an
    XGBoost classifier of observed rows (A, X), labelled 1, against shifted rows
    (A + delta, X), labelled 0. The classes are balanced, so the odds
    (1 - p) / p estimate p(a - delta | x) / p(a | x), and alpha = odds - 1.
    The folds are by unit, so both rows of a unit fall in the same fold.
    Tuned by held-out log loss, with p-hat also clipped to [c, 1 - c] as for
    the propensity plug-in."""
    import xgboost as xgb
    cols = ["A"] + covs
    Z = train[cols]
    X = pd.concat([Z, Z.assign(A=Z["A"] + D.DELTA)], ignore_index=True)
    y = np.r_[np.ones(len(Z)), np.zeros(len(Z))]

    def make(**kw):
        return xgb.XGBClassifier(n_jobs=1, random_state=SEED, **kw)
    m, best, scores = _xgb_tuned(make, X, y, np.r_[folds, folds], _neg_log_loss)

    def predict(df):
        p = m.predict_proba(df[cols])[:, 1]
        out = {method: (1 - p) / np.clip(p, 1e-12, None) - 1.0}
        for c in CLIPS:
            q = np.clip(p, c, 1 - c)
            out[f"{method}_clip{str(c)[2:]}"] = (1 - q) / q - 1.0
        return out
    return predict, best, scores


def fit_outcome(estimand, train, folds, covs):
    """The outcome regression mu-hat(a, x), shared by every alpha-hat in a
    replicate: XGBoost on (A, X), tuned by held-out squared error. Returns
    predict(df) -> (mu-hat(A, X), m(mu-hat)(X)), the second being
    mu-hat(1, X) - mu-hat(0, X) for the ATE, mu-hat(1, X) for the TSM, and
    mu-hat(A + delta, X) - mu-hat(A, X) for the shift."""
    import xgboost as xgb
    cols = ["A"] + covs

    def make(**kw):
        return xgb.XGBRegressor(n_jobs=1, random_state=SEED, **kw)

    def neg_mse(y, m, X):
        return -float(np.mean((np.asarray(y) - m.predict(X)) ** 2))
    m, best, scores = _xgb_tuned(make, train[cols], train["Y"], folds, neg_mse)

    def predict(df):
        Z = df[cols]
        mu = m.predict(Z)
        if estimand == "ATE":
            return mu, m.predict(Z.assign(A=1.0)) - m.predict(Z.assign(A=0.0))
        if estimand == "TSM":
            return mu, m.predict(Z.assign(A=1.0))
        return mu, m.predict(Z.assign(A=Z["A"] + D.DELTA)) - mu
    return predict, best, scores


# ------------------------------------------------------------- tasks ---

def task_path(block, dgp_name, n, rep, method):
    return OUT / block / dgp_name / f"n{n}" / f"rep{rep:03d}_{method}.json"


def _done(task):
    """A task is done when its result file exists. An outcome task from before
    mu0 was stored (needed for the remainder and the semi-oracle) is redone."""
    path = task_path(*task)
    if not path.exists():
        return False
    if task[4] == "outcome":
        npz = path.with_suffix(".npz")
        return npz.exists() and "mu0" in np.load(npz).files
    return True


def _mkdir(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _outcome_task(estimand, dgp_name, d, cf, inner, covs):
    """Cross-fit mu-hat; store on the sample what the one-step estimator and
    the oracle need: out-of-fold mu-hat and m(mu-hat), and the truth. Returns
    (record fields, arrays)."""
    g = D.DGPS[dgp_name]
    n = len(d)
    mu, m_mu, bests, scores = np.empty(n), np.empty(n), [], []
    for k in range(K):
        tr = cf != k
        predict, best, s = fit_outcome(estimand, d[tr], inner[tr], covs)
        mu[~tr], m_mu[~tr] = predict(d[~tr])
        bests.append(best)
        scores.append(s)
    X, a = d[covs].to_numpy(), d["A"].to_numpy()
    mu0, m0 = D._mu(a, X), D.m_mu0(estimand, a, X)
    arrays = {"y": d["Y"].to_numpy(), "mu": mu, "m_mu": m_mu, "alpha0": g.true_alpha(estimand, a, X),
              "mu0": mu0, "m_mu0": m0}
    fields = {"rows": [], "rmse_mu": float(np.sqrt(np.mean((mu - mu0) ** 2))),
              "rmse_m_mu": float(np.sqrt(np.mean((m_mu - m0) ** 2))), "best": bests, "grid": GRIDS["xgb"],
              "scores": scores}
    return fields, arrays


def _alpha_rows(alphas, dgp_name, estimand, test, covs, alpha0, weak, pi):
    """Test-set error of each alpha-hat variant: {variant: {rmse, rmse_weak,
    max_abs_alpha_hat}}. For a binary treatment A is integrated out given X:
    E[(alpha-hat - alpha0)^2 | X]."""
    out = {}
    for v, a in alphas.items():
        if pi is None:
            sq, big = (a - alpha0) ** 2, np.abs(a).max()
        else:
            g, X = D.DGPS[dgp_name], test[covs].to_numpy()
            e1 = a[0] - g.true_alpha(estimand, np.ones(len(X)), X)
            e0 = a[1] - g.true_alpha(estimand, np.zeros(len(X)), X)
            sq, big = pi * e1**2 + (1 - pi) * e0**2, max(np.abs(a[0]).max(), np.abs(a[1]).max())
        out[v] = {"rmse": float(np.sqrt(sq.mean())),
                  "rmse_weak": float(np.sqrt(sq[weak].mean())) if weak.any() else None,
                  "max_abs_alpha_hat": float(big)}
    return out


def run_task(block, dgp_name, n, rep, method, refit=False):
    """refit: reuse the setting a finished Riesz task chose by CV, refit it and
    rewrite its results with every current variant (e.g. the clipped ones).
    The record keeps the tuning's grid, scores and seconds."""
    path = task_path(block, dgp_name, n, rep, method)
    estimand = BLOCKS[block]["estimand"]
    t0 = time.time()
    old = json.loads(path.read_text()) if refit else None
    try:
        d, cf, inner = sample_data(dgp_name, n, rep)
        covs = D.DGPS[dgp_name].covariates
        if method == "outcome":
            fields, arrays = _outcome_task(estimand, dgp_name, d, cf, inner, covs)
            np.savez(_mkdir(path).with_suffix(".npz"), **arrays)
            rec = {"block": block, "dgp": dgp_name, "n": n, "rep": rep, "method": method, **fields,
                   "design": "crossfit5", "seconds": time.time() - t0}
            path.write_text(json.dumps(rec))
            return path, False, rec["seconds"]
        test, alpha0, weak, pi = test_data(dgp_name, estimand)
        fit = (fit_riesz if method in RIESZ else
               fit_plugin_classif if method == "plugin_classif" else fit_plugin_binary)
        a_obs = d["A"].to_numpy()
        on_est, fold_rows, best, scores = {}, [], [], []
        for k in range(K):
            tr = cf != k
            predict, b, s = (fit_riesz(method, estimand, d[tr], inner[tr], covs, old["best"][k], old["scores"][k])
                             if old else fit(method, estimand, d[tr], inner[tr], covs))
            best.append(b)
            scores.append(s)
            # Out-of-fold alpha-hat at the observed (A, X), for the one-step estimator.
            for v, a in predict(d[~tr]).items():
                on_est.setdefault(v, np.empty(n))[~tr] = (a if estimand == "SHIFT"
                                                         else a_obs[~tr] * a[0] + (1 - a_obs[~tr]) * a[1])
            fold_rows.append(_alpha_rows(predict(test), dgp_name, estimand, test, covs, alpha0, weak, pi))
        np.savez(_mkdir(path).with_suffix(".npz"), **on_est)
        # alpha-hat's test-set error: the mean over the five fold fits.
        rows = [{"variant": v, "rmse": float(np.mean([f[v]["rmse"] for f in fold_rows])),
                 "rmse_weak": (float(np.mean([f[v]["rmse_weak"] for f in fold_rows])) if weak.any() else None),
                 "max_abs_alpha_hat": float(max(f[v]["max_abs_alpha_hat"] for f in fold_rows))}
                for v in fold_rows[0]]
        rec = {"block": block, "dgp": dgp_name, "n": n, "rep": rep, "method": method, "rows": rows,
               "best": best, "grid": GRIDS["xgb"] if method.startswith("plugin_") else GRIDS[method],
               "scores": scores, "seconds": time.time() - t0}
        if old:
            rec.update(grid=old["grid"], seconds=old["seconds"], refit_seconds=time.time() - t0,
                       refit_of={r["variant"]: r["rmse"] for r in old["rows"]})
    except Exception:
        rec = {"block": block, "dgp": dgp_name, "n": n, "rep": rep, "method": method,
               "error": traceback.format_exc(), "seconds": time.time() - t0}
        if old:  # keep the finished task's record
            print(f"refit failed, kept the old record: {path}\n{rec['error']}", flush=True)
            return path, True, rec["seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return path, "error" in rec, rec["seconds"]


# Rough relative cost, so the slowest tasks start first.
COST = {"riesznet": 10, "rieszboost": 6, "rieszboost_kl": 6, "augforestriesz": 3, "forestriesz": 3, "krrr": 3, "outcome": 2}


def cmd_run(args):
    lo, hi = map(int, args.reps.split(":"))
    tasks = [(b, g, n, r, m) for b in args.blocks for g in BLOCKS[b]["dgps"] for n in BLOCKS[b]["ns"]
             for r in range(lo, hi) for m in ["outcome"] + BLOCKS[b]["methods"]
             if (not args.methods or m in args.methods) and (not args.dgps or g in args.dgps)]
    if args.shard:
        # A fixed slice for one array task: the same task lands in the same
        # shard on every submission, whatever the other arguments.
        k, n_shards = map(int, args.shard.split("/"))
        tasks = [t for t in tasks if zlib.crc32(repr(t).encode()) % n_shards == k]
    if args.refit:
        # Finished Riesz tasks whose results predate a variant (e.g. the clipped ones).
        def stale(t):
            if t[4] not in RIESZ or not _done(t):
                return False
            rec = json.loads(task_path(*t).read_text())
            return "error" not in rec and f"{t[4]}_clip01" not in {r["variant"] for r in rec["rows"]}
        tasks = [t for t in tasks if stale(t)]
    else:
        tasks = [t for t in tasks if args.overwrite or not _done(t)]
    tasks.sort(key=lambda t: -COST.get(t[4], 1) * t[2])
    print(f"{len(tasks)} {'refit ' if args.refit else ''}tasks on {args.jobs} processes", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for fut in as_completed([pool.submit(run_task, *t, refit=args.refit) for t in tasks]):
            path, err, sec = fut.result()
            done += 1
            if err or done % 20 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)} {time.time() - t0:6.0f}s] {'ERROR ' if err else ''}"
                      f"{path.relative_to(OUT)} {sec:.0f}s", flush=True)


# --------------------------------------------------------- summaries ---

def load():
    # Only methods still in a block, so a dropped method's old files are ignored.
    recs = [r for r in (json.loads(p.read_text()) for p in OUT.rglob("*.json"))
            if r["method"] in ["outcome"] + BLOCKS[r["block"]]["methods"]]
    errs = [r for r in recs if "error" in r]
    rows = [{**{k: r[k] for k in ("block", "dgp", "n", "rep", "method", "seconds")}, **row}
            for r in recs if "error" not in r for row in r["rows"]]
    return pd.DataFrame(rows), recs, errs


def _mean_mcse(x):
    x = x.dropna()
    return f"{x.mean():.3f} ({x.std(ddof=1) / np.sqrt(len(x)):.3f})" if len(x) > 1 else f"{x.mean():.3f}"


def paired(df, a, b, col="rmse"):
    """Mean of rmse[a] - rmse[b] over replicates, its Monte Carlo SE, and z."""
    w = df.pivot_table(index="rep", columns="variant", values=col)
    if a not in w or b not in w:
        return None
    d = (w[a] - w[b]).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, d.mean() / se if se > 0 else np.nan, len(d)


# The contrasts AIMS.md makes predictions about, as (block, a, b): rmse[a] -
# rmse[b], negative when a is better.
RIESZ_ATE = ["rieszboost", "riesznet", "krrr", "riesztree", "augforestriesz", "forestriesz"]


def headline(m):
    """The variant a learner is reported by. riesznet's |alpha-hat| is capped
    at 1/0.01, the bound of the reference plug-in (clipped at 0.01): uncapped,
    the few replicates where it drifts dominate every mean (AIMS.md section 6).
    The other Riesz learners barely change under that cap, so they are
    reported as fitted."""
    return f"{m}_clip01" if m == "riesznet" else m


CONTRASTS = list(dict.fromkeys(
    [("ate", headline(m), "plugin_xgb_clip01") for m in RIESZ_ATE]
    + [("ate", "plugin_logit", "plugin_xgb_clip01")]
    # Clipping alpha-hat, against the same learner unclipped and against the
    # plug-in clipped to the same bounds.
    + [(b, f"{m}_clip{c}", ref)
       for b, ms in [("ate", RIESZ_ATE), ("tsm", ["rieszboost", "rieszboost_kl"]), ("tsm_align", ["rieszboost", "rieszboost_kl"])]
       for m in ms for c in ("01", "05") for ref in (m, f"plugin_xgb_clip{c}")]
    + [("shift", f"{m}_clip{c}", ref) for m in ("rieszboost", "riesznet", "krrr") for c in ("01", "05")
       for ref in (m, f"plugin_classif_clip{c}")]
    + [(b, a, ref) for b in ("tsm", "tsm_align")
       for a, ref in [("rieszboost_kl", "rieszboost"), ("rieszboost_kl", "plugin_xgb_clip01"),
                      ("rieszboost", "plugin_xgb_clip01")]]
    + [("shift", headline(m), ref) for ref in ("plugin_classif", "plugin_classif_clip05")
       for m in ("rieszboost", "riesznet", "krrr")]
))


def contrast_table(df, col):
    out = []
    for block, a, b in CONTRASTS:
        sub = df[df.block == block]
        for (g, n), cell in sub.groupby(["dgp", "n"]):
            res = paired(cell, a, b, col)
            if res is None:
                continue
            mean, se, z, reps = res
            sd = se * np.sqrt(reps)
            out.append({"block": block, "dgp": g, "n": n, "contrast": f"{a} - {b}", "diff": mean, "MCSE": se,
                        "z": z, "reps": reps, "reps for |z|=3": int(np.ceil((3 * sd / mean) ** 2)) if mean and np.isfinite(sd) else None})
    out = pd.DataFrame(out)
    return out.astype({"reps for |z|=3": "Int64"}) if len(out) else out


def load_psi(recs):
    """One row per (replicate, alpha-hat variant): the one-step estimate
    psi-hat = mean[m(mu-hat) + alpha-hat (Y - mu-hat)] on the estimation sample,
    its influence-function SE, and its error. The oracle plugs in alpha0 with
    the same mu-hat.

    With mu0 stored (outcome tasks from this version on), also:
      rem       mean[(alpha-hat - alpha0)(mu0 - mu-hat)], the bias of psi-hat
                given the fits (by the Riesz representation of m), estimated
                on the estimation sample;
      rem_tail  the part of rem from the 10% of units with the largest |alpha0|;
      *_semi    the semi-oracle: the same with mu-hat replaced by mu0, which
                has zero bias, so its spread is alpha-hat's variance cost alone;
      s, B, shrink, resid
                the split rem = s * B + mean[r e], with e = mu0 - mu-hat. s is
                the least-squares fit through the origin of alpha-hat - alpha0
                on -alpha0 (s > 0: alpha-hat shrunk toward zero), r the
                residual alpha-hat - alpha0 + s alpha0, and B = -mean[alpha0 e],
                which estimates E[m(mu-hat)] - psi0, the bias of the
                outcome-only plug-in. B is shared by every alpha-hat in a
                replicate; shrink = s * B and resid = mean[r e]."""
    keys = ("block", "dgp", "n", "rep")
    outcomes = {tuple(r[k] for k in keys): r for r in recs if r["method"] == "outcome" and "error" not in r}
    arrays = {}

    def one_step(key, variant, alpha):
        o = arrays[key]
        psi0 = D.true_psi(key[1], BLOCKS[key[0]]["estimand"])
        phi = o["m_mu"] + alpha * (o["y"] - o["mu"])
        psi, se = phi.mean(), phi.std(ddof=1) / np.sqrt(len(phi))
        err = psi - psi0
        row = {**dict(zip(keys, key)), "variant": variant, "psi": psi, "se": se, "err": err,
               "sqerr": err**2, "cover": float(abs(err) <= 1.96 * se)}
        if "mu0" in o:
            a0, e = o["alpha0"], o["mu0"] - o["mu"]
            prod = (alpha - a0) * e
            tail = np.abs(a0) >= np.quantile(np.abs(a0), 0.9)
            phi_s = o["m_mu0"] + alpha * (o["y"] - o["mu0"])
            se_s, err_s = phi_s.std(ddof=1) / np.sqrt(len(phi_s)), phi_s.mean() - psi0
            s, B = -np.sum((alpha - a0) * a0) / np.sum(a0**2), -np.mean(a0 * e)
            row.update(rem=prod.mean(), rem_tail=prod[tail].sum() / len(prod), err_semi=err_s,
                       sqerr_semi=err_s**2, cover_semi=float(abs(err_s) <= 1.96 * se_s),
                       s=s, B=B, shrink=s * B, resid=np.mean((alpha - a0 + s * a0) * e))
        return row

    rows = []
    for r in recs:
        key = tuple(r[k] for k in keys)
        if "error" in r or key not in outcomes:
            continue
        if key not in arrays:
            arrays[key] = dict(np.load(task_path(*key, "outcome").with_suffix(".npz")))
        if r["method"] == "outcome":
            rows.append(one_step(key, "oracle", arrays[key]["alpha0"]))
            continue
        a = dict(np.load(task_path(*key, r["method"]).with_suffix(".npz")))
        m = r["method"]
        if m in RIESZ and f"{m}_clip01" not in a:
            # A Riesz task run before the clipped variants existed: clip its
            # stored alpha-hat at the observed (A, X). Its alpha-hat RMSE on
            # the test set needs a rerun.
            cap = {c: _alpha_cap(BLOCKS[key[0]]["estimand"], c) for c in CLIPS}
            a.update({f"{m}_clip{str(c)[2:]}": np.clip(a[m], -cap[c], cap[c]) for c in CLIPS})
        rows.extend(one_step(key, v, alpha) for v, alpha in a.items())
    return pd.DataFrame(rows)


def psi_table(sub):
    """Bias, empirical SE, RMSE, mean SE over empirical SE, and 95% coverage
    of psi-hat, each a column block with methods side by side. Also two
    measures a few drifting replicates cannot dominate, as they do the RMSE
    (and a paired z, which then sits near -1 or 1): the median |error|, and
    the number of blow-ups, replicates whose error is further than 5 times
    the oracle's empirical SE from that method's median error in the same
    cell (so a steady bias is not counted)."""
    g = sub.groupby(["dgp", "n", "variant"])
    oracle_sd = sub[sub.variant == "oracle"].groupby(["dgp", "n"]).psi.std(ddof=1).rename("oracle_sd")
    med = g.err.median().rename("med")
    blow = (sub.join(oracle_sd, on=["dgp", "n"]).join(med, on=["dgp", "n", "variant"])
            .assign(b=lambda d: (d.err - d.med).abs() > 5 * d.oracle_sd))
    t = pd.DataFrame({
        "bias": g.err.mean(),
        "empSE": g.psi.std(ddof=1),
        "RMSE": np.sqrt(g.sqerr.mean()),
        "SE/empSE": g.se.mean() / g.psi.std(ddof=1),
        "cover": g.cover.mean(),
        "median |error|": g.err.apply(lambda e: e.abs().median()),
        "blow-ups (error > 5 oracle empSE from its median)": blow.groupby(["dgp", "n", "variant"]).b.sum(),
    })
    return t


def edge_table(recs):
    """Share of fits whose selected setting sits at the smallest or largest
    value of a tuned hyperparameter, by block and method. A task's record
    holds one selected setting per cross-fitting fold, and each counts as a fit."""
    def bests(r):
        return r["best"] if isinstance(r["best"], list) else [r["best"]]
    out = []
    for (block, method), grp in itertools.groupby(
            sorted((r for r in recs if "error" not in r and any(bests(r))), key=lambda r: (r["block"], r["method"])),
            key=lambda r: (r["block"], r["method"])):
        grp = list(grp)
        for hp in bests(grp[0])[0]:
            vals = sorted({s[hp] for s in grp[0]["grid"]})
            picks = [b[hp] for r in grp for b in bests(r)]
            out.append({"block": block, "method": method, "hyperparameter": hp, "grid": vals,
                        "share at low": np.mean([p == vals[0] for p in picks]),
                        "share at high": np.mean([p == vals[-1] for p in picks]), "fits": len(picks)})
    return pd.DataFrame(out)


def cmd_summarize(args):
    df, recs, errs = load()
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 40)
    print(f"{len(recs)} task files, {len(errs)} errors")
    for e in errs[:5]:
        print(e["block"], e["dgp"], e["n"], e["rep"], e["method"], e["error"].strip().splitlines()[-1])
    for block in BLOCKS:
        sub = df[df.block == block]
        if sub.empty:
            continue
        for col in ("rmse", "rmse_weak"):
            t = sub.groupby(["dgp", "n", "variant"])[col].apply(_mean_mcse).unstack("variant")
            t = t.reindex([g for g in BLOCKS[block]["dgps"] if g in t.index.get_level_values(0)], level=0)
            print(f"\n== {block}: mean {col} of alpha-hat (MCSE), replicates: "
                  f"{sub.groupby(['dgp', 'n']).rep.nunique().min()}-{sub.groupby(['dgp', 'n']).rep.nunique().max()}")
            print(t.to_string())
        # How often the cap on a Riesz learner's |alpha-hat| binds somewhere on the test set.
        raw = sub[sub.variant.isin(list(RIESZ))]
        if not raw.empty:
            est = BLOCKS[block]["estimand"]
            t = pd.DataFrame({f"cap {_alpha_cap(est, c):g} (clip{str(c)[2:]})":
                              raw.assign(b=raw.max_abs_alpha_hat > _alpha_cap(est, c)).groupby(["dgp", "n", "variant"]).b.mean()
                              for c in CLIPS})
            t = t.unstack("variant").reindex([g for g in BLOCKS[block]["dgps"] if g in t.index.get_level_values(0)], level=0)
            print(f"\n== {block}: share of Riesz fits whose |alpha-hat| exceeds the cap somewhere on the test set")
            print(t.to_string(float_format=lambda v: f"{v:.2f}"))
        print(f"\n== {block}: seconds per task")
        print(sub.drop_duplicates(["dgp", "n", "rep", "method"]).groupby(["method", "n"]).seconds.mean().round(1).to_string())
    for col in ("rmse", "rmse_weak"):
        print(f"\n== paired contrasts in {col} (negative: the first method is better)")
        print(contrast_table(df, col).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n== tuning grid edges")
    print(edge_table(recs).to_string(index=False))
    df.to_csv(OUT / "rows.csv", index=False)

    psi = load_psi(recs)
    if psi.empty:
        return
    out = pd.DataFrame([{k: r[k] for k in ("block", "dgp", "n", "rmse_mu", "rmse_m_mu")}
                        for r in recs if r["method"] == "outcome" and "error" not in r])
    print("\n== outcome regression: mean RMSE of mu-hat(A, X) and of m(mu-hat) against the truth")
    print(out.groupby(["block", "dgp", "n"]).mean().round(3).to_string())
    for block in BLOCKS:
        sub = psi[psi.block == block]
        if sub.empty:
            continue
        t = psi_table(sub)
        order = [g for g in BLOCKS[block]["dgps"] if g in t.index.get_level_values(0)]
        psi0 = ", ".join(f"{g} {D.true_psi(g, BLOCKS[block]['estimand']):.4f}" for g in BLOCKS[block]["dgps"])
        print(f"\n== {block}: one-step estimator, replicates: {sub.groupby(['dgp', 'n']).rep.nunique().min()}"
              f"\npsi0: {psi0}")
        for col in t.columns:
            w = t[col].unstack("variant").reindex(order, level=0)
            w = w[["oracle"] + [c for c in w.columns if c != "oracle"]]
            print(f"\n-- {col}")
            print(w.to_string(float_format=lambda v: f"{v:.3f}"))
        if "rem" in sub and sub.rem.notna().any():
            g = sub.dropna(subset=["rem"]).groupby(["dgp", "n", "variant"])
            r = pd.DataFrame({
                "remainder (bias given the fits)": g.rem.mean(),
                "share of remainder from top-10% |alpha0|": g.rem_tail.mean() / g.rem.mean(),
                "semi-oracle RMSE (true mu)": np.sqrt(g.sqerr_semi.mean()),
                "semi-oracle coverage": g.cover_semi.mean(),
                "shrinkage s": g.s.mean(),
                "shrinkage term s*B": g.shrink.mean(),
                "residual term mean[r e]": g.resid.mean(),
            })
            print(f"\n== {block}: remainder and semi-oracle, replicates: "
                  f"{sub.dropna(subset=['rem']).groupby(['dgp', 'n']).rep.nunique().min()}"
                  f"\nremainder = s*B + mean[r e] (see load_psi); B, the outcome-only plug-in's bias, by cell:")
            print(sub.dropna(subset=["rem"]).drop_duplicates(["dgp", "n", "rep"]).groupby(["dgp", "n"]).B.mean()
                  .reindex(order, level=0).to_string(float_format=lambda v: f"{v:.3f}"))
            for col in r.columns:
                w = r[col].unstack("variant").reindex(order, level=0)
                w = w[["oracle"] + [c for c in w.columns if c != "oracle"]]
                print(f"\n-- {col}")
                print(w.to_string(float_format=lambda v: f"{v:.3f}"))
            # B is shared within a replicate, so across methods a larger s should
            # mean a larger remainder when B > 0 and a smaller one when B < 0.
            v = r.index.get_level_values("variant")
            est = r[v.isin([headline(m) for m in RIESZ]) | (v.str.startswith("plugin_") & v.str.endswith("_clip01"))]
            print("\n-- across methods (Riesz learners, plug-ins clipped at 0.01), correlation of s with the remainder")
            print(est.groupby(level=["dgp", "n"]).apply(
                lambda c: np.corrcoef(c["shrinkage s"], c["remainder (bias given the fits)"])[0, 1])
                .reindex(order, level=0).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n== paired contrasts in squared error of psi-hat (negative: the first method is better)")
    print(contrast_table(psi, "sqerr").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    psi.to_csv(OUT / "psi.csv", index=False)
    if args.compare:
        compare(df, psi, Path(args.compare))


def _paired_change(w, a, b):
    """Mean of w[b] - w[a] over replicates and its paired z."""
    d = (w[b] - w[a]).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() / se if se > 0 else np.nan


def compare(df, psi, ref):
    """Pair this run with an earlier one on the same datasets, e.g. results/pilot
    (single-split tuning), replicate by replicate. Reads the earlier run's
    rows.csv and psi.csv, written by its own summarize."""
    keys = ["block", "dgp", "n", "rep", "variant"]
    print(f"\n== comparison with {ref} (ref) on the same datasets")

    def design(out):
        p = next(out.rglob("rep000_outcome.json"), None)
        return json.loads(p.read_text()).get("design", "split") if p else None
    if design(OUT) != design(ref):
        print(f"designs differ ({design(OUT)} here, {design(ref)} in ref): the replicates pair by seed, "
              "and this run's sample is ref's training sample, but psi-hat is computed differently, "
              "so the data check is skipped")
    # The datasets must match: the estimation-sample outcomes, and the alpha-hat
    # of the untuned logistic plug-in, which depends on the training sample only.
    checked = bad = 0
    for p in sorted(OUT.rglob("rep00[0-4]_outcome.npz")) if design(OUT) == design(ref) else []:
        q = ref / p.relative_to(OUT)
        if not q.exists():
            continue
        checked += 1
        bad += not np.allclose(np.load(p)["y"], np.load(q)["y"], rtol=0, atol=1e-12)
        pl, ql = (x.with_name(x.name.replace("outcome", "plugin_logit")) for x in (p, q))
        if pl.exists() and ql.exists():
            a, b = np.load(pl), np.load(ql)
            bad += not all(np.allclose(a[k], b[k], rtol=0, atol=1e-10) for k in b.files)
    print(f"pairing check: {checked} replicate cells, {bad} that differ by more than 1e-10")

    rows = pd.read_csv(ref / "rows.csv")[keys + ["rmse"]].merge(df[keys + ["rmse"]], on=keys, suffixes=("_ref", ""))
    out = []
    for (b, v, g, n), c in rows.groupby(["block", "variant", "dgp", "n"]):
        ch, z = _paired_change(c.set_index("rep"), "rmse_ref", "rmse")
        out.append({"block": b, "variant": v, "dgp": g, "n": n, "ref": c.rmse_ref.mean(), "this run": c.rmse.mean(),
                    "change": ch, "z": z, "reps": len(c)})
    t = pd.DataFrame(out)
    t = t[~t.variant.str.endswith(("_clip05", "_clip01")) | (t.variant == "plugin_xgb_clip01")]
    print("\n-- alpha-hat RMSE, ref -> this run, paired (negative change: this run is better)")
    print(t.set_index(["block", "variant", "dgp", "n"]).to_string(float_format=lambda v: f"{v:.3f}"))
    # An untuned learner (the logistic plug-in) differs only by rounding, which a z would call resolved.
    real = t.change.abs() > 1e-9
    t["dir"] = np.where(real & (t.z <= -3), "better", np.where(real & (t.z >= 3), "worse", "unresolved"))
    print("\n-- cells by resolved change (|z| >= 3)")
    print(t.groupby(["variant", "dir"]).size().unstack(fill_value=0).to_string())

    p = pd.read_csv(ref / "psi.csv")[keys + ["sqerr", "err", "cover"]].merge(
        psi[keys + ["sqerr", "err", "cover"]], on=keys, suffixes=("_ref", ""))
    REF = "plugin_xgb_clip01"
    out = []
    for (b, g, n), c in p.groupby(["block", "dgp", "n"]):
        w = {col: c.pivot_table(index="rep", columns="variant", values=col) for col in ("sqerr_ref", "sqerr")}
        for v, cv in c.groupby("variant"):
            row = {"block": b, "dgp": g, "n": n, "variant": v,
                   "RMSE ref": np.sqrt(cv.sqerr_ref.mean()), "RMSE": np.sqrt(cv.sqerr.mean()),
                   "bias ref": cv.err_ref.mean(), "bias": cv.err.mean(),
                   "cover ref": cv.cover_ref.mean(), "cover": cv.cover.mean()}
            if REF in w["sqerr"] and v != REF:
                row["z ref"] = _paired_change(w["sqerr_ref"], REF, v)[1]
                row["z"] = _paired_change(w["sqerr"], REF, v)[1]
            out.append(row)
    print(f"\n-- psi-hat, ref -> this run; z is paired, on squared error, against {REF} (negative: better)")
    print(pd.DataFrame(out).set_index(["block", "dgp", "n", "variant"]).to_string(float_format=lambda v: f"{v:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    r.add_argument("--dgps", nargs="*")
    r.add_argument("--methods", nargs="*")
    r.add_argument("--reps", default="0:20")
    r.add_argument("--jobs", type=int, default=9)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--shard", help="K/N: run only the tasks in shard K of N (0-based), for a job array")
    r.add_argument("--refit", action="store_true",
                   help="refit finished Riesz tasks at their CV-chosen setting, to add newer variants")
    s = sp.add_parser("summarize")
    s.add_argument("--compare", help="an earlier run's results directory (e.g. results/pilot) to pair with")
    args = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
