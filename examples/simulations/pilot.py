"""Pilot: does each candidate DGP order the methods the way AIMS.md predicts?

The measure is the RMSE of alpha-hat against alpha0 on a fixed test set of
20,000 fresh draws from the DGP, so it estimates the L2(P) error of alpha-hat,
which is what the Riesz loss targets. For a binary treatment, A is integrated
out exactly given X, using alpha-hat(1, x), alpha-hat(0, x) and pi(x), so the
only Monte Carlo noise in the measure comes from X. It is also reported on the
weak-overlap rows of the test set alone: min(pi, 1 - pi) < 0.05 for the ATE,
pi < 0.05 for the TSM, |alpha0| > 3 for the shift.

The one-step estimator follows ../lee_schuler_10cov: every nuisance is fit on
the training sample of n, and psi-hat = mean[m(mu-hat) + alpha-hat (Y - mu-hat)]
is evaluated on an independent estimation sample of n, with an
influence-function SE and a 95% Wald interval. One XGBoost outcome regression
per replicate (the task "outcome") is shared by every alpha-hat, so a method's
gap to the oracle, which plugs in alpha0, is the cost of estimating alpha
alone. The true psi0 is exact for a binary treatment and a 4M-draw Monte
Carlo integral for the shift (dgp.true_psi).

Blocks (see dgp.py for the DGPs):
  ate    ATE at n = 1000, claims 1 and 2. Plug-ins (XGBoost and main-terms
         logistic regression, each unclipped and clipped at 0.01 and 0.05)
         and five Riesz learners.
  tsm    TSM at A = 1, n in {250, 1000, 4000}, claim 3. rieszboost with the
         squared and the KL loss, and the XGBoost plug-in.
  shift  Additive shift by 0.5, n = 1000, claim 1. The classification density
         ratio (unclipped and clipped at 0.01 and 0.05) and three Riesz
         learners.

Tuning, the same for every learner: 5-fold cross-validation on the training
sample. Each setting is fit on four folds and scored on the fifth, and the
setting with the smallest mean held-out loss is refit on the whole sample.
The loss is the squared Riesz loss for Riesz learners (`score`), log loss
for the classifiers, and squared error for the outcome and exposure models.
XGBoost early-stops on the held-out fold, and the refit uses the mean
stopping iteration. The earlier results (results/pilot) tuned on a single
20% split, which picked noisy winners from large grids. The datasets are the
same, so the two sets of results pair replicate by replicate.

Each (block, dgp, n, rep, method) is one task and writes one JSON file, plus
one .npz of its predictions on the estimation sample, so a rerun fills only
the gaps. Results go to $RIESZ_SIM_OUT/pilot_cv5 (default: results/pilot_cv5).

Usage:
  python pilot.py run --blocks ate tsm shift --reps 0:20 --jobs 9
  python pilot.py run --reps 0:50 --shard 3/8 --jobs 32   # one of 8 array tasks
  python pilot.py summarize
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
# Set RIESZ_SIM_OUT to put results elsewhere, e.g. cluster scratch storage.
OUT = Path(os.environ.get("RIESZ_SIM_OUT", HERE / "results")) / "pilot_cv5"
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
        methods=["plugin_xgb", "plugin_logit", "rieszboost", "riesznet", "krrr", "riesztree", "forestriesz"]),
    "tsm": dict(
        estimand="TSM", ns=[250, 1000, 4000],
        dgps=["additive", "additive_g2", "interaction"],
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
GRIDS = {
    "xgb": _grid(max_depth=[1, 2, 3, 5, 8], learning_rate=[0.03, 0.1, 0.3, 1.0]),
    "rieszboost": _grid(max_depth=[1, 2, 3, 5, 8], reg_lambda=[0.0, 1.0, 10.0, 100.0, 1000.0]),
    "riesznet": _grid(learning_rate=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2]),
    "krrr": _grid(length_scale_mult=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
    "riesztree": _grid(max_leaf_nodes=[3, 4, 6, 8, 12, 16, 24, 32, 64], min_samples_leaf=[5, 20, 50]),
    "forestriesz": _grid(min_samples_leaf=[1, 2, 5, 10, 20, 50, 100]),
}
GRIDS["rieszboost_kl"] = GRIDS["rieszboost"]


# ------------------------------------------------------------- data ---

def train_data(dgp_name, n, rep):
    """The training sample and its tuning folds. The 20% split drawn first was
    the single-split tuning of results/pilot; it is still drawn so that the
    folds come from the same random stream position on every run."""
    rng = np.random.default_rng([BASE_SEED, zlib.crc32(dgp_name.encode()), n, rep])
    d = D.DGPS[dgp_name].draw(n, rng)
    rng.random(n)
    folds = rng.permutation(np.arange(n) % K)
    return d, folds


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


def est_data(dgp_name, n, rep):
    """The estimation sample: n fresh draws, independent of the training sample,
    on which the one-step estimator is evaluated."""
    return D.DGPS[dgp_name].draw(n, np.random.default_rng([BASE_SEED, zlib.crc32(dgp_name.encode()), n, rep, 1]))


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


def _forestriesz(est, covs, train, min_samples_leaf):
    from forestriesz import ForestRieszRegressor
    return ForestRieszRegressor(estimand=est, min_samples_leaf=min_samples_leaf, n_jobs=1, random_state=SEED)


RIESZ = {"rieszboost": _rieszboost("squared"), "rieszboost_kl": _rieszboost("kl"), "riesznet": _riesznet,
         "krrr": _krrr, "riesztree": _riesztree, "forestriesz": _forestriesz}


# A fit function takes (method, estimand, train, folds, covs) and returns
# (predict, selected setting, CV score of every setting). predict(df)
# maps each variant to alpha-hat at (A, X) for a shift, and to
# (alpha-hat(1, x), alpha-hat(0, x)) for a binary treatment.

def fit_riesz(method, estimand, train, folds, covs):
    cols = ["A"] + covs
    est = _estimand(estimand, covs)

    def fit_score(s, k):
        part = train[folds != k]
        m = RIESZ[method](est, covs, part, **s).fit(part[cols])
        return float(m.score(train[folds == k][cols])), None
    best, scores, _ = _tune(GRIDS[method], fit_score)
    m = RIESZ[method](est, covs, train, **best).fit(train[cols])

    def predict(df):
        Z = df[cols]
        if estimand == "SHIFT":
            return {method: m.predict(Z)}
        return {method: (m.predict(Z.assign(A=1.0)), m.predict(Z.assign(A=0.0)))}
    return predict, best, scores


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
        for c in (0.01, 0.05):
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
        for c in (0.01, 0.05):
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


def _mkdir(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _outcome_task(estimand, dgp_name, train, folds, est, covs):
    """Fit mu-hat; store on the estimation sample what the one-step estimator
    and the oracle need. Returns (record fields, arrays)."""
    g = D.DGPS[dgp_name]
    predict, best, scores = fit_outcome(estimand, train, folds, covs)
    mu, m_mu = predict(est)
    X, a = est[covs].to_numpy(), est["A"].to_numpy()
    mu0, m0 = D._mu(a, X), D.m_mu0(estimand, a, X)
    arrays = {"y": est["Y"].to_numpy(), "mu": mu, "m_mu": m_mu, "alpha0": g.true_alpha(estimand, a, X)}
    fields = {"rows": [], "rmse_mu": float(np.sqrt(np.mean((mu - mu0) ** 2))),
              "rmse_m_mu": float(np.sqrt(np.mean((m_mu - m0) ** 2))), "best": best, "grid": GRIDS["xgb"],
              "scores": scores}
    return fields, arrays


def run_task(block, dgp_name, n, rep, method):
    path = task_path(block, dgp_name, n, rep, method)
    estimand = BLOCKS[block]["estimand"]
    t0 = time.time()
    try:
        train, folds = train_data(dgp_name, n, rep)
        est = est_data(dgp_name, n, rep)
        covs = D.DGPS[dgp_name].covariates
        if method == "outcome":
            fields, arrays = _outcome_task(estimand, dgp_name, train, folds, est, covs)
            np.savez(_mkdir(path).with_suffix(".npz"), **arrays)
            rec = {"block": block, "dgp": dgp_name, "n": n, "rep": rep, "method": method, **fields,
                   "seconds": time.time() - t0}
            path.write_text(json.dumps(rec))
            return path, False, rec["seconds"]
        test, alpha0, weak, pi = test_data(dgp_name, estimand)
        fit = (fit_riesz if method in RIESZ else
               fit_plugin_classif if method == "plugin_classif" else fit_plugin_binary)
        predict, best, scores = fit(method, estimand, train, folds, covs)
        alphas = predict(test)
        # alpha-hat at the observed (A, X) of the estimation sample, for the one-step estimator.
        a_est = est["A"].to_numpy()
        on_est = {k: (v if estimand == "SHIFT" else a_est * v[0] + (1 - a_est) * v[1])
                  for k, v in predict(est).items()}
        np.savez(_mkdir(path).with_suffix(".npz"), **on_est)
        rows = []
        for k, v in alphas.items():
            if pi is None:
                sq, big = (v - alpha0) ** 2, np.abs(v).max()
            else:
                # Integrate A out given X: E[(alpha-hat - alpha0)^2 | X].
                g, X = D.DGPS[dgp_name], test[covs].to_numpy()
                e1 = v[0] - g.true_alpha(estimand, np.ones(len(X)), X)
                e0 = v[1] - g.true_alpha(estimand, np.zeros(len(X)), X)
                sq, big = pi * e1**2 + (1 - pi) * e0**2, max(np.abs(v[0]).max(), np.abs(v[1]).max())
            rows.append({"variant": k, "rmse": float(np.sqrt(sq.mean())),
                         "rmse_weak": float(np.sqrt(sq[weak].mean())) if weak.any() else None,
                         "max_abs_alpha_hat": float(big)})
        rec = {"block": block, "dgp": dgp_name, "n": n, "rep": rep, "method": method, "rows": rows,
               "best": best, "grid": GRIDS["xgb"] if method.startswith("plugin_") else GRIDS[method],
               "scores": scores, "seconds": time.time() - t0}
    except Exception:
        rec = {"block": block, "dgp": dgp_name, "n": n, "rep": rep, "method": method,
               "error": traceback.format_exc(), "seconds": time.time() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return path, "error" in rec, rec["seconds"]


# Rough relative cost, so the slowest tasks start first.
COST = {"riesznet": 10, "rieszboost": 6, "rieszboost_kl": 6, "forestriesz": 3, "krrr": 3, "outcome": 2}


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
    tasks = [t for t in tasks if args.overwrite or not task_path(*t).exists()]
    tasks.sort(key=lambda t: -COST.get(t[4], 1) * t[2])
    print(f"{len(tasks)} tasks on {args.jobs} processes", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for fut in as_completed([pool.submit(run_task, *t) for t in tasks]):
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
RIESZ_ATE = ["rieszboost", "riesznet", "krrr", "riesztree", "forestriesz"]
CONTRASTS = (
    [("ate", m, "plugin_xgb_clip01") for m in RIESZ_ATE]
    + [("ate", "plugin_logit", "plugin_xgb_clip01")]
    + [("tsm", "rieszboost_kl", "rieszboost"), ("tsm", "rieszboost_kl", "plugin_xgb_clip01"),
       ("tsm", "rieszboost", "plugin_xgb_clip01")]
    + [("shift", m, ref) for ref in ("plugin_classif", "plugin_classif_clip05")
       for m in ("rieszboost", "riesznet", "krrr")]
)


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
                        "z": z, "reps": reps, "reps for |z|=3": int(np.ceil((3 * sd / mean) ** 2)) if mean else None})
    return pd.DataFrame(out)


def load_psi(recs):
    """One row per (replicate, alpha-hat variant): the one-step estimate
    psi-hat = mean[m(mu-hat) + alpha-hat (Y - mu-hat)] on the estimation sample,
    its influence-function SE, and its error. The oracle plugs in alpha0 with
    the same mu-hat."""
    keys = ("block", "dgp", "n", "rep")
    outcomes = {tuple(r[k] for k in keys): r for r in recs if r["method"] == "outcome" and "error" not in r}
    arrays = {}

    def one_step(key, variant, alpha):
        o = arrays[key]
        phi = o["m_mu"] + alpha * (o["y"] - o["mu"])
        psi, se = phi.mean(), phi.std(ddof=1) / np.sqrt(len(phi))
        err = psi - D.true_psi(key[1], BLOCKS[key[0]]["estimand"])
        return {**dict(zip(keys, key)), "variant": variant, "psi": psi, "se": se, "err": err,
                "sqerr": err**2, "cover": float(abs(err) <= 1.96 * se)}

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
        a = np.load(task_path(*key, r["method"]).with_suffix(".npz"))
        rows.extend(one_step(key, v, a[v]) for v in a.files)
    return pd.DataFrame(rows)


def psi_table(sub):
    """Bias, empirical SE, RMSE, mean SE over empirical SE, and 95% coverage
    of psi-hat, each a column block with methods side by side."""
    g = sub.groupby(["dgp", "n", "variant"])
    t = pd.DataFrame({
        "bias": g.err.mean(),
        "empSE": g.psi.std(ddof=1),
        "RMSE": np.sqrt(g.sqerr.mean()),
        "SE/empSE": g.se.mean() / g.psi.std(ddof=1),
        "cover": g.cover.mean(),
    })
    return t


def edge_table(recs):
    """Share of fits whose selected setting sits at the smallest or largest
    value of a tuned hyperparameter, by block and method."""
    out = []
    for (block, method), grp in itertools.groupby(
            sorted((r for r in recs if "error" not in r and r["best"]), key=lambda r: (r["block"], r["method"])),
            key=lambda r: (r["block"], r["method"])):
        grp = list(grp)
        for hp in grp[0]["best"]:
            vals = sorted({s[hp] for s in grp[0]["grid"]})
            picks = [r["best"][hp] for r in grp]
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
    print("\n== paired contrasts in squared error of psi-hat (negative: the first method is better)")
    print(contrast_table(psi, "sqerr").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    psi.to_csv(OUT / "psi.csv", index=False)


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
    sp.add_parser("summarize")
    args = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
