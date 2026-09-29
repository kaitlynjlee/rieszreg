"""rieszboost_l2, riesznet, the non-augmented ForestRiesz and the classification
density ratio on the Lee & Schuler 10-confounder continuous-exposure DGP
(dgp.py), for the ASE and the LASE.

Protocol, as in the RieszBoost ICML manuscript (Section 3.4, Appendix C.2.3)
and in ../lee_schuler_10cov: each replicate draws an estimation set of 500 and
then an independent training set of 500. Every learner is tuned by 5-fold
cross-validation on the training set, over the grid below, then refit on the
whole training set at the selected setting and evaluated on the estimation
set. The outcome regression is one tuned XGBoost model per replicate, shared
by every representer and by both estimands, and each representer goes into
the one-step (DML) estimator. The oracle plugs in the true representer, so
its gap to a learner is the cost of estimating alpha alone.

Each (replicate, learner) pair is one task and caches its estimation-set
predictions and the CV risk of every setting, so `summarize` assembles the
tables and adding a learner never refits the others.

  python run.py run --reps 0:2 --jobs 8      # a smoke test
  python run.py summarize                    # tables, tuning checks
"""

from __future__ import annotations

import _env  # noqa: F401  (threads and import paths; must precede numpy)

import argparse
import hashlib
import importlib.metadata as md
import inspect
import itertools
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dgp

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("RIESZ_SIM_OUT", HERE))
CACHE = OUT_ROOT / "cache" / "lee_schuler_10cov_shift"
RESULTS = OUT_ROOT / "results" / "lee_schuler_10cov_shift"

BASE_SEED = 20260929
N, K = 500, 5
ESTIMANDS = ("ASE", "LASE")
SEED = 0              # every learner with internal randomness
VALID_FRAC = 0.2      # share of each training fold held out for early stopping


def _grid(**axes):
    return [dict(zip(axes, v)) for v in itertools.product(*axes.values())]


# ------------------------------------------------------------ settings ---
# The manuscript's settings (Appendix C.2.3), as in ../lee_schuler_10cov.
# Boosted learners: patience 200, cap 20000 trees, subsample 0.9. RieszNet: 3
# hidden layers of 200, ELU, Adam with weight decay 1e-3, patience 10, cap 1000
# epochs. The authors' ASE script capped trees at 10000; the manuscript and
# their LASE script say 20000.

BOOST = dict(max_iter=20000, patience=200, subsample=0.9)
NET = dict(max_iter=1000, patience=10, hidden_sizes=(200, 200, 200), weight_decay=1e-3, batch_size=64)

XGB_GRID = _grid(learning_rate=[1e-5, 1e-4, 1e-3, 1e-2], max_depth=[3, 5, 7])
RIESZNET_GRID = _grid(learning_rate=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
# Every grid is the one ../lee_schuler_10cov uses. ForestRiesz is not in the
# manuscript; its basis (see _basis) is fixed, not tuned.
FOREST_GRID = _grid(min_samples_leaf=[5, 10, 20, 50, 100])
FOREST_DEGREE = 2


# ------------------------------------------------------------ learners ---
# Each takes (train, test, **setting) and returns (predictions on test as a
# DataFrame, iterations kept by early stopping, or None). Representer
# predictions are alpha at (A, X) and at (A + delta, X), which is all the
# Riesz loss and the one-step estimator need.

def _estimand(name):
    from rieszreg import AdditiveShift, LocalShift
    kw = dict(treatment="A", covariates=tuple(dgp.COV))
    if name == "ASE":
        return AdditiveShift(delta=dgp.DELTA, **kw)
    return LocalShift(delta=dgp.DELTA, threshold=dgp.THRESHOLD, **kw)


def _shifted(Z):
    return Z.assign(A=Z["A"] + dgp.DELTA)


def _alpha_preds(est, test):
    Z = test[["A"] + dgp.COV]
    return pd.DataFrame({"alpha": est.predict(Z), "alpha_shift": est.predict(_shifted(Z))})


def outcome_xgb(train, test, learning_rate, max_depth):
    import xgboost as xgb
    valid = np.random.default_rng(SEED).random(len(train)) < VALID_FRAC
    cols = ["A"] + dgp.COV
    fit_part, val = train[~valid], train[valid]
    m = xgb.XGBRegressor(n_estimators=BOOST["max_iter"], learning_rate=learning_rate, max_depth=max_depth,
                         subsample=BOOST["subsample"], early_stopping_rounds=BOOST["patience"],
                         n_jobs=1, random_state=SEED)
    m.fit(fit_part[cols], fit_part["Y"], eval_set=[(val[cols], val["Y"])], verbose=False)
    Z = test[cols]
    preds = pd.DataFrame({"mu": m.predict(Z), "mu_shift": m.predict(_shifted(Z))})
    return preds, m.best_iteration + 1


def classification(train, test, learning_rate, max_depth):
    """The manuscript's indirect comparator for the ASE (Appendix B.2.2): an
    XGBoost classifier of observed rows (A, X), labelled 1, against shifted
    rows (A + delta, X), labelled 0. The classes are balanced, so the odds
    (1 - p) / p estimate p(a - delta | x) / p(a | x), and alpha = odds - 1.
    Tuned by log loss on the held-out fold's observed and shifted rows. Its
    early-stopping split is by individual, so both rows of a unit fall on the
    same side, as in the authors' DensityRatioXGB_ES."""
    import xgboost as xgb
    valid = np.random.default_rng(SEED).random(len(train)) < VALID_FRAC
    cols = ["A"] + dgp.COV

    def stack(df):
        Z = df[cols]
        return pd.concat([Z, _shifted(Z)], ignore_index=True), np.r_[np.ones(len(Z)), np.zeros(len(Z))]

    (Xf, yf), (Xv, yv) = stack(train[~valid]), stack(train[valid])
    m = xgb.XGBClassifier(n_estimators=BOOST["max_iter"], learning_rate=learning_rate, max_depth=max_depth,
                          subsample=BOOST["subsample"], early_stopping_rounds=BOOST["patience"],
                          eval_metric="logloss", n_jobs=1, random_state=SEED)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    Z = test[cols]
    p, p_shift = m.predict_proba(Z)[:, 1], m.predict_proba(_shifted(Z))[:, 1]
    preds = pd.DataFrame({"alpha": (1 - p) / p - 1, "alpha_shift": (1 - p_shift) / p_shift - 1,
                          "p": p, "p_shift": p_shift})
    return preds, m.best_iteration + 1


def riesznet(estimand):
    def fit(train, test, learning_rate):
        from riesznet import RieszNet
        n = RieszNet(estimand=_estimand(estimand), hidden_sizes=NET["hidden_sizes"], activation="elu",
                     learning_rate=learning_rate, weight_decay=NET["weight_decay"], epochs=NET["max_iter"],
                     batch_size=NET["batch_size"], early_stopping_rounds=NET["patience"],
                     validation_fraction=VALID_FRAC, snapshot_epochs=[], random_state=SEED).fit(train[["A"] + dgp.COV])
        return _alpha_preds(n, test), n.best_iteration_ + 1
    return fit


# ForestRiesz's basis. The published ForestRiesz fits alpha(a, x) =
# theta(x)' phi(a) with a forest that splits on x only, and a basis phi in the
# treatment; ../lee_schuler_10cov uses [1 - a, a]. A continuous exposure
# needs a basis we choose, and the choice uses only the estimand (delta, t),
# never the DGP:
# - ASE: polynomials in a of degree FOREST_DEGREE. Each leaf's theta is then
#   the L2 projection of alpha_0 onto those polynomials. At degree 1 the
#   solution is delta (a - mean_leaf(A)) / var_leaf(A), the first-order
#   expansion of any location-family density ratio; degree 2 adds the
#   curvature a density ratio has (here exp(linear in a) - 1). Degree 0 is
#   the constant basis, whose moment is identically zero for a shift.
# - LASE: alpha_0 is 0 for a >= t + delta (alpha enters the loss there only
#   through alpha^2), and it can jump at a = t and a = t + delta, since m()
#   evaluates alpha only below t + delta and adds -alpha(a) below t. So the
#   basis is polynomials of degree FOREST_DEGREE on a < t plus a constant on
#   t <= a < t + delta. That interval holds about 12% of rows, a few per
#   leaf, and higher degrees there made the leaf solves unstable: on
#   replicates 0-7 at leaf size 20, alpha-hat RMSE was 0.22 with a constant
#   there, 0.29 with degree 1 and 0.41 with degree 2 (up to 7.5 at leaf size
#   10). Degree 2 on a < t and for the ASE was chosen on the same replicates
#   (ASE, leaf size 50: 0.126 at degree 1, 0.105 at degree 2, 0.147 at 3).
# The powers use (a - t) / A_SCALE, a fixed rescaling that keeps the leaf
# solves well conditioned; the span of the basis does not depend on it.
A_SCALE = 3.0


def _basis(estimand, degree):
    t, d = dgp.THRESHOLD, dgp.DELTA
    u = lambda f: (f[:, 0] - t) / A_SCALE                  # noqa: E731
    if estimand == "ASE":
        return [lambda f, k=k: u(f) ** k for k in range(degree + 1)]
    below = lambda f: (f[:, 0] < t).astype(float)                          # noqa: E731
    mid = lambda f: ((f[:, 0] >= t) & (f[:, 0] < t + d)).astype(float)     # noqa: E731
    return [lambda f, k=k: below(f) * u(f) ** k for k in range(degree + 1)] + [mid]


def forestriesz(estimand):
    """Non-augmented ForestRiesz (Chernozhukov et al. 2022) on EconML's GRF,
    with the basis above, splitting on the covariates only."""
    def fit(train, test, min_samples_leaf):
        from forestriesz import ForestRieszRegressor
        f = ForestRieszRegressor(estimand=_estimand(estimand), riesz_feature_fns=_basis(estimand, FOREST_DEGREE),
                                 split_feature_indices=tuple(range(1, 1 + len(dgp.COV))),
                                 min_samples_leaf=min_samples_leaf, n_jobs=1, random_state=SEED)
        f.fit(train[["A"] + dgp.COV])
        return _alpha_preds(f, test), None
    return fit


def _riesz_rows(kind, df, alpha, alpha_shift):
    """Squared Riesz loss alpha^2 - 2 m(alpha) per row, for alpha arrays of any
    trailing shape (rows, or rows x trees). m(alpha) = alpha(A + delta, X) -
    alpha(A, X), times 1(A < t) for the LASE."""
    w = np.ones(len(df)) if kind == "ASE" else (df["A"].to_numpy() < dgp.THRESHOLD).astype(float)
    if np.ndim(alpha) == 2:
        w = w[:, None]
    return alpha ** 2 - 2 * w * (alpha_shift - alpha)


def row_loss(kind, data, preds):
    """Per-row held-out loss that CV minimizes: squared error for the outcome
    regression, log loss on the observed and shifted rows for the classifier,
    and the squared Riesz loss otherwise."""
    if kind == "outcome":
        return (data["Y"].to_numpy() - preds["mu"].to_numpy()) ** 2
    if kind == "classification":
        return -np.log(preds["p"].to_numpy()) - np.log1p(-preds["p_shift"].to_numpy())
    return _riesz_rows(kind, data, preds["alpha"].to_numpy(), preds["alpha_shift"].to_numpy())


# ---------------------------------------------------- rieszboost_l2 ---
# rieszboost with the grid ../lee_schuler_10cov settled on for the ATE and ATT
# (see its README): an L2 penalty on leaf values, patience 50, subsample 0.8
# of individuals per round. It is the starting grid here; the pilot's tuning
# check says whether it has to move for the shift estimands.
#
# Patience costs no extra fits: each fit runs at the largest patience, its
# validation Riesz loss after every tree is computed from per-tree
# predictions, and early stopping at each smaller patience is replayed on
# that path. Boosting is sequential with a seeded row subsample, so the first
# t trees of that fit are the trees a fit stopped at t would have.
L2_GRID = _grid(learning_rate=[3e-3, 1e-2, 3e-2, 1e-1], max_depth=[1, 2, 3, 5],
                reg_lambda=[10.0, 30.0, 100.0, 300.0], patience=[50])
L2_SUBSAMPLE = 0.8    # share of individuals drawn per round


def _eta_path(b, Z):
    """rows x trees: the booster's eta after each tree, from per-tree
    predictions (each includes the base margin once, recovered from the
    first two trees)."""
    import xgboost as xgb
    booster = b.predictor_.booster
    d = xgb.DMatrix(np.asarray(b._features(Z), dtype=float))
    per = np.column_stack([booster.predict(d, iteration_range=(k, k + 1))
                           for k in range(booster.num_boosted_rounds())]).astype(float)
    base = per[:, 0] + per[:, 1] - booster.predict(d, iteration_range=(0, 2)).astype(float)
    return base[:, None] + np.cumsum(per - base[:, None], axis=1)


def _alpha_path(b, df):
    Z = df[["A"] + dgp.COV]
    return {c: np.asarray(b.loss_.link_to_alpha(_eta_path(b, z)))
            for c, z in (("alpha", Z), ("alpha_shift", _shifted(Z)))}


def _stop(path, patience):
    """Trees kept by early stopping with this patience on a validation-loss path."""
    best, best_i = np.inf, 0
    for i, v in enumerate(path):
        if v < best:
            best, best_i = v, i
        elif i - best_i >= patience:
            break
    return best_i + 1


def _l2_fit(estimand, train, learning_rate, max_depth, reg_lambda):
    """One fit at the largest patience, on 80% of `train`, stopping on the
    other 20%; returns the booster and its validation-loss path."""
    from rieszboost import RieszBooster
    valid = np.random.default_rng(SEED).random(len(train)) < VALID_FRAC
    cols = ["A"] + dgp.COV
    b = RieszBooster(estimand=_estimand(estimand), n_estimators=BOOST["max_iter"], learning_rate=learning_rate,
                     max_depth=max_depth, reg_lambda=reg_lambda, subsample=L2_SUBSAMPLE,
                     early_stopping_rounds=max(s["patience"] for s in L2_GRID),
                     random_state=SEED).fit(train[~valid][cols], eval_set=train[valid][cols])
    v = train[valid].reset_index(drop=True)
    path = _riesz_rows(estimand, v, **_alpha_path(b, v)).mean(axis=0)
    return b, path


def rieszboost_l2(estimand):
    def select(train, test):
        folds = np.arange(len(train)) * K // len(train)
        patiences = sorted({s["patience"] for s in L2_GRID})
        bases = list(dict.fromkeys((s["learning_rate"], s["max_depth"], s["reg_lambda"]) for s in L2_GRID))
        losses = {str(s): [] for s in L2_GRID}
        iters = {str(s): [] for s in L2_GRID}
        replay_ok = []
        for lr, depth, lam in bases:
            for k in range(K):
                val = train[folds == k].reset_index(drop=True)
                b, path = _l2_fit(estimand, train[folds != k].reset_index(drop=True), lr, depth, lam)
                replay_ok.append(_stop(path, max(patiences)) == b.best_iteration_ + 1)
                a = _alpha_path(b, val)
                for p in patiences:
                    # at the fit's own patience, xgboost's stop; smaller ones are replayed
                    t = b.best_iteration_ + 1 if p == max(patiences) else _stop(path, p)
                    s = str(dict(learning_rate=lr, max_depth=depth, reg_lambda=lam, patience=p))
                    losses[s].append(_riesz_rows(estimand, val, a["alpha"][:, t - 1], a["alpha_shift"][:, t - 1]))
                    iters[s].append(t)
        risk = {s: float(np.concatenate(v).mean()) for s, v in losses.items()}
        best = L2_GRID[[str(s) for s in L2_GRID].index(min(risk, key=risk.get))]
        b, path = _l2_fit(estimand, train, best["learning_rate"], best["max_depth"], best["reg_lambda"])
        t = b.best_iteration_ + 1 if best["patience"] == max(patiences) else _stop(path, best["patience"])
        a = _alpha_path(b, test)
        preds = pd.DataFrame({c: a[c][:, t - 1] for c in ("alpha", "alpha_shift")})
        return preds, {"best": best, "cv_risk": risk, "cv_iters": iters, "iters": t,
                       "replay_matches_booster": float(np.mean(replay_ok))}
    return select


# name -> (fit function, grid, loss kind, iteration cap[, selector]). A
# component with a selector is fit by selector(train, test) instead of
# cv_select(fit, grid, ...). Slowest first, so a parallel run starts the long
# tasks early. The classifier has no LASE counterpart (the manuscript says
# none exists), so it runs for the ASE only.
COMPONENTS = {
    **{f"rieszboost_l2_{e}": (None, L2_GRID, e, BOOST["max_iter"], rieszboost_l2(e)) for e in ESTIMANDS},
    "outcome": (outcome_xgb, XGB_GRID, "outcome", BOOST["max_iter"]),
    "classification_ASE": (classification, XGB_GRID, "classification", BOOST["max_iter"]),
    **{f"riesznet_{e}": (riesznet(e), RIESZNET_GRID, e, NET["max_iter"]) for e in ESTIMANDS},
    **{f"forestriesz_{e}": (forestriesz(e), FOREST_GRID, e, None) for e in ESTIMANDS},
}
LEARNERS = ("rieszboost_l2", "riesznet", "forestriesz", "classification")

# Source hashed into every component's cache key, and per learner the extra
# helpers and rieszreg packages it uses.
SHARED = ("_estimand", "_shifted", "_alpha_preds", "_riesz_rows", "row_loss", "cv_select", "data")
EXTRA = {"rieszboost_l2": ("_eta_path", "_alpha_path", "_stop", "_l2_fit"), "forestriesz": ("_basis",)}
PACKAGES = {"rieszboost_l2": ("rieszreg", "rieszboost"), "riesznet": ("rieszreg", "riesznet"),
            "forestriesz": ("rieszreg", "forestriesz")}


# ----------------------------------------------------------- one task ---

def data(rep):
    """Estimation set, then training set."""
    rng = np.random.default_rng([BASE_SEED, rep])
    return dgp.draw(N, rng), dgp.draw(N, rng)


def cv_select(fit, grid, kind, train, test):
    """K-fold CV over `grid` by mean held-out loss; refit the best setting on
    all of `train` and predict `test`."""
    folds = np.arange(len(train)) * K // len(train)     # rows are iid, so contiguous folds
    risk, iters = {}, {}
    for s in grid:
        losses, its = [], []
        for k in range(K):
            val = train[folds == k].reset_index(drop=True)
            p, it = fit(train[folds != k].reset_index(drop=True), val, **s)
            losses.append(row_loss(kind, val, p))
            its.append(it)
        risk[str(s)], iters[str(s)] = float(np.concatenate(losses).mean()), its
    best = grid[[str(s) for s in grid].index(min(risk, key=risk.get))]
    preds, it = fit(train, test, **best)
    return preds, {"best": best, "cv_risk": risk, "cv_iters": iters, "iters": it}


def _versions():
    out = {}
    for p in ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "econml"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def _stem(comp):
    return comp.rsplit("_", 1)[0] if comp.endswith(ESTIMANDS) else comp


def cache_key(comp):
    """Everything that determines a component's fit except the replicate: its
    learner source and settings, the data and CV code, the package versions,
    and the source of the rieszreg packages it uses."""
    fit, grid, kind, cap = COMPONENTS[comp][:4]
    stem = _stem(comp)
    fns = (stem if stem != "outcome" else "outcome_xgb",) + SHARED + EXTRA.get(stem, ())
    parts = [inspect.getsource(globals()[f]) for f in fns]
    parts += [repr((grid, kind, cap, BOOST, NET, BASE_SEED, N, K, SEED, VALID_FRAC, L2_SUBSAMPLE, A_SCALE, FOREST_DEGREE)),
              inspect.getsource(dgp), repr(sorted(_versions().items()))]
    for pkg in PACKAGES.get(stem, ()):
        parts += [p.read_text() for p in sorted((_env.REPO / "packages" / pkg / "python" / pkg).rglob("*.py"))]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


def cache_path(comp, rep, key):
    return CACHE / comp / f"rep{rep}_{key}.pkl"


def _write(path, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.to_pickle(out, tmp)
    tmp.replace(path)                 # atomic: a killed job leaves the previous file in place


def run_task(rep, comp, key, overwrite=False):
    path = cache_path(comp, rep, key)
    if path.exists() and not overwrite:
        return
    fit, grid, kind, _ = COMPONENTS[comp][:4]
    selector = COMPONENTS[comp][4] if len(COMPONENTS[comp]) > 4 else None
    est, train = data(rep)
    t0 = time.perf_counter()
    try:
        preds, record = selector(train, est) if selector else cv_select(fit, grid, kind, train, est)
        out = {"preds": {c: preds[c].to_numpy() for c in preds}, "record": record}
    except Exception as err:          # a failed fit is recorded as missing, and the run continues
        out = {"preds": None, "record": {"error": repr(err)}}
    out["seconds"] = time.perf_counter() - t0
    _write(path, out)
    if overwrite:                     # drop this pair's files from older code, so one version remains
        for old in (CACHE / comp).glob(f"rep{rep}_*.pkl"):
            if old != path:
                old.unlink()
    print(f"done rep={rep} {comp} {out['seconds']:.0f}s{' FAILED' if out['preds'] is None else ''}", flush=True)


def run(reps, comps, jobs, overwrite=False):
    """Fit every (replicate, component) pair not yet cached under the current
    key. With overwrite, refit every pair and replace its cache file."""
    keys = {c: cache_key(c) for c in comps}
    tasks = [(r, c, keys[c], overwrite) for c in comps for r in reps
             if overwrite or not cache_path(c, r, keys[c]).exists()]
    print(f"{len(tasks)} tasks to run{' (overwriting)' if overwrite else ''}", flush=True)
    if jobs == 1:
        for t in tasks:
            run_task(*t)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(lambda t: _task_process(*t), tasks))


def _task_process(rep, comp, key, overwrite):
    """Fit one (replicate, component) pair in its own Python process. A crash
    in native code (xgboost aborted once with "stack smashing detected" on
    Savio) then loses that fit alone instead of every worker in a shared
    pool. The pair is retried once and, if it crashes again, recorded as a
    failed fit, which summarize counts in its `failed` column."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "task", str(rep), comp]
    if overwrite:
        cmd.append("--overwrite")
    for attempt in (1, 2):
        code = subprocess.run(cmd).returncode
        if code == 0:
            return
        how = f"signal {signal.Signals(-code).name}" if code < 0 else f"exit code {code}"
        print(f"rep={rep} {comp}: process ended with {how} (attempt {attempt})", flush=True)
    _write(cache_path(comp, rep, key), {"preds": None, "seconds": float("nan"),
                                        "record": {"error": f"fit process crashed twice ({how})"}})
    print(f"done rep={rep} {comp} FAILED (process crashed twice)", flush=True)


# ------------------------------------------------------------- summary ---

# The manuscript's Tables 3 and 4 (1000 replicates), for reference. Its
# rieszboost ran the manuscript grid, which this study replaces with
# rieszboost_l2. Its "Indirect" row is the classification comparator. Its
# RieszNet LASE row has MAE above RMSE, which averages of per-replicate values
# cannot do, so one of those two numbers is a typo.
PAPER = {
    ("ASE", "rieszboost"): dict(alpha_rmse=0.249, alpha_mae=0.188, pct_bias=-1.21, rmse=0.105, coverage=0.838),
    ("ASE", "classification"): dict(alpha_rmse=0.269, alpha_mae=0.199, pct_bias=-1.14, rmse=0.113, coverage=0.843),
    ("ASE", "riesznet"): dict(alpha_rmse=2.496, alpha_mae=0.534, pct_bias=1.89, rmse=0.313, coverage=0.774),
    ("LASE", "rieszboost"): dict(alpha_rmse=0.252, alpha_mae=0.143, pct_bias=0.49, rmse=0.180, coverage=0.905),
    ("LASE", "riesznet"): dict(alpha_rmse=0.331, alpha_mae=0.411, pct_bias=-1.14, rmse=0.394, coverage=0.934),
}


def load(comp):
    key = cache_key(comp)
    return {int(f.name[3:].split("_")[0]): pd.read_pickle(f) for f in (CACHE / comp).glob(f"rep*_{key}.pkl")}


def one_step(estimand, d, mu, alpha):
    a, y = d["A"].to_numpy(), d["Y"].to_numpy()
    tau, resid = mu["mu_shift"] - mu["mu"], y - mu["mu"]
    if estimand == "ASE":
        phi = tau + alpha * resid
        est = phi.mean()
    else:
        below = (a < dgp.THRESHOLD).astype(float)
        p = below.mean()
        est = np.mean(below * tau + alpha * resid) / p
        phi = (below * (tau - est) + alpha * resid) / p
    return float(est), float(phi.std(ddof=1) / np.sqrt(len(y)))


def estimates():
    """One row per (replicate, estimand, method)."""
    mu = load("outcome")
    fits = {c: load(c) for c in COMPONENTS if c != "outcome"}
    rows = []
    for rep in sorted(mu):
        if mu[rep]["preds"] is None:
            continue
        d, _ = data(rep)
        X, A = d[dgp.COV].to_numpy(), d["A"].to_numpy()
        mu0 = {"mu": dgp.outcome(A, X), "mu_shift": dgp.outcome(A + dgp.DELTA, X)}
        for e in ESTIMANDS:
            a0 = dgp.true_alpha(e, A, X)
            for method in ("oracle_mu0", "oracle") + LEARNERS:
                if method.startswith("oracle"):
                    alpha = a0
                else:
                    got = fits.get(f"{method}_{e}", {}).get(rep)
                    if got is None:
                        continue
                    if got["preds"] is None:
                        rows.append(dict(rep=rep, estimand=e, method=method, failed=True))
                        continue
                    alpha = got["preds"]["alpha"]
                # oracle_mu0 also uses the true outcome regression: a check on
                # the pipeline and on Monte Carlo noise, since with alpha_0 the
                # one-step estimator is unbiased for any independent mu-hat.
                est, se = one_step(e, d, mu0 if method == "oracle_mu0" else mu[rep]["preds"], alpha)
                rows.append(dict(rep=rep, estimand=e, method=method, failed=False, est=est, se=se,
                                 alpha_rmse=float(np.sqrt(np.mean((alpha - a0) ** 2))),
                                 alpha_mae=float(np.mean(np.abs(alpha - a0)))))
    return pd.DataFrame(rows)


def table(df):
    """Performance per (estimand, method), each measure with its Monte Carlo SE."""
    rows = []
    for (e, m), g in df.groupby(["estimand", "method"], sort=False):
        ok = g[~g.failed]
        R = len(ok)
        err = ok.est.to_numpy() - dgp.PSI[e]
        rmse = np.sqrt(np.mean(err ** 2))
        cov = np.mean(np.abs(err) <= 1.96 * ok.se.to_numpy())
        rows.append(dict(
            estimand=e, method=m, reps=R, failed=int(g.failed.sum()),
            alpha_rmse=ok.alpha_rmse.mean(), alpha_rmse_mcse=ok.alpha_rmse.std(ddof=1) / np.sqrt(R),
            alpha_mae=ok.alpha_mae.mean(), alpha_mae_mcse=ok.alpha_mae.std(ddof=1) / np.sqrt(R),
            pct_bias=100 * err.mean() / dgp.PSI[e], pct_bias_mcse=100 * err.std(ddof=1) / np.sqrt(R) / dgp.PSI[e],
            rmse=rmse, rmse_mcse=np.std(err ** 2, ddof=1) / np.sqrt(R) / (2 * rmse),
            emp_se=err.std(ddof=1), mean_se=ok.se.mean(),
            coverage=cov, coverage_mcse=np.sqrt(cov * (1 - cov) / R)))
    return pd.DataFrame(rows)


def tuning():
    """Per component, the settings CV selected, the share of selections on the
    low or high edge of each tuned axis, and the share of refits that reached
    the iteration cap. A consistent edge means the grid should move."""
    lines = []
    for comp, spec in COMPONENTS.items():
        grid, cap = spec[1], spec[3]
        recs = [g["record"] for g in load(comp).values() if g["preds"] is not None]
        if not recs:
            continue
        lines.append(f"\n{comp}: {len(recs)} replicates")
        lines.append(pd.Series([str(r["best"]) for r in recs]).value_counts(normalize=True).round(2).to_string())
        for axis in grid[0]:
            vals = sorted({s[axis] for s in grid})
            if len(vals) > 1:
                chosen = np.array([r["best"][axis] for r in recs])
                lines.append(f"  {axis}: at lowest {np.mean(chosen == vals[0]):.2f}, at highest {np.mean(chosen == vals[-1]):.2f}")
        if cap:
            its = np.array([r["iters"] for r in recs])
            lines.append(f"  refit iterations: median {int(np.median(its))}, share at cap {np.mean(its >= cap):.2f}")
            if "learning_rate" in grid[0] and np.any(its >= cap):
                lr = np.array([r["best"]["learning_rate"] for r in recs])
                by = ", ".join(f"{v:g}: {np.mean(its[lr == v] >= cap):.2f}" for v in sorted(set(lr)))
                lines.append(f"  share of refits at cap, by selected learning rate: {by}")
        lines.append(regret(recs, grid))
    return "\n".join(lines)


def regret(recs, grid):
    """Median regret of each setting: its CV risk minus the smallest CV risk
    in the same replicate, taken over replicates. When the risk surface is
    flat, selection frequencies put settings on an edge by chance; the setting
    with the smallest median regret is the one the edge rule should check."""
    risk = pd.DataFrame([r["cv_risk"] for r in recs])
    reg = risk.sub(risk.min(axis=1), axis=0).median().sort_values()
    best = grid[[str(s) for s in grid].index(reg.index[0])]
    edges = [f"{axis} at {'lowest' if best[axis] == min(v) else 'highest'}"
             for axis in best for v in [[s[axis] for s in grid]]
             if len(set(v)) > 1 and best[axis] in (min(v), max(v))]
    out = ["  median regret (CV risk above the replicate's best), smallest five:"]
    out += [f"    {k}: {v:.4f}" for k, v in reg.head(5).items()]
    out.append(f"  smallest median regret: {reg.index[0]} -> " + ("; ".join(edges) + " (on the edge)" if edges else "interior"))
    return "\n".join(out)


def cache_status():
    """What is in the cache, per component: files written by the current code
    (matching cache key), files from other code or package versions, and
    failed fits with their first error."""
    lines = [f"cache: {CACHE}" + ("" if CACHE.exists() else "  (does not exist)")]
    if "RIESZ_SIM_OUT" not in os.environ:
        lines.append("RIESZ_SIM_OUT is not set; savio.sh writes to /global/scratch/users/$USER/rieszreg_sim")
    for comp in COMPONENTS:
        files = list((CACHE / comp).glob("rep*.pkl"))
        key = cache_key(comp)
        current = [f for f in files if f.stem.endswith(key)]
        failed = [pd.read_pickle(f)["record"].get("error") for f in current]
        failed = [e for e in failed if e]
        line = f"  {comp:20s} {len(current):5d} current, {len(files) - len(current):5d} with another key, {len(failed)} failed"
        if failed:
            line += f"; first error: {failed[0][:200]}"
        lines.append(line)
    return "\n".join(lines)


def summarize():
    df = estimates()
    if df.empty:
        print("No results to summarize. What the cache holds:")
        print(cache_status())
        print("'another key' means the files came from different code or package versions than this "
              "checkout and environment; rerun, or summarize with the environment that wrote them.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "estimates.csv", index=False)
    t = table(df)
    t.to_csv(RESULTS / "table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("Representer accuracy (manuscript Table 3) and one-step estimator (Table 4), with MCSEs:")
    print(t.round(3).to_string(index=False))
    paper = pd.DataFrame([dict(estimand=e, method=m, **v) for (e, m), v in PAPER.items()])
    print("\nThe manuscript, for reference:")
    print(paper.to_string(index=False))
    print(tuning())


def status():
    """Per component: cache files under the current key and under each older
    key, with the newest file's time. Older keys come from different code or
    package versions and are ignored by summarize."""
    import datetime
    print(f"cache: {CACHE}")
    for comp in COMPONENTS:
        new = cache_key(comp)
        by_key = {}
        for f in (CACHE / comp).glob("rep*.pkl"):
            by_key.setdefault(f.stem.rsplit("_", 1)[1], []).append(f.stat().st_mtime)
        parts = []
        for k, ts in sorted(by_key.items(), key=lambda kv: (kv[0] != new, -max(kv[1]))):
            when = datetime.datetime.fromtimestamp(max(ts)).strftime("%m-%d %H:%M")
            parts.append(f"{'current' if k == new else k}: {len(ts)} (newest {when})")
        print(f"  {comp:20s} " + ("; ".join(parts) if parts else "none"))


def rekey(comps, from_key=None, dry_run=False):
    """Copy each component's cache files from an older cache key to the
    current one, keeping the originals.

    The cache key hashes the whole source of the rieszreg packages a learner
    uses, so any edit there (a docstring, a save/load change) orphans every
    cached fit, even when the fits themselves would not change. Use this only
    after checking that the change leaves the fits unchanged, for example by
    refitting one replicate with the new code and comparing it with the old
    cache file. With several older keys present, pass --from-key.
    """
    for comp in comps:
        new = cache_key(comp)
        files = sorted((CACHE / comp).glob("rep*.pkl"))
        keys = sorted({f.stem.rsplit("_", 1)[1] for f in files} - {new})
        if from_key is None and len(keys) != 1:
            print(f"{comp}: {len(keys)} older keys {keys}; pass --from-key to pick one")
            continue
        src = from_key or keys[0]
        moved = 0
        for f in files:
            if f.stem.endswith(src):
                rep = int(f.name[3:].split("_")[0])
                dst = cache_path(comp, rep, new)
                if not dst.exists():
                    if not dry_run:
                        shutil.copy2(f, dst)
                    moved += 1
        print(f"{comp}: {'would copy' if dry_run else 'copied'} {moved} files from key {src} to {new}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reps", default="0:2", help="half-open range: 0:100 runs replicates 0 to 99")
    r.add_argument("--components", nargs="*", default=None, help=f"default: all of {list(COMPONENTS)}")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--overwrite", action="store_true",
                   help="refit and replace cached pairs, deleting their files from older code")
    sub.add_parser("summarize")
    t = sub.add_parser("task", help="fit one (replicate, component) pair; used internally by run")
    t.add_argument("rep", type=int)
    t.add_argument("component")
    t.add_argument("--overwrite", action="store_true")
    sub.add_parser("status", help="cached fits per learner, by cache key")
    k = sub.add_parser("rekey", help="reuse cached fits after a code change verified not to alter them")
    k.add_argument("--components", nargs="+", required=True)
    k.add_argument("--from-key", default=None)
    k.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        lo, hi = (int(v) for v in a.reps.split(":"))
        run(range(lo, hi), a.components or list(COMPONENTS), a.jobs, a.overwrite)
    elif a.cmd == "task":
        run_task(a.rep, a.component, cache_key(a.component), a.overwrite)
    elif a.cmd == "rekey":
        rekey(a.components, a.from_key, a.dry_run)
    elif a.cmd == "status":
        status()
    else:
        summarize()
