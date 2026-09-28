"""The binary-treatment simulation of Lee & Schuler (2025), Section 3.1, run
with the same four sources of alpha-hat as the main study.

DGP (one covariate):
    X ~ Uniform(0, 1)
    A | X ~ Bernoulli(expit(-0.02 X - X^2 + 4 log(X + 0.3) + 1.5))
    Y | A, X ~ N(5X + 9XA + 5 sin(pi X) + 25(A - 2), 1)
    psi_ATE = 29.502, psi_ATT = 30.786 (the paper's values; checked below).

Protocol, as in the paper: n = 1000 per replicate, the first 500 rows train
every learner and the other 500 are the estimation set (one split, no swap).
Every boosted learner (outcome regression, propensity score, rieszboost) is
tuned by 5-fold CV on the training half over learning rate {0.001, 0.01, 0.1,
0.25} x trees {10, 30, 50, 75, 100, 150, 200} x depth {3, 5, 7}, with no early
stopping, then refit on the whole training half. The tree counts are read off
one 200-tree fit per (learning rate, depth). riesznet uses the main study's
grid and early stopping, tuned the same way.

  ../../.venv/bin/python lee_schuler.py --reps 0:100 --jobs 9
"""

from __future__ import annotations

import _env  # noqa: F401

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import learners as L
from run import one_step, propensity_alpha

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "lee_schuler"
BASE_SEED = 20250107
N, N_TRAIN, K = 1000, 500, 5
PSI = {"ATE": 29.502, "ATT": 30.786}
LRS = [0.001, 0.01, 0.1, 0.25]
TREES = [10, 30, 50, 75, 100, 150, 200]
DEPTHS = [3, 5, 7]
COV = ["X"]


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def propensity(x):
    return expit(-0.02 * x - x**2 + 4.0 * np.log(x + 0.3) + 1.5)


def mu(a, x):
    return 5 * x + 9 * x * a + 5 * np.sin(np.pi * x) + 25 * (a - 2)


def draw(n, rng):
    x = rng.uniform(0, 1, n)
    a = rng.binomial(1, propensity(x)).astype(float)
    y = mu(a, x) + rng.normal(0, 1, n)
    return pd.DataFrame({"A": a, "X": x, "Y": y})


def true_alpha(estimand, a, x):
    pi = propensity(x)
    return a / pi - (1 - a) / (1 - pi) if estimand == "ATE" else a - (1 - a) * pi / (1 - pi)


def check_truth(n=4_000_000, seed=0):
    x = np.random.default_rng(seed).uniform(0, 1, n)
    tau, pi = mu(1, x) - mu(0, x), propensity(x)
    return {"ATE": tau.mean(), "ATT": (pi * tau).mean() / pi.mean()}


# ------------------------------------------------ boosted learners ---
# Each returns predictions on `test` at every tree count in TREES.

def _outcome_path(train, test, lr, depth):
    import xgboost as xgb
    cols = ["A"] + COV
    m = xgb.XGBRegressor(n_estimators=max(TREES), learning_rate=lr, max_depth=depth,
                         reg_lambda=L.L2_PENALTY, n_jobs=1, random_state=L.SEED)
    m.fit(train[cols], train["Y"])
    Z = test[cols]
    return {M: pd.DataFrame({"mu": m.predict(Z, iteration_range=(0, M)),
                             "mu1": m.predict(Z.assign(A=1.0), iteration_range=(0, M)),
                             "mu0": m.predict(Z.assign(A=0.0), iteration_range=(0, M))})
            for M in TREES}


def _propensity_path(train, test, lr, depth):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=max(TREES), learning_rate=lr, max_depth=depth,
                          reg_lambda=L.L2_PENALTY, n_jobs=1, random_state=L.SEED)
    m.fit(train[COV], train["A"])
    return {M: pd.DataFrame({"pi": m.predict_proba(test[COV], iteration_range=(0, M))[:, 1]})
            for M in TREES}


def _rieszboost_path(estimand, train, test, lr, depth):
    from rieszboost import ATE, ATT, RieszBooster
    est = RieszBooster(estimand={"ATE": ATE, "ATT": ATT}[estimand](treatment="A", covariates=("X",)),
                       n_estimators=max(TREES), learning_rate=lr, max_depth=depth,
                       reg_lambda=L.L2_PENALTY, random_state=L.SEED).fit(train[["A"] + COV])
    Z = test[["A"] + COV]
    p, p1, p0 = (est.predict_path(z, TREES) for z in (Z, Z.assign(A=1.0), Z.assign(A=0.0)))
    return {M: pd.DataFrame({"alpha": p[:, j], "alpha1": p1[:, j], "alpha0": p0[:, j]})
            for j, M in enumerate(TREES)}


def _riesznet(estimand, train, test, learning_rate, weight_decay):
    from riesznet import ATE, ATT, RieszNet
    est = RieszNet(estimand={"ATE": ATE, "ATT": ATT}[estimand](treatment="A", covariates=("X",)),
                   hidden_sizes=(64, 64), activation="elu", learning_rate=learning_rate,
                   weight_decay=weight_decay, epochs=500, batch_size=64,
                   early_stopping_rounds=L.PATIENCE, validation_fraction=L.VALID_FRAC,
                   snapshot_epochs=[], random_state=L.SEED).fit(train[["A"] + COV])
    Z = test[["A"] + COV]
    return pd.DataFrame({"alpha": est.predict(Z), "alpha1": est.predict(Z.assign(A=1.0)),
                         "alpha0": est.predict(Z.assign(A=0.0))})


def _loss(kind, data, preds):
    if kind == "outcome":
        return (data["Y"].to_numpy() - preds["mu"].to_numpy()) ** 2
    if kind == "propensity":
        p = np.clip(preds["pi"].to_numpy(), 1e-12, 1 - 1e-12)
        a = data["A"].to_numpy()
        return -(a * np.log(p) + (1 - a) * np.log(1 - p))
    return L.riesz_loss(kind, data["A"].to_numpy(),
                        *(preds[c].to_numpy() for c in ("alpha", "alpha1", "alpha0")))


def tune_boosted(kind, train, test, folds):
    """5-fold CV over (lr, depth, trees) on train; refit the best on all of
    train and predict test. `kind` is outcome, propensity, ATE or ATT."""
    path = {"outcome": _outcome_path, "propensity": _propensity_path}.get(
        kind, lambda tr, te, lr, d: _rieszboost_path(kind, tr, te, lr, d))
    risk = {}
    for lr, depth in itertools.product(LRS, DEPTHS):
        per_M = {M: [] for M in TREES}
        for k in range(K):
            tr, va = train[folds != k], train[folds == k]
            for M, pr in path(tr, va, lr, depth).items():
                per_M[M].append(_loss(kind, va, pr))
        for M in TREES:
            risk[(lr, depth, M)] = float(np.concatenate(per_M[M]).mean())
    lr, depth, M = min(risk, key=risk.get)
    return path(train, test, lr, depth)[M], dict(learning_rate=lr, max_depth=depth, trees=M)


def tune_riesznet(estimand, train, test, folds):
    risk = {}
    for s in L.RIESZNET_GRID:
        losses = [_loss(estimand, train[folds == k],
                        _riesznet(estimand, train[folds != k], train[folds == k], **s))
                  for k in range(K)]
        risk[json.dumps(s)] = float(np.concatenate(losses).mean())
    best = json.loads(min(risk, key=risk.get))
    return _riesznet(estimand, train, test, **best), best


# ------------------------------------------------------------- run ---

def run_rep(rep: int) -> None:
    path = OUT_DIR / f"rep{rep}.pkl"
    if path.exists():
        return
    t0 = time.perf_counter()
    rng = np.random.default_rng([BASE_SEED, rep])
    d = draw(N, rng)
    train, test = d.iloc[:N_TRAIN].reset_index(drop=True), d.iloc[N_TRAIN:].reset_index(drop=True)
    folds = rng.permutation(np.arange(N_TRAIN) % K)
    a, x = test["A"].to_numpy(), test["X"].to_numpy()

    mu_hat, mu_set = tune_boosted("outcome", train, test, folds)
    pi_hat, pi_set = tune_boosted("propensity", train, test, folds)
    settings = {"outcome": mu_set, "propensity": pi_set}
    rows = []
    for estimand in ("ATE", "ATT"):
        a0 = true_alpha(estimand, a, x)
        rb, settings[f"rieszboost_{estimand}"] = tune_boosted(estimand, train, test, folds)
        rn, settings[f"riesznet_{estimand}"] = tune_riesznet(estimand, train, test, folds)
        sources = {"oracle": a0, "propensity": propensity_alpha(estimand, a, pi_hat["pi"].to_numpy())["alpha"].to_numpy(),
                   "rieszboost": rb["alpha"].to_numpy(), "riesznet": rn["alpha"].to_numpy()}
        for method, alpha in sources.items():
            est, se = one_step(estimand, test, mu_hat, alpha)
            rows.append(dict(rep=rep, estimand=estimand, method=method, est=est, se=se,
                             alpha_rmse=float(np.sqrt(np.mean((alpha - a0) ** 2))),
                             alpha_mae=float(np.mean(np.abs(alpha - a0))),
                             product_term=float(-np.mean((alpha - a0) * (mu_hat["mu"].to_numpy() - mu(a, test["X"].to_numpy()))))))
    nuis = dict(rep=rep, mu_rmse=float(np.sqrt(np.mean((mu_hat["mu"].to_numpy() - mu(a, x)) ** 2))),
                pi_rmse=float(np.sqrt(np.mean((pi_hat["pi"].to_numpy() - propensity(x)) ** 2))))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"estimates": pd.DataFrame(rows), "settings": settings, "nuisance": nuis,
                  "seconds": time.perf_counter() - t0}, path)
    print(f"done rep={rep} in {time.perf_counter() - t0:.0f}s", flush=True)


def summarize() -> None:
    cells = [pd.read_pickle(f) for f in sorted(OUT_DIR.glob("rep*.pkl"))]
    e = pd.concat([c["estimates"] for c in cells], ignore_index=True)
    e["err"] = e["est"] - e["estimand"].map(PSI)
    e["cover"] = e["err"].abs() <= 1.96 * e["se"]
    g = e.groupby(["estimand", "method"])
    R = g.size()
    out = pd.DataFrame({
        "reps": R, "bias": g.err.mean(), "bias_mcse": g.err.std(ddof=1) / np.sqrt(R),
        "emp_se": g.est.std(ddof=1), "mean_se": g.se.mean(), "rmse": np.sqrt(g.err.apply(lambda v: (v**2).mean())),
        "coverage": g.cover.mean(), "alpha_rmse": g.alpha_rmse.mean(), "alpha_mae": g.alpha_mae.mean(),
        "product_term": g.product_term.mean()})
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    n = pd.DataFrame([c["nuisance"] for c in cells])
    print(f"\nmu RMSE {n.mu_rmse.mean():.3f}   pi RMSE {n.pi_rmse.mean():.3f}")
    print("\nPaper, Tables 1-3 (500 reps): alpha RMSE ATE 0.920 / indirect 1.402; ATT 0.435 / 0.636")
    print("  ATE: RieszBoost mean 29.522, RMSE 0.187, cov 0.940; indirect 29.539, 0.260, 0.902")
    print("  ATT: RieszBoost mean 30.786, RMSE 0.177, cov 0.950; indirect 30.793, 0.191, 0.942")
    s = pd.DataFrame([{k: json.dumps(v) for k, v in c["settings"].items()} for c in cells])
    for col in s:
        print(f"\n{col}:"); print(s[col].value_counts(normalize=True).head(5).round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:100")
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        print("truth check (MC):", {k: round(v, 3) for k, v in check_truth().items()})
        lo, hi = (int(v) for v in args.reps.split(":"))
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs, batch_size=1)(delayed(run_rep)(r) for r in range(lo, hi))
        summarize()
