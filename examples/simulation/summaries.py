"""Every table and figure, each a summary of the stored per-cell results.

Tables are returned as DataFrames, with Monte Carlo SEs (MCSEs) next to each
performance measure. `python summaries.py` prints them all.
"""

from __future__ import annotations

import _env  # noqa: F401

import json

import numpy as np
import pandas as pd

import dgp
import learners as L
from run import load_results

METHOD_ORDER = ["oracle", "propensity", "rieszboost", "riesznet"]
# Edges that are hard limits of the hyperparameter are exempt from the edge
# rule: the grid cannot extend past them.
HARD_LIMITS = {("weight_decay", 0.0), ("max_depth", 1)}


def true_values(dgp_name: str = dgp.DEFAULT) -> dict[str, float]:
    return {e: dgp.DGPS[dgp_name].true_psi(e)[0] for e in ("ATE", "ATT")}


# ---------------------------------------------------------- tuning ---

def edge_check(tuning: pd.DataFrame) -> pd.DataFrame:
    """For each learner type and tuned hyperparameter, the share of fits whose
    selected value is the smallest or the largest in the grid."""
    sel = tuning[tuning["selected"] == True]  # noqa: E712
    rows = []
    for learner, g in sel.groupby("learner"):
        grid = L.LEARNERS[learner][2]
        for hp in grid[0]:
            values = sorted({s[hp] for s in grid})
            chosen = g["setting"].map(lambda s: json.loads(s)[hp])
            for n, c in chosen.groupby(g["n"]):
                lo, hi = values[0], values[-1]
                rows.append(dict(
                    learner=learner, hyperparameter=hp, n=n, grid=str(values),
                    at_smallest=(c == lo).mean(), at_largest=(c == hi).mean(),
                    smallest_exempt=(hp, lo) in HARD_LIMITS,
                    modal=c.mode().iloc[0], fits=len(c)))
    return pd.DataFrame(rows)


def regret_profile(tuning: pd.DataFrame) -> pd.DataFrame:
    """Median, over replicates, of each setting's cross-validated regret: its
    risk minus the smallest risk in its grid on the same dataset. The median
    shows the typical fit. The mean is dominated by rare fits that diverge,
    which cross-validation rejects anyway."""
    t = tuning.assign(regret=tuning["cv_risk"]
                      - tuning.groupby(["n", "rep", "learner"])["cv_risk"].transform("min"))
    return (t.groupby(["learner", "n", "setting"])["regret"]
            .agg(median="median", mean="mean").reset_index())


def best_setting_edges(tuning: pd.DataFrame) -> pd.DataFrame:
    """The edge rule on the regret profile: for each learner type and n, the
    setting with the smallest median regret, and whether any of its tuned
    hyperparameters sits at the smallest or largest grid value."""
    prof = regret_profile(tuning)
    rows = []
    for (learner, n), g in prof.groupby(["learner", "n"]):
        best = json.loads(g.loc[g["median"].idxmin(), "setting"])
        grid = L.LEARNERS[learner][2]
        edges = []
        for hp, v in best.items():
            values = sorted({s[hp] for s in grid})
            if v in (values[0], values[-1]) and (hp, v) not in HARD_LIMITS:
                edges.append(f"{hp}={v:g} ({'smallest' if v == values[0] else 'largest'})")
        rows.append(dict(learner=learner, n=n, best=json.dumps(best),
                         edges="; ".join(edges) or "interior"))
    return pd.DataFrame(rows)


def cap_check(tuning: pd.DataFrame, frac: float = 0.9) -> pd.DataFrame:
    """Share of selected fits in which early stopping kept at least `frac` of
    the iteration cap in some cross-fitting fold."""
    sel = tuning[tuning["selected"] == True]  # noqa: E712
    return (sel.assign(at_cap=sel["iters_max"] >= frac * sel["cap"])
            .groupby(["learner", "n"])
            .agg(at_cap=("at_cap", "mean"), median_iters=("iters_mean", "median"),
                 cap=("cap", "first"))
            .reset_index())


def selection_table(tuning: pd.DataFrame) -> pd.DataFrame:
    """How often each setting is selected, by learner type and n."""
    sel = tuning[tuning["selected"] == True]  # noqa: E712
    return (sel.groupby(["learner", "n"])["setting"].value_counts(normalize=True)
            .rename("share").reset_index())


# ----------------------------------------------------- performance ---

def alpha_accuracy(est: pd.DataFrame) -> pd.DataFrame:
    """RMSE of alpha-hat against alpha_0 within each dataset, averaged over
    datasets. The oracle has zero error by construction and is left out."""
    e = est[est["method"] != "oracle"]
    g = e.groupby(["estimand", "n", "method"])["alpha_rmse"]
    return pd.DataFrame({"alpha_rmse": g.mean(), "mcse": g.std(ddof=1) / np.sqrt(g.count()),
                         "reps": g.count()}).reset_index()


def psi_performance(est: pd.DataFrame, dgp_name: str = dgp.DEFAULT, n_boot: int = 500,
                    seed: int = 0) -> pd.DataFrame:
    """Bias, EmpSE, ModSE/EmpSE, RMSE, RMSE relative to the oracle, and
    coverage of the one-step estimator, each with its MCSE."""
    psi0 = true_values(dgp_name)
    rng = np.random.default_rng(seed)
    rows = []
    for (estimand, n), g in est.groupby(["estimand", "n"]):
        theta = psi0[estimand]
        wide_err = g.pivot(index="rep", columns="method", values="est") - theta
        for method in METHOD_ORDER:
            m = g[g["method"] == method]
            failed = int(m["est"].isna().sum())
            m = m.dropna(subset=["est", "se"])
            R = len(m)
            err = m["est"].to_numpy() - theta
            se = m["se"].to_numpy()
            emp = err.std(ddof=1)
            mod = np.sqrt(np.mean(se**2))
            mse = np.mean(err**2)
            cover = np.abs(err) <= 1.96 * se
            mcse_emp = emp / np.sqrt(2 * (R - 1))
            mcse_mod = np.std(se**2, ddof=1) / np.sqrt(R) / (2 * mod)
            ratio = mod / emp
            # RMSE relative to the oracle, paired on rep, bootstrapped over reps
            paired = wide_err[[method, "oracle"]].dropna().to_numpy()
            rel = np.sqrt(np.mean(paired[:, 0] ** 2) / np.mean(paired[:, 1] ** 2))
            idx = rng.integers(0, len(paired), size=(n_boot, len(paired)))
            boot = np.sqrt((paired[idx, 0] ** 2).mean(1) / (paired[idx, 1] ** 2).mean(1))
            rows.append(dict(
                estimand=estimand, n=n, method=method, reps=R, failed=failed,
                bias=err.mean(), bias_mcse=emp / np.sqrt(R),
                emp_se=emp, emp_se_mcse=mcse_emp,
                se_ratio=ratio, se_ratio_mcse=ratio * np.hypot(mcse_mod / mod, mcse_emp / emp),
                rmse=np.sqrt(mse), rmse_mcse=np.std(err**2, ddof=1) / np.sqrt(R) / (2 * np.sqrt(mse)),
                rel_rmse=rel, rel_rmse_mcse=boot.std(ddof=1),
                coverage=cover.mean(), coverage_mcse=np.sqrt(cover.mean() * (1 - cover.mean()) / R),
            ))
    return pd.DataFrame(rows)


def nuisance_accuracy(nuis: pd.DataFrame) -> pd.DataFrame:
    g = nuis.groupby(["nuisance", "n"])["rmse"]
    return pd.DataFrame({"rmse": g.mean(), "mcse": g.std(ddof=1) / np.sqrt(g.count())}).reset_index()


# ----------------------------------------------------------- pilot ---

def pilot_contrasts(est: pd.DataFrame, k: float = 5.0, n_sim_planned: int | None = None,
                    dgp_name: str = dgp.DEFAULT) -> pd.DataFrame:
    """Resolvability of every contrast the mockups promise (see pilot_check.py)."""
    from pilot_check import _pilot_verdict, pilot_contrast
    psi0 = true_values(dgp_name)
    # pilot_check keys true values by a DGP column; here the estimand plays that role
    e = est.assign(DGP=est["estimand"])
    out = []
    pairs = [("rieszboost", "riesznet"), ("rieszboost", "propensity"), ("riesznet", "propensity"),
             ("rieszboost", "oracle"), ("riesznet", "oracle"), ("propensity", "oracle")]
    for a, b in pairs:
        out.append(pilot_contrast(e, "mse", a, b, psi0, k=k, n_sim_planned=n_sim_planned,
                                  by=["DGP", "n"], method="method").assign(display="psi MSE"))
    for m in METHOD_ORDER:
        out.append(pilot_contrast(e, "coverage", m, None, psi0, k=k, n_sim_planned=n_sim_planned,
                                  by=["DGP", "n"], method="method").assign(display="coverage vs 0.95"))
        out.append(pilot_contrast(e, "bias", m, None, psi0, k=k, n_sim_planned=n_sim_planned,
                                  by=["DGP", "n"], method="method").assign(display="bias vs 0"))
    # alpha-hat RMSE contrasts: d_i is the per-dataset difference in RMSE
    for (estimand, n), g in est.groupby(["estimand", "n"]):
        w = g.pivot(index="rep", columns="method", values="alpha_rmse")
        for a, b in pairs[:3]:
            d = (w[a] - w[b]).dropna().to_numpy()
            out.append(pd.DataFrame([dict(DGP=estimand, n=n, display="alpha RMSE",
                                          **_pilot_verdict(d, "alpha_rmse", a, b, k, n_sim_planned))]))
    res = pd.concat(out, ignore_index=True).rename(columns={"DGP": "estimand"})
    return res


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else dgp.DEFAULT
    pd.set_option("display.width", 250)
    r = load_results(name)
    print(f"DGP: {name}")
    print(f"cells: {r['estimates'][['n', 'rep']].drop_duplicates().groupby('n').size().to_dict()}")
    print("\n== edge check ==")
    print(edge_check(r["tuning"]).round(2).to_string(index=False))
    print("\n== edge rule on median regret ==")
    print(best_setting_edges(r["tuning"]).to_string(index=False))
    print("\n== early-stopping cap check ==")
    print(cap_check(r["tuning"]).round(2).to_string(index=False))
    print("\n== nuisance RMSE ==")
    print(nuisance_accuracy(r["nuisance"]).round(3).to_string(index=False))
    print("\n== alpha-hat RMSE ==")
    print(alpha_accuracy(r["estimates"]).round(3).to_string(index=False))
    print("\n== psi-hat performance ==")
    print(psi_performance(r["estimates"], name).round(3).to_string(index=False))
