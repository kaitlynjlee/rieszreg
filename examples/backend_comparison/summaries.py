"""The summaries behind each display in backend_comparison.ipynb.

Every function reads the tidy tables from simulation.run_simulation(), so
each display can be recomputed from results.csv and tuning.csv without
refitting. Performance measures and their Monte Carlo SEs follow Morris,
White & Crowther (2019), Stat Med 38(11), Table 6.
"""

import numpy as np
import pandas as pd

from estimands import ESTIMANDS
from learners import OUTCOME_LEARNER, RIESZ_LEARNERS
from simulation import true_values

CELL = ["dgp", "n", "estimand", "learner"]
Z_95 = 1.959964


# ---------------------------------------------------------------- DGP diagnostics

def describe_dgp(dgp, n=200_000, seed=0):
    """Diagnostics of one DGP on a single large draw: treatment prevalence,
    overlap, signal-to-noise, how nonlinear mu is, the size of the true
    representers, and each psi_0 with the Monte Carlo SE of its computation."""
    Y, A, X = dgp.sample(n, np.random.default_rng(seed))
    pi = dgp.propensity(X)
    mu = dgp.outcome_regression(A, X)
    design = np.column_stack([np.ones(n), A, X])
    mu_linear = design @ np.linalg.lstsq(design, mu, rcond=None)[0]
    row = {
        "P(A = 1)": A.mean(),
        "pi(X), 1st percentile": np.quantile(pi, 0.01),
        "pi(X), 99th percentile": np.quantile(pi, 0.99),
        "share of pi(X) at a clip bound": np.mean((pi <= dgp.pi_clip[0]) | (pi >= dgp.pi_clip[1])),
        "signal-to-noise, Var(mu) / Var(Y)": mu.var() / Y.var(),
        "linear share of Var(mu)": mu_linear.var() / mu.var(),
    }
    for name, estimand in ESTIMANDS.items():
        alpha = estimand.true_representer(dgp, A, X)
        psi, mc_se = true_values(dgp.name)[name]
        row |= {f"{name}: psi_0": psi, f"{name}: MC SE of psi_0": mc_se,
                f"{name}: sd of alpha_0": alpha.std(), f"{name}: max |alpha_0|": np.abs(alpha).max()}
    return pd.Series(row, name=dgp.name)


# ---------------------------------------------------------------- tuning checks

def edge_check(tuning):
    """The edge rule. For each learner and tuned hyperparameter, the share of
    fits whose selected value is the smallest or the largest in its grid. A
    consistent edge choice means the grid should move in that direction. In a
    two-point grid every value is an edge, so the check cannot pass."""
    grids = {OUTCOME_LEARNER.name: OUTCOME_LEARNER.grid} | {name: l.grid for name, l in RIESZ_LEARNERS.items()}
    rows = []
    for (dgp, estimand, learner), s in tuning[tuning.selected].groupby(["dgp", "estimand", "learner"]):
        for param, values in grids[learner].items():
            param = param.removeprefix("estimator__")
            rows.append({"dgp": dgp, "estimand": estimand, "learner": learner, "hyperparameter": param,
                         "grid": values, "n_fits": len(s),
                         "at_smallest": np.mean(s[param] == min(values)),
                         "at_largest": np.mean(s[param] == max(values))})
    return pd.DataFrame(rows).set_index(["dgp", "estimand", "learner", "hyperparameter"])


def early_stopping_check(tuning):
    """Early stopping acts as a grid over the number of trees or epochs. If
    the kept iteration keeps reaching the cap, the learner wants more
    iterations; raise the learning rate rather than the cap."""
    s = tuning[tuning.selected].assign(frac=lambda d: d.best_iteration / d.iteration_cap)
    return s.groupby(["dgp", "estimand", "learner"]).agg(
        n_fits=("frac", "size"),
        cap=("iteration_cap", "max"),
        median_iteration=("best_iteration", "median"),
        max_iteration=("best_iteration", "max"),
        share_above_90pct_of_cap=("frac", lambda f: np.mean(f >= 0.9)),
        median_fold_seconds=("seconds", "median"),
    )


# ---------------------------------------------------------------- performance

def estimator_performance(results):
    """Performance of psi_hat per (dgp, n, estimand, learner), each measure
    with its Monte Carlo SE (columns ending in _mcse).

    n_sim     replicates with an estimate; n_failed, replicates without one
    bias      mean of psi_hat - psi_0
    emp_se    SD of psi_hat across replicates (the true sampling SE)
    mod_se    root mean of the estimated variances (the method's own SE)
    rmse      root mean squared error of psi_hat
    coverage  share of 95% Wald intervals psi_hat +/- 1.96 se containing psi_0
    """
    def measures(g):
        n_failed = int(g.est.isna().sum())
        g = g.dropna(subset=["est", "se"])
        m = len(g)
        err = g.est - g.truth
        emp_se, mod_se, mse = g.est.std(ddof=1), np.sqrt(np.mean(g.se ** 2)), np.mean(err ** 2)
        coverage = np.mean(err.abs() <= Z_95 * g.se)
        return pd.Series({
            "n_sim": m, "n_failed": n_failed,
            "bias": err.mean(), "bias_mcse": emp_se / np.sqrt(m),
            "emp_se": emp_se, "emp_se_mcse": emp_se / np.sqrt(2 * (m - 1)),
            "mod_se": mod_se, "mod_se_mcse": np.sqrt(np.var(g.se ** 2, ddof=1) / (4 * m * mod_se ** 2)),
            "rmse": np.sqrt(mse), "rmse_mcse": np.std(err ** 2, ddof=1) / np.sqrt(m) / (2 * np.sqrt(mse)),
            "coverage": coverage, "coverage_mcse": np.sqrt(coverage * (1 - coverage) / m),
        })
    return results.groupby(CELL, sort=False).apply(measures, include_groups=False)


def nuisance_accuracy(results):
    """Mean error of the cross-fitted nuisances against the truth, per
    (dgp, n, estimand, learner), with Monte Carlo SEs. mu_rmse is the same
    for every Riesz learner, since they share the outcome regression."""
    g = results.dropna(subset=["alpha_rmse"]).groupby(CELL, sort=False)
    return pd.DataFrame({
        "alpha_rmse": g.alpha_rmse.mean(),
        "alpha_rmse_mcse": g.alpha_rmse.std(ddof=1) / np.sqrt(g.size()),
        "alpha_mae": g.alpha_mae.mean(),
        "mu_rmse": g.mu_rmse.mean(),
        "minutes_to_fit_alpha": g.fit_seconds.mean() / 60,
    })


def with_mcse(table, digits=3):
    """Merges each `x` column with its `x_mcse` column into "x (mcse)"."""
    out = table.copy()
    for col in [c for c in table if f"{c}_mcse" in table]:
        out[col] = [f"{v:.{digits}f} ({s:.{digits}f})" for v, s in zip(table[col], table[f"{col}_mcse"])]
        out = out.drop(columns=f"{col}_mcse")
    return out


# ---------------------------------------------------------------- paired contrasts

PAIRED_MEASURES = {
    # measure: per-replicate value whose mean is the displayed quantity
    "bias":       lambda r: r.est - r.truth,
    "MSE":        lambda r: (r.est - r.truth) ** 2,
    "coverage":   lambda r: ((r.est - r.truth).abs() <= Z_95 * r.se).astype(float),
    "alpha RMSE": lambda r: r.alpha_rmse,
}


def paired_contrasts(results, a="xgboost", b="riesznet", k=5):
    """Learner `a` minus learner `b`, paired on replicate. Both learners see
    the same datasets and the same mu_hat, so each difference d is formed
    within a replicate.

    For each measure: the mean of d (the gap between the two learners), its
    Monte Carlo SE sd(d) / sqrt(n_sim), and z = |gap| / MCSE. n_sim_needed
    is how many replicates would show the gap at k Monte Carlo SEs, sized on
    the lower 95% bound of |gap| so a lucky pilot does not undersize the run.
    When z < 2 the run cannot tell the gap from zero, and n_sim_needed is
    noise.
    """
    rows = []
    for measure, value in PAIRED_MEASURES.items():
        wide = (results.assign(value=value).dropna(subset=["value"])
                .pivot_table(index=["dgp", "n", "estimand", "rep"], columns="learner", values="value"))
        d = (wide[a] - wide[b]).dropna()
        for (dgp, n, estimand), dk in d.groupby(level=["dgp", "n", "estimand"]):
            gap, sd = dk.mean(), dk.std(ddof=1)
            mcse = sd / np.sqrt(len(dk))
            lower = max(0.0, abs(gap) - Z_95 * mcse)
            if sd == 0:   # d never varied (e.g. both learners always covered): its spread is unknown, not zero
                z = needed = np.nan
            else:
                z, needed = abs(gap) / mcse, np.ceil((k * sd / lower) ** 2) if lower > 0 else np.inf
            rows.append({"dgp": dgp, "n": n, "estimand": estimand, "measure": measure, "n_sim": len(dk),
                         "gap": gap, "gap_mcse": mcse, "z": z, "n_sim_needed": needed})
    return pd.DataFrame(rows).set_index(["dgp", "n", "estimand", "measure"])


# ---------------------------------------------------------------- figure

LEARNER_STYLE = {"xgboost": dict(color="#2a78d6", marker="o"), "riesznet": dict(color="#eb6834", marker="s")}


def plot_errors(results):
    """psi_hat - psi_0 in every replicate, one panel per (dgp, estimand).
    Grey lines join the two learners' estimates on the same dataset. The
    black bar is the mean error (the bias) +/- 1.96 Monte Carlo SEs."""
    import matplotlib.pyplot as plt

    panels = list(results.groupby(["dgp", "estimand"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4), sharey=True, squeeze=False)
    learners = [l for l in RIESZ_LEARNERS if l in set(results.learner)]
    for ax, ((dgp, estimand), g) in zip(axes[0], panels):
        err = g.assign(err=g.est - g.truth).pivot_table(index="rep", columns="learner", values="err")[learners]
        ax.plot(range(len(learners)), err.T.to_numpy(), color="#c3c2b7", linewidth=0.8, zorder=1)
        for x, learner in enumerate(learners):
            e = err[learner].dropna()
            ax.scatter(np.full(len(e), x), e, s=24, zorder=2, edgecolor="white", linewidth=0.8,
                       **LEARNER_STYLE.get(learner, {}))
            half = Z_95 * e.std(ddof=1) / np.sqrt(len(e))
            ax.errorbar(x + 0.18, e.mean(), yerr=half, fmt="o", markersize=4, color="black",
                        capsize=3, linewidth=1.5, zorder=3)
        ax.axhline(0, color="#6b6a64", linewidth=1, linestyle="--", zorder=0)
        ax.set_xticks(range(len(learners)), learners)
        ax.set_xlim(-0.5, len(learners) - 0.3)
        ax.set_title(f"{dgp} DGP, {estimand}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8e7e1", linewidth=0.6)
    axes[0][0].set_ylabel(r"$\hat\psi - \psi_0$")
    fig.tight_layout()
    return fig
