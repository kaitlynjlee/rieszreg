"""Write the paper's tables (.tex) and figures (.pdf) from the stored results.

Run with matplotlib available, which the workspace venv does not pin:
  uv run --with matplotlib python outputs.py [--out DIR]
"""

from __future__ import annotations

import _env  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import learners as L
import summaries as S
from run import load_results

HERE = Path(__file__).resolve().parent
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs,
# light mode). Colour follows the method in every display.
COLORS = {"propensity": "#2a78d6", "rieszboost": "#eb6834", "riesznet": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"oracle": "oracle", "propensity": "propensity", "rieszboost": "rieszboost",
          "riesznet": "riesznet"}
HP_NAMES = {"max_depth": "depth", "learning_rate": "learning rate", "weight_decay": "weight decay"}
LEARNER_NAMES = {"outcome": "outcome", "propensity": "propensity",
                 "rieszboost_ATE": "rieszboost, ATE", "rieszboost_ATT": "rieszboost, ATT",
                 "riesznet_ATE": "riesznet, ATE", "riesznet_ATT": "riesznet, ATT"}


def _fmt(x, digits):
    return f"{x:.{digits}f}".replace("-", "$-$") if np.isfinite(x) else "--"


def _cell(x, se, digits):
    return f"{_fmt(x, digits)} ({_fmt(se, digits)})"


# ------------------------------------------------------------ tables ---

def psi_table(perf: pd.DataFrame, estimand: str) -> str:
    rows = []
    for n, g in perf[perf["estimand"] == estimand].groupby("n"):
        for i, m in enumerate(S.METHOD_ORDER):
            r = g[g["method"] == m].iloc[0]
            rel = "1" if m == "oracle" else _cell(r.rel_rmse, r.rel_rmse_mcse, 2)
            rows.append(" & ".join([
                str(n) if i == 0 else "", LABELS[m],
                _cell(r.bias, r.bias_mcse, 3), _cell(r.emp_se, r.emp_se_mcse, 3),
                _cell(r.se_ratio, r.se_ratio_mcse, 2), _cell(r.rmse, r.rmse_mcse, 3),
                rel, _cell(r.coverage, r.coverage_mcse, 3)]) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    head = (r"\begin{tabular}{llcccccc}" "\n" r"\toprule" "\n"
            r"$n$ & $\hat\alpha$ & Bias & EmpSE & SE ratio & RMSE & RMSE / oracle & Coverage \\"
            "\n" r"\midrule")
    return "\n".join([head, *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


def alpha_table(acc: pd.DataFrame) -> str:
    methods = ["propensity", "rieszboost", "riesznet"]
    rows = []
    for estimand, g in acc.groupby("estimand"):
        for i, (n, h) in enumerate(g.groupby("n")):
            cells = [_cell(*h[h.method == m][["alpha_rmse", "mcse"]].iloc[0].to_numpy(), 3)
                     for m in methods]
            rows.append(" & ".join([estimand if i == 0 else "", str(n), *cells]) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    head = (r"\begin{tabular}{llccc}" "\n" r"\toprule" "\n"
            r"Estimand & $n$ & propensity & rieszboost & riesznet \\" "\n" r"\midrule")
    return "\n".join([head, *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


def tuning_table(tuning: pd.DataFrame) -> str:
    """Edge rule on median regret, and the early-stopping cap check, by learner and n."""
    best = S.best_setting_edges(tuning)
    caps = S.cap_check(tuning)
    rows = []
    for learner in L.LEARNERS:
        for i, n in enumerate(sorted(best.n.unique())):
            b = best[(best.learner == learner) & (best.n == n)].iloc[0]
            c = caps[(caps.learner == learner) & (caps.n == n)].iloc[0]
            setting = ", ".join(f"{HP_NAMES[k]} {v:g}" for k, v in json.loads(b.best).items())
            edge = "none" if b.edges == "interior" else ", ".join(
                HP_NAMES[e.split("=")[0]] + " (" + e.split("(")[1] for e in b.edges.split("; "))
            rows.append(" & ".join([LEARNER_NAMES[learner] if i == 0 else "", str(n), setting,
                                    edge, _fmt(c.at_cap, 2)]) + r" \\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    head = (r"\begin{tabular}{llllc}" "\n" r"\toprule" "\n"
            r"Learner & $n$ & Smallest median regret & On an edge & At cap \\" "\n" r"\midrule")
    return "\n".join([head, *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


# ----------------------------------------------------------- figures ---

def alpha_figure(acc: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharex=True)
    for ax, estimand in zip(axes, ["ATE", "ATT"]):
        g = acc[acc.estimand == estimand]
        for m in ["propensity", "rieszboost", "riesznet"]:
            h = g[g.method == m].sort_values("n")
            ax.errorbar(h.n, h.alpha_rmse, yerr=2 * h.mcse, color=COLORS[m], lw=2, marker="o",
                        ms=5, mec="white", mew=1, capsize=0, elinewidth=1.5, label=LABELS[m])
            ax.annotate(LABELS[m], (h.n.iloc[-1], h.alpha_rmse.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([500, 1000, 2000], ["500", "1000", "2000"])
        ax.minorticks_off()
        ax.set_xlim(420, 3600)
        ax.set_title(estimand, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("$n$")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        yt = np.round(np.geomspace(g.alpha_rmse.min() * 0.9, g.alpha_rmse.max() * 1.1, 5), 2)
        ax.set_yticks(yt, [f"{v:g}" for v in yt])
    axes[0].set_ylabel(r"mean RMSE of $\hat\alpha$")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_all(out: Path, dgp_name: str = S.dgp.DEFAULT) -> None:
    r = load_results(dgp_name)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    perf = S.psi_performance(r["estimates"], dgp_name)
    acc = S.alpha_accuracy(r["estimates"])
    (out / "tables" / "psi_ate.tex").write_text(psi_table(perf, "ATE"))
    (out / "tables" / "psi_att.tex").write_text(psi_table(perf, "ATT"))
    (out / "tables" / "alpha.tex").write_text(alpha_table(acc))
    (out / "tables" / "tuning.tex").write_text(tuning_table(r["tuning"]))
    alpha_figure(acc, out / "figures" / "alpha_rmse.pdf")
    reps = r["estimates"].groupby("n")["rep"].nunique().to_dict()
    failed = int(r["tuning"]["failed"].sum())
    (out / "tables" / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\nrepsN{chr(64 + i)}}}{{{reps[n]}}}\n"
                for i, n in enumerate(sorted(reps), 1))
        + f"\\newcommand{{\\nfailed}}{{{failed}}}\n")
    print(f"wrote tables and figures to {out}; reps per n = {reps}; failed fold fits = {failed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results")
    write_all(ap.parse_args().out)
