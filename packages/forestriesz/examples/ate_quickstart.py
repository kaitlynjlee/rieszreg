"""Quickstart: ATE on a synthetic binary-treatment DGP.

Run from the repo root: ``.venv/bin/python forestriesz/examples/ate_quickstart.py``

Demonstrates the locally linear sieve fit, which is the right configuration
for ATE / ATT and other difference-style estimands. The forest splits on
covariates only; the [1{T=0}, 1{T=1}] sieve resolves treatment within each
leaf, recovering the inverse propensity representer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forestriesz import ATE, ForestRieszRegressor
from forestriesz.feature_fns import default_riesz_features


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(-0.02 * x - x ** 2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    estimand = ATE(treatment="a", covariates=("x",))
    fr = ForestRieszRegressor(
        estimand=estimand,
        riesz_feature_fns=default_riesz_features(estimand),
        n_estimators=500,
        min_samples_leaf=10,
        max_samples=0.5,
        random_state=0,
    )
    fr.fit(df)
    alpha_hat = fr.predict(df)
    truth = a / pi - (1 - a) / (1 - pi)

    print(f"alpha_hat range      : [{alpha_hat.min():.3f}, {alpha_hat.max():.3f}]")
    print(f"truth     range      : [{truth.min():.3f}, {truth.max():.3f}]")
    print(f"correlation w/ truth : {np.corrcoef(alpha_hat, truth)[0, 1]:.3f}")
    print(f"RMSE vs truth        : {float(np.sqrt(np.mean((alpha_hat - truth) ** 2))):.3f}")
    print()
    print(fr.diagnose(df).summary())


if __name__ == "__main__":
    main()
