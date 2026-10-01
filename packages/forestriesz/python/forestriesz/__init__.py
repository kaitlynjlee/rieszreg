"""forestriesz: random-forest Riesz regression.

Two forest learners:

- ``AugForestRieszRegressor`` — the recommended one. Works for every
  estimand and every built-in loss with no extra setup.
- ``ForestRieszRegressor`` — the reference implementation of ForestRiesz
  (Chernozhukov, Newey, Quintas-Martínez, Syrgkanis, ICML 2022) on EconML's
  GRF, kept for comparison. Supports ATE / ATT / TSM out of the box and
  honest confidence intervals via ``predict_interval``.

    from forestriesz import AugForestRieszRegressor, ATE
    forest = AugForestRieszRegressor(estimand=ATE(treatment="treated"))
    forest.fit(Z)                  # Z: treatment column + covariates
    alpha_hat = forest.predict(Z)

Every estimand, loss and diagnostic from ``rieszreg`` is re-exported here.
"""

from __future__ import annotations

from rieszreg.user_api import *  # noqa: F401,F403
from rieszreg.user_api import __all__ as _shared

from .aug_backend import AugForestRieszBackend
from .aug_estimator import AugForestRieszRegressor
from .backend import ForestRieszBackend
from .diagnostics import ForestDiagnostics
from .estimator import ForestRieszRegressor

__all__ = [
    *_shared,
    "AugForestRieszRegressor",
    "AugForestRieszBackend",
    "ForestRieszRegressor",
    "ForestRieszBackend",
    "ForestDiagnostics",
]
