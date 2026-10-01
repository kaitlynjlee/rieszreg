"""riesztree: single-tree Riesz regression.

`RieszTreeRegressor` fits one decision tree to the Riesz representer α̂ of
a causal estimand. The tree is easy to inspect (``get_n_leaves()``,
``get_depth()``, ``feature_importances_``) and works with every built-in
loss.

    from riesztree import RieszTreeRegressor, ATE
    tree = RieszTreeRegressor(estimand=ATE(treatment="treated"), max_depth=4)
    tree.fit(Z)                  # Z: treatment column + covariates
    alpha_hat = tree.predict(Z)

Every estimand, loss and diagnostic from ``rieszreg`` is re-exported here.
Importing this module registers the predictor loader for
``rieszreg.RieszEstimator.load``.
"""

from __future__ import annotations

from rieszreg.user_api import *  # noqa: F401,F403
from rieszreg.user_api import __all__ as _shared

from .backend import RieszTreeBackend
from .diagnostics import TreeDiagnostics
from .estimator import RieszTreeRegressor
from .predictor import RieszTreePredictor  # noqa: F401  (registers the loader)

__all__ = [*_shared, "RieszTreeRegressor", "RieszTreeBackend", "TreeDiagnostics"]
