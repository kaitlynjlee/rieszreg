"""rieszboost: gradient boosting for Riesz representers.

`RieszBooster` is a sklearn-style estimator. Tell it what to estimate
(the estimand), fit it on the treatment + covariate columns, and predict
the Riesz representer α̂ for each row:

    from rieszboost import RieszBooster, ATE
    booster = RieszBooster(estimand=ATE(treatment="treated"))
    booster.fit(Z)                 # Z: treatment column + covariates
    alpha_hat = booster.predict(Z)

Every estimand, loss and diagnostic from `rieszreg` is re-exported here.
"""

from rieszreg.user_api import *  # noqa: F401,F403
from rieszreg.user_api import __all__ as _shared
from rieszreg.backends import register_predictor_loader as _register

# So `RieszEstimator.load` works after `import rieszboost` alone.
_register("xgboost", "rieszboost.backends.xgboost:XGBoostPredictor.load")
_register("sklearn", "rieszboost.backends.sklearn:SklearnPredictor.load")

__all__ = [*_shared, "RieszBooster", "XGBoostBackend", "SklearnBackend"]


# xgboost loads lazily, so `import rieszboost` stays cheap.
_LAZY = {
    "RieszBooster": ("estimator", "RieszBooster"),
    "XGBoostBackend": ("backends", "XGBoostBackend"),
    "SklearnBackend": ("backends", "SklearnBackend"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        from importlib import import_module
        return getattr(import_module(f"{__name__}.{mod_name}"), attr)
    raise AttributeError(f"module 'rieszboost' has no attribute {name!r}")
