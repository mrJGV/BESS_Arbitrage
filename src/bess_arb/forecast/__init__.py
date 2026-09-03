"""The point forecaster: features, their publication instants, and LightGBM.

Kept out of :mod:`bess_arb.policy` on purpose. A policy is a price vector
(CLAUDE.md invariant 2) and the package that says so is eighty lines long;
putting a gradient-boosting trainer inside it would make that claim harder to
read than the code stating it. :class:`bess_arb.policy.forecast.ForecastPolicy`
is the thin adapter, and it lives with the other policies.

LightGBM is imported in :mod:`bess_arb.forecast.lgbm` and nowhere else, on the
same reasoning as the solver boundary: an ``import-linter`` contract and
``tests/test_import_boundary.py`` keep it there, and :func:`build_forecaster`
imports that module inside the function so a run that only wants
:func:`build_features` never pays for the library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bess_arb.config import Config
from bess_arb.forecast.features import FeatureTable, build_features
from bess_arb.series import load_exogenous, load_price_history, regime_of
from bess_arb.timeline import Regime

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from bess_arb.forecast.lgbm import PriceForecaster

__all__ = [
    "FeatureTable",
    "build_features",
    "build_forecaster",
]


def build_forecaster(config: Config, regime_name: str) -> PriceForecaster:
    """The forecaster for one regime, wired from the config and the snapshot.

    Build it **once per regime** and reuse it across the degradation sweep:
    the forecast does not depend on ``c_deg``, so refitting per sweep point
    triples the cost of a run for numbers that are identical by construction.
    """
    from bess_arb.forecast.lgbm import ForecastSpec, PriceForecaster

    spec = config.forecast
    return PriceForecaster(
        prices=load_price_history(config, regime_name),
        exog=load_exogenous(config, regime_name),
        regime=regime_of(config, regime_name),
        level_regime=level_regime(config),
        spec=ForecastSpec(
            lags_days=spec.lags_days,
            refit_days=spec.refit_days,
            min_train_days=spec.min_train_days,
            min_deviation_days=spec.min_deviation_days,
            level_params=dict(spec.level),
            deviation_params=dict(spec.deviation),
            level_rounds=spec.level_rounds,
            deviation_rounds=spec.deviation_rounds,
            validation_days=spec.validation_days,
            early_stopping_rounds=spec.early_stopping_rounds,
            seed=config.seed,
            threads=spec.threads,
        ),
    )


def level_regime(config: Config) -> Regime:
    """The grid the level stage works on: the coarsest the config declares.

    That is the grid on which the whole history exists and on which the
    published forecasts are still issued, so it is where the long-horizon
    signal lives. Derived from the configured regimes rather than written as
    "hourly", so the day a third regime appears this does not quietly keep
    pointing at the wrong one.
    """
    return max((window.regime for window in config.data.regimes), key=lambda r: r.dt_h)
