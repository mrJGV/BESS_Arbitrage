"""The forecast policy's quantile source: the point forecast, shifted.

There is no quantile model. The forecaster answers at tau by shifting its own
point forecast by the tau-quantile of its own past errors, taken over periods
that had already cleared before the gate. Three things have to hold, and each
of them failed at some point while this was built: the shift is causal, it is
bucketed by time of day rather than pooled, and when there are too few
residuals to take a quantile of, the forecaster declines rather than returning
a family of identical vectors.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.forecast.lgbm import ForecastSpec, PriceForecaster
from bess_arb.policy.floor import FloorPolicy
from bess_arb.policy.forecast import ForecastPolicy
from bess_arb.timeline import Regime, to_market_time, utc_index

HOURLY = Regime("hourly", 1.0)
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)

FIRST = dt.date(2023, 1, 1)
LAST = dt.date(2024, 6, 30)

TINY = ForecastSpec(
    lags_days=(2, 3, 4),
    refit_days=60,
    min_train_days=200,
    min_deviation_days=30,
    min_residual_obs=30,
    level_params={"learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 20},
    deviation_params={"learning_rate": 0.1, "num_leaves": 8},
    level_rounds=15,
    deviation_rounds=10,
    validation_days=30,
    early_stopping_rounds=5,
    seed=1,
    threads=1,
)


def _prices() -> pd.Series:
    index = utc_index(FIRST, LAST, HOURLY)
    local = to_market_time(index)
    hour = np.asarray(local.hour, dtype=np.float64)
    doy = np.asarray(local.dayofyear, dtype=np.float64)
    rng = np.random.default_rng(11)
    return pd.Series(
        60.0
        + 40.0 * np.sin((hour - 4.0) * np.pi / 12.0)
        + 10.0 * np.sin(doy * 2.0 * np.pi / 365.0)
        + rng.normal(0.0, 10.0, len(index)),
        index=index,
        name="price_eur_mwh",
    )


def _exog(prices: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    return pd.DataFrame(
        {
            "demand_forecast_mw": 25000.0 + rng.normal(0.0, 2000.0, len(prices)),
            "wind_forecast_mw": 6000.0 + rng.normal(0.0, 2500.0, len(prices)),
            "solar_forecast_mw": 4000.0 + rng.normal(0.0, 2000.0, len(prices)),
        },
        index=prices.index,
    )


def _forecaster(prices: pd.Series) -> PriceForecaster:
    return PriceForecaster(prices, _exog(prices), HOURLY, HOURLY, TINY)


def _window(day: dt.date) -> pd.DatetimeIndex:
    return utc_index(day, day + dt.timedelta(days=1), HOURLY)


def _warm(forecaster: PriceForecaster, upto: dt.date, days: int) -> None:
    """Walk the backtest forwards so the residual buffer fills causally."""
    for offset in range(days, 0, -1):
        day = upto - dt.timedelta(days=offset)
        forecaster.forecast(day, _window(day))


def test_a_cold_forecaster_declines_rather_than_returning_a_flat_family() -> None:
    """The residual buffer fills only as the backtest asks for forecasts.

    On the opening days there is nothing to take a quantile of. Shifting by
    nothing would return the *same* vector at every tau, so the K scenario
    solves would be K copies of one solve and every curve would collapse to a
    single step — the degenerate case, arrived at from the other direction.
    Declining hands the day to the floor, which has years of history.
    """
    prices = _prices()
    forecaster = _forecaster(prices)
    day = dt.date(2023, 12, 1)

    assert forecaster.forecast(day, _window(day)) is not None
    assert forecaster.forecast_quantile(day, _window(day), 0.9) is None


def test_a_warmed_forecaster_answers_and_the_family_is_monotone() -> None:
    prices = _prices()
    forecaster = _forecaster(prices)
    day = dt.date(2024, 3, 1)
    _warm(forecaster, day, 40)

    window = _window(day)
    family = np.array([forecaster.forecast_quantile(day, window, t) for t in TAUS])

    assert not np.isnan(family).any()
    assert np.all(np.diff(family, axis=0) >= -1e-9)


def test_the_shift_is_not_a_single_scalar_across_the_day() -> None:
    """Bucketed, not pooled, and this is the test that says so.

    A pooled quantile adds one number to every period, so each scenario is a
    parallel copy of the same day. The optimiser trades on differences between
    periods and a constant cancels in every difference, so all K solves return
    the identical dispatch and every curve collapses. Bucketing by time of day
    makes the spread fan out, which is what puts steps in the curve at all.
    """
    prices = _prices()
    forecaster = _forecaster(prices)
    day = dt.date(2024, 3, 1)
    _warm(forecaster, day, 40)

    window = _window(day)
    low = forecaster.forecast_quantile(day, window, 0.1)
    high = forecaster.forecast_quantile(day, window, 0.9)
    assert low is not None and high is not None

    width = high - low
    assert width.std() > 1e-6, "the shift was constant across the whole window"


def test_the_residual_shift_reads_nothing_after_the_gate() -> None:
    """Invariant 1. A residual needs a price that has already cleared.

    Prices from the gate onwards are replaced with an absurd value. If any of
    them reached the residual buffer, the shift would move.
    """
    prices = _prices()
    day = dt.date(2024, 3, 1)
    gate = pd.Timestamp(
        dt.datetime.combine(day - dt.timedelta(days=1), dt.time(12)),
        tz="Europe/Madrid",
    ).tz_convert("UTC")

    honest = _forecaster(prices)
    _warm(honest, day, 40)

    poisoned_prices = prices.copy()
    poisoned_prices.loc[poisoned_prices.index >= gate] = 9999.0
    poisoned = _forecaster(poisoned_prices)
    _warm(poisoned, day, 40)

    window = _window(day)
    a = honest.forecast_quantile(day, window, 0.9)
    b = poisoned.forecast_quantile(day, window, 0.9)
    assert a is not None and b is not None
    assert a == pytest.approx(b)


def test_the_ladder_is_reported() -> None:
    """Bucket, pooled fallback and declined, counted like every other rung."""
    prices = _prices()
    forecaster = _forecaster(prices)
    day = dt.date(2024, 3, 1)
    _warm(forecaster, day, 40)
    forecaster.forecast_quantile(day, _window(day), 0.5)

    diagnostics = forecaster.diagnostics()

    assert "residual_bucket" in diagnostics
    assert "residual_pooled" in diagnostics
    assert "residual_declined" in diagnostics


def test_the_policy_falls_back_to_the_floor_when_the_forecaster_declines() -> None:
    """The whole point of declining: the floor answers instead.

    And it must be the *same* floor the chart plots against, not a second
    climatology living inside the forecast policy.
    """
    prices = _prices()
    floor = FloorPolicy(prices)
    policy = ForecastPolicy(_forecaster(prices), FloorPolicy(prices))
    day = dt.date(2023, 12, 1)
    window = _window(day)

    believed = policy.prices_for_quantile(day, window, 0.9)

    assert believed == pytest.approx(floor.prices_for_quantile(day, window, 0.9))
    assert policy.quantile_diagnostics()["fallback_calls"] == 1
    assert policy.quantile_diagnostics()["forecast_calls"] == 0


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.5, 2.0])
def test_a_tau_outside_the_open_unit_interval_is_refused(tau: float) -> None:
    prices = _prices()
    forecaster = _forecaster(prices)
    day = dt.date(2024, 3, 1)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        forecaster.forecast_quantile(day, _window(day), tau)
