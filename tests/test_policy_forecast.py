"""The forecast policy: a price vector, and the floor when it has none.

What is checked here is the *adapter*, not the model. Whether the forecast is
any good is a question for the backtest; whether the policy is still a policy —
one method, one vector, no constraint of its own, and a fallback that is the
project's own null hypothesis rather than a second one — is a question about
structure, and it is the one CLAUDE.md invariant 2 rests on.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.policy import POLICY_NAMES, ForecastPolicy, PricePolicy, build_policy
from bess_arb.policy.floor import FloorPolicy
from bess_arb.timeline import Regime, to_market_time, utc_index

QUARTER_HOURLY = Regime("quarter_hourly", 0.25)

FIRST_DAY = dt.date(2025, 1, 1)
LAST_DAY = dt.date(2025, 3, 31)


def _prices() -> pd.Series:
    index = utc_index(FIRST_DAY, LAST_DAY, QUARTER_HOURLY)
    local = to_market_time(index)
    hour = np.asarray(local.hour) + np.asarray(local.minute) / 60.0
    return pd.Series(
        60.0 + 40.0 * np.sin((hour - 4.0) * np.pi / 12.0),
        index=index,
        name="price_eur_mwh",
    )


class _Stub:
    """A forecaster that answers a fixed vector, or declines.

    Standing in for LightGBM on purpose: this file is about what the policy
    does with the two possible answers, and a real fit would make that
    slower to run and harder to read without testing anything more.
    """

    def __init__(self, value: float | None) -> None:
        self._value = value
        self.asked: list[dt.date] = []
        self.asked_quantiles: list[tuple[dt.date, float]] = []

    def forecast(self, day: dt.date, window: pd.DatetimeIndex) -> np.ndarray | None:
        self.asked.append(day)
        if self._value is None:
            return None
        return np.full(len(window), self._value)

    def forecast_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> np.ndarray | None:
        """The point answer, spread around itself, or the same decline.

        Enough to stand in for the residual shift: what the policy does with a
        quantile answer is identical whatever produced it, and what it does
        with a *declined* one is the branch this file exists to pin.
        """
        self.asked_quantiles.append((day, tau))
        if self._value is None:
            return None
        return np.full(len(window), self._value + 10.0 * (tau - 0.5))

    def diagnostics(self) -> dict[str, int]:
        return {"trained": 0, "level_only": 0, "no_history": 0}


def _window(day: dt.date) -> pd.DatetimeIndex:
    return utc_index(day, day + dt.timedelta(days=1), QUARTER_HOURLY)


def test_the_forecast_policy_is_registered_between_the_other_two() -> None:
    """The ladder is the chart: no information, a forecast, perfect foresight."""
    assert POLICY_NAMES == ("floor", "forecast", "oracle")


def test_it_satisfies_the_policy_protocol() -> None:
    policy = ForecastPolicy(_Stub(50.0), FloorPolicy(_prices()))

    assert isinstance(policy, PricePolicy)
    assert policy.name == "forecast"


def test_it_returns_what_the_forecaster_says() -> None:
    day = dt.date(2025, 3, 1)
    policy = ForecastPolicy(_Stub(42.0), FloorPolicy(_prices()))

    believed = policy.prices_for(day, _window(day))

    np.testing.assert_array_equal(believed, np.full(len(_window(day)), 42.0))
    assert policy.diagnostics()["forecast_days"] == 1
    assert policy.diagnostics()["fallback_days"] == 0


def test_it_falls_back_to_the_floor_and_not_to_something_of_its_own() -> None:
    """The fallback is the *same* climatological vector the chart plots against.

    Asserted by comparing against the floor directly. A second average living
    inside this policy could drift from the floor without any test noticing,
    and "the forecast beat the floor" would stop comparing like with like.
    """
    day = dt.date(2025, 3, 1)
    prices = _prices()
    floor = FloorPolicy(prices)
    policy = ForecastPolicy(_Stub(None), FloorPolicy(prices))

    believed = policy.prices_for(day, _window(day))

    np.testing.assert_array_equal(believed, floor.prices_for(day, _window(day)))
    assert policy.diagnostics()["fallback_days"] == 1
    assert policy.diagnostics()["forecast_days"] == 0


def test_the_decision_day_is_passed_through_unchanged() -> None:
    """The gate belongs to the day being decided, and only the policy knows it."""
    policy = ForecastPolicy(stub := _Stub(10.0), FloorPolicy(_prices()))
    days = [dt.date(2025, 3, 1), dt.date(2025, 3, 2)]

    for day in days:
        policy.prices_for(day, _window(day))

    assert stub.asked == days


def test_building_a_forecast_policy_without_a_forecaster_is_refused() -> None:
    """A clear error beats a policy that silently becomes the floor.

    The forecaster is expensive and regime-specific and must be shared across
    a degradation sweep, so it is built by the caller; a default here would
    hide a three-fold cost or, worse, a second chart bar that is the floor
    under another name.
    """
    with pytest.raises(ValueError, match="needs a fitted forecaster"):
        build_policy("forecast", _prices())


def test_the_other_policies_still_build_without_one() -> None:
    prices = _prices()

    assert build_policy("floor", prices).name == "floor"
    assert build_policy("oracle", prices).name == "oracle"
