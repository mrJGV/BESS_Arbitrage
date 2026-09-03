"""The forecast policy: a point forecast, and the floor when there is none.

The third bar of the chart, and the only one that has to be *earned*. The
oracle reads prices that have cleared; the floor reads an average of prices
that cleared long ago; this one has to say what tomorrow will cost using only
what was on the wire at noon today. Everything difficult about that is next
door in :mod:`bess_arb.forecast` — what may be read, and when it became
readable. What is left here is nine lines, which is the point: the policy is
still a price vector into the same optimiser (CLAUDE.md invariant 2), and the
forecaster is not permitted to change the horizon, the cadence or a
constraint.

The fallback, and why it is the floor
-------------------------------------

Below a year of history the forecaster declines to answer rather than
returning a model fitted on three weeks. Something still has to be bid, and
the honest something is the null hypothesis: **the forecast policy falls back
to the floor's own climatological vector**, so on the days it knows nothing it
scores what knowing nothing is worth, and the chart's middle bar is never
flattered by a period the forecaster could not really cover.

Injecting the floor rather than reimplementing an average is deliberate. A
second climatology here would be a second null hypothesis, free to drift from
the one the chart plots against, and "the forecast beat the floor" would stop
being a comparison of like with like.

The fallback is counted and reported. In the headline quarter-hourly regime it
never fires — its first decision day has 3.7 years of prices behind it — and
in the hourly regime it covers the first year, which is stated with the
result rather than left for a reader to work out.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

import pandas as pd

from bess_arb.model.spec import FloatArray
from bess_arb.policy.floor import FloorPolicy

__all__ = ["ForecastPolicy", "Forecaster"]


@runtime_checkable
class Forecaster(Protocol):
    """All the policy needs of a forecaster, declared where it is needed.

    Two methods, named here rather than importing
    :class:`bess_arb.forecast.lgbm.PriceForecaster`. That keeps LightGBM out of
    the policy layer's type surface as well as its imports, and it says exactly
    what the policy is allowed to ask for: a vector, or an admission that there
    is none. A forecaster with a wider interface could not be used differently
    without changing this file, which is the same guarantee
    :class:`bess_arb.policy.PricePolicy` gives one level up.
    """

    def forecast(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray | None: ...

    def diagnostics(self) -> dict[str, int]: ...


class ForecastPolicy:
    """Forecast prices for the window, falling back to the floor."""

    name = "forecast"

    def __init__(self, forecaster: Forecaster, fallback: FloorPolicy) -> None:
        self._forecaster = forecaster
        self._fallback = fallback
        self._days = {"forecast": 0, "fallback": 0}

    def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray:
        """What the policy believes the window will cost, decided at the gate."""
        believed = self._forecaster.forecast(day, window)
        if believed is None:
            self._days["fallback"] += 1
            return self._fallback.prices_for(day, window)
        self._days["forecast"] += 1
        return believed

    def diagnostics(self) -> dict[str, int]:
        """Days served by the model and by the floor, plus the model's own.

        Reported on every run. ``fallback_days`` is the count that matters to
        a reader: it is how much of the result was not actually forecast.
        """
        return {
            "forecast_days": self._days["forecast"],
            "fallback_days": self._days["fallback"],
            **self._forecaster.diagnostics(),
        }
