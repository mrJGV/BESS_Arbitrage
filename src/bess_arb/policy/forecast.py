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

    def forecast_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> FloatArray | None: ...

    def diagnostics(self) -> dict[str, int]: ...


class ForecastPolicy:
    """Forecast prices for the window, falling back to the floor."""

    name = "forecast"

    def __init__(self, forecaster: Forecaster, fallback: FloorPolicy) -> None:
        self._forecaster = forecaster
        self._fallback = fallback
        self._days = {"forecast": 0, "fallback": 0}
        self._quantile_days = {"forecast": 0, "fallback": 0}
        self._scenario_days = {"forecast": 0, "fallback": 0}

    def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray:
        """What the policy believes the window will cost, decided at the gate."""
        believed = self._forecaster.forecast(day, window)
        if believed is None:
            self._days["fallback"] += 1
            # The floor records its own vector into the shared pool, which is
            # right: on this day that vector is what this policy bid, so the
            # error it later shows is this policy's error.
            return self._fallback.prices_for(day, window)
        self._days["forecast"] += 1
        # v3's residual pool, recorded at the one place a belief is formed.
        # `record` keeps the first entry per day, so the fallback branch above
        # and this one cannot both claim a day.
        self._fallback.scenario_pool().record(day, window, believed)
        return believed

    def prices_for_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> FloatArray:
        """v2.5's scenario source: the same fallback shape as :meth:`prices_for`.

        A day too young for the point forecast is too young for the quantile
        forecast too — there is no separate model to be short of history for —
        so it falls back to the floor's own quantile ladder, which is already
        causal and already answers at any tau, and it does the same when the
        residual buffer behind the shift is still too thin. Injecting the same
        :class:`FloorPolicy` instance
        used by :meth:`prices_for`'s fallback keeps both null hypotheses
        identical, for the reason given in the module docstring: a second one
        drifting from the first would stop being a comparison of like with
        like.
        """
        believed = self._forecaster.forecast_quantile(day, window, tau)
        if believed is None:
            self._quantile_days["fallback"] += 1
            return self._fallback.prices_for_quantile(day, window, tau)
        self._quantile_days["forecast"] += 1
        return believed

    def price_scenarios(
        self, day: dt.date, window: pd.DatetimeIndex, n_scenarios: int
    ) -> FloatArray | None:
        """v3's joint scenario source: this policy's belief plus its own errors.

        Delegates the residual pool to the injected :class:`FloorPolicy`'s
        :class:`~bess_arb.scenarios.BeliefResiduals`, but records **this
        policy's** belief into it, which is what makes the pool the forecast
        policy's error distribution rather than the floor's.

        Three things follow from the pool holding whole *windows*, and each was
        a decision that would otherwise need its own code:

        - **The two horizon days are paired.** One recorded belief spans D and
          D+1 from a single decision, so a sampled residual carries the
          observed dependence across the horizon boundary. Lead 1 is genuinely
          harder than lead 0 — at the gate the D+1 exogenous bundle covering
          D+1 has not been published — and drawing the two halves
          independently would fabricate an independence the data does not
          show.
        - **Fallback days are in the pool.** On a day the forecaster declined,
          the belief recorded is the floor's, so the residual is the floor's
          error. That is correct rather than a leak: the object is *this
          policy's* causal error, and on that day this policy bid the floor's
          vector. In the headline quarter-hourly regime it never arises; in
          the hourly regime it covers the first year, and
          :meth:`diagnostics` already says how much.
        - **The scenario mean is the point forecast.** The pool is centred, so
          v3 differs from v2 in dispersion alone and cannot quietly collect a
          causal bias correction the point forecaster deliberately does not
          make.
        """
        believed = self.prices_for(day, window)
        pool = self._fallback.scenario_pool()
        residuals = pool.sample(day, window, n_scenarios)
        if residuals is None:
            self._scenario_days["fallback"] += 1
            return None
        self._scenario_days["forecast"] += 1
        return believed[None, :] + residuals

    def scenario_diagnostics(self) -> dict[str, int]:
        """v3's ladder counts: days bid as a distribution, days bid as v2."""
        return {
            "scenario_forecast_days": self._scenario_days["forecast"],
            "scenario_fallback_days": self._scenario_days["fallback"],
            **self._fallback.scenario_pool().diagnostics(),
        }

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

    def quantile_diagnostics(self) -> dict[str, int]:
        """The quantile-call counterpart of :meth:`diagnostics`.

        Kept separate rather than merged in: :meth:`prices_for` is called
        once per decision day, but :meth:`prices_for_quantile` is called once
        per *(day, tau)* pair — see
        :func:`bess_arb.bid.scenarios.solve_curves` — so the two counters are
        in different units and merging them would silently misreport one of
        them as the other.
        """
        return {
            "forecast_calls": self._quantile_days["forecast"],
            "fallback_calls": self._quantile_days["fallback"],
        }
