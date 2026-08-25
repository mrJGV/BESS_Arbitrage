"""The rolling-horizon backtest.

``docs/DECISIONS.md`` §2.1 and §2.6. For each delivery day D:

1. build the UTC window covering D and D+1 — 48 hours on an ordinary day;
2. ask the policy what it believes those prices are;
3. solve, from the state of charge the previous day left behind;
4. **implement day D only**, and settle it at the prices that cleared;
5. carry the SoC at the end of D into the next day's solve.

The two-day window with one day implemented removes the end-of-horizon
artefact without inventing a terminal value function. A free terminal SoC
worth nothing empties the battery every midnight and puts a visible cliff in
any dispatch plot; a cyclic condition removes the accounting artefact but not
the economic one, forcing an unwind even when the next day opens expensive.
Solving two days and keeping one costs a second day's solve and needs no new
parameter, which is why it is the protocol here.

Warm-up: the first seven days are simulated and then dropped from every
metric, so the arbitrary 50% opening SoC cannot show up in the result.
Dropped from the *metrics*, not skipped — skipping them would just move the
arbitrary state to day eight.

Why the model is pooled by window length
----------------------------------------

CLAUDE.md forbids rebuilding the model inside the loop, and the backend is
constructed with a fixed ``n_periods``. Those two facts collide on the days
DST moves: a delivery day is 23, 24 or 25 hours long, so a two-day window is
47, 48 or 49 periods (188, 192 or 196 quarter-hourly). One model cannot serve
all three, and padding a short window with invented periods is worse than
holding three models.

So :class:`_ModelPool` holds one instance per distinct window length. There
are exactly three per regime, both are built within the first year, and every
subsequent day of the backtest re-solves an existing one — which is the
property the rule is protecting. A pool that grew would mean the window
length was being derived wrongly, so its size is asserted in the tests rather
than assumed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bess_arb.backtest.metrics import settle_profit
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryMILP, get_backend
from bess_arb.model.spec import (
    BatteryParams,
    FloatArray,
    SolverConfig,
    SolveStatus,
)
from bess_arb.policy import PricePolicy
from bess_arb.series import delivery_days
from bess_arb.timeline import Regime, periods_in_day, utc_index

__all__ = ["BacktestResult", "DayResult", "run_backtest"]


@dataclass(frozen=True, slots=True)
class DayResult:
    """One implemented delivery day."""

    day: dt.date
    warmup: bool
    periods: int
    """Periods implemented — this day's own count, 23/24/25 or 92/96/100."""
    window_periods: int
    hours: float
    status: SolveStatus
    profit_eur: float
    """Settled at realised prices, whatever the policy believed."""
    believed_objective_eur: float
    """The solver's objective over the whole window, at the policy's prices.

    Kept because the difference between this and the settled profit is the
    cost of being wrong, and it is free to record.
    """
    charged_mwh: float
    discharged_mwh: float
    soc_start_mwh: float
    soc_end_mwh: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Every implemented day of one policy at one degradation cost."""

    policy: str
    regime: Regime
    params: BatteryParams
    protocol: HorizonConfig
    days: tuple[DayResult, ...]
    window_lengths: tuple[int, ...]
    solver_name: str | None = None
    solver_version: str | None = None

    @property
    def evaluated(self) -> tuple[DayResult, ...]:
        """The days that count: everything past the warm-up."""
        return tuple(day for day in self.days if not day.warmup)

    def frame(self) -> pd.DataFrame:
        """Per-day records, indexed by delivery day.

        For plotting and for reading a run that looks wrong; the metrics
        never go through it.
        """
        return pd.DataFrame(
            [
                {
                    "day": day.day,
                    "warmup": day.warmup,
                    "periods": day.periods,
                    "window_periods": day.window_periods,
                    "hours": day.hours,
                    "status": day.status.value,
                    "profit_eur": day.profit_eur,
                    "believed_objective_eur": day.believed_objective_eur,
                    "charged_mwh": day.charged_mwh,
                    "discharged_mwh": day.discharged_mwh,
                    "soc_start_mwh": day.soc_start_mwh,
                    "soc_end_mwh": day.soc_end_mwh,
                }
                for day in self.days
            ]
        ).set_index("day")


class _ModelPool:
    """One :class:`BatteryMILP` per distinct window length.

    See the module docstring. The pool is not a cache in the optimisation
    sense — nothing is ever evicted, and it reaches its final size of three
    within the first year of any regime.
    """

    def __init__(
        self,
        backend_name: str,
        params: BatteryParams,
        dt_h: float,
        solver: SolverConfig | None,
    ) -> None:
        self._backend = get_backend(backend_name)
        self._params = params
        self._dt_h = dt_h
        self._solver = solver
        self._models: dict[int, BatteryMILP] = {}

    def get(self, n_periods: int) -> BatteryMILP:
        model = self._models.get(n_periods)
        if model is None:
            model = self._backend(
                self._params, n_periods, self._dt_h, solver=self._solver
            )
            self._models[n_periods] = model
        return model

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(sorted(self._models))


def run_backtest(
    prices: pd.Series,
    policy: PricePolicy,
    params: BatteryParams,
    regime: Regime,
    protocol: HorizonConfig,
    *,
    backend: str = "pyomo",
    solver: SolverConfig | None = None,
    first_day: dt.date | None = None,
    last_day: dt.date | None = None,
    on_day: Callable[[int, int, DayResult], None] | None = None,
) -> BacktestResult:
    """Walk the delivery days, implementing one at a time.

    ``prices`` are the realised prices — used for settlement always, and for
    the decision only if ``policy`` chooses to look at them. ``first_day`` and
    ``last_day`` bound the *decision* days; the horizon reaches one day past
    ``last_day``, so that day's prices must be in the series too.
    """
    days = _decision_days(prices, regime, protocol, first_day, last_day)
    pool = _ModelPool(backend, params, regime.dt_h, solver)

    soc = protocol.soc_initial_mwh(params.e_max_mwh)
    warmup_until = protocol.warmup_days
    records: list[DayResult] = []
    solver_name: str | None = None
    solver_version: str | None = None

    for position, day in enumerate(days):
        horizon_end = day + dt.timedelta(days=protocol.window_days - 1)
        implement_end = day + dt.timedelta(days=protocol.implement_days - 1)
        window = utc_index(day, horizon_end, regime)
        n_implement = len(utc_index(day, implement_end, regime))

        believed = policy.prices_for(day, window)
        if believed.shape != (len(window),):
            raise ValueError(
                f"{policy.name} returned {believed.shape[0]} prices for a "
                f"{len(window)}-period window on {day}"
            )

        solution = pool.get(len(window)).solve(believed, soc)
        solver_name = solution.solver_name
        solver_version = solution.solver_version

        p_c = solution.p_c_mw[:n_implement]
        p_d = solution.p_d_mw[:n_implement]
        realised = _realised(prices, window[:n_implement], day)

        soc_end = float(solution.soc_mwh[n_implement - 1])
        record = DayResult(
            day=day,
            warmup=position < warmup_until,
            periods=n_implement,
            window_periods=len(window),
            hours=n_implement * regime.dt_h,
            status=solution.status,
            profit_eur=settle_profit(p_c, p_d, realised, regime.dt_h, params),
            believed_objective_eur=solution.objective,
            charged_mwh=float(p_c.sum()) * regime.dt_h,
            discharged_mwh=float(p_d.sum()) * regime.dt_h,
            soc_start_mwh=soc,
            soc_end_mwh=soc_end,
        )
        records.append(record)
        if on_day is not None:
            on_day(position, len(days), record)
        soc = soc_end

    return BacktestResult(
        policy=policy.name,
        regime=regime,
        params=params,
        protocol=protocol,
        days=tuple(records),
        window_lengths=pool.lengths,
        solver_name=solver_name,
        solver_version=solver_version,
    )


def _decision_days(
    prices: pd.Series,
    regime: Regime,
    protocol: HorizonConfig,
    first_day: dt.date | None,
    last_day: dt.date | None,
) -> list[dt.date]:
    """Days that can be decided: the horizon must fit inside the data.

    The last ``window_days - 1`` days of the snapshot cannot be decided,
    because their horizon runs past the end of the prices. Truncating here
    rather than tolerating a short final window keeps every day in the run
    solved under the identical protocol, which is what §2.4 means by "same
    cadence".
    """
    available = delivery_days(prices)
    if not available:
        raise ValueError("the price series is empty")

    covered = set(available)
    expected = {
        available[0] + dt.timedelta(days=n)
        for n in range((available[-1] - available[0]).days + 1)
    }
    if covered != expected:
        missing = sorted(expected - covered)
        raise ValueError(
            f"the price series skips {len(missing)} delivery day(s), first "
            f"{missing[0]}; the backtest needs a contiguous calendar"
        )

    horizon_tail = dt.timedelta(days=protocol.window_days - 1)
    days = [day for day in available if day + horizon_tail <= available[-1]]
    if first_day is not None:
        days = [day for day in days if day >= first_day]
    if last_day is not None:
        days = [day for day in days if day <= last_day]

    if len(days) <= protocol.warmup_days:
        raise ValueError(
            f"{len(days)} decidable day(s) is not more than the "
            f"{protocol.warmup_days}-day warm-up; nothing would be measured"
        )

    # Cheap, and it fails on the one class of bug worth failing on here.
    for day in (days[0], days[-1]):
        if periods_in_day(day, regime) < 1:
            raise ValueError(f"{day} derives no periods at dt_h={regime.dt_h}")
    return days


def _realised(prices: pd.Series, window: pd.DatetimeIndex, day: dt.date) -> FloatArray:
    values = prices.reindex(window)
    if values.isna().any():
        raise KeyError(
            f"settlement for {day}: {int(values.isna().sum())} of "
            f"{len(window)} implemented periods have no realised price"
        )
    return np.asarray(values.to_numpy(dtype=np.float64))
