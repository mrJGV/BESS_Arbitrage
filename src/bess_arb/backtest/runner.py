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
from functools import partial

import numpy as np
import pandas as pd

from bess_arb.backtest.metrics import settle_profit
from bess_arb.bid import (
    QUANTILE_LEVELS,
    BidCurves,
    ScenarioSolves,
    deliver,
    solve_curves,
    solve_scenarios,
)
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryMILP, BidCurveMILP, get_backend, get_curve_backend
from bess_arb.model.spec import (
    BatteryParams,
    FloatArray,
    Solution,
    SolverConfig,
    SolveStatus,
)
from bess_arb.policy import PricePolicy, QuantilePolicy, ScenarioPolicy
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
    clipped_mwh: float = 0.0
    """Cleared but not delivered, and zero under fixed-schedule settlement.

    Only a bid curve can promise what the state of charge cannot meet: a
    schedule comes out of the optimiser already feasible. Under curve
    settlement this is the honesty number for the "clip, no penalty" rule —
    see :mod:`bess_arb.bid.deliver`.
    """

    curve_steps: float = 1.0
    """Mean distinct quantities per period, 1.0 for a fixed schedule.

    A curve of one step is a schedule wearing a limit price, so a run whose
    mean sits near 1.0 produced no option value to measure.
    """


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
    bidding: str = "schedule"
    """Which settlement the run used — recorded, because the three are not
    comparable numbers unless a reader can see which produced which."""

    n_scenarios: int = 0
    """Scenarios per window under ``bidding="joint"``, zero otherwise.

    Recorded with the result because a joint run's curve resolution is capped
    by it: a curve cannot carry more distinct steps than the sample has
    trajectories, so ``curve_steps`` must always be read against this.
    """

    schedule_fallback_days: int = 0
    """Days a ``"joint"`` run bid the fixed schedule for want of a distribution.

    The residual pool fills only as decisions are made *and clear*, so the
    opening stretch of every joint run is bid as v2. This is how much of the
    result is not actually v3, and it is reported rather than described — the
    same discipline §2.5 applies to the floor's own fallbacks.
    """

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
                    "clipped_mwh": day.clipped_mwh,
                    "curve_steps": day.curve_steps,
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


class _CurveModelPool:
    """One :class:`BidCurveMILP` per ``(window length, scenario count)``.

    The same build-once discipline as :class:`_ModelPool`, and the same reason
    for pooling on window length: a two-day window is 47, 48 or 49 periods and
    one model cannot serve all three. The scenario count joins the key because
    it sizes the first stage as well as the recourse; in practice it is
    constant across a run, so the pool still reaches three entries and stops.

    Built lazily, so a run that never bids an optimised curve never imports
    the optimiser -- which keeps the extra Pyomo construction cost off every
    v1 and v2 run.
    """

    def __init__(
        self,
        backend_name: str,
        params: BatteryParams,
        dt_h: float,
        solver: SolverConfig | None,
    ) -> None:
        self._backend_name = backend_name
        self._params = params
        self._dt_h = dt_h
        self._solver = solver
        self._backend: type[BidCurveMILP] | None = None
        self._models: dict[tuple[int, int], BidCurveMILP] = {}

    def get(self, n_periods: int, n_scenarios: int) -> BidCurveMILP:
        key = (n_periods, n_scenarios)
        model = self._models.get(key)
        if model is None:
            if self._backend is None:
                self._backend = get_curve_backend(self._backend_name)
            model = self._backend(
                self._params,
                n_periods,
                n_scenarios,
                self._dt_h,
                solver=self._solver,
            )
            self._models[key] = model
        return model

    @property
    def keys(self) -> tuple[tuple[int, int], ...]:
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
    bidding: str = "schedule",
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
    n_scenarios: int = 5,
) -> BacktestResult:
    """Walk the delivery days, implementing one at a time.

    ``prices`` are the realised prices — used for settlement always, and for
    the decision only if ``policy`` chooses to look at them. ``first_day`` and
    ``last_day`` bound the *decision* days; the horizon reaches one day past
    ``last_day``, so that day's prices must be in the series too.

    ``bidding`` selects what the policy commits, and it is the only thing
    v2.5 changes about the protocol:

    ``"schedule"``
        v1 and v2. One solve on the policy's price vector, and the dispatch
        it returns is implemented as-is — equivalent to bidding at the
        market's price limits (``docs/DECISIONS.md`` §2.3).
    ``"curve"``
        One solve per quantile level, assembled into a price-quantity curve
        per period, cleared at the prices that actually settled, then clipped
        to what the state of charge can deliver. The policy must implement
        :class:`~bess_arb.policy.QuantilePolicy`.
    ``"joint"``
        v3. One solve per *scenario*, where the S scenarios are a joint
        sample from the policy's own predictive law rather than a comonotone
        sweep of marginals — see :class:`~bess_arb.scenarios.BeliefResiduals`.
        The clearing, clipping and settlement path is byte-identical to
        ``"curve"``; only where the price vectors came from is different,
        which is what makes a ``"curve"`` versus ``"joint"`` comparison a
        statement about the *dependence assumption* and nothing else. The
        policy must implement :class:`~bess_arb.policy.ScenarioPolicy`.

        A day whose residual pool is still too thin falls back to
        ``"schedule"`` for that day and is counted. Falling back to
        ``"curve"`` instead would mix two scenario constructions inside one
        run and make the result a statement about neither.

    Everything else is held identical across the three, which is what keeps
    the comparison between them a statement about bidding rather than about
    three different backtests. The oracle is the check: its curve is
    degenerate under both extensions, so all three modes must return the same
    profit for it to the last cent.
    """
    if bidding not in ("schedule", "curve", "joint", "optimised"):
        raise ValueError(
            "bidding must be 'schedule', 'curve', 'joint' or 'optimised', got "
            f"{bidding!r}"
        )

    # Narrowed once, here, rather than re-tested per day: the loop then
    # branches on the narrowed value, which is both cheaper and the form the
    # type checker can follow.
    curve_policy: QuantilePolicy | None = None
    joint_policy: ScenarioPolicy | None = None
    if bidding == "curve":
        if not isinstance(policy, QuantilePolicy):
            raise TypeError(
                f"the {policy.name} policy cannot bid curves: it has no "
                "prices_for_quantile method. Curve settlement needs K price "
                "vectors per window, one per quantile level."
            )
        curve_policy = policy
    elif bidding in ("joint", "optimised"):
        if not isinstance(policy, ScenarioPolicy):
            raise TypeError(
                f"the {policy.name} policy cannot bid joint scenarios: it has "
                "no price_scenarios method. Joint settlement needs S whole "
                "price trajectories per window, drawn from the policy's own "
                "predictive law."
            )
        joint_policy = policy
    days = _decision_days(prices, regime, protocol, first_day, last_day)
    pool = _ModelPool(backend, params, regime.dt_h, solver)
    curve_pool = _CurveModelPool(backend, params, regime.dt_h, solver)

    soc = protocol.soc_initial_mwh(params.e_max_mwh)
    warmup_until = protocol.warmup_days
    records: list[DayResult] = []
    solver_name: str | None = None
    solver_version: str | None = None
    joint_fallback = 0

    for position, day in enumerate(days):
        horizon_end = day + dt.timedelta(days=protocol.window_days - 1)
        implement_end = day + dt.timedelta(days=protocol.implement_days - 1)
        window = utc_index(day, horizon_end, regime)
        n_implement = len(utc_index(day, implement_end, regime))

        model = pool.get(len(window))
        realised = _realised(prices, window[:n_implement], day)

        # partial rather than a lambda: it binds this day's model and SoC at
        # construction, so the callable cannot pick up the next iteration's
        # values if it is ever held past the call.
        solve_window = partial(model.solve, soc_initial=soc)

        settle = partial(
            _implement_curves,
            prices=prices,
            window=window,
            day=day,
            n_implement=n_implement,
            soc=soc,
            regime=regime,
            params=params,
        )

        if curve_policy is not None:
            curves, solves = solve_curves(
                solve_window,
                curve_policy,
                day,
                window,
                quantile_levels=_levels_for(policy, quantile_levels),
            )
            objective, status, name, version = _middle_of(solves)
            outcome = settle(
                curves,
                believed_objective=objective,
                status=status,
                solver_name=name,
                solver_version=version,
            )
        elif joint_policy is not None:
            scenarios = joint_policy.price_scenarios(day, window, n_scenarios)
            if scenarios is None:
                # No distribution yet — bid the fixed schedule, which is what
                # v1 and v2 bid every day, and count it. Falling back to the
                # quantile sweep would put two scenario constructions inside
                # one run.
                joint_fallback += 1
                outcome = _implement_schedule(
                    policy, solve_window, day, window, n_implement
                )
            elif scenarios.shape[1] != len(window):
                raise ValueError(
                    f"{policy.name} returned scenarios over {scenarios.shape[1]} "
                    f"periods for a {len(window)}-period window on {day}"
                )
            elif bidding == "joint":
                curves, solves = solve_scenarios(solve_window, scenarios)
                objective, status, name, version = _middle_of(solves)
                outcome = settle(
                    curves,
                    believed_objective=objective,
                    status=status,
                    solver_name=name,
                    solver_version=version,
                )
            else:
                # v3 stage 2. One solve, and the curve comes back as a
                # decision rather than as S dispatches stitched together.
                chosen = curve_pool.get(len(window), scenarios.shape[0]).solve(
                    scenarios, soc
                )
                curves = BidCurves(
                    prices=chosen.band_prices, quantities=chosen.quantities
                )
                outcome = settle(
                    curves,
                    # The expected profit over the scenario set, which is what
                    # this optimiser was shown. Settlement still recomputes
                    # from the cleared dispatch — §4, and the distinction is
                    # sharper here than anywhere else in the project, because
                    # this objective is an expectation over beliefs and could
                    # not be a profit even in principle.
                    believed_objective=chosen.objective,
                    status=chosen.status,
                    solver_name=chosen.solver_name,
                    solver_version=chosen.solver_version,
                )
        else:
            outcome = _implement_schedule(
                policy, solve_window, day, window, n_implement
            )

        solver_name = outcome.solver_name
        solver_version = outcome.solver_version

        record = DayResult(
            day=day,
            warmup=position < warmup_until,
            periods=n_implement,
            window_periods=len(window),
            hours=n_implement * regime.dt_h,
            status=outcome.status,
            profit_eur=settle_profit(
                outcome.p_c, outcome.p_d, realised, regime.dt_h, params
            ),
            believed_objective_eur=outcome.believed_objective,
            charged_mwh=float(outcome.p_c.sum()) * regime.dt_h,
            discharged_mwh=float(outcome.p_d.sum()) * regime.dt_h,
            soc_start_mwh=soc,
            soc_end_mwh=outcome.soc_end,
            clipped_mwh=outcome.clipped_mwh,
            curve_steps=outcome.curve_steps,
        )
        records.append(record)
        if on_day is not None:
            on_day(position, len(days), record)
        soc = outcome.soc_end

    return BacktestResult(
        policy=policy.name,
        regime=regime,
        params=params,
        protocol=protocol,
        days=tuple(records),
        window_lengths=pool.lengths,
        solver_name=solver_name,
        solver_version=solver_version,
        bidding=bidding,
        n_scenarios=n_scenarios if bidding in ("joint", "optimised") else 0,
        schedule_fallback_days=joint_fallback,
    )


@dataclass(frozen=True, slots=True)
class _Implemented:
    """What one day's bidding path produced, before it becomes a record.

    The three bidding modes differ only in how these fields are obtained, so
    naming them makes the settlement below identical for all three rather than
    duplicated three times — which is the property that keeps a mode-to-mode
    comparison a statement about bidding.
    """

    p_c: FloatArray
    p_d: FloatArray
    soc_end: float
    believed_objective: float
    status: SolveStatus
    clipped_mwh: float
    curve_steps: float
    solver_name: str | None
    solver_version: str | None


def _implement_schedule(
    policy: PricePolicy,
    solve_window: Callable[[FloatArray], Solution],
    day: dt.date,
    window: pd.DatetimeIndex,
    n_implement: int,
) -> _Implemented:
    """v1 and v2: one solve on the policy's price vector, dispatched as-is."""
    believed = policy.prices_for(day, window)
    if believed.shape != (len(window),):
        raise ValueError(
            f"{policy.name} returned {believed.shape[0]} prices for a "
            f"{len(window)}-period window on {day}"
        )
    solution = solve_window(believed)
    return _Implemented(
        p_c=solution.p_c_mw[:n_implement],
        p_d=solution.p_d_mw[:n_implement],
        soc_end=float(solution.soc_mwh[n_implement - 1]),
        believed_objective=solution.objective,
        status=solution.status,
        # A schedule comes out of the optimiser already feasible, so there is
        # nothing to clip, and a one-step curve is what a fixed schedule is.
        clipped_mwh=0.0,
        curve_steps=1.0,
        solver_name=solution.solver_name,
        solver_version=solution.solver_version,
    )


def _implement_curves(
    curves: BidCurves,
    prices: pd.Series,
    window: pd.DatetimeIndex,
    day: dt.date,
    n_implement: int,
    soc: float,
    regime: Regime,
    params: BatteryParams,
    *,
    believed_objective: float,
    status: SolveStatus,
    solver_name: str | None,
    solver_version: str | None,
) -> _Implemented:
    """Clear the curves at realised prices, then repair for SoC.

    Shared by all three curve modes deliberately -- v2.5's comonotone sweep,
    v3's joint sample, and v3 stage 2's optimised curve. They differ in how
    the curve was arrived at and in nothing after that, so this function is
    where "only the construction changed" stops being a claim and becomes a
    fact about the code. It is also why the residual clipping of the optimised
    curve is comparable with the constructed ones: the same repair, applied to
    a curve that was chosen so as not to need it.
    """
    # Clear and settle the implemented day only. The horizon's second day was
    # solved to remove the end-of-horizon artefact and is discarded here
    # exactly as it is under a fixed schedule.
    net = curves.clear(_realised(prices, window, day))[:n_implement]
    delivered = deliver(net, soc, regime.dt_h, params)
    return _Implemented(
        p_c=delivered.p_c_mw,
        p_d=delivered.p_d_mw,
        soc_end=float(delivered.soc_mwh[n_implement - 1]),
        believed_objective=believed_objective,
        status=status,
        clipped_mwh=delivered.clipped_mwh,
        curve_steps=float(curves.step_counts[:n_implement].mean()),
        solver_name=solver_name,
        solver_version=solver_version,
    )


def _middle_of(
    solves: ScenarioSolves,
) -> tuple[float, SolveStatus, str | None, str | None]:
    """The representative scenario solve, for the columns a run reports.

    The middle solve's objective, so ``believed_objective`` keeps meaning
    "what the policy believed it was worth" rather than becoming a sum over
    scenarios that no single belief corresponds to. Under a quantile sweep
    that is the median scenario; under a joint sample it is an arbitrary
    draw, which is honest -- a sample has no median trajectory to point at.
    """
    middle = solves.solutions[len(solves.solutions) // 2]
    return middle.objective, middle.status, middle.solver_name, middle.solver_version


def _levels_for(
    policy: PricePolicy, quantile_levels: tuple[float, ...]
) -> tuple[float, ...]:
    """The quantile levels to sweep for one policy.

    The oracle is the exception, and it is worth spending a branch on: perfect
    foresight has no uncertainty for a quantile to describe, so all K of its
    vectors are identical and K-1 of the solves would be repeats of the first.
    Sweeping one level costs a fifth as much and produces the identical
    one-step curve — which is also the run whose profit must match the
    fixed-schedule oracle exactly.
    """
    if policy.name == "oracle":
        return (0.5,)
    return quantile_levels


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
