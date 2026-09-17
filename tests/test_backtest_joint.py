"""Joint-scenario settlement in the rolling loop -- v3's arm of the backtest.

``run_backtest(bidding="joint")`` differs from ``bidding="curve"`` in exactly
one place: where the S price vectors come from. The clearing, the SoC repair,
the settlement and the accounting are the same code, called from the same
helper, which is what makes a curve-versus-joint comparison a statement about
the *dependence assumption* rather than about two different backtests.

Three things are pinned here. The oracle identity, which says the clearing
path is right, so that whatever the joint arm scores is a statement about
bidding rather than a suspected defect. The fallback, which is how much of a run was not actually
bid as v3. And determinism, because a seeded resample that is not reproducible
is not a measurement.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.backtest.metrics import summarise
from bess_arb.backtest.runner import BacktestResult, run_backtest
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryParams, FloatArray
from bess_arb.policy.floor import FloorPolicy
from bess_arb.policy.oracle import OraclePolicy
from bess_arb.scenarios import ScenarioConfig
from bess_arb.timeline import MARKET_TZ, Regime, utc_index

HOURLY = Regime("hourly", 1.0)
BATTERY = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=17.0)
PROTOCOL = HorizonConfig(
    window_days=2, implement_days=1, warmup_days=0, soc_initial_fraction=0.5
)
SCENARIOS = ScenarioConfig(
    n_scenarios=5, min_scenario_days=10, centre_residuals=True, seed=20260916
)


def _prices(first: str, last: str) -> pd.Series:
    index = utc_index(first, last, HOURLY)
    hour = np.asarray(index.tz_convert(MARKET_TZ).hour, dtype=float)
    rng = np.random.default_rng(20260916)
    return pd.Series(
        60.0
        + 40.0 * np.sin((hour - 5.0) * np.pi / 12.0)
        + rng.normal(0.0, 15.0, len(index)),
        index=index,
        name="price_eur_mwh",
    )


def _run(
    policy: object, prices: pd.Series, bidding: str, **kwargs: object
) -> BacktestResult:
    return run_backtest(
        prices,
        policy,  # type: ignore[arg-type]
        BATTERY,
        HOURLY,
        PROTOCOL,
        bidding=bidding,
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_oracle_settles_identically_under_joint_and_schedule() -> None:
    """The free correctness check, extended to v3.

    Perfect foresight has a point mass for a predictive law, so a sample of
    any size holds one trajectory, the curve is one step at the price that
    cleared, and clearing it returns the schedule it started from. Any
    disagreement here is a clearing bug and would invalidate every joint
    number the backtest produces.
    """
    prices = _prices("2024-01-01", "2024-01-20")

    schedule = summarise(_run(OraclePolicy(prices), prices, "schedule"))
    joint = summarise(_run(OraclePolicy(prices), prices, "joint", n_scenarios=5))

    assert joint.profit_eur == pytest.approx(schedule.profit_eur, abs=1e-6)
    assert joint.clipped_mwh == pytest.approx(0.0, abs=1e-9)
    assert joint.mean_curve_steps == pytest.approx(1.0)


def test_a_policy_without_scenarios_is_refused_by_name() -> None:
    """The narrowing happens once, before the loop, and says what is missing."""
    prices = _prices("2024-01-01", "2024-01-20")

    class _PointOnly:
        name = "point_only"

        def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray:
            return np.zeros(len(window))

    with pytest.raises(TypeError, match="price_scenarios"):
        _run(_PointOnly(), prices, "joint")


def test_an_unknown_bidding_mode_names_every_mode() -> None:
    prices = _prices("2024-01-01", "2024-01-20")
    with pytest.raises(ValueError, match="'schedule', 'curve', 'joint' or 'optimised'"):
        _run(OraclePolicy(prices), prices, "curves")


def test_the_opening_days_fall_back_to_the_schedule_and_are_counted() -> None:
    """The pool fills only as decisions are made *and clear*.

    So the first stretch of any joint run is bid as v2, and the count is
    reported rather than described -- the same discipline the floor's own
    fallbacks get. Falling back to the quantile sweep instead would put two
    scenario constructions inside one run.
    """
    prices = _prices("2024-01-01", "2024-03-01")
    policy = FloorPolicy(prices, scenarios=SCENARIOS)
    result = _run(policy, prices, "joint", n_scenarios=5)

    assert result.schedule_fallback_days > 0
    assert result.schedule_fallback_days < len(result.days)
    # A fallback day bids a schedule, so it clips nothing and carries one step.
    fell_back = [
        d for d in result.days if d.curve_steps == 1.0 and d.clipped_mwh == 0.0
    ]
    assert len(fell_back) >= result.schedule_fallback_days


def test_the_scenario_count_is_recorded_with_the_result() -> None:
    """``curve_steps`` is capped by it, so it must be readable beside it.

    A curve cannot carry more distinct steps than the sample has trajectories,
    so ``step_counts`` means nothing read without it.
    """
    prices = _prices("2024-01-01", "2024-03-01")
    policy = FloorPolicy(prices, scenarios=SCENARIOS)
    result = _run(policy, prices, "joint", n_scenarios=7)

    assert result.bidding == "joint"
    assert result.n_scenarios == 7
    assert max(d.curve_steps for d in result.days) <= 7.0

    schedule = _run(FloorPolicy(prices), prices, "schedule")
    assert schedule.n_scenarios == 0
    assert schedule.schedule_fallback_days == 0


def test_two_identical_joint_runs_agree_to_the_cent() -> None:
    """Invariant 8 at the level a reader cares about.

    The draw is seeded on the decision day, so a re-run reproduces it. Without
    this a joint number could not be quoted at all.
    """
    prices = _prices("2024-01-01", "2024-03-01")
    profits = [
        summarise(
            _run(
                FloorPolicy(prices, scenarios=SCENARIOS), prices, "joint", n_scenarios=5
            )
        ).profit_eur
        for _ in range(2)
    ]
    assert profits[0] == pytest.approx(profits[1], abs=1e-9)


def test_the_joint_curves_are_livelier_than_the_comonotone_ones() -> None:
    """The mechanism v3 exists for, checked rather than argued.

    A comonotone sweep moves every period's price together, so a period's rank
    within the day rarely changes and its dispatch rarely does either. A joint
    sample carries the idiosyncratic component too, so more periods get a
    curve with real shape. Whether that *pays* is the backtest's question;
    that it happens at all is this test's.
    """
    prices = _prices("2024-01-01", "2024-04-01")

    curve = summarise(_run(FloorPolicy(prices, scenarios=SCENARIOS), prices, "curve"))
    joint = summarise(
        _run(FloorPolicy(prices, scenarios=SCENARIOS), prices, "joint", n_scenarios=5)
    )
    assert joint.mean_curve_steps > curve.mean_curve_steps


def test_the_oracle_settles_identically_under_the_optimised_mode() -> None:
    """The identity extended to the second formulation.

    This is the strongest of the three oracle checks, because it crosses a
    formulation boundary rather than a construction one: the stochastic
    program handed a single scenario must reduce to the plain window MILP, and
    its curve must clear back to the schedule that produced it. A drift
    between the two formulations would show up here and nowhere else in the
    backtest.
    """
    prices = _prices("2024-01-01", "2024-01-20")

    schedule = summarise(_run(OraclePolicy(prices), prices, "schedule"))
    optimised = summarise(
        _run(OraclePolicy(prices), prices, "optimised", n_scenarios=5)
    )

    assert optimised.profit_eur == pytest.approx(schedule.profit_eur, abs=1e-6)
    assert optimised.clipped_mwh == pytest.approx(0.0, abs=1e-9)


def test_the_optimised_mode_clips_less_than_the_constructed_one() -> None:
    """v3 stage 2's whole reason for existing, as a property rather than a run.

    Stage 1 clears a curve assembled from independent per-scenario solves, so
    the plan it commits is one no scenario endorsed and the SoC repair has a
    great deal to do. Stage 2 enforces feasibility per scenario *inside* the
    curve choice, so on the sampled acceptance patterns there is nothing to
    repair. It is not zero -- a realised day can mix bands across periods in a
    way no sampled scenario did -- but it must be markedly less.
    """
    prices = _prices("2024-01-01", "2024-04-01")

    constructed = summarise(
        _run(FloorPolicy(prices, scenarios=SCENARIOS), prices, "joint", n_scenarios=5)
    )
    optimised = summarise(
        _run(
            FloorPolicy(prices, scenarios=SCENARIOS),
            prices,
            "optimised",
            n_scenarios=5,
        )
    )
    assert optimised.clipped_mwh < constructed.clipped_mwh


def test_the_optimised_mode_needs_the_same_scenario_capability() -> None:
    prices = _prices("2024-01-01", "2024-01-20")

    class _PointOnly:
        name = "point_only"

        def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray:
            return np.zeros(len(window))

    with pytest.raises(TypeError, match="price_scenarios"):
        _run(_PointOnly(), prices, "optimised")
