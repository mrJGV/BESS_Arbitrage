"""Scenario orchestration: K solves in, one curve out.

``bid/scenarios.py`` owns two things worth testing on their own. It calls the
injected solver exactly once per quantile level and nothing else — which is
CLAUDE.md invariant 2 held at this layer, since the only thing differing
between those calls is the price vector. And it refuses a scenario family that
is not monotone in tau, which is the guard standing where the tau labels still
exist to check against.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.bid.scenarios import QUANTILE_LEVELS, solve_curves
from bess_arb.model.spec import FloatArray, Solution, SolveStatus

DAY = dt.date(2024, 6, 1)
WINDOW = pd.date_range("2024-06-01", periods=4, freq="h", tz="UTC")


class _Recording:
    """A solver that charges below the median price and sells above it."""

    def __init__(self) -> None:
        self.seen: list[FloatArray] = []

    def __call__(self, prices: FloatArray) -> Solution:
        self.seen.append(prices.copy())
        middle = float(np.median(prices))
        p_c = np.where(prices < middle, 10.0, 0.0)
        p_d = np.where(prices > middle, 10.0, 0.0)
        return Solution(
            status=SolveStatus.OPTIMAL,
            objective=0.0,
            p_c_mw=p_c,
            p_d_mw=p_d,
            soc_mwh=np.zeros_like(prices),
            dt_h=1.0,
        )


class _Widening:
    """A well-behaved quantile source: fans out around a fixed day shape."""

    shape = np.array([30.0, 20.0, 90.0, 60.0])

    def prices_for_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> FloatArray:
        return self.shape * (0.5 + tau)


class _Crossing:
    """A source whose tau=0.9 falls below its tau=0.7 in one period.

    What a fallback ladder produces when different taus read different rungs
    of it — the floor policy did exactly this before its threshold was made
    tau-independent.
    """

    def prices_for_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> FloatArray:
        prices = np.array([30.0, 20.0, 90.0, 60.0]) * (0.5 + tau)
        if tau > 0.8:
            prices[2] = 1.0
        return prices


def test_the_solver_is_called_once_per_quantile_level() -> None:
    """K levels, K solves, and each one differs only in the prices it is given."""
    solver = _Recording()

    curves, solves = solve_curves(solver, _Widening(), DAY, WINDOW)

    assert len(solver.seen) == len(QUANTILE_LEVELS)
    assert len(solves.solutions) == len(QUANTILE_LEVELS)
    assert solves.quantile_levels == QUANTILE_LEVELS
    assert curves.n_periods == len(WINDOW)
    assert curves.n_steps == len(QUANTILE_LEVELS)


def test_scenario_prices_are_passed_through_in_ascending_tau_order() -> None:
    """The curve's steps mean nothing if the levels arrive out of order."""
    solver = _Recording()

    solve_curves(solver, _Widening(), DAY, WINDOW)

    seen = np.array(solver.seen)
    assert np.all(np.diff(seen, axis=0) >= -1e-9)


def test_a_family_that_falls_as_tau_rises_is_refused() -> None:
    """The guard that would have caught the floor's ladder-crossing bug.

    ``build_curves`` sorts each period's pairs by price before pairing them
    with quantities, so a scrambled family is silently reinterpreted rather
    than rejected. The check therefore belongs here.
    """
    with pytest.raises(ValueError, match="must not fall as tau rises"):
        solve_curves(_Recording(), _Crossing(), DAY, WINDOW)


def test_a_single_level_gives_a_one_step_curve() -> None:
    """The oracle's path: no sweep to run, so one solve is the whole curve."""
    curves, solves = solve_curves(
        _Recording(), _Widening(), DAY, WINDOW, quantile_levels=(0.5,)
    )

    assert curves.n_steps == 1
    assert len(solves.solutions) == 1


def test_quantile_levels_must_ascend() -> None:
    with pytest.raises(ValueError, match="must ascend"):
        solve_curves(_Recording(), _Widening(), DAY, WINDOW, quantile_levels=(0.9, 0.1))


def test_quantile_levels_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        solve_curves(_Recording(), _Widening(), DAY, WINDOW, quantile_levels=())
