"""Δt is a parameter, and this is the test whose only job is to prove it.

The market went from hourly to quarter-hourly market time units on
1 October 2025, so the code has to run unchanged at Δt = 1 and Δt = 0.25.
An implicit "one period = one hour" is the most likely bug in the project
and would not announce itself: it would show up as a plausible number.

The same four hours, expressed as four hourly periods and as sixteen
quarter-hourly ones, must give the same optimum. With prices constant inside
each hour and non-negative, refining the grid cannot create value: any finer
schedule averages back to an hourly one with the same objective, an SoC path
that stays inside its window because it is linear between two feasible
endpoints, and no simultaneous charge and discharge. Equality is therefore
the correct expectation, not an approximation.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arb.model import SolveStatus
from conftest import ModelFactory

TOL = 1e-6
HOURLY_PRICES = np.array([10.0, 10.0, 100.0, 100.0])
GOLDEN_OPTIMUM = 1500.0

# Distinct in every hour, so the comparison is not resting on the golden
# case's symmetry. Deliberately all positive: under *negative* prices the
# finer grid can genuinely beat the coarser one, by discharging in one
# quarter to make room to be paid for charging in the next — churn that
# exclusivity forbids inside a single period but not across two. That is a
# real property of the market, not a Δt bug, so it does not belong in a test
# whose subject is Δt.
RAMP_PRICES = np.array([40.0, 5.0, 12.0, 95.0, 130.0, 60.0])


def _quarter_hourly(hourly: np.ndarray) -> np.ndarray:
    """The same price path, resampled to four periods per hour."""
    return np.repeat(hourly, 4)


def test_golden_case_is_the_same_at_quarter_hourly_resolution(
    make_model: ModelFactory,
) -> None:
    hourly = make_model(n_periods=4, dt_h=1.0).solve(HOURLY_PRICES, soc_initial=0.0)
    quarterly = make_model(n_periods=16, dt_h=0.25).solve(
        _quarter_hourly(HOURLY_PRICES), soc_initial=0.0
    )

    assert hourly.status is SolveStatus.OPTIMAL
    assert quarterly.status is SolveStatus.OPTIMAL
    assert hourly.objective == pytest.approx(GOLDEN_OPTIMUM, abs=TOL)
    assert quarterly.objective == pytest.approx(GOLDEN_OPTIMUM, abs=TOL)


def test_dt_invariance_on_a_non_degenerate_price_path(
    make_model: ModelFactory,
) -> None:
    hourly = make_model(n_periods=6, dt_h=1.0).solve(RAMP_PRICES, soc_initial=5.0)
    quarterly = make_model(n_periods=24, dt_h=0.25).solve(
        _quarter_hourly(RAMP_PRICES), soc_initial=5.0
    )

    assert hourly.objective == pytest.approx(quarterly.objective, abs=TOL)
    # Same energy moved, whatever the period length.
    assert hourly.charged_mwh == pytest.approx(quarterly.charged_mwh, abs=TOL)
    assert hourly.discharged_mwh == pytest.approx(quarterly.discharged_mwh, abs=TOL)


def test_period_count_is_never_assumed_to_be_a_day(make_model: ModelFactory) -> None:
    """DST days have 23 or 25 hours (92 or 100 quarter-hourly periods).

    The model takes ``n_periods`` from its caller and must not care. This is
    the modelling-layer half of that invariant; the calendar half arrives
    with the timeline module.
    """
    for n_periods in (23, 25, 92, 100):
        prices = np.tile([10.0, 100.0], n_periods // 2 + 1)[:n_periods]
        solution = make_model(n_periods=n_periods, dt_h=1.0).solve(
            prices, soc_initial=0.0
        )

        assert solution.status is SolveStatus.OPTIMAL
        assert solution.n_periods == n_periods
