"""Curve settlement in the rolling loop.

``run_backtest(bidding="curve")`` replaces the fixed schedule with a
price-quantity curve per period, cleared at the prices that settled and then
clipped to what the state of charge can deliver. Everything else — the window,
the cadence, the warm-up, the model pool — is held identical, and that is what
makes the two modes comparable at all.

The test that matters is the oracle's. Perfect foresight has no uncertainty
for a quantile to describe, so its curve is a single step at the price that
cleared and the two settlements must agree exactly. It is the check that says
the clearing path is right, and it is what makes the *negative* v2.5 result
trustworthy rather than a suspected defect: the measured shortfall of the
curve modes is a statement about bidding, not about this code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_arb.backtest.metrics import summarise
from bess_arb.backtest.runner import run_backtest
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryParams
from bess_arb.policy.floor import FloorPolicy
from bess_arb.policy.oracle import OraclePolicy
from bess_arb.timeline import MARKET_TZ, Regime, utc_index

HOURLY = Regime("hourly", 1.0)
BATTERY = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=17.0)
PROTOCOL = HorizonConfig(
    window_days=2, implement_days=1, warmup_days=0, soc_initial_fraction=0.5
)


def _prices(first: str, last: str) -> pd.Series:
    """A daily shape with day-to-day variation, so the scenarios differ."""
    index = utc_index(first, last, HOURLY)
    hour = np.asarray(index.tz_convert(MARKET_TZ).hour, dtype=float)
    rng = np.random.default_rng(20260908)
    return pd.Series(
        60.0 + 40.0 * np.sin((hour - 5.0) * np.pi / 12.0)
        + rng.normal(0.0, 15.0, len(index)),
        index=index,
        name="price_eur_mwh",
    )


def test_the_oracle_settles_identically_under_both_modes() -> None:
    """The free correctness check the whole extension is built around.

    The oracle's belief does not depend on tau, so all its scenarios are one
    scenario, its curve is one step at the realised price, and clearing it
    returns the schedule it started from. Any disagreement here is a clearing
    bug, and it would invalidate every curve number measured.
    """
    prices = _prices("2024-01-01", "2024-01-20")

    schedule = summarise(
        run_backtest(
            prices, OraclePolicy(prices), BATTERY, HOURLY, PROTOCOL,
            bidding="schedule",
        )
    )
    curve = summarise(
        run_backtest(
            prices, OraclePolicy(prices), BATTERY, HOURLY, PROTOCOL,
            bidding="curve",
        )
    )

    assert curve.profit_eur == pytest.approx(schedule.profit_eur, abs=1e-6)
    assert curve.discharged_mwh == pytest.approx(schedule.discharged_mwh, abs=1e-6)
    assert curve.clipped_mwh == pytest.approx(0.0, abs=1e-9)
    assert curve.mean_curve_steps == pytest.approx(1.0)


def test_the_oracle_sweeps_one_level_not_five() -> None:
    """K-1 of the oracle's solves would be repeats, so they are not run.

    Asserted through the curve it produces: a single step means a single
    scenario was solved.
    """
    prices = _prices("2024-01-01", "2024-01-12")

    result = run_backtest(
        prices, OraclePolicy(prices), BATTERY, HOURLY, PROTOCOL, bidding="curve"
    )

    assert all(day.curve_steps == pytest.approx(1.0) for day in result.evaluated)


def test_the_floor_produces_real_curves_and_records_the_clip() -> None:
    """A policy with genuine uncertainty gets more than one step somewhere.

    Both diagnostics are the ones the v2.5 result is read through: a mean step
    count near 1.0 means the scenario sweep produced no curve, and the clipped
    fraction is what the "clip, no penalty" settlement is knowingly not
    charging for.
    """
    prices = _prices("2024-01-01", "2024-02-20")

    result = run_backtest(
        prices, FloorPolicy(prices), BATTERY, HOURLY, PROTOCOL, bidding="curve"
    )
    metrics = summarise(result)

    assert metrics.mean_curve_steps > 1.0
    assert metrics.clipped_mwh >= 0.0
    assert metrics.clipped_fraction >= 0.0


def test_a_schedule_run_reports_no_clip_and_one_step() -> None:
    """The fixed-schedule path is untouched: its dispatch is feasible already."""
    prices = _prices("2024-01-01", "2024-01-15")

    metrics = summarise(
        run_backtest(
            prices, FloorPolicy(prices), BATTERY, HOURLY, PROTOCOL,
            bidding="schedule",
        )
    )

    assert metrics.clipped_mwh == pytest.approx(0.0)
    assert metrics.mean_curve_steps == pytest.approx(1.0)


def test_the_settlement_mode_is_recorded_on_the_result() -> None:
    """Two numbers that are not comparable must say which is which."""
    prices = _prices("2024-01-01", "2024-01-12")

    for mode in ("schedule", "curve"):
        result = run_backtest(
            prices, OraclePolicy(prices), BATTERY, HOURLY, PROTOCOL, bidding=mode
        )
        assert result.bidding == mode


def test_a_policy_without_quantiles_cannot_bid_curves() -> None:
    """Refused up front rather than failing on the first day of a long run."""

    class _PointOnly:
        name = "point_only"

        def prices_for(self, day: object, window: pd.DatetimeIndex) -> np.ndarray:
            return np.zeros(len(window))

    prices = _prices("2024-01-01", "2024-01-12")

    with pytest.raises(TypeError, match="cannot bid curves"):
        run_backtest(
            prices, _PointOnly(), BATTERY, HOURLY, PROTOCOL,
            bidding="curve",
        )


def test_an_unknown_bidding_mode_is_refused() -> None:
    prices = _prices("2024-01-01", "2024-01-12")

    with pytest.raises(ValueError, match="must be 'schedule' or 'curve'"):
        run_backtest(
            prices, OraclePolicy(prices), BATTERY, HOURLY, PROTOCOL,
            bidding="limit_order",
        )
