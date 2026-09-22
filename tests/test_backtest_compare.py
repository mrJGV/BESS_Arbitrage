"""The headline share and its interval, checked against arithmetic done by hand.

The bootstrap is tested in two ways. Its bookkeeping is compared against a
naive version that builds every resample's day indices explicitly, so the
prefix-sum shortcut cannot drift from what a circular block bootstrap is. Its
behaviour is pinned on inputs whose answer is known without resampling: a
margin that is the same every day has an interval of zero width, and a margin
that is pure noise has an interval that straddles zero.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from bess_arb.backtest.compare import (
    GapShare,
    compare_to_floor,
    gap_share,
    resampled_totals,
)
from bess_arb.backtest.runner import BacktestResult, DayResult
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryParams, SolveStatus
from bess_arb.timeline import Regime

HOURLY = Regime("hourly", 1.0)
BATTERY = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=17.0)
PROTOCOL = HorizonConfig(
    window_days=2, implement_days=1, warmup_days=1, soc_initial_fraction=0.5
)
SEED = 20260819
TOL = 1e-9


def _run(policy: str, profits: list[float], *, first: int = 0) -> BacktestResult:
    """One warm-up day at a silly profit, then ``profits``, one per day."""
    days = [0.0, *profits]
    return BacktestResult(
        policy=policy,
        regime=HOURLY,
        params=BATTERY,
        protocol=PROTOCOL,
        days=tuple(
            DayResult(
                day=dt.date(2025, 1, 1) + dt.timedelta(days=first + index),
                warmup=index == 0,
                periods=24,
                window_periods=48,
                hours=24.0,
                status=SolveStatus.OPTIMAL,
                profit_eur=1e6 if index == 0 else profit,
                believed_objective_eur=profit,
                charged_mwh=20.0,
                discharged_mwh=17.0,
                soc_start_mwh=10.0,
                soc_end_mwh=10.0,
            )
            for index, profit in enumerate(days)
        ),
        window_lengths=(48,),
    )


def _compare(
    policy: BacktestResult,
    floor: BacktestResult,
    bound: BacktestResult,
    *,
    seed: int = SEED,
) -> GapShare:
    return compare_to_floor(
        policy,
        floor,
        bound,
        block_days=7,
        resamples=2000,
        confidence=0.95,
        seed=seed,
    )


def _interval(result: GapShare) -> tuple[float, float]:
    assert result.interval is not None
    return result.interval


def test_the_share_is_the_fraction_of_the_gap_between_floor_and_bound() -> None:
    """Floor 800, bound 1,000, policy 850: a quarter of the €200 gap."""
    assert gap_share(850.0, 800.0, 1000.0) == pytest.approx(0.25, abs=TOL)
    assert gap_share(800.0, 800.0, 1000.0) == pytest.approx(0.0, abs=TOL)
    assert gap_share(1000.0, 800.0, 1000.0) == pytest.approx(1.0, abs=TOL)


def test_losing_to_the_floor_is_a_negative_share() -> None:
    assert gap_share(780.0, 800.0, 1000.0) == pytest.approx(-0.1, abs=TOL)


def test_no_gap_makes_the_share_undefined() -> None:
    """Not infinite: above the market's spread both references stay idle."""
    assert gap_share(0.0, 0.0, 0.0) is None
    assert gap_share(5.0, 10.0, 10.0) is None


def test_the_comparison_reads_only_the_days_after_the_warm_up() -> None:
    """The warm-up day carries €1m in every run; none of it may reach the share."""
    floor = _run("floor", [800.0] * 20)
    bound = _run("oracle", [1000.0] * 20)
    policy = _run("forecast", [850.0] * 20)

    result = _compare(policy, floor, bound)

    assert result.days == 20
    assert result.share == pytest.approx(0.25, abs=TOL)
    assert result.margin_points == pytest.approx(0.05, abs=TOL)
    # €50 a day on 10 MW over 20 days of 24 hours, annualised on 8,766 h.
    expected = 50.0 * 20 / (480.0 / 8766.0) / 10.0
    assert result.margin_eur_per_mw_year == pytest.approx(expected, abs=1e-6)


def test_a_margin_that_never_varies_has_an_interval_of_zero_width() -> None:
    """Every resample has the same days' worth of the same numbers."""
    floor = _run("floor", [800.0] * 30)
    bound = _run("oracle", [1000.0] * 30)
    policy = _run("forecast", [850.0] * 30)

    result = _compare(policy, floor, bound)

    assert _interval(result) == pytest.approx((0.25, 0.25), abs=TOL)
    assert result.at_or_below_floor == 0.0


def test_a_margin_that_is_pure_noise_straddles_zero() -> None:
    rng = np.random.default_rng(7)
    floor_days = rng.normal(1000.0, 200.0, 300)
    noise = rng.normal(0.0, 100.0, 300)
    noise -= noise.mean()
    floor = _run("floor", list(floor_days))
    bound = _run("oracle", list(floor_days + 150.0))
    policy = _run("forecast", list(floor_days + noise))

    result = _compare(policy, floor, bound)
    low, high = _interval(result)

    assert low < 0.0 < high
    assert result.at_or_below_floor is not None
    assert 0.2 < result.at_or_below_floor < 0.8


def test_a_consistent_margin_through_noise_excludes_zero() -> None:
    rng = np.random.default_rng(11)
    floor_days = rng.normal(1000.0, 200.0, 300)
    floor = _run("floor", list(floor_days))
    bound = _run("oracle", list(floor_days + 150.0))
    policy = _run("forecast", list(floor_days + 40.0 + rng.normal(0.0, 20.0, 300)))

    result = _compare(policy, floor, bound)

    assert _interval(result)[0] > 0.0
    assert result.at_or_below_floor == 0.0


def test_the_interval_is_reproducible_from_the_seed() -> None:
    """Invariant 8, and a check that the seed is really what drives the draw."""
    rng = np.random.default_rng(3)
    floor_days = rng.normal(1000.0, 200.0, 100)
    floor = _run("floor", list(floor_days))
    bound = _run("oracle", list(floor_days + 150.0))
    policy = _run("forecast", list(floor_days + rng.normal(10.0, 80.0, 100)))

    first = _compare(policy, floor, bound)
    again = _compare(policy, floor, bound)
    other = _compare(policy, floor, bound, seed=SEED + 1)

    assert first == again
    assert first.interval != other.interval


def test_the_summary_dictionary_carries_the_interval() -> None:
    floor = _run("floor", [800.0] * 30)
    bound = _run("oracle", [1000.0] * 30)
    policy = _run("forecast", [850.0] * 30)

    row = _compare(policy, floor, bound).as_dict()

    assert row["share_of_gap"] == pytest.approx(0.25)
    assert row["share_of_gap_interval"] == pytest.approx([0.25, 0.25])
    assert row["block_days"] == 7
    assert row["confidence"] == 0.95


def test_runs_over_different_days_are_refused() -> None:
    floor = _run("floor", [800.0] * 10)
    bound = _run("oracle", [1000.0] * 10)
    policy = _run("forecast", [850.0] * 10, first=1)

    with pytest.raises(ValueError, match="same delivery days"):
        _compare(policy, floor, bound)


@pytest.mark.parametrize(("n_days", "block"), [(23, 7), (21, 7), (5, 1), (4, 9)])
def test_the_resampled_totals_match_explicit_circular_blocks(
    n_days: int, block: int
) -> None:
    """The prefix-sum shortcut against building every resample's indices.

    Both sides draw the same block starts from the same generator state, so
    this checks the bookkeeping — wrap-around, the shortened last block, a
    block longer than the run — and not the randomness.
    """
    rng = np.random.default_rng(5)
    daily = rng.normal(size=(3, n_days))
    resamples = 50

    fast = resampled_totals(daily, block, resamples, np.random.default_rng(99))

    length = min(block, n_days)
    n_blocks = -(-n_days // length)
    starts = np.random.default_rng(99).integers(0, n_days, size=(resamples, n_blocks))
    slow = np.empty((resamples, 3))
    for row in range(resamples):
        indices = np.concatenate(
            [(start + np.arange(length)) % n_days for start in starts[row]]
        )[:n_days]
        slow[row] = daily[:, indices].sum(axis=1)

    np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_a_block_as_long_as_the_run_only_rotates_it() -> None:
    """Every resample is the whole run, so every total is the run's total."""
    daily = np.arange(12, dtype=float).reshape(2, 6)

    totals = resampled_totals(daily, 6, 20, np.random.default_rng(1))

    np.testing.assert_allclose(totals, np.tile(daily.sum(axis=1), (20, 1)))
