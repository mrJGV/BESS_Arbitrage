"""The floor is climatological, and it is causal.

Two properties, and the second is the one that matters. A climatology is
trivially easy to compute over the whole sample, and the version that peeks
is indistinguishable from the version that does not by looking at the
numbers: both produce a smooth daily shape and a plausible result. So the
peek is tested for directly, by giving the policy a series whose future
contradicts its past and asserting that the future has no effect.

CLAUDE.md invariant 1 is written on *timestamps*, so the cut is at noon
market-local on D-1 and prices stamped after that are unavailable even though
they were published the previous day and are genuinely known. That strictness
is deliberate — see the module docstring of ``policy/floor.py`` — and the
boundary case is pinned here rather than left to be rediscovered.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.policy import build_policy
from bess_arb.policy.floor import FloorPolicy
from bess_arb.timeline import MARKET_TZ, Regime, gate_close_utc, utc_index

HOURLY = Regime("hourly", 1.0)
QUARTER_HOURLY = Regime("quarter_hourly", 0.25)


def _series(first: str, last: str, regime: Regime, values: np.ndarray) -> pd.Series:
    index = utc_index(first, last, regime)
    assert len(values) == len(index)
    return pd.Series(values, index=index, name="price_eur_mwh")


def _ramp(first: str, last: str, regime: Regime, *, scale: float = 1.0) -> pd.Series:
    """A price that is the local hour of day, times a scale.

    Deterministic and readable: the climatology of a series like this is the
    series itself, so a wrong key or a wrong timezone shows up as a number
    that is off by exactly the offset.
    """
    index = utc_index(first, last, regime)
    local = index.tz_convert(MARKET_TZ)
    return pd.Series(
        np.asarray(local.hour, dtype=float) * scale, index=index, name="price_eur_mwh"
    )


def test_the_climatology_is_the_local_hour_of_day() -> None:
    """A price equal to the local hour must average back to the local hour."""
    prices = _ramp("2022-01-01", "2022-02-28", HOURLY)
    policy = FloorPolicy(prices)
    window = utc_index("2022-02-20", "2022-02-21", HOURLY)

    believed = policy.prices_for(dt.date(2022, 2, 20), window)

    expected = np.asarray(window.tz_convert(MARKET_TZ).hour, dtype=float)
    assert believed == pytest.approx(expected)


def test_a_utc_keyed_climatology_would_fail_this() -> None:
    """The positive control for the timezone.

    In February Madrid is UTC+1, so keying on the UTC hour shifts the whole
    profile by one period. Asserting the shift is *not* present is what makes
    the previous test a test of the zone rather than of arithmetic.
    """
    prices = _ramp("2022-01-01", "2022-02-28", HOURLY)
    window = utc_index("2022-02-20", "2022-02-21", HOURLY)

    believed = FloorPolicy(prices).prices_for(dt.date(2022, 2, 20), window)

    utc_keyed = np.asarray(window.hour, dtype=float)
    assert not np.allclose(believed, utc_keyed)


def test_prices_after_the_gate_cannot_move_the_climatology() -> None:
    """The lookahead test, done by contradiction.

    Everything from the gate onward is replaced by a number nothing else in
    the series comes near. If any of it leaked into the average the believed
    prices would move, and they must not move at all.
    """
    honest = _ramp("2022-01-01", "2022-03-31", HOURLY)
    day = dt.date(2022, 3, 20)
    cutoff = gate_close_utc(day)

    poisoned = honest.copy()
    poisoned.loc[poisoned.index >= cutoff] = 10_000.0

    window = utc_index(day, day + dt.timedelta(days=1), HOURLY)
    from_honest = FloorPolicy(honest).prices_for(day, window)
    from_poisoned = FloorPolicy(poisoned).prices_for(day, window)

    assert from_poisoned == pytest.approx(from_honest)
    assert from_honest.max() < 100.0


def test_the_price_stamped_exactly_at_the_gate_is_excluded() -> None:
    """The boundary, pinned: 'no timestamp later than noon' cuts at noon.

    One period at exactly the gate instant, made extreme. Including it would
    move that hour's average by a visible amount, so the assertion is on a
    number and not on a tolerance.
    """
    day = dt.date(2022, 3, 20)
    cutoff = gate_close_utc(day)
    flat = _series(
        "2022-03-01",
        "2022-03-31",
        HOURLY,
        np.zeros(len(utc_index("2022-03-01", "2022-03-31", HOURLY))),
    )
    flat.loc[cutoff] = 1_000.0

    window = utc_index(day, day + dt.timedelta(days=1), HOURLY)
    believed = FloorPolicy(flat).prices_for(day, window)

    assert believed == pytest.approx(np.zeros(len(window)))


def test_the_average_expands_as_the_backtest_walks_forward() -> None:
    """Later days see more history, and the mean moves accordingly.

    Guards the opposite failure to the lookahead one: a policy that captured
    the prefix once and never advanced would also pass every causality check
    above while quietly freezing the climatology at day one.
    """
    index = utc_index("2022-01-01", "2022-01-31", HOURLY)
    # The level doubles halfway through the month, so the causal mean at the
    # end must sit strictly above the causal mean near the start.
    values = np.where(index < index[len(index) // 2], 10.0, 20.0)
    prices = pd.Series(values, index=index, name="price_eur_mwh")
    policy = FloorPolicy(prices)

    early = policy.prices_for(
        dt.date(2022, 1, 5), utc_index("2022-01-05", "2022-01-06", HOURLY)
    )
    late = policy.prices_for(
        dt.date(2022, 1, 30), utc_index("2022-01-30", "2022-01-31", HOURLY)
    )

    assert early.mean() == pytest.approx(10.0)
    assert 10.0 < late.mean() < 20.0


def test_the_climatology_does_not_depend_on_the_order_days_are_asked_for() -> None:
    """No hidden cursor.

    The obvious implementation advances an accumulator day by day and is
    silently wrong when a caller asks out of order — which a chart, a test or
    a parallel sweep all do.
    """
    prices = _ramp("2022-01-01", "2022-03-31", HOURLY)
    days = [dt.date(2022, 3, d) for d in (10, 25, 3, 18)]
    windows = {day: utc_index(day, day + dt.timedelta(days=1), HOURLY) for day in days}

    forward = FloorPolicy(prices)
    backward = FloorPolicy(prices)
    ascending = {day: forward.prices_for(day, windows[day]) for day in sorted(days)}
    descending = {
        day: backward.prices_for(day, windows[day])
        for day in sorted(days, reverse=True)
    }

    for day in days:
        assert ascending[day] == pytest.approx(descending[day]), day


def test_the_window_spans_two_days_and_both_are_priced() -> None:
    """A 48-hour horizon gets 48 climatological prices, not 24 repeated."""
    prices = _ramp("2022-01-01", "2022-02-28", HOURLY, scale=1.0)
    day = dt.date(2022, 2, 20)
    window = utc_index(day, day + dt.timedelta(days=1), HOURLY)

    believed = FloorPolicy(prices).prices_for(day, window)

    assert len(believed) == 48
    assert believed[:24] == pytest.approx(believed[24:])


@pytest.mark.parametrize(
    ("day", "periods"),
    [(dt.date(2022, 3, 27), 23), (dt.date(2022, 10, 30), 25)],
)
def test_transition_days_are_priced_period_by_period(
    day: dt.date, periods: int
) -> None:
    """23 and 25 hours, and a price for each.

    The floor is keyed on wall-clock time rather than on an index into the
    day precisely so this needs no special case: the hour that does not exist
    is simply never asked for, and the hour that happens twice is asked for
    twice and answered the same way both times.
    """
    prices = _ramp("2022-01-01", "2022-12-31", HOURLY)
    window = utc_index(day, day, HOURLY)
    assert len(window) == periods

    believed = FloorPolicy(prices).prices_for(day, window)

    assert len(believed) == periods
    assert np.isfinite(believed).all()


def test_the_quarter_hourly_regime_keys_on_the_quarter() -> None:
    """96 distinct climatological prices a day, not 24 repeated four times."""
    index = utc_index("2025-10-01", "2025-12-31", QUARTER_HOURLY)
    local = index.tz_convert(MARKET_TZ)
    values = np.asarray(local.hour * 4 + local.minute // 15, dtype=float)
    prices = pd.Series(values, index=index, name="price_eur_mwh")

    day = dt.date(2025, 12, 20)
    window = utc_index(day, day, QUARTER_HOURLY)
    believed = FloorPolicy(prices).prices_for(day, window)

    assert len(believed) == 96
    assert len(np.unique(np.round(believed, 6))) == 96


def test_no_history_at_all_gives_a_flat_vector() -> None:
    """The first day of the sample has nothing before its gate.

    A flat vector means zero spread, so the optimiser correctly declines to
    trade. Inventing a number here would put a fabricated decision inside the
    warm-up, where nobody would ever look at it again.
    """
    prices = _ramp("2022-01-01", "2022-01-31", HOURLY)
    day = dt.date(2022, 1, 1)
    window = utc_index(day, day + dt.timedelta(days=1), HOURLY)

    policy = FloorPolicy(prices)
    believed = policy.prices_for(day, window)

    assert believed == pytest.approx(np.zeros(len(window)))
    assert policy.diagnostics()["no_history"] == len(window)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_the_gate_holds_whatever_resolution_the_index_carries(unit: str) -> None:
    """A regression, and the reason the snapshot is the case that finds it.

    ``DatetimeIndex.asi8`` returns integers in the index's *own* unit while
    ``Timestamp.value`` is always nanoseconds, and a Parquet round trip hands
    back microseconds. Comparing them directly puts the gate a thousand times
    too early, every price appears to predate it, and the floor reads the
    entire future — with no exception raised and a perfectly plausible daily
    shape coming out. Every resolution must give the same answer.
    """
    honest = _ramp("2022-01-01", "2022-03-31", HOURLY)
    day = dt.date(2022, 3, 20)

    poisoned = honest.copy()
    poisoned.loc[poisoned.index >= gate_close_utc(day)] = 10_000.0
    poisoned.index = pd.DatetimeIndex(poisoned.index).as_unit(unit)

    window = utc_index(day, day + dt.timedelta(days=1), HOURLY).as_unit(unit)
    believed = FloorPolicy(poisoned).prices_for(day, window)

    assert believed.max() < 100.0


def test_the_diagnostics_count_every_period_they_priced() -> None:
    """The caveat is reported as a number, so it can be checked."""
    prices = _ramp("2022-01-01", "2022-06-30", HOURLY)
    policy = FloorPolicy(prices)
    day = dt.date(2022, 6, 1)
    window = utc_index(day, day + dt.timedelta(days=1), HOURLY)

    policy.prices_for(day, window)

    assert sum(policy.diagnostics().values()) == len(window)


def test_the_registry_builds_the_floor_by_name() -> None:
    prices = _ramp("2022-01-01", "2022-01-31", HOURLY)

    policy = build_policy("floor", prices)

    assert policy.name == "floor"
    with pytest.raises(ValueError, match="unknown policy"):
        build_policy("hunch", prices)
