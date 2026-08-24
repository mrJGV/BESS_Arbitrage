"""The rest of the calendar contract: the gate, the grid, and what it rejects.

``test_dst.py`` covers the transition days. This file covers the other two
things ``timeline`` is responsible for — the 12:00 D-1 information gate that
invariant 1 rests on, and refusing to hand out a malformed index. The
rejections matter as much as the happy path: a naive timestamp or a silently
dropped hour is the shape a lookahead bug arrives in, and a frozen snapshot
must not be writable from one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from bess_arb.timeline import (
    MARKET_TZ,
    Regime,
    day_index,
    delivery_day,
    gate_close_utc,
    hours_in_day,
    periods_in_day,
    to_market_time,
    utc_index,
    validate_index,
)

HOURLY = Regime("hourly", 1.0)
QUARTER_HOURLY = Regime("quarter_hourly", 0.25)


# --- the information gate ------------------------------------------------


def test_the_gate_is_noon_market_local_on_the_day_before() -> None:
    gate = gate_close_utc(dt.date(2024, 1, 15))
    local = gate.tz_convert(MARKET_TZ)

    assert local.date() == dt.date(2024, 1, 14)
    assert (local.hour, local.minute) == (12, 0)


def test_the_gate_tracks_the_offset_rather_than_a_fixed_hour() -> None:
    """Noon in Madrid is 11:00 UTC in winter and 10:00 UTC in summer.

    A gate pinned to a constant UTC hour would be an hour *late* in summer,
    admitting information the bidder did not have. See the ``timeline``
    module docstring on why local noon is the conservative reading of
    "12:00 CET".
    """
    assert gate_close_utc(dt.date(2024, 1, 15)).hour == 11
    assert gate_close_utc(dt.date(2024, 7, 15)).hour == 10


@pytest.mark.parametrize(
    "day",
    [
        dt.date(2024, 1, 15),
        dt.date(2024, 7, 15),
        dt.date(2025, 10, 26),  # the 25-hour day
        dt.date(2026, 3, 29),  # the 23-hour day
        dt.date(2025, 10, 27),  # the day after a fall-back
        dt.date(2026, 3, 30),  # the day after a spring-forward
    ],
)
def test_the_gate_closes_before_the_day_it_decides(day: dt.date) -> None:
    """The property invariant 1 actually needs, checked across transitions.

    Bids for day D close before D starts, by roughly twelve hours. "Roughly"
    is the point: on a transition day the wall-clock distance is 11 or 13
    hours, so a test written as an exact offset would be asserting the bug.
    """
    first_period = day_index(day, HOURLY)[0]
    gate = gate_close_utc(day)

    assert gate < first_period
    assert pd.Timedelta(hours=10) <= first_period - gate <= pd.Timedelta(hours=14)


def test_every_gate_in_a_year_precedes_its_day() -> None:
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D").date

    for day in days:
        assert gate_close_utc(day) < day_index(day, QUARTER_HOURLY)[0]


# --- days, instants, and the difference ----------------------------------


def test_a_day_is_a_date_and_an_instant_is_not() -> None:
    """``datetime`` subclasses ``date``, so this has to be refused explicitly.

    An unguarded ``isinstance(day, date)`` accepts a timestamp and discards
    its time — which on a transition day turns a 23-hour day into a 24-hour
    one without complaint.
    """
    assert periods_in_day("2024-06-15", HOURLY) == 24
    assert periods_in_day(dt.date(2024, 6, 15), HOURLY) == 24

    with pytest.raises(TypeError, match="not a datetime"):
        periods_in_day(dt.datetime(2024, 6, 15, 0, 0), HOURLY)
    with pytest.raises(TypeError, match="not a datetime"):
        periods_in_day(pd.Timestamp("2024-06-15", tz="UTC"), HOURLY)


def test_the_delivery_day_is_local_not_utc() -> None:
    """A Spanish day starts at 22:00 or 23:00 UTC on the previous date.

    Grouping on the UTC date would misfile the first one or two periods of
    every single day of the year, not just the transition ones.
    """
    winter = day_index(dt.date(2024, 1, 15), HOURLY)
    summer = day_index(dt.date(2024, 7, 15), HOURLY)

    assert str(winter[0]) == "2024-01-14 23:00:00+00:00"
    assert str(summer[0]) == "2024-07-14 22:00:00+00:00"
    assert set(delivery_day(winter)) == {dt.date(2024, 1, 15)}
    assert set(delivery_day(summer)) == {dt.date(2024, 7, 15)}


def test_period_labels_are_period_starts() -> None:
    """OMIE numbers period 1 as H1Q1, 00:00-00:15 local. Labels match that."""
    local = to_market_time(day_index(dt.date(2024, 6, 15), QUARTER_HOURLY))

    assert str(local[0]).startswith("2024-06-15 00:00:00")
    assert str(local[1]).startswith("2024-06-15 00:15:00")
    assert str(local[-1]).startswith("2024-06-15 23:45:00")


def test_an_index_spans_its_days_inclusively() -> None:
    index = utc_index("2024-06-15", "2024-06-17", HOURLY)

    assert len(index) == 72
    assert set(delivery_day(index)) == {
        dt.date(2024, 6, 15),
        dt.date(2024, 6, 16),
        dt.date(2024, 6, 17),
    }


def test_a_reversed_range_is_an_error() -> None:
    with pytest.raises(ValueError, match="precedes"):
        utc_index("2024-06-17", "2024-06-15", HOURLY)


# --- regimes -------------------------------------------------------------


@pytest.mark.parametrize("dt_h", [0.0, -1.0, 25.0])
def test_an_impossible_period_length_is_rejected(dt_h: float) -> None:
    with pytest.raises(ValueError):
        Regime("nonsense", dt_h)


def test_a_period_length_that_does_not_divide_a_short_day_is_an_error() -> None:
    """Two-hour periods fit a 24-hour day and not a 23-hour one.

    Rather than round 11.5 to something plausible, say so. The market has no
    such regime today, but the failure mode — a period length that only works
    on ordinary days — is exactly what this module exists to catch.
    """
    two_hourly = Regime("two_hourly", 2.0)

    assert periods_in_day("2024-06-15", two_hourly) == 12
    with pytest.raises(ValueError, match="whole periods"):
        periods_in_day("2026-03-29", two_hourly)


def test_hours_in_day_is_derived_not_tabulated() -> None:
    assert hours_in_day("2024-06-15") == 24.0
    assert hours_in_day("2026-03-29") == 23.0
    assert hours_in_day("2025-10-26") == 25.0


# --- what validate_index refuses -----------------------------------------


def test_a_well_formed_index_validates() -> None:
    index = utc_index("2025-10-24", "2025-10-28", QUARTER_HOURLY)

    validate_index(index, QUARTER_HOURLY, first_day="2025-10-24", last_day="2025-10-28")


def test_a_naive_index_is_refused() -> None:
    naive = pd.date_range("2024-06-15", periods=24, freq="1h")

    with pytest.raises(ValueError, match="timezone-naive"):
        validate_index(naive, HOURLY)


def test_a_local_index_is_refused() -> None:
    """Storing Europe/Madrid loses one of the two 02:00s every October."""
    local = pd.date_range("2024-06-15", periods=24, freq="1h", tz=MARKET_TZ)

    with pytest.raises(ValueError, match="must be stored in UTC"):
        validate_index(local, HOURLY)


def test_an_unsorted_index_is_refused() -> None:
    index = utc_index("2024-06-15", "2024-06-16", HOURLY)

    with pytest.raises(ValueError, match="not sorted"):
        validate_index(index[::-1], HOURLY)


def test_a_duplicated_timestamp_is_refused() -> None:
    index = utc_index("2024-06-15", "2024-06-16", HOURLY)
    doubled = index.append(index[-1:]).sort_values()

    with pytest.raises(ValueError, match="duplicate"):
        validate_index(doubled, HOURLY)


def test_a_missing_period_is_refused() -> None:
    """The failure a raw API pull actually produces: one row quietly absent."""
    index = utc_index("2024-06-15", "2024-06-16", HOURLY)
    gapped = index.delete(10)

    with pytest.raises(ValueError, match="uniformly spaced"):
        validate_index(gapped, HOURLY)


def test_the_wrong_regime_is_refused() -> None:
    index = utc_index("2024-06-15", "2024-06-16", HOURLY)

    with pytest.raises(ValueError, match="uniformly spaced"):
        validate_index(index, QUARTER_HOURLY)


def test_a_short_index_is_refused_against_a_declared_span() -> None:
    index = utc_index("2024-06-15", "2024-06-16", HOURLY)

    with pytest.raises(ValueError, match="expected"):
        validate_index(index, HOURLY, first_day="2024-06-15", last_day="2024-06-17")
