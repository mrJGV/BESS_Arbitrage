"""Invariant 4, turned into a test: never 24 periods per day.

Twice a year a Spanish delivery day is 23 or 25 hours long — 92 or 100
quarter-hourly periods. Code that assumes 24 breaks on those days *silently*:
no exception, just a day's worth of prices shifted by an hour and an
unexplained step in profit. This file is the guard, and it runs against the
real transition dates inside the frozen data window rather than against
invented ones.

Both cases are genuinely present in the data. The quarter-hourly regime,
which produces the headline, contains 2025-10-26 (100 periods) and
2026-03-29 (92 periods) — so neither branch of this test is hypothetical.

The transition dates are written out by hand *and* derived from the zone
database, and the two are asserted equal. Either alone would be weak: a
hardcoded list goes stale when the window moves, and a derivation that
silently returned nothing would let every other assertion here vacuously
pass.
"""

from __future__ import annotations

import datetime as dt

import pytest

from bess_arb.timeline import (
    Regime,
    day_index,
    delivery_day,
    dst_transition_days,
    hours_in_day,
    periods_in_day,
    periods_per_day,
    to_market_time,
    utc_index,
)

HOURLY = Regime("hourly", 1.0)
QUARTER_HOURLY = Regime("quarter_hourly", 0.25)

# The data window: hourly to 30 September 2025, quarter-hourly after it.
WINDOW_FIRST_DAY = dt.date(2022, 1, 1)
WINDOW_LAST_DAY = dt.date(2026, 8, 20)

# Last Sunday of March: 02:00 local becomes 03:00, so the day loses an hour.
SPRING_FORWARD = (
    dt.date(2022, 3, 27),
    dt.date(2023, 3, 26),
    dt.date(2024, 3, 31),
    dt.date(2025, 3, 30),
    dt.date(2026, 3, 29),
)
# Last Sunday of October: 03:00 local becomes 02:00, so the day gains one.
FALL_BACK = (
    dt.date(2022, 10, 30),
    dt.date(2023, 10, 29),
    dt.date(2024, 10, 27),
    dt.date(2025, 10, 26),
)


def test_the_derived_transition_days_are_the_real_ones() -> None:
    """The list below is hand-written; the module derives its own from the zone.

    Asserting they agree is what stops every other test in this file from
    passing vacuously against an empty derivation.
    """
    derived = dst_transition_days(WINDOW_FIRST_DAY, WINDOW_LAST_DAY)

    assert derived == tuple(sorted(SPRING_FORWARD + FALL_BACK))


@pytest.mark.parametrize("day", SPRING_FORWARD)
def test_spring_forward_days_are_short(day: dt.date) -> None:
    assert hours_in_day(day) == 23.0
    assert periods_in_day(day, HOURLY) == 23
    assert periods_in_day(day, QUARTER_HOURLY) == 92


@pytest.mark.parametrize("day", FALL_BACK)
def test_fall_back_days_are_long(day: dt.date) -> None:
    assert hours_in_day(day) == 25.0
    assert periods_in_day(day, HOURLY) == 25
    assert periods_in_day(day, QUARTER_HOURLY) == 100


@pytest.mark.parametrize("day", SPRING_FORWARD + FALL_BACK)
@pytest.mark.parametrize("regime", [HOURLY, QUARTER_HOURLY], ids=lambda r: r.name)
def test_no_transition_day_has_the_default_period_count(
    day: dt.date, regime: Regime
) -> None:
    """The invariant stated as an assertion, not as a comment."""
    default = round(24.0 / regime.dt_h)

    assert periods_in_day(day, regime) != default


@pytest.mark.parametrize("day", SPRING_FORWARD + FALL_BACK)
@pytest.mark.parametrize("regime", [HOURLY, QUARTER_HOURLY], ids=lambda r: r.name)
def test_the_day_index_has_as_many_rows_as_the_day_has_periods(
    day: dt.date, regime: Regime
) -> None:
    """The count and the grid are computed separately; they must agree.

    ``periods_in_day`` divides an elapsed duration, ``day_index`` steps a UTC
    range between two converted midnights. Nothing forces them to match
    except that both are right.
    """
    index = day_index(day, regime)

    assert len(index) == periods_in_day(day, regime)
    assert index.is_monotonic_increasing
    assert not index.has_duplicates


def test_the_repeated_local_hour_appears_twice_on_a_fall_back_day() -> None:
    """What a 25-hour day actually *is*, not just how long it is.

    On 26 October 2025 local 02:00 happens twice, at +02:00 and again at
    +01:00. A correct index contains both, distinguishable by UTC instant and
    by offset. Storing local time is what loses one of them.
    """
    local = to_market_time(day_index(dt.date(2025, 10, 26), HOURLY))
    repeated = [stamp for stamp in local if stamp.hour == 2]

    assert len(repeated) == 2
    assert repeated[0] != repeated[1]
    assert repeated[0].utcoffset() != repeated[1].utcoffset()


def test_the_skipped_local_hour_is_absent_on_a_spring_forward_day() -> None:
    """29 March 2026 has no local 02:00 at all; the index must not invent one."""
    local = to_market_time(day_index(dt.date(2026, 3, 29), HOURLY))

    assert [stamp for stamp in local if stamp.hour == 2] == []
    assert [stamp for stamp in local if stamp.hour == 1] != []
    assert [stamp for stamp in local if stamp.hour == 3] != []


def test_the_repeated_quarter_hours_are_all_present() -> None:
    """The fall-back hour is four periods in the quarter-hourly regime, twice."""
    local = to_market_time(day_index(dt.date(2025, 10, 26), QUARTER_HOURLY))

    assert len([stamp for stamp in local if stamp.hour == 2]) == 8


@pytest.mark.parametrize("regime", [HOURLY, QUARTER_HOURLY], ids=lambda r: r.name)
def test_a_transition_day_groups_into_exactly_one_delivery_day(
    regime: Regime,
) -> None:
    """UTC-date grouping would split these days; local grouping must not.

    A 25-hour day starting at 22:00 UTC on the 25th straddles two UTC dates.
    Grouping on the UTC date would report two partial days and no error.
    """
    for day in (dt.date(2025, 10, 26), dt.date(2026, 3, 29)):
        index = day_index(day, regime)
        days = delivery_day(index)

        assert set(days) == {day}
        assert len(days) == periods_in_day(day, regime)
        assert len(set(index.date)) == 2  # it really does straddle two UTC dates


@pytest.mark.parametrize("year", [2022, 2023, 2024, 2025])
@pytest.mark.parametrize("regime", [HOURLY, QUARTER_HOURLY], ids=lambda r: r.name)
def test_a_year_of_periods_adds_up(year: int, regime: Regime) -> None:
    """Per-day counts and the continuous grid must agree over a whole year.

    The two errors this catches are opposite and both plausible: a grid that
    quietly drops the missing hour in March, and per-day counts that add a
    day the grid does not have. One 23-hour day and one 25-hour day cancel,
    so the year still totals 24 h/day — which is exactly why a year-level
    check alone would not be enough, and why the per-day assertions above
    exist too.
    """
    first = dt.date(year, 1, 1)
    last = dt.date(year, 12, 31)

    per_day = periods_per_day(first, last, regime)
    grid = utc_index(first, last, regime)

    assert int(per_day.sum()) == len(grid)
    assert len(per_day) == (last - first).days + 1
