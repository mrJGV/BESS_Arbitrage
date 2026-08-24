"""The project's calendar: UTC storage, Europe/Madrid delivery days.

Named ``timeline`` and not ``calendar`` so it cannot shadow the stdlib module
of that name — the same reasoning that makes ``model/pyomo.py`` safe only
under a src-layout.

Why this module exists
----------------------

A delivery day is a *local* object and a time series is a *UTC* one, and the
two disagree twice a year. On the last Sunday of March a Spanish day is 23
hours long, on the last Sunday of October it is 25. Code that divides a year
by 24 is wrong on those days, and wrong quietly: the error surfaces as an
unexplained jump in profit rather than as an exception. So nothing here
counts periods — everything *derives* them from the offset the zone actually
had, and :func:`periods_in_day` is the only place that arithmetic happens
(CLAUDE.md invariant 4).

The division of labour: **timestamps are stored and reasoned about in UTC**,
which is uniform and has no ambiguous or missing hours, and **days are
delimited in Europe/Madrid**, which is what the market means by a day. Every
index this module returns is tz-aware UTC; :func:`to_market_time` is the only
sanctioned way to present one.

Both DST cases are in the frozen data window, so the tests are not
hypothetical: the quarter-hourly regime contains 2025-10-26 (100 periods) and
2026-03-29 (92 periods).

A note on the gate hour
-----------------------

:func:`gate_close_utc` returns 12:00 *Europe/Madrid* on D-1, so it is 11:00
UTC in winter and 10:00 UTC in summer. ``docs/DECISIONS.md`` §2.2 writes the
gate as "12:00 CET", which is unambiguous only in winter — Spain is on CEST
from late March to late October, and a literal fixed UTC+1 would place the
gate at 11:00 UTC all year.

Local noon is the conservative reading of the two. In summer it closes the
information set an hour *earlier* than a fixed UTC+1 would, so a feature that
passes the no-lookahead test under this definition passes under the other as
well. It is also what the market actually does: OMIE quotes session times in
peninsular local time. If that ever needs to change it is one constant here
and nowhere else.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

__all__ = [
    "GATE_CLOSE_LOCAL_HOUR",
    "INDEX_NAME",
    "MARKET_TZ",
    "UTC",
    "DayLike",
    "Regime",
    "day_bounds",
    "day_index",
    "delivery_day",
    "dst_transition_days",
    "gate_close_utc",
    "hours_in_day",
    "periods_in_day",
    "periods_per_day",
    "to_market_time",
    "utc_index",
    "validate_index",
]

MARKET_TZ = ZoneInfo("Europe/Madrid")
"""Where a delivery day begins and ends. Presentation only — never storage."""

UTC = dt.UTC

INDEX_NAME = "datetime_utc"
"""Every index this module builds carries this name, so a frame that has been
through a merge or a Parquet round trip still says which convention it is in."""

GATE_CLOSE_LOCAL_HOUR = 12
"""Day-ahead bid submission closes at noon, market local time. See the module
docstring for why local noon rather than a fixed UTC+1."""

DayLike = dt.date | str
"""A delivery day: a ``date``, or an ISO ``YYYY-MM-DD`` string.

Deliberately *not* an instant. A tz-aware timestamp does not name a day
without a convention for which zone decides, and silently picking one is the
bug this module exists to prevent. Use :func:`delivery_day` to go from
instants to days.
"""


@dataclass(frozen=True, slots=True)
class Regime:
    """A market time-unit regime: a name and a period length.

    The two that exist are hourly (``dt_h = 1.0``, to 30 September 2025) and
    quarter-hourly (``dt_h = 0.25``, from 1 October 2025, when SDAC moved to
    15-minute MTUs). Neither is hardcoded here — both are built from
    ``config/params.yaml``, because invariant 7 puts every numeric there and
    because the boundary date is a fact about the market that a reader should
    be able to find in one place.
    """

    name: str
    dt_h: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("regime name must not be empty")
        if not self.dt_h > 0.0:
            raise ValueError(f"dt_h must be positive, got {self.dt_h}")
        if self.dt_h > 24.0:
            raise ValueError(f"dt_h must not exceed a day, got {self.dt_h}")

    @property
    def step(self) -> pd.Timedelta:
        """The period length as a pandas offset, for ``date_range`` and diffs."""
        return pd.Timedelta(hours=self.dt_h)


def _as_day(day: DayLike) -> dt.date:
    """Normalise a delivery day, refusing instants.

    ``datetime`` (and therefore ``pd.Timestamp``) subclasses ``date``, so an
    unguarded ``isinstance(day, date)`` would silently accept a timestamp and
    throw its time away — which on a transition day is exactly how a 23-hour
    day gets treated as a 24-hour one.
    """
    if isinstance(day, str):
        return dt.date.fromisoformat(day)
    if isinstance(day, dt.datetime):
        raise TypeError(
            "a delivery day must be a date, not a datetime: an instant does "
            "not name a day without saying which zone decides. Use "
            "delivery_day() to map instants to days."
        )
    if isinstance(day, dt.date):
        return day
    raise TypeError(f"expected a date or an ISO date string, got {type(day).__name__}")


def _local_midnight(day: dt.date) -> pd.Timestamp:
    """The instant a Spanish delivery day starts.

    Safe on both transition days: Madrid switches at 02:00/03:00, so local
    midnight is never ambiguous and never skipped.
    """
    return pd.Timestamp(dt.datetime.combine(day, dt.time()), tz=MARKET_TZ)


def _day_range(first_day: DayLike, last_day: DayLike) -> list[dt.date]:
    first = _as_day(first_day)
    last = _as_day(last_day)
    if last < first:
        raise ValueError(f"last_day {last} precedes first_day {first}")
    return [first + dt.timedelta(days=n) for n in range((last - first).days + 1)]


def hours_in_day(day: DayLike) -> float:
    """Wall-clock length of a delivery day: 23, 24 or 25.

    Measured as the elapsed time between consecutive local midnights, so the
    answer comes from the zone database rather than from a rule about which
    Sunday it is.
    """
    d = _as_day(day)
    start = _local_midnight(d)
    end = _local_midnight(d + dt.timedelta(days=1))
    return float((end - start).total_seconds()) / 3600.0


def periods_in_day(day: DayLike, regime: Regime) -> int:
    """Market time units in a delivery day — 23/24/25 or 92/96/100.

    The one place in the project where periods-per-day is computed. Anything
    that needs the number asks here; nothing multiplies 24 by anything.
    """
    hours = hours_in_day(day)
    exact = hours / regime.dt_h
    count = round(exact)
    if abs(exact - count) > 1e-9:
        raise ValueError(
            f"regime {regime.name!r} with dt_h={regime.dt_h} does not divide "
            f"the {hours:g}-hour day {_as_day(day)} into whole periods"
        )
    return int(count)


def day_index(day: DayLike, regime: Regime) -> pd.DatetimeIndex:
    """The UTC timestamps of one delivery day's periods.

    Each label is the *start* of its period, matching how OMIE numbers them
    (period 1 is H1Q1, 00:00-00:15 local).
    """
    d = _as_day(day)
    return pd.date_range(
        start=_local_midnight(d).tz_convert(UTC),
        periods=periods_in_day(d, regime),
        freq=regime.step,
        name=INDEX_NAME,
    )


def utc_index(
    first_day: DayLike, last_day: DayLike, regime: Regime
) -> pd.DatetimeIndex:
    """The UTC index spanning a range of delivery days, both ends inclusive.

    Built by converting the two bounding local midnights to UTC and stepping
    uniformly between them. That is DST-correct for free: UTC has no
    transitions, so the irregularity lives entirely in the endpoints and the
    index needs no special-casing in the middle.
    """
    first = _as_day(first_day)
    last = _as_day(last_day)
    if last < first:
        raise ValueError(f"last_day {last} precedes first_day {first}")
    return pd.date_range(
        start=_local_midnight(first).tz_convert(UTC),
        end=_local_midnight(last + dt.timedelta(days=1)).tz_convert(UTC),
        freq=regime.step,
        inclusive="left",
        name=INDEX_NAME,
    )


def day_bounds(
    first_day: DayLike, last_day: DayLike
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Half-open local bounds ``[start, end)`` covering whole delivery days.

    What an API request wants: asking ESIOS for
    ``2024-01-01T00:00:00+01:00`` to ``2024-02-01T00:00:00+01:00`` asks for
    January as Spain defines it. Asking in UTC would clip an hour off each
    end, and asking with a naive string lets the server pick a zone.
    """
    first = _as_day(first_day)
    last = _as_day(last_day)
    if last < first:
        raise ValueError(f"last_day {last} precedes first_day {first}")
    return _local_midnight(first), _local_midnight(last + dt.timedelta(days=1))


def periods_per_day(first_day: DayLike, last_day: DayLike, regime: Regime) -> pd.Series:
    """Periods per delivery day over a range, indexed by local date.

    Useful as a diagnostic and as manifest evidence: summing it must give the
    row count of the corresponding Parquet file, which is a stronger check
    than "the file is about the right size".
    """
    days = _day_range(first_day, last_day)
    return pd.Series(
        [periods_in_day(d, regime) for d in days],
        index=pd.Index(days, name="delivery_day"),
        name="periods",
        dtype="int64",
    )


def dst_transition_days(first_day: DayLike, last_day: DayLike) -> tuple[dt.date, ...]:
    """The days in the range that are not 24 hours long.

    Derived from the zone, not from a hardcoded list, so it stays right if the
    window moves — or if Europe ever does abolish the changeover. The test
    pins the known dates against this, which is what stops a derivation that
    silently returns nothing from passing.
    """
    return tuple(d for d in _day_range(first_day, last_day) if hours_in_day(d) != 24.0)


def gate_close_utc(day: DayLike) -> pd.Timestamp:
    """When the information set for delivery day ``day`` closes.

    Noon market-local on D-1, as UTC. No input used to decide day ``day`` may
    carry a timestamp at or after this instant (invariant 1). See the module
    docstring on why this is local noon rather than a fixed UTC+1.
    """
    eve = _as_day(day) - dt.timedelta(days=1)
    local = pd.Timestamp(
        dt.datetime.combine(eve, dt.time(GATE_CLOSE_LOCAL_HOUR)), tz=MARKET_TZ
    )
    return local.tz_convert(UTC)


def delivery_day(index: pd.DatetimeIndex) -> pd.Index:
    """Which local delivery day each UTC instant belongs to.

    The grouping key for anything that is per-day: it puts 22:00 UTC on
    31 December into 1 January, which is what the market means and what naive
    UTC-date grouping gets wrong every single day of the year.
    """
    _require_utc(index)
    return pd.Index(index.tz_convert(MARKET_TZ).date, name="delivery_day")


def to_market_time(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Present a stored UTC index in Europe/Madrid. Presentation only."""
    _require_utc(index)
    return index.tz_convert(MARKET_TZ)


def _require_utc(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(index).__name__}")
    if index.tz is None:
        raise ValueError(
            "index is timezone-naive; this project stores UTC and a naive "
            "index is the shape a lookahead bug arrives in"
        )


def validate_index(
    index: pd.DatetimeIndex,
    regime: Regime,
    *,
    first_day: DayLike | None = None,
    last_day: DayLike | None = None,
) -> None:
    """Assert an index is a well-formed UTC series for ``regime``.

    Checks tz, ordering, uniqueness and uniform spacing; if both days are
    given, checks the index is exactly the one :func:`utc_index` would build.
    Raises rather than returning a flag — a malformed index should stop a
    frozen snapshot from being written, not be noted in a log.
    """
    _require_utc(index)
    if str(index.tz) not in ("UTC", "utc"):
        raise ValueError(f"index must be stored in UTC, got {index.tz}")
    if not index.is_monotonic_increasing:
        raise ValueError("index is not sorted")
    if index.has_duplicates:
        raise ValueError("index contains duplicate timestamps")

    if len(index) > 1:
        gaps = index.to_series().diff().dropna().unique()
        if len(gaps) != 1 or pd.Timedelta(gaps[0]) != regime.step:
            raise ValueError(
                f"index is not uniformly spaced at {regime.step} "
                f"(regime {regime.name!r}); found gaps {sorted(set(gaps))}"
            )

    if first_day is not None and last_day is not None:
        expected = utc_index(first_day, last_day, regime)
        if len(index) != len(expected):
            raise ValueError(
                f"index has {len(index)} rows, expected {len(expected)} for "
                f"{_as_day(first_day)}..{_as_day(last_day)} at "
                f"dt_h={regime.dt_h}"
            )
        if not index.equals(expected):
            raise ValueError("index does not match the expected UTC grid")
