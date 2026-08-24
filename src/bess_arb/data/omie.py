"""OMIE ``marginalpdbc`` files — the independent cross-check on ESIOS.

Two sources for the same number is the only way to find out that one of them
means something slightly different from what you assumed. ESIOS is the
primary because it also carries the forecast series; OMIE is the market
operator publishing its own clearing prices, so an agreement between them
rules out a whole class of quiet error — wrong indicator, wrong geography,
wrong sign convention, an off-by-one hour from a timezone slip.

The off-by-one is the one that matters and the reason this parser exists in
this shape. **OMIE files carry no timestamps.** A row says "day 2024-06-15,
period 7" and nothing more, so reading one requires knowing how many periods
that day had and when each began — which is exactly what
:mod:`bess_arb.timeline` computes. Mapping OMIE's period numbers through the
same calendar the rest of the project uses turns the cross-check into a test
of that calendar too: on 26 October the file has 25 rows, and if the timeline
disagrees the parse fails loudly instead of shifting a day's prices by an
hour.

File format, from OMIE's own layout: a ``MARGINALPDBC;`` header, then
semicolon-separated ``year;month;day;period;price_pt;price_es;`` rows, then a
``*`` terminator. Prices are €/MWh. **The Portuguese price is the fifth field
and the Spanish price the sixth** — they are equal whenever the
Spain-Portugal interconnector is uncongested, which is most days, so a
transposition here would pass a casual eyeball and fail only on the days that
matter.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pandas as pd
import requests

from bess_arb.timeline import INDEX_NAME, DayLike, Regime, day_index, periods_in_day

__all__ = [
    "DEFAULT_BASE_URL",
    "OmieError",
    "fetch_marginalpdbc",
    "fetch_marginalpdbc_range",
    "marginalpdbc_url",
    "parse_marginalpdbc",
]

DEFAULT_BASE_URL = "https://www.omie.es/es/file-download"

_HEADER = "MARGINALPDBC"
_TERMINATOR = "*"


class OmieError(RuntimeError):
    """A file could not be fetched, or did not parse as ``marginalpdbc``."""


def marginalpdbc_url(day: DayLike, *, base_url: str = DEFAULT_BASE_URL) -> str:
    """Download URL for one delivery day's marginal price file."""
    stamp = _as_date(day).strftime("%Y%m%d")
    return f"{base_url}?parents=marginalpdbc&filename=marginalpdbc_{stamp}.1"


def _as_date(day: DayLike) -> dt.date:
    if isinstance(day, str):
        return dt.date.fromisoformat(day)
    if isinstance(day, dt.datetime):
        raise TypeError("a delivery day must be a date, not a datetime")
    if isinstance(day, dt.date):
        return day
    raise TypeError(f"expected a date or an ISO date string, got {type(day).__name__}")


def parse_marginalpdbc(text: str, day: DayLike, regime: Regime) -> pd.DataFrame:
    """Parse one file into a UTC-indexed frame of Portuguese and Spanish prices.

    The row count is checked against the calendar rather than against 24: a
    23-row file in March and a 25-row file in October are correct, and a
    24-row file on either of those days is the bug.
    """
    expected_day = _as_date(day)
    expected_periods = periods_in_day(expected_day, regime)

    rows: list[tuple[int, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(_HEADER) or line.startswith(_TERMINATOR):
            continue
        fields = [field.strip() for field in line.split(";") if field.strip() != ""]
        if len(fields) < 6:
            continue
        year, month, dom, period = (int(field) for field in fields[:4])
        if dt.date(year, month, dom) != expected_day:
            raise OmieError(
                f"file for {expected_day} contains a row dated "
                f"{dt.date(year, month, dom)}"
            )
        rows.append((period, _as_price(fields[4]), _as_price(fields[5])))

    if not rows:
        raise OmieError(f"no data rows parsed for {expected_day}")

    rows.sort()
    periods = [period for period, _, _ in rows]
    if periods != list(range(1, len(rows) + 1)):
        raise OmieError(
            f"{expected_day}: period numbers are not 1..{len(rows)}, got "
            f"{periods[:5]}..{periods[-5:]}"
        )
    if len(rows) != expected_periods:
        raise OmieError(
            f"{expected_day}: file has {len(rows)} periods, the calendar says "
            f"{expected_periods} for regime {regime.name!r}. A DST day read as "
            "an ordinary one shifts a whole day of prices."
        )

    return pd.DataFrame(
        {
            "price_pt_eur_mwh": [pt for _, pt, _ in rows],
            "price_es_eur_mwh": [es for _, _, es in rows],
        },
        index=day_index(expected_day, regime),
    )


def _as_price(field: str) -> float:
    """OMIE writes decimals with a point here; accept a comma regardless."""
    try:
        return float(field.replace(",", "."))
    except ValueError as error:
        raise OmieError(f"cannot read {field!r} as a price") from error


def fetch_marginalpdbc(
    day: DayLike,
    regime: Regime,
    *,
    session: requests.Session | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 60.0,
) -> pd.DataFrame:
    """Download and parse one delivery day. No token needed — OMIE is public."""
    url = marginalpdbc_url(day, base_url=base_url)
    getter = session if session is not None else requests
    try:
        response = getter.get(url, timeout=timeout_s)
    except requests.RequestException as error:
        raise OmieError(f"GET {url} failed: {error}") from error
    if response.status_code != 200:
        raise OmieError(f"GET {url} returned {response.status_code}")
    return parse_marginalpdbc(response.text, day, regime)


def fetch_marginalpdbc_range(
    first_day: DayLike,
    last_day: DayLike,
    regime: Regime,
    *,
    session: requests.Session | None = None,
    base_url: str = DEFAULT_BASE_URL,
    progress: bool = False,
) -> pd.DataFrame:
    """Download and parse a range of delivery days, both ends inclusive."""
    frames = [
        fetch_marginalpdbc(day, regime, session=session, base_url=base_url)
        for day in _each_day(first_day, last_day, progress=progress)
    ]
    if not frames:
        raise OmieError("empty day range")
    return pd.concat(frames).sort_index().rename_axis(INDEX_NAME)


def _each_day(
    first_day: DayLike, last_day: DayLike, *, progress: bool
) -> Iterator[dt.date]:
    first = _as_date(first_day)
    last = _as_date(last_day)
    if last < first:
        raise ValueError(f"last_day {last} precedes first_day {first}")
    for offset in range((last - first).days + 1):
        day = first + dt.timedelta(days=offset)
        if progress:
            print(f"  omie {day}", flush=True)
        yield day
