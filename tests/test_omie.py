"""The OMIE parser, offline.

``marginalpdbc`` files carry no timestamps — a row says "2024-06-15, period
7" and stops there. Everything that could go wrong in reading one is
therefore a question about the calendar: which instant is period 7, and how
many periods does this day have? These tests answer both against synthetic
files, so they run without a network and fail for one reason at a time.

Synthetic files prove the parser handles the shapes; they cannot prove the
shapes are real. That evidence comes from the snapshot's ESIOS/OMIE
cross-check, which reads real files over a real month and is recorded in
``data/manifest.json``. Keeping the network out of the suite is deliberate —
a test that needs OMIE to be up is a test that fails for reasons unrelated to
this repository.
"""

from __future__ import annotations

import datetime as dt

import pytest

from bess_arb.data.omie import OmieError, marginalpdbc_url, parse_marginalpdbc
from bess_arb.timeline import Regime, to_market_time

HOURLY = Regime("hourly", 1.0)

ORDINARY_DAY = dt.date(2024, 6, 15)
SPRING_FORWARD = dt.date(2024, 3, 31)  # 23 periods
FALL_BACK = dt.date(2024, 10, 27)  # 25 periods


def _file(day: dt.date, n_periods: int, *, pt: float = 50.0, es: float = 50.0) -> str:
    """A synthetic ``marginalpdbc`` file with the real envelope."""
    rows = [
        f"{day.year};{day.month:02d};{day.day:02d};{period};{pt + period};"
        f"{es + period};"
        for period in range(1, n_periods + 1)
    ]
    return "\n".join(["MARGINALPDBC;", *rows, "*"])


def test_the_url_names_the_delivery_day() -> None:
    url = marginalpdbc_url(ORDINARY_DAY)

    assert url.endswith("marginalpdbc_20240615.1")


def test_an_ordinary_day_parses_onto_the_utc_grid() -> None:
    frame = parse_marginalpdbc(_file(ORDINARY_DAY, 24), ORDINARY_DAY, HOURLY)

    assert len(frame) == 24
    assert str(frame.index[0]) == "2024-06-14 22:00:00+00:00"
    assert str(to_market_time(frame.index)[0]).startswith("2024-06-15 00:00:00")


def test_the_spanish_price_is_the_sixth_field_not_the_fifth() -> None:
    """The two are equal on most days, so a transposition hides in plain sight.

    Only the congested days would reveal it, and by then the snapshot is
    frozen. Hence a fixture where they differ by construction.
    """
    frame = parse_marginalpdbc(
        _file(ORDINARY_DAY, 24, pt=100.0, es=200.0), ORDINARY_DAY, HOURLY
    )

    assert frame["price_pt_eur_mwh"].iloc[0] == 101.0
    assert frame["price_es_eur_mwh"].iloc[0] == 201.0


def test_a_short_day_has_twenty_three_periods() -> None:
    frame = parse_marginalpdbc(_file(SPRING_FORWARD, 23), SPRING_FORWARD, HOURLY)

    assert len(frame) == 23
    assert [stamp.hour for stamp in to_market_time(frame.index)].count(2) == 0


def test_a_long_day_has_twenty_five_periods() -> None:
    frame = parse_marginalpdbc(_file(FALL_BACK, 25), FALL_BACK, HOURLY)

    assert len(frame) == 25
    assert [stamp.hour for stamp in to_market_time(frame.index)].count(2) == 2


@pytest.mark.parametrize(
    ("day", "n_periods"),
    [(SPRING_FORWARD, 24), (FALL_BACK, 24), (ORDINARY_DAY, 23), (ORDINARY_DAY, 25)],
)
def test_a_period_count_that_contradicts_the_calendar_is_refused(
    day: dt.date, n_periods: int
) -> None:
    """The failure mode this parser exists to catch.

    A 24-row file on a transition day is not a rounding problem: read
    without complaint it shifts a whole day of prices by an hour, and the
    error surfaces much later as an unexplained step in profit.
    """
    with pytest.raises(OmieError, match="the calendar says"):
        parse_marginalpdbc(_file(day, n_periods), day, HOURLY)


def test_rows_for_the_wrong_day_are_refused() -> None:
    text = _file(dt.date(2024, 6, 16), 24)

    with pytest.raises(OmieError, match="contains a row dated"):
        parse_marginalpdbc(text, ORDINARY_DAY, HOURLY)


def test_out_of_order_rows_are_sorted_not_rejected() -> None:
    """Order is not part of the format's guarantee; completeness is."""
    lines = _file(ORDINARY_DAY, 24).splitlines()
    shuffled = "\n".join([lines[0], *reversed(lines[1:-1]), lines[-1]])

    frame = parse_marginalpdbc(shuffled, ORDINARY_DAY, HOURLY)

    assert frame.index.is_monotonic_increasing
    assert frame["price_es_eur_mwh"].iloc[0] == 51.0


def test_a_gap_in_the_period_numbers_is_refused() -> None:
    lines = _file(ORDINARY_DAY, 24).splitlines()
    del lines[7]  # period 7 goes missing; 23 rows numbered 1..24

    with pytest.raises(OmieError, match="not 1"):
        parse_marginalpdbc("\n".join(lines), ORDINARY_DAY, HOURLY)


def test_an_empty_file_is_refused() -> None:
    with pytest.raises(OmieError, match="no data rows"):
        parse_marginalpdbc("MARGINALPDBC;\n*\n", ORDINARY_DAY, HOURLY)


def test_a_comma_decimal_is_read() -> None:
    text = "MARGINALPDBC;\n" + "\n".join(
        f"2024;06;15;{period};50,25;60,75;" for period in range(1, 25)
    )

    frame = parse_marginalpdbc(text, ORDINARY_DAY, HOURLY)

    assert frame["price_es_eur_mwh"].iloc[0] == 60.75
