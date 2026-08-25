"""Reading the frozen snapshot back onto the calendar it was written on.

``tests/test_snapshot.py`` checks the files are what the manifest claims.
This checks the *loader* — that what the backtest receives is a validated UTC
series on the regime's own grid, and that it refuses rather than repairs.

The refusals matter more than the successes here. A loader that quietly
forward-fills a gap, or hands back a timezone-naive index, produces a
backtest that runs to completion and reports a number derived from prices
that were never published.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from bess_arb.config import load_config
from bess_arb.series import (
    PRICE_COLUMN,
    SnapshotMissingError,
    delivery_days,
    load_prices,
    regime_of,
)
from bess_arb.timeline import MARKET_TZ, periods_in_day, utc_index

CONFIG = load_config()

pytestmark = pytest.mark.skipif(
    not CONFIG.data.parquet_path("hourly").is_file(),
    reason="no frozen snapshot in the working tree",
)


@pytest.mark.parametrize("regime_name", ["hourly", "quarter_hourly"])
def test_the_prices_come_back_on_the_regimes_own_grid(regime_name: str) -> None:
    window = CONFIG.data.regime(regime_name)
    prices = load_prices(CONFIG, regime_name)

    assert prices.name == PRICE_COLUMN
    assert str(prices.index.tz) == "UTC"
    assert prices.index.equals(
        utc_index(window.first_day, window.last_day, window.regime)
    )
    assert not prices.isna().any()


@pytest.mark.parametrize("regime_name", ["hourly", "quarter_hourly"])
def test_the_delivery_days_are_local_days_and_contiguous(regime_name: str) -> None:
    """22:00 UTC on 31 December belongs to 1 January, not to 31 December."""
    window = CONFIG.data.regime(regime_name)
    days = delivery_days(load_prices(CONFIG, regime_name))

    assert days[0] == window.first_day
    assert days[-1] == window.last_day
    assert len(days) == (window.last_day - window.first_day).days + 1


def test_the_first_period_of_a_day_is_local_midnight() -> None:
    """The one that a UTC-date group-by gets wrong every day of the year."""
    prices = load_prices(CONFIG, "hourly")
    day = dt.date(2024, 6, 15)
    first = prices.index[
        prices.index >= utc_index(day, day, CONFIG.data.regime("hourly").regime)[0]
    ][0]

    assert first.tz_convert(MARKET_TZ).hour == 0
    assert first.tz_convert(MARKET_TZ).date() == day


@pytest.mark.parametrize(
    ("regime_name", "day", "expected"),
    [
        ("hourly", dt.date(2024, 3, 31), 23),
        ("hourly", dt.date(2024, 10, 27), 25),
        ("quarter_hourly", dt.date(2025, 10, 26), 100),
        ("quarter_hourly", dt.date(2026, 3, 29), 92),
    ],
)
def test_transition_days_carry_their_own_period_count(
    regime_name: str, day: dt.date, expected: int
) -> None:
    """Read back from the loaded series, not from the calendar alone."""
    regime = regime_of(CONFIG, regime_name)
    prices = load_prices(CONFIG, regime_name)
    local_dates = pd.Index(prices.index.tz_convert(MARKET_TZ).date)

    assert int((local_dates == day).sum()) == expected == periods_in_day(day, regime)


def test_the_regime_comes_from_the_config_and_not_from_the_file() -> None:
    """Δt is declared, never inferred from row spacing."""
    assert regime_of(CONFIG, "hourly").dt_h == 1.0
    assert regime_of(CONFIG, "quarter_hourly").dt_h == 0.25


def test_an_unknown_regime_is_an_error() -> None:
    with pytest.raises(KeyError, match="unknown regime"):
        load_prices(CONFIG, "half_hourly")


def test_a_missing_snapshot_says_so_rather_than_returning_nothing(
    tmp_path: object,
) -> None:
    """An empty series would look like a market with no prices in it."""
    import dataclasses

    elsewhere = dataclasses.replace(
        CONFIG.data, directory=CONFIG.data.directory / "nope"
    )

    with pytest.raises(SnapshotMissingError, match="no snapshot"):
        load_prices(dataclasses.replace(CONFIG, data=elsewhere), "hourly")
