"""Reading the frozen snapshot back.

Deliberately *not* in :mod:`bess_arb.data`. That package is the one-shot pull
and is not modified after the snapshot commit (CLAUDE.md invariant 6);
reading a committed Parquet is ordinary application code that every later
slice needs — the backtest now, the forecaster and the chart afterwards — and
putting it there would make the frozen package a place edits keep landing.

Nothing here fills a gap or interpolates. A snapshot that fails
:func:`~bess_arb.timeline.validate_index` is a snapshot the backtest must not
run on, because the failure mode of a silently patched index is a profit
number nobody can trace back to a price.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from bess_arb.config import Config, RegimeWindow
from bess_arb.timeline import Regime, delivery_day, validate_index

__all__ = ["SnapshotMissingError", "delivery_days", "load_prices", "regime_of"]

PRICE_COLUMN = "price_eur_mwh"


class SnapshotMissingError(FileNotFoundError):
    """The regime's Parquet is not in the working tree."""


def regime_of(config: Config, regime_name: str) -> Regime:
    """The period length declared for a regime, from config and not guessed."""
    window: RegimeWindow = config.data.regime(regime_name)
    return window.regime


def load_prices(config: Config, regime_name: str) -> pd.Series:
    """Realised day-ahead prices for one regime, on its own UTC grid.

    Validated against the calendar on the way out: tz, ordering, uniqueness,
    uniform spacing at the regime's ``dt_h``, and the exact row count the
    delivery days imply. That last check is what a 23- or 25-hour day would
    fail, and it costs milliseconds once per run.
    """
    window = config.data.regime(regime_name)
    path: Path = config.data.parquet_path(regime_name)
    if not path.is_file():
        raise SnapshotMissingError(
            f"no snapshot for regime {regime_name!r} at {path}. The frozen "
            "snapshot is committed; a clone should already have it."
        )

    frame = pd.read_parquet(path)
    if PRICE_COLUMN not in frame.columns:
        raise KeyError(
            f"{path.name} has no {PRICE_COLUMN!r} column; found {list(frame.columns)}"
        )

    index = pd.DatetimeIndex(frame.index)
    validate_index(
        index,
        window.regime,
        first_day=window.first_day,
        last_day=window.last_day,
    )

    prices = frame[PRICE_COLUMN].astype("float64")
    if prices.isna().any():
        raise ValueError(
            f"{path.name}: {int(prices.isna().sum())} missing prices; the "
            "snapshot is supposed to be gap-free"
        )
    prices.index = index
    prices.name = PRICE_COLUMN
    return prices


def delivery_days(prices: pd.Series) -> list[dt.date]:
    """The local delivery days a UTC series covers, in order.

    Grouping is by market-local day, so 22:00 UTC on 31 December belongs to
    1 January — which is what the market means and what a UTC-date group-by
    gets wrong every day of the year.
    """
    days = delivery_day(pd.DatetimeIndex(prices.index))
    return sorted(set(days))
