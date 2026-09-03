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

__all__ = [
    "EXOGENOUS_COLUMNS",
    "PRICE_COLUMN",
    "SnapshotMissingError",
    "delivery_days",
    "load_exogenous",
    "load_price_history",
    "load_prices",
    "regime_of",
]

PRICE_COLUMN = "price_eur_mwh"

EXOGENOUS_COLUMNS = (
    "demand_forecast_mw",
    "wind_forecast_mw",
    "solar_forecast_mw",
)
"""The published D+1 forecast series, in the order the manifest lists them."""


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


def load_price_history(config: Config, regime_name: str) -> pd.Series:
    """Every regime's realised prices, put on one regime's grid.

    The backtest reads one regime because a result belongs to one market
    design; the *forecaster* reads all of them, because history before the
    regime boundary is history — every price in it predates every gate in the
    quarter-hourly window, so using it is not a licence, it is just data.
    Without this the headline regime would train on 10.5 months.

    Two conversions, and only one of them is lossless:

    - **coarse to fine** — repeating each hourly price across its four
      quarters. That is exact reconstruction rather than interpolation, and
      only because of a measured fact: indicator 600 was already on a
      15-minute grid before the market moved, with the four values inside
      each hour identical to 0.0000 EUR/MWh until 30 September 2025 (the
      manifest records the downsample as lossless). The hourly regime ends on
      exactly that day, so the repetition puts back what was collapsed.
    - **fine to coarse** — averaging quarters into hours. That one does lose
      information, which is why it is not how the quarter-hourly forecast is
      produced; see :mod:`bess_arb.forecast.lgbm`.

    The result is validated for uniform spacing, so a gap where two regimes
    meet fails here rather than becoming a silently wrong lag.
    """
    target = regime_of(config, regime_name)
    pieces = [
        _on_grid(load_prices(config, window.name), window.regime, target)
        for window in config.data.regimes
    ]
    history = pd.concat(pieces).sort_index()
    history = history[~history.index.duplicated(keep="last")]
    history.index = pd.DatetimeIndex(history.index).as_unit("ns")
    validate_index(pd.DatetimeIndex(history.index), target)
    history.name = PRICE_COLUMN
    return history


def _on_grid(prices: pd.Series, source: Regime, target: Regime) -> pd.Series:
    """Move one regime's prices onto another's period length."""
    if source.dt_h == target.dt_h:
        return prices
    if source.dt_h > target.dt_h:
        index = pd.date_range(
            start=prices.index[0],
            end=prices.index[-1] + source.step - target.step,
            freq=target.step,
            name=prices.index.name,
        )
        return prices.reindex(index, method="ffill")
    return prices.resample(target.step).mean().dropna()


def load_exogenous(config: Config, regime_name: str) -> pd.DataFrame:
    """The published D+1 forecast series covering every regime, on their grid.

    They are *not* put on the price grid here. The D+1 family is published
    hourly and stayed hourly when the market moved to 15-minute units, so
    resampling it would be inventing intra-hour detail that was never
    published; the feature builder reads the hour containing each period
    instead. ``docs/DECISIONS.md`` §5.3 and §5.4 — one grid per file, and the
    forward-fill left to this slice to decide deliberately.
    """
    frames = [_exogenous_for(config, window.name) for window in config.data.regimes]
    exog = pd.concat(frames).sort_index()
    exog = exog[~exog.index.duplicated(keep="last")]
    exog.index = pd.DatetimeIndex(exog.index).as_unit("ns")
    missing = [c for c in EXOGENOUS_COLUMNS if c not in exog.columns]
    if missing:
        raise KeyError(f"the snapshot has no exogenous column(s) {missing}")
    return exog.loc[:, list(EXOGENOUS_COLUMNS)]


def _exogenous_for(config: Config, regime_name: str) -> pd.DataFrame:
    """One regime's exogenous block, from whichever file carries it.

    A regime whose own grid matches the published series keeps them in its
    price file; one whose grid is finer gets a sibling file on the published
    grid. Both cases are handled by looking, rather than by encoding which
    regime is which — that is a property of the snapshot, and the snapshot is
    frozen but not the only one that could exist.
    """
    frames: list[pd.DataFrame] = []
    for path in (
        config.data.parquet_path(regime_name),
        *config.data.exogenous_paths(regime_name),
    ):
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        present = [c for c in EXOGENOUS_COLUMNS if c in frame.columns]
        if present:
            frames.append(frame.loc[:, present])
    if not frames:
        raise SnapshotMissingError(
            f"no exogenous series for regime {regime_name!r} in {config.data.directory}"
        )
    return pd.concat(frames, axis=1)


def delivery_days(prices: pd.Series) -> list[dt.date]:
    """The local delivery days a UTC series covers, in order.

    Grouping is by market-local day, so 22:00 UTC on 31 December belongs to
    1 January — which is what the market means and what a UTC-date group-by
    gets wrong every day of the year.
    """
    days = delivery_day(pd.DatetimeIndex(prices.index))
    return sorted(set(days))
