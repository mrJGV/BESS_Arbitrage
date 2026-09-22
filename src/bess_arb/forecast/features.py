"""Features, and the instant each one became knowable.

This module exists to make CLAUDE.md invariant 1 *checkable* rather than
promised. Every column it produces is accompanied by an ``available_at``
timestamp: the instant the value could first have been read by someone
standing in the market. :meth:`FeatureTable.violations` then compares that
against the gate and returns what, if anything, leaks —
``tests/test_no_lookahead.py`` asserts the answer is empty.

The rule, in one line
---------------------

**A feature is admissible for delivery day D if and only if
``available_at <= gate(D)``**, where ``gate(D)`` is noon market-local on D-1
(:func:`bess_arb.timeline.gate_close_utc`).

That needs one distinction the invariant's own wording leaves implicit. A
series has a *valid time* — the period it describes — and a *publication
time* — when its value was first knowable. The invariant is read on
publication time, for every series alike:

- The "Prevision diaria D+1" family (indicators 1775/1777/1779) is published
  once per day covering the following day, so a value describing 20:00 on
  day D was on the wire the previous morning. That is the whole reason a
  forecast is usable at all.
- A day-ahead **price** is not known when its period is delivered either. It
  is known when OMIE publishes the auction result: the whole of delivery day
  X, every period at once, about an hour after the auction for X closes at
  noon on X-1 (:func:`bess_arb.timeline.price_published_utc`). So at the
  gate for D every price through D-1 has been public for about a day, and
  none of D's has.

The same reading governs the training filter. A row's target is admissible
for a fit made at ``gate(D)`` iff its own delivery day had been published by
then, which is every day through D-1. It is also the reading
:mod:`bess_arb.policy.floor` applies to its climatology and
:mod:`bess_arb.scenarios` to its residual pool, so every policy reads the same
history.

**What is assumed, and where it would break.** ``docs/DECISIONS.md`` §5.3
records the D+1 family as fixed at vintage D-1 and never rewritten — it is the
only family that survives the gate at all, and the rolling one was rejected on
measured evidence. What §5.3 does not fix is the *hour* on D-1, and the frozen
snapshot cannot supply it: it stores valid times and has no publication column.
So the noon deadline is one input this module takes on trust. It is
therefore written down as a constant, :data:`EXOGENOUS_PUBLISHED_AT_GATE`,
rather than buried in an expression: if
the D+1 bundle turned out to land at 14:00 on D-1, this is the single line
that would change, and every exogenous feature would become inadmissible
together rather than one of them quietly staying in. The price publication
hour is the other, held in :data:`bess_arb.timeline.PRICE_PUBLICATION_LOCAL_HOUR`
for the same reason; unlike the exogenous deadline it is inert for the model,
since any hour between the auction close and the next gate admits the same
days.

Two horizons, two information sets
----------------------------------

The backtest solves D and D+1 and implements D. At the gate for D the D+1
forecast bundle covering **D** has been published; the one covering **D+1**
has not — it goes out the following morning. So the second day of the window
is forecast from calendar and lagged prices alone. That is not a modelling
preference, it is what is on the wire, and it is why the tables are built per
``lead_days``: lead 0 carries the exogenous block and lead 1 does not.

The lag features are anchored on the *same* gate for both leads, so a column
named ``price_lag1d`` means the same history in both tables and only the
exogenous block differs.

Why lags start at one day, and not at zero
------------------------------------------

At the gate for D the last delivery day whose prices are public is D-1,
published the previous afternoon. So the shortest admissible lag is one day,
and ``price_lag1d`` carries yesterday's whole curve, evening peak included.
A ``price_lag0d`` column would be day D's own prices, which publish an hour
*after* the gate: that is the lookahead bug in this project, and
:func:`build_features` refuses a lag below :data:`FIRST_KNOWN_LAG_DAYS` rather
than trusting the config file. The test suite asks for one deliberately and
checks that it is caught.

Earlier versions of this module read the invariant on delivery time and so
started the lags at two days, leaving only the morning of D-1 in a separate
feature. That was a conservative choice, not an information-set requirement,
and it threw away the single most informative predictor a battery has:
yesterday's evening peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bess_arb.model.spec import FloatArray
from bess_arb.timeline import (
    UTC,
    Regime,
    delivery_day,
    gate_close_utc,
    price_published_index,
    price_published_utc,
    to_market_time,
)

__all__ = [
    "ALWAYS_KNOWN",
    "EXOGENOUS_COLUMNS",
    "EXOGENOUS_PUBLISHED_AT_GATE",
    "FIRST_KNOWN_LAG_DAYS",
    "TIME_OF_DAY_KEYED",
    "FeatureTable",
    "Violation",
    "build_features",
]

FIRST_KNOWN_LAG_DAYS = 1
"""The shortest admissible price lag, in delivery days.

At the gate for D the last delivery day whose prices have been published is
D-1; day D's own publish an hour after the gate. See the module docstring;
this constant is the difference between a working backtest and a number that
cannot be earned.
"""

EXOGENOUS_PUBLISHED_AT_GATE = True
"""Whether the D+1 forecast bundle for day D is on the wire by ``gate(D)``.

``docs/DECISIONS.md`` §5.3 fixes the vintage at D-1; what the snapshot cannot
confirm is the hour within that day, since it stores valid times only. See the
module docstring. Setting this ``False`` makes the whole exogenous block
inadmissible rather than silently shifting it, which is the failure mode worth
having.
"""

ALWAYS_KNOWN: tuple[str, ...] = ("month", "day_of_week", "time_of_day", "is_weekend")
"""Columns derived from the calendar alone, knowable arbitrarily far ahead.

Enumerated rather than inferred. :meth:`FeatureTable.violations` treats a
missing ``available_at`` as "always known", so a leaking column could hide by
declaring itself calendrical; the check pins this tuple against the columns
that actually carry no timestamp.
"""

TIME_OF_DAY_KEYED: tuple[str, ...] = ("price_lag", "price_hod_mean_recent")
"""Prefixes of the columns whose value is keyed on local time of day.

Everything else here is a daily or hourly aggregate and is by construction
constant inside an hour. The distinction is structural, not empirical, which
is why it is declared rather than sniffed from the data: it tells
:func:`bess_arb.forecast.lgbm._to_deviation` which columns carry intra-hour
signal to subtract an hour mean from, and a schema that depended on what a
particular snapshot happened to contain would change shape with the data.
"""

EXOGENOUS_COLUMNS: tuple[str, ...] = (
    "demand_forecast_mw",
    "wind_forecast_mw",
    "solar_forecast_mw",
    "residual_demand_mw",
    "residual_demand_dev_mw",
    "renewable_share",
)
"""The published-forecast block. Present at lead 0 only."""


@dataclass(frozen=True, slots=True)
class Violation:
    """One column that carries information from after its own gate."""

    column: str
    rows: int
    worst_lateness: pd.Timedelta
    example_target: pd.Timestamp
    example_available_at: pd.Timestamp
    example_gate: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"{self.column}: {self.rows} row(s) readable only after the gate, "
            f"worst by {self.worst_lateness}; e.g. the value for "
            f"{self.example_target} was knowable at {self.example_available_at}, "
            f"gate {self.example_gate}"
        )


@dataclass(frozen=True, slots=True)
class FeatureTable:
    """Feature values, their publication instants, and the target.

    ``values`` and ``available_at`` share an index; ``available_at`` carries
    every column except the calendrical ones. A column absent from it is
    always known, and :meth:`violations` insists that set is exactly
    :data:`ALWAYS_KNOWN`, so the exemption cannot be claimed by anything else.
    """

    values: pd.DataFrame
    available_at: pd.DataFrame
    gate: pd.Series
    """The gate each row is decided under: noon local on D-1, in UTC."""
    settled_at: pd.Series
    """When the row's target became observable — the publication of its own
    delivery day's prices, about 13:00 local on the day before delivery."""
    target: pd.Series
    lead_days: int
    regime: Regime

    @property
    def columns(self) -> list[str]:
        return list(self.values.columns)

    def violations(self) -> list[Violation]:
        """Every column that can be read only after its gate has closed.

        The mechanical form of invariant 1. Returns the offenders rather than
        raising, so a test can name them and a deliberately broken table can
        be shown to be caught.
        """
        undeclared = set(self.values.columns) - set(self.available_at.columns)
        if undeclared != set(ALWAYS_KNOWN):
            raise ValueError(
                "the columns claiming to need no publication instant are not "
                f"the declared calendrical set: {sorted(undeclared)} vs "
                f"{sorted(ALWAYS_KNOWN)}"
            )

        # Both sides as naive UTC integers: a tz-aware Series hands back an
        # object array of Timestamps, which compares elementwise in Python and
        # would turn a vector check into a slow one that still works — the
        # kind of thing nobody notices until the table is a hundred thousand
        # rows long.
        gate = self.gate.to_numpy(dtype="datetime64[ns]")
        found: list[Violation] = []
        for column in self.available_at.columns:
            stamps = self.available_at[column].to_numpy(dtype="datetime64[ns]")
            late = stamps > gate
            if not late.any():
                continue
            first = int(np.argmax(late))
            found.append(
                Violation(
                    column=column,
                    rows=int(late.sum()),
                    worst_lateness=pd.Timedelta((stamps[late] - gate[late]).max()),
                    example_target=pd.Timestamp(self.values.index[first]),
                    example_available_at=pd.Timestamp(stamps[first]),
                    example_gate=pd.Timestamp(gate[first]),
                )
            )
        return found

    def usable(self) -> pd.Series:
        """Rows with a target and at least one lagged price behind them.

        The opening days of any history have neither. Dropping them is not a
        judgement call: there is nothing to learn from a row whose price
        features are all missing.
        """
        priced = [c for c in self.values.columns if c.startswith("price_")]
        return self.target.notna() & self.values[priced].notna().any(axis=1)

    def trainable_before(self, cutoff: pd.Timestamp) -> pd.Series:
        """Rows whose target had already been published at ``cutoff``.

        The training-set filter, and the second place a leak could enter: a
        model fitted for day D must not have seen a price that had not been
        published by ``gate(D)``. Every such row's own features predate its
        own gate, which precedes its publication, so this one comparison is
        sufficient.
        """
        return self.usable() & (self.settled_at <= cutoff)


def build_features(
    prices: pd.Series,
    exog: pd.DataFrame | None,
    regime: Regime,
    *,
    lead_days: int,
    lags_days: tuple[int, ...],
) -> FeatureTable:
    """Feature table for targets at horizon-day offset ``lead_days``.

    ``prices`` is the realised price history on ``regime``'s grid — the whole
    of it, not a slice. Causality is enforced per row by ``available_at`` and
    per fit by :meth:`FeatureTable.trainable_before`, which is what lets one
    table serve every refit instead of rebuilding it per decision day.
    ``exog`` is the hourly published-forecast frame, or ``None`` to omit the
    block.
    """
    if lead_days < 0:
        raise ValueError(f"lead_days must be non-negative, got {lead_days}")
    if not lags_days:
        raise ValueError("lags_days must not be empty")
    too_recent = sorted(k for k in lags_days if k < FIRST_KNOWN_LAG_DAYS)
    if too_recent:
        raise ValueError(
            f"price lag(s) {too_recent} reach into a delivery day whose prices "
            f"are not published when the gate closes; the shortest known lag is "
            f"{FIRST_KNOWN_LAG_DAYS} day. See bess_arb.forecast.features."
        )

    index = pd.DatetimeIndex(prices.index).as_unit("ns")
    if index.tz is None:
        raise ValueError("the price history must be tz-aware UTC")
    lags = tuple(sorted(set(lags_days)))

    local = to_market_time(index)
    day = pd.Index(delivery_day(index))
    time_of_day = np.asarray(local.hour * 100 + local.minute, dtype=np.int64)

    pivot = _price_pivot(day, time_of_day, prices.to_numpy(dtype=np.float64))
    days = pd.Index(pivot.index)
    position = pd.Series(np.arange(len(days)), index=days)

    row = np.asarray(position.reindex(day).to_numpy(), dtype=np.int64)
    column = np.asarray(pivot.columns.get_indexer(time_of_day), dtype=np.int64)
    grid = pivot.to_numpy()

    day_published = _day_published_utc(days)
    day_gate = _day_gate_utc(days)
    anchor = row - lead_days  # position of the decision day D
    gate = _utc(_stamp(day_gate, anchor))

    values = pd.DataFrame(index=index)
    stamps = pd.DataFrame(index=index)

    values["month"] = np.asarray(local.month, dtype=np.int64)
    values["day_of_week"] = np.asarray(local.dayofweek, dtype=np.int64)
    values["time_of_day"] = time_of_day
    values["is_weekend"] = (np.asarray(local.dayofweek) >= 5).astype(np.int64)

    # --- lagged realised prices, keyed on local wall-clock time ------------
    # Matching on (hour, minute) rather than on a fixed row offset is what
    # keeps this right across DST: on the 23-hour day one key is simply
    # absent and arrives as NaN, which LightGBM handles natively, instead of
    # silently pairing 03:00 with the previous day's 02:00.
    #
    # A lag's publication instant is that of the delivery day it reads: the
    # whole of D-k became public at once, the afternoon before D-k started.
    for k in lags:
        values[f"price_lag{k}d"] = _cell(grid, anchor - k, column)
        stamps[f"price_lag{k}d"] = _utc(_stamp(day_published, anchor - k))

    # The aggregates over the lag window are knowable once their newest day
    # is, which is what stamps them at the newest lag's publication.
    newest, span = lags[0], len(lags)
    by_time = pivot.shift(newest).rolling(span, min_periods=1).mean().to_numpy()
    values["price_hod_mean_recent"] = _cell(by_time, anchor, column)
    stamps["price_hod_mean_recent"] = _utc(_stamp(day_published, anchor - newest))

    daily = pivot.mean(axis=1)
    values[f"price_day_mean_lag{newest}d"] = _row(daily.to_numpy(), anchor - newest)
    stamps[f"price_day_mean_lag{newest}d"] = _utc(
        _stamp(day_published, anchor - newest)
    )

    values["price_day_mean_recent"] = _row(
        daily.shift(newest).rolling(span, min_periods=1).mean().to_numpy(), anchor
    )
    stamps["price_day_mean_recent"] = _utc(_stamp(day_published, anchor - newest))

    # --- published forecasts, day D only -----------------------------------
    if exog is not None and lead_days == 0:
        block = _exogenous(exog, index, day)
        published = (
            gate
            if EXOGENOUS_PUBLISHED_AT_GATE
            else np.full(len(index), np.datetime64("NaT", "ns"))
        )
        for name in EXOGENOUS_COLUMNS:
            values[name] = block[name].to_numpy()
            stamps[name] = published

    return FeatureTable(
        values=values,
        available_at=stamps,
        gate=pd.Series(gate, index=index, name="gate"),
        # A row's target is knowable when its delivery day's prices publish,
        # not when its period is delivered — the same reading as the lags.
        settled_at=pd.Series(
            price_published_index(index), index=index, name="settled_at"
        ),
        target=pd.Series(
            prices.to_numpy(dtype=np.float64), index=index, name="price_eur_mwh"
        ),
        lead_days=lead_days,
        regime=regime,
    )


def _price_pivot(
    day: pd.Index, time_of_day: np.ndarray, values: FloatArray
) -> pd.DataFrame:
    """Prices as delivery day x local time-of-day, on a gap-free day index.

    ``first`` on a duplicate key rather than a mean: the 25-hour October day
    repeats one local hour, and averaging the two would invent a price that
    never cleared. Taking the first is a choice; inventing a number is not
    available.
    """
    frame = pd.DataFrame({"day": np.asarray(day), "tod": time_of_day, "price": values})
    pivot = frame.groupby(["day", "tod"])["price"].first().unstack()
    span = pd.date_range(min(pivot.index), max(pivot.index), freq="D").date
    return pivot.reindex(span)


def _day_published_utc(days: pd.Index) -> np.ndarray:
    """The instant each delivery day's prices were published, as a lookup array."""
    return np.array(
        [price_published_utc(d).tz_localize(None) for d in days],
        dtype="datetime64[ns]",
    )


def _day_gate_utc(days: pd.Index) -> np.ndarray:
    """``gate_close_utc`` for each delivery day, as a lookup array."""
    return np.array(
        [gate_close_utc(d).tz_localize(None) for d in days], dtype="datetime64[ns]"
    )


def _stamp(table: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Gather publication instants, ``NaT`` where the row does not exist."""
    out = np.full(len(rows), np.datetime64("NaT", "ns"))
    ok = (rows >= 0) & (rows < len(table))
    out[ok] = table[rows[ok]]
    return out


def _utc(stamps: np.ndarray) -> pd.DatetimeIndex:
    """Label naive UTC instants as UTC.

    The lookup arrays are naive because NumPy has no time zone, but nothing
    outside this module should ever see a naive timestamp — that is the shape
    a lookahead bug arrives in (see :mod:`bess_arb.timeline`), and here it
    would also make ``available_at`` incomparable with ``settled_at``.
    """
    return pd.DatetimeIndex(stamps).tz_localize(UTC)


def _cell(grid: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> FloatArray:
    out = np.full(len(rows), np.nan)
    ok = (rows >= 0) & (rows < grid.shape[0]) & (columns >= 0)
    out[ok] = grid[rows[ok], columns[ok]]
    return np.asarray(out, dtype=np.float64)


def _row(vector: np.ndarray, rows: np.ndarray) -> FloatArray:
    out = np.full(len(rows), np.nan)
    ok = (rows >= 0) & (rows < len(vector))
    out[ok] = vector[rows[ok]]
    return np.asarray(out, dtype=np.float64)


def _exogenous(
    exog: pd.DataFrame, index: pd.DatetimeIndex, day: pd.Index
) -> pd.DataFrame:
    """The published forecasts, put on the target grid without resampling.

    The D+1 family is published hourly and stayed hourly after the market
    moved to 15-minute units, so a quarter-hourly target has to read the hour
    that contains it. Done by flooring the UTC stamp rather than by
    ``ffill``: flooring cannot walk across a missing hour, so a hole in the
    exogenous series arrives as NaN instead of as the previous hour's value
    repeated four more times.
    """
    rows = exog.reindex(pd.DatetimeIndex(index).floor("h"))
    demand = rows["demand_forecast_mw"].to_numpy(dtype=np.float64)
    wind = rows["wind_forecast_mw"].to_numpy(dtype=np.float64)
    solar = rows["solar_forecast_mw"].to_numpy(dtype=np.float64)
    residual = demand - wind - solar

    out = pd.DataFrame(index=index)
    out["demand_forecast_mw"] = demand
    out["wind_forecast_mw"] = wind
    out["solar_forecast_mw"] = solar
    out["residual_demand_mw"] = residual
    # Deviation from the day's own mean, because arbitrage is decided by the
    # *shape* of a day and not by its level. Every period of day D sits in
    # the same publication, so this reads nothing the gate has not released.
    day_mean = pd.Series(residual, index=index).groupby(pd.Index(day)).transform("mean")
    out["residual_demand_dev_mw"] = residual - day_mean.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["renewable_share"] = np.where(demand > 0.0, (wind + solar) / demand, np.nan)
    return out
