"""The no-information floor: a climatological price vector.

``docs/DECISIONS.md`` §2.5. The floor is the null hypothesis — without it,
"the forecast captures 84% of the bound" cannot be contradicted, because a
high number reads as skill and a low one as a hard market and nothing
separates the two.

**It is a price vector, not a heuristic.** §2.5 describes the *effect* —
charge in the historically cheapest periods, discharge in the historically
dearest — and it would be easy to write that directly as "charge the N
cheapest periods". That would be a second dispatch rule with its own implicit
constraints, and the comparison with the bound would stop meaning anything.
So instead the hour-of-day × month average is handed to the same optimiser as
every other policy, and the cheap-periods behaviour falls out.

Three choices worth stating
---------------------------

**The average is causal.** For delivery day D it uses only prices stamped
before noon (market local) on D-1, which is CLAUDE.md invariant 1 applied
without an exception. The literal reading of the invariant is on *timestamps*,
so this discards the afternoon and evening of D-1 even though those prices
were published the day before and are genuinely known at the gate. That costs
half a day out of a multi-year average and is worth paying: the alternative is
a floor that needs a carve-out in the no-lookahead test slice 4 installs, and
a test with a carve-out in it guards less than it appears to.

The cost of being causal is that the average is thin in the first weeks of the
backtest and converges over the first year. That is a real weakness, and it
runs in the *unflattering* direction for honesty — a weak floor makes the
forecast policy look better — so it is measured rather than argued about:
:meth:`FloorPolicy.diagnostics` counts how often each fallback fired.

**The average expands; it does not roll.** Every price before the gate
counts, with no trailing window and no decay. A lookback length would be a
tuning knob on the null hypothesis, and a tuned null hypothesis is one that
can be moved until the headline looks right.

**The key is local wall-clock time.** ``(month, hour, minute)`` in
Europe/Madrid, because the shape of a price day is a local phenomenon — the
solar trough and the evening peak sit at local clock times, not UTC ones. For
the hourly regime the minute is always zero, so the key reduces to exactly the
hour-of-day × month of §2.5; for the quarter-hourly regime it extends to the
96 quarters without an index into the day, which is what keeps it correct on
the 92- and 100-period transition days.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from bess_arb.model.spec import FloatArray
from bess_arb.timeline import MARKET_TZ, gate_close_utc

__all__ = ["FloorPolicy"]


class FloorPolicy:
    """Climatological prices, using only what predates the gate."""

    name = "floor"

    def __init__(self, prices: pd.Series, *, min_observations: int = 1) -> None:
        if min_observations < 1:
            raise ValueError(
                f"min_observations must be at least 1, got {min_observations}"
            )
        self._min_observations = min_observations

        index = pd.DatetimeIndex(prices.index)
        if not index.is_monotonic_increasing:
            raise ValueError("the price series must be sorted to be read causally")

        local = index.tz_convert(MARKET_TZ)
        values = np.asarray(prices.to_numpy(dtype=np.float64))
        stamps = _nanoseconds(index)

        month = np.asarray(local.month, dtype=np.int64)
        hour = np.asarray(local.hour, dtype=np.int64)
        minute = np.asarray(local.minute, dtype=np.int64)

        # Two key levels, so a month with no history yet can still fall back
        # on the daily shape rather than on a flat line.
        time_of_day = hour * 100 + minute
        self._by_month_time = _prefix_sums(month * 10_000 + time_of_day, stamps, values)
        self._by_time = _prefix_sums(time_of_day, stamps, values)
        self._all = _prefix_sums(np.zeros_like(month), stamps, values)

        self._fallbacks: dict[str, int] = {
            "month_time": 0,
            "time": 0,
            "grand_mean": 0,
            "no_history": 0,
        }

    def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray:
        """The climatological price of each period in ``window``.

        One cutoff for the whole window: the gate closes once, for the day
        being decided, and the second day of a 48-hour horizon does not get
        to see further ahead than the first.
        """
        cutoff = int(gate_close_utc(day).as_unit("ns").value)
        local = window.tz_convert(MARKET_TZ)
        month = np.asarray(local.month, dtype=np.int64)
        hour = np.asarray(local.hour, dtype=np.int64)
        minute = np.asarray(local.minute, dtype=np.int64)

        time_keys = hour * 100 + minute
        month_time_keys = month * 10_000 + time_keys

        out = np.empty(len(window), dtype=np.float64)
        for position in range(len(window)):
            out[position] = self._climatological(
                int(month_time_keys[position]), int(time_keys[position]), cutoff
            )
        return out

    def _climatological(self, month_time: int, time: int, cutoff: int) -> float:
        """The fallback ladder, most specific first."""
        for level, key, table in (
            ("month_time", month_time, self._by_month_time),
            ("time", time, self._by_time),
            ("grand_mean", 0, self._all),
        ):
            mean = _mean_before(table, key, cutoff, self._min_observations)
            if mean is not None:
                self._fallbacks[level] += 1
                return mean

        # No price at all predates the gate. A flat vector is the honest
        # answer: with no history there is no spread to arbitrage, and the
        # optimiser correctly does nothing. These days are inside the warm-up.
        self._fallbacks["no_history"] += 1
        return 0.0

    def diagnostics(self) -> dict[str, int]:
        """How many periods were priced at each level of the fallback ladder.

        Reported with the result. A large ``time`` or ``grand_mean`` count
        means the floor spent a meaningful part of the backtest without a
        populated month-of-year average, which weakens it — and a weak floor
        flatters everything measured against it.
        """
        return dict(self._fallbacks)


def _nanoseconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer nanoseconds since the epoch, whatever resolution came in.

    ``DatetimeIndex.asi8`` returns the raw integers *in the index's own unit*,
    and a Parquet round trip hands back microsecond resolution while
    ``Timestamp.value`` is always nanoseconds. Comparing the two directly is
    wrong by a factor of a thousand, which puts every gate a thousand times
    too early — so the whole series appears to predate the cut and the floor
    silently reads the future. Both sides are pinned to nanoseconds here and
    at the one place a cutoff is built.
    """
    return np.asarray(index.as_unit("ns").asi8, dtype=np.int64)


class _PrefixSums:
    """Per-key sorted timestamps with a running sum, for causal means.

    A cumulative sum per key answers "the mean of this key's observations
    before instant t" in one ``searchsorted``, for any t and in any order.
    The alternative — a group-by per delivery day — is O(days x rows) and
    tempts one into carrying mutable state that only works if days are
    visited in order, which the tests deliberately do not do.
    """

    __slots__ = ("cumsum", "stamps")

    def __init__(self, stamps: np.ndarray, cumsum: np.ndarray) -> None:
        self.stamps = stamps
        self.cumsum = cumsum


def _prefix_sums(
    keys: np.ndarray, stamps: np.ndarray, values: np.ndarray
) -> dict[int, _PrefixSums]:
    table: dict[int, _PrefixSums] = {}
    for key in np.unique(keys):
        selected = keys == key
        table[int(key)] = _PrefixSums(
            stamps=stamps[selected], cumsum=np.cumsum(values[selected])
        )
    return table


def _mean_before(
    table: dict[int, _PrefixSums], key: int, cutoff: int, minimum: int
) -> float | None:
    """Mean of a key's observations strictly before ``cutoff``, or ``None``.

    ``side="left"`` is the load-bearing argument: an observation stamped
    exactly at the gate instant is excluded. That is the invariant's own
    wording — *no timestamp later than* noon on D-1 — read strictly at the
    boundary, where a reading has to be chosen and the conservative one costs
    nothing.
    """
    entry = table.get(key)
    if entry is None:
        return None
    count = int(np.searchsorted(entry.stamps, cutoff, side="left"))
    if count < minimum:
        return None
    return float(entry.cumsum[count - 1] / count)
