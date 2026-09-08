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

    def __init__(
        self,
        prices: pd.Series,
        *,
        min_observations: int = 1,
        min_quantile_observations: int = 10,
    ) -> None:
        if min_observations < 1:
            raise ValueError(
                f"min_observations must be at least 1, got {min_observations}"
            )
        if min_quantile_observations < 1:
            raise ValueError(
                "min_quantile_observations must be at least 1, got "
                f"{min_quantile_observations}"
            )
        self._min_observations = min_observations
        # Serves the whole tau sweep, not one level — see
        # `_min_observations_for_quantile` on why this must not depend on tau.
        # Ten covers QUANTILE_LEVELS' most demanding level, tau=0.1.
        self._min_quantile_observations = min_quantile_observations

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
        self._quantile_fallbacks: dict[str, int] = {
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
        month_time_keys, time_keys = self._keys_for(window)

        out = np.empty(len(window), dtype=np.float64)
        for position in range(len(window)):
            out[position] = self._climatological(
                int(month_time_keys[position]), int(time_keys[position]), cutoff
            )
        return out

    def prices_for_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> FloatArray:
        """The causal tau-quantile of each period's climatology.

        v2.5's scenario source for the floor (the bid-curve extension to
        ``docs/DECISIONS.md`` §2.3): a low tau gives a cheap, low-spread
        version of the day's shape and a high tau a dear, wide one, so a sweep
        over tau traces the floor's own bid curve through the same fallback
        ladder as the mean — most-specific bucket first, same gate, same
        strict "before the cutoff" reading.
        """
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must lie strictly between 0 and 1, got {tau}")
        cutoff = int(gate_close_utc(day).as_unit("ns").value)
        month_time_keys, time_keys = self._keys_for(window)

        out = np.empty(len(window), dtype=np.float64)
        for position in range(len(window)):
            out[position] = self._climatological_quantile(
                int(month_time_keys[position]), int(time_keys[position]), cutoff, tau
            )
        return out

    @staticmethod
    def _keys_for(window: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Month-time and time-of-day bucket keys, shared by both ladders."""
        local = window.tz_convert(MARKET_TZ)
        month = np.asarray(local.month, dtype=np.int64)
        hour = np.asarray(local.hour, dtype=np.int64)
        minute = np.asarray(local.minute, dtype=np.int64)
        time_keys = hour * 100 + minute
        return month * 10_000 + time_keys, time_keys

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

    def _climatological_quantile(
        self, month_time: int, time: int, cutoff: int, tau: float
    ) -> float:
        """The quantile fallback ladder — same buckets, its own trust threshold."""
        minimum = self._min_observations_for_quantile()
        for level, key, table in (
            ("month_time", month_time, self._by_month_time),
            ("time", time, self._by_time),
            ("grand_mean", 0, self._all),
        ):
            value = _quantile_before(table, key, cutoff, minimum, tau)
            if value is not None:
                self._quantile_fallbacks[level] += 1
                return value

        self._quantile_fallbacks["no_history"] += 1
        return 0.0

    def _min_observations_for_quantile(self) -> int:
        """How many causal observations a bucket needs before any of its
        quantiles are trusted, as opposed to falling back to a coarser bucket.

        ``self._min_observations`` (default 1) is the threshold the *mean*
        uses, and reusing it here is wrong for a reason that only appears once
        the same buckets serve a quantile as well as a mean: **the two
        statistics do not become well defined at the same count.** A mean is
        usable at n=1; a quantile *spread* is identically zero there, so a
        one-observation bucket does not give a noisy scenario family, it gives
        no family at all — every tau returns the same number and the period's
        bid curve collapses to a single step.

        That failure was perverse in the way that makes it hard to notice: a
        bucket with *no* observations correctly fell through to the coarser
        time-of-day bucket and got a healthy estimate, while the bucket beside
        it holding *one* observation passed the test and produced a degenerate
        one. Measured on the headline quarter-hourly regime, where each
        (month, time-of-day) bucket occurs exactly once in the 11-month
        snapshot: 96 of 192 periods came back with zero q90-q10 spread on the
        second and third of **every month** — 24 days, 3.24% of all periods.

        **The threshold does not depend on tau, and that is load-bearing.**
        Scaling it with how extreme tau is — ``ceil(1 / min(tau, 1 - tau))``,
        ten points for tau=0.1 and two for the median — is the statistically
        natural rule and it is wrong here, because it lets **different taus
        land on different rungs of the ladder**. Measured: with nine
        observations in the August bucket, tau=0.3/0.5/0.7 read that bucket
        (night prices around EUR 180/MWh) while tau=0.1 and tau=0.9 fell
        through to the all-months bucket, whose 90th percentile is EUR
        148.92 — *below* the fine bucket's 70th. The family stopped being
        monotone in tau, which is to say it stopped being a quantile family,
        and :func:`bess_arb.bid.curve.build_curves` cannot see it because it
        sorts by price before pairing.

        So one number, applied to every tau, chosen to serve the most
        demanding level in the sweep: at ``QUANTILE_LEVELS``'s tau=0.1 a
        bucket needs about ten points for the tail to be distinguishable at
        all. A sweep reaching further into the tails needs this raised, which
        is why it is a constructor argument rather than a literal.
        """
        return max(self._min_observations, self._min_quantile_observations)

    def diagnostics(self) -> dict[str, int]:
        """How many periods were priced at each level of the fallback ladder.

        Reported with the result. A large ``time`` or ``grand_mean`` count
        means the floor spent a meaningful part of the backtest without a
        populated month-of-year average, which weakens it — and a weak floor
        flatters everything measured against it.
        """
        return dict(self._fallbacks)

    def quantile_diagnostics(self) -> dict[str, int]:
        """The same ladder count, kept separate for the quantile calls.

        Mixing this into :meth:`diagnostics` would conflate two different
        questions — how often the mean climatology was thin, and how often
        the quantile one was — under one counter that answers neither
        honestly.
        """
        return dict(self._quantile_fallbacks)


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
    """Per-key sorted timestamps, a running sum, and the raw values.

    A cumulative sum per key answers "the mean of this key's observations
    before instant t" in one ``searchsorted``, for any t and in any order.
    The alternative — a group-by per delivery day — is O(days x rows) and
    tempts one into carrying mutable state that only works if days are
    visited in order, which the tests deliberately do not do.

    ``values`` is kept alongside the cumulative sum for the same key and in
    the same time order, because a mean decomposes into a running sum but a
    quantile does not: answering "the tau-quantile of this key's observations
    before instant t" needs the observations themselves, not an aggregate of
    them. The counts here are small enough — a bucket never holds more than a
    few thousand observations over the whole snapshot — that slicing the
    prefix and calling :func:`numpy.quantile` on it is cheap; see
    :func:`_quantile_before`.
    """

    __slots__ = ("cumsum", "stamps", "values")

    def __init__(
        self, stamps: np.ndarray, cumsum: np.ndarray, values: np.ndarray
    ) -> None:
        self.stamps = stamps
        self.cumsum = cumsum
        self.values = values


def _prefix_sums(
    keys: np.ndarray, stamps: np.ndarray, values: np.ndarray
) -> dict[int, _PrefixSums]:
    table: dict[int, _PrefixSums] = {}
    for key in np.unique(keys):
        selected = keys == key
        table[int(key)] = _PrefixSums(
            stamps=stamps[selected],
            cumsum=np.cumsum(values[selected]),
            values=values[selected],
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


def _quantile_before(
    table: dict[int, _PrefixSums], key: int, cutoff: int, minimum: int, tau: float
) -> float | None:
    """The tau-quantile of a key's observations strictly before ``cutoff``.

    Same cutoff convention as :func:`_mean_before` — ``side="left"`` excludes
    an observation stamped exactly at the gate. The prefix ``values[:count]``
    is unsorted by value (it is sorted by time, which is what the cutoff
    needs); :func:`numpy.quantile` sorts it internally, which is fine at these
    bucket sizes and would not be at a full-series scale.
    """
    entry = table.get(key)
    if entry is None:
        return None
    count = int(np.searchsorted(entry.stamps, cutoff, side="left"))
    if count < minimum:
        return None
    return float(np.quantile(entry.values[:count], tau))
