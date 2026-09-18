"""The headline: how much of the gap between the floor and the bound a policy closes.

``% of bound`` puts the floor at 85.7% on the headline regime, so a forecast
that adds 1.6 points of it reads as 87.3% and looks like most of the answer.
The quantity a forecast is actually responsible for is the stretch between the
two references, so this module reports

    share of gap = (policy − floor) / (bound − floor)

which is 0 for the floor, 1 for perfect foresight, and negative for a policy
that loses to knowing nothing. It is a ratio of settled profits over the same
delivery days, so the annualisation, the battery size and the day count all
cancel out of it.

Whether that figure is distinguishable from zero
------------------------------------------------

At the central ``c_deg`` the forecast's margin over the floor is tens of euros a
day against a daily profit above a thousand, so the point estimate alone does
not say whether the forecast earned anything. The interval comes from a
**paired circular block bootstrap over delivery days**:

- *Paired*: every resample draws the same days for the policy, the floor and
  the bound, so the day-to-day variation they share — a wide-spread week lifts
  all three — cancels in the ratio instead of widening the interval.
- *Blocks*: a day's profit is not independent of the days before it. The state
  of charge a policy ends a day with is chosen against the next day's prices
  and carried into it, and price regimes (a windy week, a gas move) persist for
  days. Resampling single days would treat that dependence as noise and
  understate the interval. A block keeps ``block_days`` consecutive days
  together; a week also keeps the weekday/weekend pattern inside each block.
- *Circular*: blocks wrap from the last day to the first, so every day is
  equally likely to be drawn. A plain moving-block scheme under-samples the
  ends of the run.

The interval is the percentile interval of the resampled shares. A block
length is a judgement, not a derivation; the result is only as good as the
blocks are long enough to carry the dependence, and a longer block gives a
wider interval if that dependence is persistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from bess_arb.backtest.metrics import summarise
from bess_arb.backtest.runner import BacktestResult
from bess_arb.model.spec import FloatArray

__all__ = ["GapShare", "compare_to_floor", "gap_share", "resampled_totals"]

_GAP_TOLERANCE_EUR = 1e-9


def gap_share(policy_eur: float, floor_eur: float, bound_eur: float) -> float | None:
    """``(policy − floor) / (bound − floor)``, or ``None`` when there is no gap.

    Undefined rather than infinite when the bound earns no more than the floor:
    at a degradation cost above the market's spread both stay idle, and the
    truthful answer is that there was nothing to close — the same reasoning
    that makes ``% of bound`` undefined when the bound earns nothing.
    """
    gap = bound_eur - floor_eur
    if gap <= _GAP_TOLERANCE_EUR:
        return None
    return (policy_eur - floor_eur) / gap


@dataclass(frozen=True, slots=True)
class GapShare:
    """One policy against the floor and the bound, at one degradation cost."""

    policy: str
    c_deg_eur_mwh: float
    days: int
    share: float | None
    """The point estimate: the share of the floor-to-bound gap the policy closes."""
    margin_points: float | None
    """``(policy − floor) / bound``, the same margin in points of the bound."""
    margin_eur_per_mw_year: float
    interval: tuple[float, float] | None
    """Percentile interval of the resampled shares at ``confidence``.

    ``None`` when some resample has no gap to divide by. On a real run the gap
    is summed over hundreds of days and this does not happen; on a short test
    run it can, and an interval computed over the resamples that happen to
    have a gap would be conditioned on the answer.
    """
    at_or_below_floor: float | None
    """Fraction of resamples in which the policy does not beat the floor.

    A summary of the bootstrap distribution, not a p-value: the resamples
    are centred on the observed share rather than on zero, so this is the
    one-sided counterpart of the interval and nothing more. Small means the
    margin survives resampling the days.
    """
    confidence: float
    block_days: int
    resamples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "c_deg_eur_mwh": self.c_deg_eur_mwh,
            "days": self.days,
            "share_of_gap": _rounded(self.share),
            "margin_points": _rounded(self.margin_points),
            "margin_eur_per_mw_year": round(self.margin_eur_per_mw_year, 2),
            "share_of_gap_interval": (
                None
                if self.interval is None
                else [round(self.interval[0], 6), round(self.interval[1], 6)]
            ),
            "at_or_below_floor": _rounded(self.at_or_below_floor),
            "confidence": self.confidence,
            "block_days": self.block_days,
            "resamples": self.resamples,
        }


def compare_to_floor(
    policy: BacktestResult,
    floor: BacktestResult,
    bound: BacktestResult,
    *,
    block_days: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> GapShare:
    """The policy's share of the floor-to-bound gap, with its bootstrap interval.

    The three runs must cover the same delivery days after the warm-up; a
    comparison over different days would be a comparison of calendars.
    """
    if block_days < 1:
        raise ValueError(f"block_days must be at least 1, got {block_days}")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

    runs = (policy, floor, bound)
    calendars = [tuple(day.day for day in run.evaluated) for run in runs]
    if calendars[0] != calendars[1] or calendars[0] != calendars[2]:
        raise ValueError(
            f"{policy.policy}, {floor.policy} and {bound.policy} do not cover "
            "the same delivery days after the warm-up"
        )

    daily = np.array(
        [[day.profit_eur for day in run.evaluated] for run in runs], dtype=float
    )
    policy_eur, floor_eur, bound_eur = daily.sum(axis=1)
    summary = summarise(policy)

    totals = resampled_totals(daily, block_days, resamples, np.random.default_rng(seed))
    gaps = totals[:, 2] - totals[:, 1]
    lifts = totals[:, 0] - totals[:, 1]
    interval: tuple[float, float] | None = None
    below: float | None = None
    if np.all(gaps > _GAP_TOLERANCE_EUR):
        shares = lifts / gaps
        tail = (1.0 - confidence) / 2.0
        low, high = np.quantile(shares, [tail, 1.0 - tail])
        interval = (float(low), float(high))
        below = float(np.mean(shares <= 0.0))

    return GapShare(
        policy=policy.policy,
        c_deg_eur_mwh=policy.params.c_deg_eur_mwh,
        days=daily.shape[1],
        share=gap_share(policy_eur, floor_eur, bound_eur),
        margin_points=(
            None
            if abs(bound_eur) < _GAP_TOLERANCE_EUR
            else (policy_eur - floor_eur) / bound_eur
        ),
        margin_eur_per_mw_year=(
            (policy_eur - floor_eur) / summary.years / policy.params.p_max_mw
        ),
        interval=interval,
        at_or_below_floor=below,
        confidence=confidence,
        block_days=block_days,
        resamples=resamples,
    )


def resampled_totals(
    daily: FloatArray, block_days: int, resamples: int, rng: np.random.Generator
) -> FloatArray:
    """Totals of each series over ``resamples`` circular block resamples.

    ``daily`` is ``(series, days)``; the result is ``(resamples, series)``, and
    every series is resampled with the same block starts, which is what makes
    the comparison paired.

    Each resample is ``ceil(days / block)`` blocks with uniform random starts,
    the last one cut short so the resample has exactly ``days`` days. A block
    longer than the run is shortened to the run, where every resample is a
    rotation and the interval collapses to the point, which is the right
    answer for a run too short to say anything.

    Built from prefix sums of the series laid end to end twice, so a block's
    total is one subtraction and a resample never materialises its day
    indices — a 1,361-day run at 10,000 resamples would otherwise be a
    13.6-million-element index array per series.
    """
    n_series, n_days = daily.shape
    if n_days == 0:
        raise ValueError("cannot resample a run with no days")
    block = min(block_days, n_days)
    n_blocks = math.ceil(n_days / block)
    last = n_days - (n_blocks - 1) * block

    starts = rng.integers(0, n_days, size=(resamples, n_blocks))
    prefix = np.concatenate(
        [np.zeros((n_series, 1)), np.cumsum(np.tile(daily, 2), axis=1)], axis=1
    )
    offsets = np.arange(n_days)
    full = prefix[:, offsets + block] - prefix[:, offsets]
    tail = prefix[:, offsets + last] - prefix[:, offsets]

    totals = np.empty((resamples, n_series))
    for series in range(n_series):
        totals[:, series] = (
            full[series, starts[:, :-1]].sum(axis=1) + tail[series, starts[:, -1]]
        )
    return totals


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
