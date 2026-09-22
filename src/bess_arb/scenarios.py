"""Joint price scenarios from a policy's own causal forecast errors.

v3's scenario source: **the forecast object a battery needs is a joint
trajectory over the window, not a set of marginal quantiles.** Its decision
depends on the relative ordering of prices within the window, and ordering is
a property of the joint law.

Why marginals are not enough
----------------------------

The quantile sweep builds its K scenarios by asking each period for its own
tau-quantile at a common tau. Every period sits at the same level at once, so
the family is *comonotone* -- it assumes cross-period price surprises are
perfectly correlated. The obvious alternative -- perturbing one period against
a fixed median day -- assumes they are uncorrelated. Real surprises are
neither: most of a day's surprise moves every period together, and a part of
it does not.

A joint sample assumes neither. It reproduces whatever dependence the errors
actually have, because each scenario **is** a past error.

What a scenario is
------------------

One past decision's whole error trajectory. On delivery day D the policy
believed some vector over the 48-hour window; the window later cleared; the
difference is that decision's error. :class:`BeliefResiduals` records the
belief as it is formed, derives the error once the periods have cleared, and
samples S of those trajectories to add to today's belief.

Three properties follow from recording *windows* rather than periods, and each
one is a decision that would otherwise need enforcing:

- **The two horizon days are paired.** A recorded belief spans D and D+1 from a
  single decision, so a sampled residual carries the observed dependence across
  the horizon boundary rather than gluing two independent draws together.
- **Cross-period dependence is exact, not modelled.** No factor structure, no
  copula, no correlation matrix to shrink at 192 dimensions. Whatever the
  errors do, the sample does.
- **It is policy-agnostic.** The floor believes a climatology and the forecast
  policy believes a model; both produce one vector per window, so both get a
  joint scenario source from this one object, each sampled from *its own*
  errors.

Causality
---------

The residual pool holds only decisions whose windows had been *published* in
full by this decision's gate -- the same publication-time reading of
invariant 1 that :class:`~bess_arb.policy.floor.FloorPolicy` applies to its
climatology, and for the same reason. A window over D and D+1 is known once
D+1's prices publish, on the afternoon of D. See
:meth:`BeliefResiduals._eligible`.

Centring
--------

The pool's per-period mean is subtracted by default, so the scenario set's
expectation is exactly the policy's own point belief and v3 differs from v2 in
*dispersion alone*. Leaving it uncentred would hand the policy a free causal
bias correction and would make a v3-versus-v2 comparison a statement about two
things at once. The same reasoning as ``docs/DECISIONS.md`` section 2.4's "the
oracle differs from the policy in exactly one respect", applied one layer
down.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bess_arb.model.spec import FloatArray
from bess_arb.timeline import MARKET_TZ, gate_close_utc, price_published_index

__all__ = ["BeliefResiduals", "ScenarioConfig"]


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """How a policy draws joint scenarios. From ``config/params.yaml``'s ``bid``.

    ``min_scenario_days`` is the trust threshold on the residual pool, playing
    the role ``forecast.min_train_days`` plays for the point model and
    ``min_residual_obs`` for v2.5's shift: below it the policy declines to
    produce scenarios at all and the run falls back to the fixed schedule. It
    is not a lookback *length* -- the pool expands and never forgets, for the
    reason ``docs/DECISIONS.md`` section 2.5 gives about tuning knobs on the
    null hypothesis.
    """

    n_scenarios: int = 5
    min_scenario_days: int = 30
    centre_residuals: bool = True
    seed: int = 20260819

    def __post_init__(self) -> None:
        if self.n_scenarios < 1:
            raise ValueError(f"n_scenarios must be at least 1, got {self.n_scenarios}")
        if self.min_scenario_days < 1:
            raise ValueError(
                f"min_scenario_days must be at least 1, got {self.min_scenario_days}"
            )


@dataclass(frozen=True, slots=True)
class _Belief:
    """One past decision: what was believed, over which periods.

    ``first_day_periods`` is kept because it is half of the shape a residual
    must match to be usable -- a 48-period window split 24/24 and one split
    25/23 cover the same clock hours in a different order, so adding the second
    to the first would misalign every period after the transition.
    """

    stamps_ns: NDArray[np.int64]
    first_day_periods: int
    believed: FloatArray
    published_ns: int
    """When the last delivery day of the window was published — the instant
    the whole window's realised prices became known."""


class BeliefResiduals:
    """A policy's causal error trajectories, recorded and resampled.

    One instance per policy instance. :meth:`record` is called by the policy
    every time it forms a belief -- including on days the run later discards as
    warm-up, because a warm-up day's error is as informative as any other and
    excluding it would thin the pool for no reason.
    """

    def __init__(self, prices: pd.Series, config: ScenarioConfig) -> None:
        index = pd.DatetimeIndex(prices.index)
        if not index.is_monotonic_increasing:
            raise ValueError("the price series must be sorted to be read causally")
        self._config = config
        self._price_ns = np.asarray(index.as_unit("ns").asi8, dtype=np.int64)
        self._price_values = np.asarray(prices.to_numpy(dtype=np.float64))
        self._beliefs: dict[dt.date, _Belief] = {}
        self._residuals: dict[dt.date, FloatArray] = {}
        self._rungs: dict[str, int] = {"sampled": 0, "declined": 0}

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    # -- recording ---------------------------------------------------------

    def record(
        self, day: dt.date, window: pd.DatetimeIndex, believed: FloatArray
    ) -> None:
        """Keep what was believed on ``day``, to become a residual once it clears.

        Keyed by delivery day rather than appended, so a policy shared across a
        degradation sweep records each day once instead of three identical
        times. Same reasoning as ``PriceForecaster``'s keyed rung tallies.
        """
        if day in self._beliefs:
            return
        local_dates = pd.Index(window.tz_convert(MARKET_TZ).date)
        self._beliefs[day] = _Belief(
            stamps_ns=np.asarray(window.as_unit("ns").asi8, dtype=np.int64),
            first_day_periods=int((local_dates == day).sum()),
            believed=np.asarray(believed, dtype=np.float64).copy(),
            published_ns=int(price_published_index(window).as_unit("ns").asi8.max()),
        )

    # -- sampling ----------------------------------------------------------

    def sample(
        self, day: dt.date, window: pd.DatetimeIndex, n_scenarios: int
    ) -> FloatArray | None:
        """``n_scenarios`` joint residual trajectories for ``window``, or ``None``.

        ``None`` means the pool behind the gate is thinner than
        ``min_scenario_days``, which is not an error: it is the opening stretch
        of every run, before enough decisions have been made *and cleared* for
        there to be a distribution. The caller falls back to the fixed
        schedule, which is the honest thing to bid with no distribution -- it
        is what v1 and v2 bid on every day.
        """
        pool = self.pool(day, window)
        if pool is None:
            self._rungs["declined"] += 1
            return None
        self._rungs["sampled"] += 1
        # Seeded on the decision day, not on a running counter: the draw for a
        # given day is then the same whatever order the days were visited in
        # and whatever else shares the process (CLAUDE.md invariant 8).
        rng = np.random.default_rng([self._config.seed, day.toordinal()])
        drawn = rng.integers(0, pool.shape[0], size=n_scenarios)
        return np.asarray(pool[drawn], dtype=np.float64)

    def pool(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray | None:
        """Every usable residual trajectory, as an ``(n_days, n_periods)`` array.

        Public because it is the object the tests assert on: a finite draw of S
        rows has a sample mean near zero, but the *pool* is centred exactly, so
        the "scenario expectation equals the point belief" identity is stated
        here and checked here.
        """
        rows = [
            residual
            for past in sorted(self._beliefs)
            if self._eligible(past, day, window)
            if (residual := self._residual(past)) is not None
        ]
        if len(rows) < self._config.min_scenario_days:
            return None
        pool = np.vstack(rows)
        if self._config.centre_residuals:
            pool = pool - pool.mean(axis=0, keepdims=True)
        return pool

    def _eligible(self, past: dt.date, day: dt.date, window: pd.DatetimeIndex) -> bool:
        """May ``past``'s error trajectory be used to decide ``day``?

        Two independent conditions, both load-bearing.

        **Causality.** Every period of ``past``'s window must have been
        published by this gate. Invariant 1 is read on publication time: a
        delivery day's prices become knowable when OMIE publishes them, about
        13:00 on the day before delivery, so a window over D and D+1 is known
        from the afternoon of D and is admissible for any decision whose gate
        is noon on D+1 or later. The same reading
        :func:`bess_arb.policy.floor._mean_before` applies. Both sides are
        pinned to nanoseconds here: a Parquet round trip hands back
        microsecond indexes, and comparing those against ``Timestamp.value``
        puts the gate a thousand times too early.

        **Shape.** A residual is added position by position to today's belief,
        so the two windows must describe the same clock. Equal length is
        necessary and not sufficient: a 48-period window split 24/24 and one
        split 25/23 both have 48 periods and diverge after the transition.
        Rejecting the handful of mismatched days a year is the same trade the
        backtest makes in pooling one model per window length rather than
        padding a short window -- a shifted residual would fabricate prices.
        """
        belief = self._beliefs[past]
        gate_ns = int(gate_close_utc(day).as_unit("ns").value)
        target_first_day = int(
            (pd.Index(window.tz_convert(MARKET_TZ).date) == day).sum()
        )
        return (
            belief.published_ns <= gate_ns
            and len(belief.stamps_ns) == len(window)
            and belief.first_day_periods == target_first_day
        )

    def _residual(self, past: dt.date) -> FloatArray | None:
        """``realised - believed`` over ``past``'s window, cached once computed.

        ``None`` if any period of that window has no realised price -- which
        happens at the very end of a snapshot and never in the middle, since
        :func:`bess_arb.backtest.runner._decision_days` refuses a series with
        calendar holes.
        """
        cached = self._residuals.get(past)
        if cached is not None:
            return cached
        belief = self._beliefs[past]
        position = np.searchsorted(self._price_ns, belief.stamps_ns)
        if int(position.max(initial=0)) >= len(self._price_ns):
            return None
        if not np.array_equal(self._price_ns[position], belief.stamps_ns):
            return None
        residual = self._price_values[position] - belief.believed
        if not np.isfinite(residual).all():
            return None
        self._residuals[past] = residual
        return residual

    # -- reporting ---------------------------------------------------------

    def diagnostics(self) -> dict[str, int]:
        """Days served by the joint sample and days that fell back.

        ``scenario_declined`` is the count a reader needs: it is how much of a
        v3 run was actually bid as v2, because the pool was not yet deep enough
        to hold a distribution.
        """
        return {
            "scenario_sampled": self._rungs["sampled"],
            "scenario_declined": self._rungs["declined"],
            "scenario_pool_days": len(self._residuals),
        }
