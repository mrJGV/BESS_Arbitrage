"""The quantile sources: what the floor and the oracle believe at each tau.

The bid-curve extension asks a policy for K price vectors instead of one. What
has to hold of those vectors is stricter than it looks: they must be monotone
in tau — otherwise they are not quantiles — they must still respect the gate,
and a thin bucket must produce a *usable* family rather than a degenerate one.
Each of those failed at some point during the build, and each is pinned here.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.policy import QuantilePolicy
from bess_arb.policy.floor import FloorPolicy
from bess_arb.policy.oracle import OraclePolicy
from bess_arb.timeline import Regime, price_published_index, to_market_time, utc_index

HOURLY = Regime("hourly", 1.0)
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _prices(first: dt.date, last: dt.date) -> pd.Series:
    """A daily shape with day-to-day noise, so buckets have real spread."""
    index = utc_index(first, last, HOURLY)
    local = to_market_time(index)
    hour = np.asarray(local.hour, dtype=np.float64)
    rng = np.random.default_rng(20260819)
    return pd.Series(
        60.0
        + 40.0 * np.sin((hour - 4.0) * np.pi / 12.0)
        + rng.normal(0.0, 12.0, len(index)),
        index=index,
        name="price_eur_mwh",
    )


def _window(day: dt.date) -> pd.DatetimeIndex:
    return utc_index(day, day + dt.timedelta(days=1), HOURLY)


def _family(policy: QuantilePolicy, day: dt.date) -> np.ndarray:
    window = _window(day)
    return np.array([policy.prices_for_quantile(day, window, t) for t in TAUS])


# --------------------------------------------------------------------------
# The floor
# --------------------------------------------------------------------------


def test_the_floor_satisfies_the_quantile_protocol() -> None:
    assert isinstance(
        FloorPolicy(_prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))), QuantilePolicy
    )


def test_the_floor_family_is_monotone_in_tau() -> None:
    """Not a preference: a family that falls as tau rises is not a quantile.

    ``bid.scenarios`` refuses such a family, but by then the cause is a long
    way from the symptom, so it is also asserted at the source.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 6, 30))
    floor = FloorPolicy(prices)

    for day in (dt.date(2024, 3, 2), dt.date(2024, 4, 2), dt.date(2024, 6, 15)):
        family = _family(floor, day)
        assert np.all(np.diff(family, axis=0) >= -1e-9)


def test_a_bucket_of_one_observation_does_not_produce_a_flat_family() -> None:
    """The failure the quantile threshold exists to prevent.

    A mean is well defined at n=1; a quantile *spread* is identically zero
    there, so a one-observation bucket gives no scenario family at all. The
    ladder must fall through to the coarser bucket, which at that moment holds
    a month or more of the same hour. The gate is noon on D-1, so on the 2nd
    of a month the pre-noon buckets hold exactly one observation — that is the
    day this reproduces.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    floor = FloorPolicy(prices)

    family = _family(floor, dt.date(2024, 3, 2))

    spread = family[-1] - family[0]
    assert np.all(spread > 1e-6), "a period came back with no scenario spread"


def test_the_threshold_does_not_vary_with_tau() -> None:
    """Why: a per-tau threshold lets different taus read different rungs.

    When it did, tau=0.3/0.5/0.7 read the month bucket while tau=0.1 and 0.9
    fell through to the all-months bucket, and the coarse bucket's 90th
    percentile came back below the fine bucket's 70th. Pinned by driving the
    ladder at a count that sits between the two thresholds a tau-dependent
    rule would have used.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    floor = FloorPolicy(prices, min_quantile_observations=10)

    for day in (dt.date(2024, 3, 2), dt.date(2024, 3, 5), dt.date(2024, 3, 9)):
        family = _family(floor, day)
        assert np.all(np.diff(family, axis=0) >= -1e-9)


def test_the_floor_quantiles_read_nothing_after_the_gate() -> None:
    """Invariant 1, applied to the quantile ladder as well as the mean.

    Prices published after the gate are replaced with an absurd value; the answer must
    not move. A quantile is more sensitive to this than a mean, since one
    extreme observation relocates the tail of a small sample.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    day = dt.date(2024, 3, 15)

    honest = _family(FloorPolicy(prices), day)

    poisoned = prices.copy()
    gate = pd.Timestamp(
        dt.datetime.combine(day - dt.timedelta(days=1), dt.time(12)),
        tz="Europe/Madrid",
    ).tz_convert("UTC")
    poisoned.loc[
        np.asarray(price_published_index(pd.DatetimeIndex(poisoned.index)) > gate)
    ] = 9999.0

    assert _family(FloorPolicy(poisoned), day) == pytest.approx(honest)


def test_the_quantile_ladder_is_counted_separately_from_the_mean_ladder() -> None:
    """Different units: one call per day against one call per (day, tau).

    Merging them would report one as the other, and the count of how often the
    floor was thin is the diagnostic a reader checks first.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    floor = FloorPolicy(prices)
    day = dt.date(2024, 3, 15)

    floor.prices_for(day, _window(day))
    floor.prices_for_quantile(day, _window(day), 0.5)

    assert sum(floor.diagnostics().values()) == len(_window(day))
    assert sum(floor.quantile_diagnostics().values()) == len(_window(day))


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.5])
def test_a_tau_outside_the_open_unit_interval_is_refused(tau: float) -> None:
    floor = FloorPolicy(_prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31)))
    day = dt.date(2024, 3, 15)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        floor.prices_for_quantile(day, _window(day), tau)


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------


def test_the_oracle_believes_the_same_thing_at_every_tau() -> None:
    """Perfect foresight has no uncertainty for a quantile to describe.

    This is what makes the oracle's curve a single step, and therefore what
    makes clearing it reproduce the fixed-schedule oracle exactly — the free
    check on the whole curve path.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    oracle = OraclePolicy(prices)
    day = dt.date(2024, 3, 15)
    window = _window(day)

    point = oracle.prices_for(day, window)
    for tau in TAUS:
        assert oracle.prices_for_quantile(day, window, tau) == pytest.approx(point)


def test_the_oracle_satisfies_the_quantile_protocol() -> None:
    assert isinstance(
        OraclePolicy(_prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))),
        QuantilePolicy,
    )
