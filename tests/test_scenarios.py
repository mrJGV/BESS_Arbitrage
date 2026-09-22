"""v3's joint scenario source: what may enter the pool, and what comes out.

:class:`~bess_arb.scenarios.BeliefResiduals` is the whole of v3's new
modelling content, so what it must not do is the interesting half. It must not
admit a residual from a decision whose window had not been published at this
gate (invariant 1), it must not add a residual whose window describes a different
clock (invariant 4, via the DST-length days), and it must not quietly move the
scenario set's centre away from the policy's own point belief -- which would
turn a v3-versus-v2 comparison into a statement about two things at once.

The pooled-versus-bucketed degeneracy v2.5 had cannot arise here: a joint
residual is a *vector*, so it varies across periods by construction, and the
failure mode that collapses every curve to one step has no analogue. What
replaces it is the thin-pool case, which declines rather than degenerating.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.scenarios import BeliefResiduals, ScenarioConfig
from bess_arb.timeline import (
    Regime,
    gate_close_utc,
    price_published_index,
    to_market_time,
    utc_index,
)

HOURLY = Regime("hourly", 1.0)
QUARTER = Regime("quarter_hourly", 0.25)


def _prices(first: dt.date, last: dt.date, regime: Regime = HOURLY) -> pd.Series:
    index = utc_index(first, last, regime)
    hour = np.asarray(to_market_time(index).hour, dtype=np.float64)
    rng = np.random.default_rng(20260819)
    return pd.Series(
        60.0
        + 40.0 * np.sin((hour - 4.0) * np.pi / 12.0)
        + rng.normal(0.0, 12.0, len(index)),
        index=index,
        name="price_eur_mwh",
    )


def _window(day: dt.date, regime: Regime = HOURLY) -> pd.DatetimeIndex:
    return utc_index(day, day + dt.timedelta(days=1), regime)


def _populate(
    residuals: BeliefResiduals,
    first: dt.date,
    n_days: int,
    *,
    belief: float = 50.0,
    regime: Regime = HOURLY,
) -> None:
    """Record a flat belief for ``n_days`` consecutive decisions."""
    for offset in range(n_days):
        day = first + dt.timedelta(days=offset)
        window = _window(day, regime)
        residuals.record(day, window, np.full(len(window), belief))


def _config(**overrides: object) -> ScenarioConfig:
    settings: dict[str, object] = {
        "n_scenarios": 5,
        "min_scenario_days": 10,
        "centre_residuals": True,
        "seed": 20260819,
    }
    settings.update(overrides)
    return ScenarioConfig(**settings)  # type: ignore[arg-type]


# -- causality --------------------------------------------------------------


def test_pool_admits_nothing_that_was_not_published_at_the_gate() -> None:
    """Invariant 1, as this module reads it.

    Every admitted residual's window must have been *published* by the
    decision's gate. Checked against the publication instants rather than
    against a day count, because the off-by-one that matters here is one
    delivery day's publication, not one period.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 4, 30))
    residuals = BeliefResiduals(prices, _config())
    _populate(residuals, dt.date(2024, 1, 1), 60)

    day = dt.date(2024, 3, 1)
    window = _window(day)
    gate_ns = int(gate_close_utc(day).as_unit("ns").value)
    pool = residuals.pool(day, window)
    assert pool is not None

    # Re-derive which decisions were admitted and assert on their stamps.
    admitted = [
        past
        for past in sorted(residuals._beliefs)
        if residuals._eligible(past, day, window)
    ]
    assert admitted, "nothing was admitted; the test would pass vacuously"
    for past in admitted:
        assert residuals._beliefs[past].published_ns <= gate_ns
    # And the reading is publication, not delivery: the newest admitted
    # window runs to the end of D-1, after the gate on the period clock.
    newest = residuals._beliefs[max(admitted)]
    assert int(newest.stamps_ns.max()) > gate_ns


def test_the_boundary_is_publication_not_delivery() -> None:
    """The decision two days back is admitted; the one a day back is not.

    Deciding D at noon on D-1: the window decided on D-2 covers D-2 and D-1,
    both published by the afternoon of D-2, so it is known in full. The
    window decided on D-1 covers D-1 and D, and D publishes an hour after
    the gate. Both windows end after the gate on the period clock, which is
    exactly why the cut cannot be made on period timestamps.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    residuals = BeliefResiduals(prices, _config(min_scenario_days=1))
    day = dt.date(2024, 3, 1)
    for past in (day - dt.timedelta(days=2), day - dt.timedelta(days=1)):
        residuals.record(past, _window(past), np.full(len(_window(past)), 50.0))

    assert residuals._eligible(day - dt.timedelta(days=2), day, _window(day))
    assert not residuals._eligible(day - dt.timedelta(days=1), day, _window(day))


def test_poisoning_prices_published_after_the_gate_cannot_move_the_pool() -> None:
    """The perturbation check, on the one object that reads realised prices.

    Declared publication instants are only as good as the bookkeeping behind
    them. This trusts nothing: rewrite every price published after the gate
    and require the pool to come back bit-identical.
    """
    day = dt.date(2024, 3, 1)
    window = _window(day)
    gate = gate_close_utc(day)

    clean = _prices(dt.date(2024, 1, 1), dt.date(2024, 4, 30))
    poisoned = clean.copy()
    poisoned[
        np.asarray(price_published_index(pd.DatetimeIndex(clean.index)) > gate)
    ] = -999.0

    pools = []
    for series in (clean, poisoned):
        residuals = BeliefResiduals(series, _config())
        _populate(residuals, dt.date(2024, 1, 1), 55)
        pool = residuals.pool(day, window)
        assert pool is not None
        pools.append(pool)

    assert np.array_equal(pools[0], pools[1])


# -- shape ------------------------------------------------------------------


def test_a_window_of_different_length_is_refused() -> None:
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    residuals = BeliefResiduals(prices, _config(min_scenario_days=1))

    past = dt.date(2024, 1, 10)
    short = _window(past)[:40]
    residuals.record(past, short, np.full(len(short), 50.0))

    day = dt.date(2024, 3, 1)
    assert not residuals._eligible(past, day, _window(day))


def test_a_dst_day_is_refused_against_an_ordinary_one_of_equal_length() -> None:
    """Equal length is necessary and not sufficient -- invariant 4's edge.

    The window opening on 2024-03-30 is 24 + 23 = 47 periods and the one
    opening on 2024-03-31 is 23 + 24 = 47. Same length, different clock: add
    one to the other and every period after the transition is misaligned.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 5, 31))
    residuals = BeliefResiduals(prices, _config(min_scenario_days=1))

    spring_eve = dt.date(2024, 3, 30)
    spring = dt.date(2024, 3, 31)
    eve_window = _window(spring_eve)
    spring_window = _window(spring)
    assert len(eve_window) == len(spring_window) == 47

    residuals.record(spring_eve, eve_window, np.full(47, 50.0))
    later = dt.date(2024, 5, 1)
    assert not residuals._eligible(spring_eve, later, spring_window)


def test_ordinary_days_match_each_other() -> None:
    """The negative control for the two rejections above."""
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    residuals = BeliefResiduals(prices, _config(min_scenario_days=1))
    past = dt.date(2024, 1, 10)
    residuals.record(past, _window(past), np.full(48, 50.0))
    day = dt.date(2024, 3, 1)
    assert residuals._eligible(past, day, _window(day))


# -- what comes out ---------------------------------------------------------


def test_the_centred_pool_has_exactly_zero_mean_per_period() -> None:
    """The identity that keeps v3 a statement about dispersion alone.

    The scenario set's expectation is the policy's own point belief, so v3
    cannot quietly collect a causal bias correction the way an uncentred pool
    would. Asserted on the pool rather than on a finite draw: a bootstrap of S
    rows has a sample mean near zero, and the exact statement lives here.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 4, 30))
    residuals = BeliefResiduals(prices, _config())
    _populate(residuals, dt.date(2024, 1, 1), 55)

    day = dt.date(2024, 3, 1)
    pool = residuals.pool(day, _window(day))
    assert pool is not None
    assert np.allclose(pool.mean(axis=0), 0.0, atol=1e-12)


def test_an_uncentred_pool_keeps_the_bias() -> None:
    """The negative control: centring is doing something, not decorating.

    A flat belief of 50 against a series whose mean is near 60 leaves a
    systematic positive error, and the uncentred pool must show it.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 4, 30))
    residuals = BeliefResiduals(prices, _config(centre_residuals=False))
    _populate(residuals, dt.date(2024, 1, 1), 55, belief=50.0)

    day = dt.date(2024, 3, 1)
    pool = residuals.pool(day, _window(day))
    assert pool is not None
    assert pool.mean() > 1.0


def test_a_thin_pool_declines_rather_than_degenerating() -> None:
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    residuals = BeliefResiduals(prices, _config(min_scenario_days=30))
    _populate(residuals, dt.date(2024, 1, 1), 5)

    day = dt.date(2024, 2, 1)
    assert residuals.sample(day, _window(day), 5) is None
    assert residuals.diagnostics()["scenario_declined"] == 1


def test_a_scenario_is_a_whole_trajectory_not_a_constant() -> None:
    """The failure v2.5's pooled shift had, checked not to have an analogue.

    A constant added to every period cancels in every difference, so all K
    solves return the identical dispatch and every curve collapses to one
    step. A joint residual is a vector drawn from a real past decision, so it
    varies across periods by construction -- and two draws differ from each
    other, which is what makes the K dispatches differ at all.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 4, 30))
    residuals = BeliefResiduals(prices, _config(n_scenarios=8))
    _populate(residuals, dt.date(2024, 1, 1), 55)

    day = dt.date(2024, 3, 1)
    drawn = residuals.sample(day, _window(day), 8)
    assert drawn is not None
    assert drawn.shape == (8, 48)
    # Varies within a scenario ...
    assert drawn.std(axis=1).min() > 1.0
    # ... and between scenarios.
    assert len({row.tobytes() for row in drawn}) > 1


def test_the_draw_is_seeded_on_the_day_not_on_call_order() -> None:
    """Invariant 8, in the form that survives a reordered backtest.

    Seeding on a running counter would make a day's scenarios depend on how
    many days preceded it in this process, so a sub-range re-run would not
    reproduce the full run's numbers.
    """
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 4, 30))
    day = dt.date(2024, 3, 1)
    other = dt.date(2024, 3, 2)

    draws = []
    for target in (day, other, day):
        residuals = BeliefResiduals(prices, _config())
        _populate(residuals, dt.date(2024, 1, 1), 55)
        drawn = residuals.sample(target, _window(target), 5)
        assert drawn is not None
        draws.append(drawn)

    assert np.array_equal(draws[0], draws[2])
    assert not np.array_equal(draws[0], draws[1])


def test_recording_is_idempotent_across_a_degradation_sweep() -> None:
    """One policy instance serves three c_deg points; the pool counts one day."""
    prices = _prices(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    residuals = BeliefResiduals(prices, _config(min_scenario_days=1))
    day = dt.date(2024, 1, 10)
    window = _window(day)

    residuals.record(day, window, np.full(48, 50.0))
    residuals.record(day, window, np.full(48, 999.0))

    later = dt.date(2024, 3, 1)
    pool = residuals.pool(later, _window(later))
    assert pool is not None
    assert pool.shape[0] == 1


def test_quarter_hourly_windows_work_unchanged() -> None:
    """Invariant 3: nothing here assumes an hour."""
    prices = _prices(dt.date(2025, 10, 1), dt.date(2025, 12, 31), QUARTER)
    residuals = BeliefResiduals(prices, _config(min_scenario_days=10))
    _populate(residuals, dt.date(2025, 10, 1), 50, regime=QUARTER)

    day = dt.date(2025, 12, 1)
    drawn = residuals.sample(day, _window(day, QUARTER), 5)
    assert drawn is not None
    assert drawn.shape == (5, 192)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_scenario_count_below_one_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="n_scenarios"):
        ScenarioConfig(n_scenarios=bad)
