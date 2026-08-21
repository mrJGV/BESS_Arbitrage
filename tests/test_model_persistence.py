"""The persistent re-solve must not return a stale answer.

The backtest is ~15,000 window solves of a model with a few hundred
variables, so the model is built once and each window mutates only the price
coefficients and the initial SoC. The failure mode that matters is not a
crash: it is a second solve that quietly returns the first solve's answer.
That would corrupt every window while looking entirely plausible, so it is
checked against freshly-built models rather than trusted.

This is also the test that would catch a backend which satisfies the
Protocol by rebuilding inside ``solve`` and then, one refactor later, caches
something it should not.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arb.model import BatteryParams, SolveError
from conftest import ModelFactory, profit_from_dispatch

TOL = 1e-6

PRICE_VECTORS = [
    np.array([10.0, 10.0, 100.0, 100.0]),
    np.array([100.0, 100.0, 10.0, 10.0]),
    np.array([-20.0, 5.0, 60.0, 55.0]),
    np.array([30.0, 30.0, 30.0, 30.0]),
    np.array([0.0, 250.0, -10.0, 40.0]),
]
INITIAL_SOCS = [0.0, 5.0, 10.0, 17.5, 20.0]


def test_resolving_matches_a_freshly_built_model(
    make_model: ModelFactory, golden_params: BatteryParams
) -> None:
    reused = make_model()

    for prices in PRICE_VECTORS:
        for soc_initial in INITIAL_SOCS:
            from_reused = reused.solve(prices, soc_initial)
            from_fresh = make_model().solve(prices, soc_initial)

            assert from_reused.objective == pytest.approx(from_fresh.objective, abs=TOL)
            # The schedule may legitimately differ where the optimum is
            # non-unique, so what is compared is the profit of the schedule
            # actually returned.
            assert profit_from_dispatch(
                from_reused, prices, golden_params
            ) == pytest.approx(from_reused.objective, abs=TOL)


def test_a_repeated_solve_is_not_the_previous_answer(
    make_model: ModelFactory,
) -> None:
    """The blunt version: two different windows, two different answers."""
    model = make_model()

    cheap_then_dear = model.solve(np.array([10.0, 10.0, 100.0, 100.0]), 0.0)
    flat = model.solve(np.array([30.0, 30.0, 30.0, 30.0]), 0.0)
    cheap_then_dear_again = model.solve(np.array([10.0, 10.0, 100.0, 100.0]), 0.0)

    assert flat.objective == pytest.approx(0.0, abs=TOL)
    assert cheap_then_dear.objective == pytest.approx(1500.0, abs=TOL)
    assert cheap_then_dear_again.objective == pytest.approx(1500.0, abs=TOL)


def test_initial_soc_alone_changes_the_answer(make_model: ModelFactory) -> None:
    """A stale ``soc_initial`` is the subtler half of the same bug.

    Starting full is worth an extra 20 MWh delivered at €100 minus what the
    empty start would have moved; all that matters here is that the two
    differ, and in the right direction.
    """
    model = make_model()
    prices = np.array([10.0, 10.0, 100.0, 100.0])

    from_empty = model.solve(prices, soc_initial=0.0)
    from_full = model.solve(prices, soc_initial=20.0)

    assert from_full.objective > from_empty.objective + TOL


def test_out_of_range_initial_soc_is_rejected(make_model: ModelFactory) -> None:
    """Silently clipping a nonsense SoC would hide a broken backtest loop."""
    model = make_model()
    prices = np.array([10.0, 10.0, 100.0, 100.0])

    with pytest.raises(ValueError):
        model.solve(prices, soc_initial=25.0)
    with pytest.raises(ValueError):
        model.solve(prices, soc_initial=-1.0)


def test_solver_rounding_at_the_soc_bounds_is_tolerated(
    make_model: ModelFactory,
) -> None:
    """A previous window's terminal SoC of -1e-15 is zero, not an error."""
    model = make_model()
    prices = np.array([10.0, 10.0, 100.0, 100.0])

    solution = model.solve(prices, soc_initial=-1e-12)
    assert solution.objective == pytest.approx(1500.0, abs=TOL)


def test_wrong_length_price_vector_is_rejected(make_model: ModelFactory) -> None:
    model = make_model(n_periods=4)

    with pytest.raises(ValueError):
        model.solve(np.array([10.0, 100.0]), soc_initial=0.0)


def test_non_finite_prices_are_rejected(make_model: ModelFactory) -> None:
    """A NaN from a bad join must stop the run, not produce a schedule."""
    model = make_model(n_periods=4)

    with pytest.raises(ValueError):
        model.solve(np.array([10.0, np.nan, 100.0, 100.0]), soc_initial=0.0)


def test_solve_error_is_the_projects_own_exception() -> None:
    """Callers outside ``model/`` must be able to catch failure by type."""
    assert issubclass(SolveError, RuntimeError)
