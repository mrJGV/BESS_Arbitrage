"""The bid curve object: construction, monotonisation and clearing.

``bid/curve.py`` turns K scenario solves into one price-quantity curve per
period. What has to hold is narrow and worth stating: the curve is monotone,
it clears by the auction convention, and the oracle's degenerate case comes
back as exactly the schedule it started from.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arb.bid.curve import BidCurves, build_curves


def _from_net(prices: np.ndarray, net: np.ndarray) -> BidCurves:
    """Build curves from a signed net position, splitting it as the model does."""
    return build_curves(prices, np.maximum(-net, 0.0), np.maximum(net, 0.0))


def test_quantities_come_back_non_decreasing_in_price() -> None:
    """The invariant every downstream layer relies on.

    A demand curve falls with price and a supply curve rises with it, so the
    signed net position is non-decreasing — that single condition is what
    makes the curve submittable and what ``BidCurves`` checks on construction.
    """
    prices = np.array([[12.0, 8.0], [30.0, 18.0], [120.0, 40.0]])
    net = np.array([[0.0, -10.0], [-10.0, -10.0], [0.0, -10.0]])

    curves = _from_net(prices, net)

    assert np.all(np.diff(curves.quantities, axis=1) >= -1e-9)


def test_violators_are_pooled_not_clamped() -> None:
    """Isotonic regression, and the specific values it produces.

    The scenarios are solved independently, so a dearer price vector can
    reorder the day rather than only lift it, and a period's raw quantities
    need not be monotone. Pooling violators to their weighted mean is
    symmetric between over-buying and over-selling; a running max or min would
    pick a side. ``[0, -10, 0]`` therefore becomes ``[-5, -5, 0]`` and not
    ``[0, 0, 0]`` or ``[-10, -10, 0]``.
    """
    prices = np.array([[12.0], [30.0], [120.0]])
    net = np.array([[0.0], [-10.0], [0.0]])

    curves = _from_net(prices, net)

    assert curves.quantities[0] == pytest.approx([-5.0, -5.0, 0.0])


def test_a_step_is_accepted_once_the_price_reaches_its_limit() -> None:
    """The auction convention: a bid at limit L clears at a price *equal* to L.

    Taking the step below instead would make the oracle's identity check fail
    by one step, which is the cheapest possible way to notice the convention
    is wrong.
    """
    curves = BidCurves(
        prices=np.array([[10.0, 20.0, 30.0]]),
        quantities=np.array([[-10.0, 0.0, 10.0]]),
    )

    assert curves.clear(np.array([20.0]))[0] == pytest.approx(0.0)
    assert curves.clear(np.array([19.99]))[0] == pytest.approx(-10.0)
    assert curves.clear(np.array([30.0]))[0] == pytest.approx(10.0)


def test_prices_outside_the_curve_clamp_to_its_ends() -> None:
    """Below the cheapest step, buy the most; above the dearest, sell the most.

    The realised price came in outside everything any scenario expected, and
    the outermost step is the right answer in both directions.
    """
    curves = BidCurves(
        prices=np.array([[10.0, 30.0]]),
        quantities=np.array([[-10.0, 10.0]]),
    )

    assert curves.clear(np.array([-500.0]))[0] == pytest.approx(-10.0)
    assert curves.clear(np.array([4000.0]))[0] == pytest.approx(10.0)


def test_the_oracle_degenerate_curve_returns_its_own_schedule() -> None:
    """One scenario, one step, and clearing it reproduces the schedule exactly.

    Perfect foresight has no quantile to sweep, so the oracle's curve is a
    single step at the price that cleared. This is the free correctness check
    on the whole clearing path: if it ever stops holding, the clearing code is
    wrong and not the model.
    """
    realised = np.array([10.0, 90.0, 45.0])
    schedule = np.array([-10.0, 7.0, 0.0])

    curves = _from_net(realised.reshape(1, -1), schedule.reshape(1, -1))

    assert curves.n_steps == 1
    assert curves.clear(realised) == pytest.approx(schedule)


def test_step_counts_reports_degeneracy() -> None:
    """The diagnostic that says whether the sweep produced a curve at all.

    A period whose quantity never changes across the sweep is a fixed schedule
    wearing a limit price, and a run of those means the scenario family moved
    the price level without moving the day's shape.
    """
    prices = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    net = np.array([[-10.0, -10.0], [-10.0, 0.0], [-10.0, 10.0]])

    curves = _from_net(prices, net)

    assert curves.step_counts[0] == pytest.approx(1.0)
    assert curves.step_counts[1] == pytest.approx(3.0)


def test_a_non_monotone_curve_is_refused() -> None:
    """Constructed directly, because ``build_curves`` cannot produce one."""
    with pytest.raises(ValueError, match="non-decreasing"):
        BidCurves(
            prices=np.array([[10.0, 20.0]]),
            quantities=np.array([[10.0, -10.0]]),
        )
