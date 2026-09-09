"""The feasibility repair: what the battery can actually deliver.

A cleared curve knows nothing about the state of charge, so a position it
clears can be physically impossible. ``bid/deliver.py`` clips it forward and
counts what was clipped. The two properties worth pinning are that the result
is always feasible and that one pass is enough, plus the worked example the
settlement rule was chosen on.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arb.backtest.metrics import settle_profit
from bess_arb.bid.deliver import deliver
from bess_arb.model.spec import BatteryParams

PARAMS = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=17.0)


def _cleared(prices: np.ndarray) -> np.ndarray:
    """The curve from the settlement-rule discussion, read at ``prices``."""
    return np.array(
        [
            -10.0 if prices[0] <= 30 else (-5.0 if prices[0] <= 40 else 0.0),
            -10.0 if prices[1] <= 25 else 0.0,
            10.0 if prices[2] >= 70 else (5.0 if prices[2] >= 50 else 0.0),
            10.0 if prices[3] >= 65 else 0.0,
        ]
    )


def test_the_worked_example_when_the_charge_leg_misses() -> None:
    """03:00 clears above the limit, so the evening offer cannot be met in full.

    Hand-computed when the settlement rule was chosen: the curve refuses to buy
    at 45, stores only 9.22 MWh, and can deliver 8.50 MW of the 10 MW that
    cleared at t3 — 11.50 MWh cleared but undelivered, and EUR 443 of profit.
    """
    prices = np.array([45.0, 22.0, 95.0, 80.0])

    result = deliver(_cleared(prices), 0.0, 1.0, PARAMS)

    assert result.clipped_discharge_mwh == pytest.approx(11.50, abs=0.01)
    assert result.p_d_mw[2] == pytest.approx(8.50, abs=0.01)
    assert settle_profit(
        result.p_c_mw, result.p_d_mw, prices, 1.0, PARAMS
    ) == pytest.approx(443.0, abs=0.5)


def test_the_worked_example_when_everything_clears() -> None:
    """Both charge legs fill, and the option costs nothing on a good day."""
    prices = np.array([28.0, 22.0, 95.0, 80.0])

    result = deliver(_cleared(prices), 0.0, 1.0, PARAMS)

    assert result.p_d_mw[2] == pytest.approx(10.0)
    assert result.p_d_mw[3] == pytest.approx(7.0, abs=0.01)
    assert settle_profit(
        result.p_c_mw, result.p_d_mw, prices, 1.0, PARAMS
    ) == pytest.approx(721.0, abs=0.5)


def test_the_repair_never_leaves_the_state_of_charge_window() -> None:
    """The property, over positions deliberately built to be infeasible."""
    rng = np.random.default_rng(20260819)
    for _ in range(200):
        net = rng.uniform(-15.0, 15.0, size=48)
        soc0 = float(rng.uniform(0.0, PARAMS.e_max_mwh))

        result = deliver(net, soc0, 0.25, PARAMS)

        assert np.all(result.soc_mwh >= -1e-9)
        assert np.all(result.soc_mwh <= PARAMS.e_max_mwh + 1e-9)
        assert np.all(result.p_c_mw <= PARAMS.p_max_mw + 1e-9)
        assert np.all(result.p_d_mw <= PARAMS.p_max_mw + 1e-9)
        assert np.all(result.p_c_mw >= 0.0)
        assert np.all(result.p_d_mw >= 0.0)


def test_one_pass_is_enough() -> None:
    """Re-delivering an already-delivered dispatch changes nothing.

    The repair only ever reduces quantities, so it cannot create a violation
    behind itself; a second sweep is a fixed point. If this fails, the forward
    pass is not sufficient and the module docstring's argument is wrong.
    """
    rng = np.random.default_rng(7)
    for _ in range(50):
        net = rng.uniform(-15.0, 15.0, size=32)
        soc0 = float(rng.uniform(0.0, PARAMS.e_max_mwh))

        once = deliver(net, soc0, 1.0, PARAMS)
        twice = deliver(once.p_d_mw - once.p_c_mw, soc0, 1.0, PARAMS)

        assert twice.p_c_mw == pytest.approx(once.p_c_mw)
        assert twice.p_d_mw == pytest.approx(once.p_d_mw)
        assert twice.clipped_mwh == pytest.approx(0.0, abs=1e-9)


def test_a_feasible_position_is_passed_through_untouched() -> None:
    """Nothing is clipped when the store can meet the whole position."""
    net = np.array([-10.0, -10.0, 8.0, 0.0])

    result = deliver(net, 0.0, 1.0, PARAMS)

    assert result.clipped_mwh == pytest.approx(0.0)
    assert result.p_c_mw == pytest.approx([10.0, 10.0, 0.0, 0.0])
    assert result.p_d_mw == pytest.approx([0.0, 0.0, 8.0, 0.0])


def test_a_full_battery_cannot_take_more_charge() -> None:
    """The overfill direction, which the discharge tests would not catch."""
    result = deliver(np.array([-10.0]), PARAMS.e_max_mwh, 1.0, PARAMS)

    assert result.p_c_mw[0] == pytest.approx(0.0)
    assert result.clipped_charge_mwh == pytest.approx(10.0)


def test_the_power_limit_binds_before_the_store_does() -> None:
    """A curve promising more than the connection point can carry is capped."""
    result = deliver(np.array([-25.0]), 0.0, 1.0, PARAMS)

    assert result.p_c_mw[0] == pytest.approx(PARAMS.p_max_mw)


def test_quarter_hourly_periods_scale_the_energy_not_the_power() -> None:
    """Invariant 3: nothing here may assume a one-hour period.

    At dt = 0.25 the same 10 MW moves a quarter of the energy, so a battery
    that fills in two hourly periods needs eight quarter-hourly ones.
    """
    net = np.full(8, -10.0)

    result = deliver(net, 0.0, 0.25, PARAMS)

    stored = PARAMS.eta_c * float(result.p_c_mw.sum()) * 0.25
    assert stored == pytest.approx(PARAMS.eta_c * 20.0)
    assert result.clipped_charge_mwh == pytest.approx(0.0)
