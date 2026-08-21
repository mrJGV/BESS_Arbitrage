"""The golden test. Nothing downstream is worth debugging until this passes.

Prices ``[10, 10, 100, 100]``, Δt = 1 h, SoC₀ = 0, ``c_deg`` = 0:

    charge 10 MW through both cheap hours   -> 20 MWh drawn, €200 paid
    stored: 20 · √0.85                      =  18.439 MWh
    delivered: 0.85 · 20                    =  17.0 MWh, €1,700 received
    optimum                                 =  €1,500

or in one line, ``20 MWh × (0.85 × 100 − 10)``.

**Assert on the objective, never on the dispatch vector.** The two discharge
prices are equal, so any split of the 17 MWh across the last two hours is
optimal; HiGHS returns ``p_d = [0, 0, 7, 10]``, not the ``[0, 0, 8.5, 8.5]``
one writes down by hand. A test that pinned the schedule would fail a
correct model — the non-uniqueness is a property of the instance, not a
solver quirk.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arb.model import BatteryParams, SolveStatus
from conftest import ModelFactory, profit_from_dispatch, soc_trajectory

GOLDEN_PRICES = np.array([10.0, 10.0, 100.0, 100.0])
GOLDEN_OPTIMUM = 1500.0
TOL = 1e-6


def test_golden_case_objective(make_model: ModelFactory) -> None:
    solution = make_model().solve(GOLDEN_PRICES, soc_initial=0.0)

    assert solution.status is SolveStatus.OPTIMAL
    assert solution.objective == pytest.approx(GOLDEN_OPTIMUM, abs=TOL)


def test_golden_objective_matches_its_own_dispatch(
    make_model: ModelFactory, golden_params: BatteryParams
) -> None:
    """The reported objective is the profit of the reported schedule.

    Guards against the failure mode where the objective is right and the
    dispatch that reaches the backtest is something else.
    """
    solution = make_model().solve(GOLDEN_PRICES, soc_initial=0.0)

    recomputed = profit_from_dispatch(solution, GOLDEN_PRICES, golden_params)
    assert recomputed == pytest.approx(solution.objective, abs=TOL)


def test_golden_dispatch_is_physically_consistent(
    make_model: ModelFactory, golden_params: BatteryParams
) -> None:
    """Energy balance, power limits and exclusivity, checked from outside."""
    solution = make_model().solve(GOLDEN_PRICES, soc_initial=0.0)
    p = golden_params

    assert (solution.p_c_mw >= -TOL).all()
    assert (solution.p_d_mw >= -TOL).all()
    assert (solution.p_c_mw <= p.p_max_mw + TOL).all()
    assert (solution.p_d_mw <= p.p_max_mw + TOL).all()
    assert (solution.soc_mwh >= -TOL).all()
    assert (solution.soc_mwh <= p.e_max_mwh + TOL).all()

    replayed = soc_trajectory(solution, p, soc_initial=0.0)
    assert replayed == pytest.approx(solution.soc_mwh, abs=TOL)

    simultaneous = (solution.p_c_mw > TOL) & (solution.p_d_mw > TOL)
    assert not simultaneous.any()

    # 17 MWh delivered from 20 MWh drawn: the round trip, not a coincidence.
    assert solution.charged_mwh == pytest.approx(20.0, abs=TOL)
    assert solution.discharged_mwh == pytest.approx(17.0, abs=TOL)
    assert solution.equivalent_cycles(p.e_max_mwh) == pytest.approx(0.85, abs=TOL)


def test_degradation_cost_suppresses_a_marginal_cycle(
    make_model: ModelFactory,
) -> None:
    """``c_deg`` is a minimum-spread threshold, not an accounting entry.

    The golden spread earns ``0.85·100 − 10 = €75`` per MWh drawn, i.e.
    €88.2 per MWh *delivered*. A degradation cost above that must stop the
    battery cycling entirely — which is the mechanism that sets cycles per
    year, so it is worth a test rather than a comment.
    """
    params = BatteryParams(
        p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=100.0
    )
    solution = make_model(params).solve(GOLDEN_PRICES, soc_initial=0.0)

    assert solution.objective == pytest.approx(0.0, abs=TOL)
    assert solution.discharged_mwh == pytest.approx(0.0, abs=TOL)
