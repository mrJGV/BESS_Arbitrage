"""Randomised consistency, over instances nobody chose by hand.

The golden case proves one optimum. These check the invariants that must
hold on *every* solve — the energy balance, the power and SoC bounds,
exclusivity, and the objective being the profit of the schedule actually
returned — across price paths with negative periods, spikes and flat
stretches, at both market granularities.

The seed comes from ``config/params.yaml`` (CLAUDE.md invariant 8), so a
failure is reproducible from the repository alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arb.config import load_config
from bess_arb.model import BatteryParams, SolveStatus
from conftest import ModelFactory, profit_from_dispatch, soc_trajectory

TOL = 1e-6
N_INSTANCES = 8


def _price_paths(seed: int, n_periods: int) -> list[np.ndarray]:
    """Lognormal-ish day-ahead shapes, shifted so some periods go negative."""
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(N_INSTANCES):
        level = rng.uniform(20.0, 120.0)
        spread = rng.uniform(5.0, 80.0)
        shape = np.sin(np.linspace(0.0, 4.0 * np.pi, n_periods))
        noise = rng.normal(0.0, 0.3, size=n_periods)
        paths.append(level + spread * (shape + noise) - rng.uniform(0.0, 60.0))
    return paths


@pytest.mark.parametrize(("n_periods", "dt_h"), [(96, 1.0), (192, 0.25)])
def test_every_solution_is_feasible_and_self_consistent(
    make_model: ModelFactory, n_periods: int, dt_h: float
) -> None:
    config = load_config()
    params = config.battery
    model = make_model(params, n_periods=n_periods, dt_h=dt_h)

    for prices in _price_paths(config.seed, n_periods):
        for soc_initial in (0.0, 0.5 * params.e_max_mwh, params.e_max_mwh):
            solution = model.solve(prices, soc_initial)

            assert solution.status is SolveStatus.OPTIMAL
            assert (solution.p_c_mw >= -TOL).all()
            assert (solution.p_d_mw >= -TOL).all()
            assert (solution.p_c_mw <= params.p_max_mw + TOL).all()
            assert (solution.p_d_mw <= params.p_max_mw + TOL).all()
            assert (solution.soc_mwh >= -TOL).all()
            assert (solution.soc_mwh <= params.e_max_mwh + TOL).all()

            simultaneous = (solution.p_c_mw > TOL) & (solution.p_d_mw > TOL)
            assert not simultaneous.any()

            replayed = soc_trajectory(solution, params, soc_initial)
            assert replayed == pytest.approx(solution.soc_mwh, abs=1e-6)

            assert profit_from_dispatch(solution, prices, params) == pytest.approx(
                solution.objective, abs=1e-6
            )


def test_a_higher_degradation_cost_never_increases_cycling(
    make_model: ModelFactory,
) -> None:
    """The monotonicity that makes ``c_deg`` the knob it is claimed to be.

    A linear throughput cost is a minimum-spread threshold, so raising it can
    only remove cycles, never add them. Across the published 5 / 17 / 40
    sweep, on the same prices, throughput must be non-increasing.
    """
    config = load_config()
    prices = _price_paths(config.seed, 96)[0]

    throughput = []
    for c_deg in config.c_deg_sensitivity:
        params = BatteryParams(
            p_max_mw=config.battery.p_max_mw,
            e_max_mwh=config.battery.e_max_mwh,
            eta_rt=config.battery.eta_rt,
            c_deg_eur_mwh=c_deg,
            charge_tariff_eur_mwh=config.battery.charge_tariff_eur_mwh,
        )
        solution = make_model(params, n_periods=96, dt_h=1.0).solve(prices, 0.0)
        throughput.append(solution.equivalent_cycles(params.e_max_mwh))

    assert throughput == sorted(throughput, reverse=True)
    assert throughput[0] > throughput[-1]
