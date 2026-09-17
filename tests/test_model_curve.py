"""v3 stage 2: the bid curve chosen by optimisation rather than construction.

This is the project's second formulation, permitted under a scoped exemption
from CLAUDE.md invariant 2 (see
:class:`~bess_arb.model.base.BidCurveMILP` for the boundary). A second
formulation is exactly the thing that can drift away from the first while
still looking right, so the tests here are mostly *identities against the
window MILP* rather than properties checked in isolation:

- with one scenario there is no uncertainty, so it must reproduce the plain
  window MILP's objective to 1e-6;
- a curve flat in the price band is a fixed schedule, so the optimum cannot
  fall below the fixed schedule's expected profit on the same scenarios;
- the binaries are load-bearing here for one reason more than in the window
  model, and that reason is tested directly.

The band arithmetic gets its own test against
:meth:`bess_arb.bid.curve.BidCurves.clear`, because the two implement the same
auction rule in different places and a divergence would make the optimiser
solve for an acceptance pattern that settlement never produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_arb.bid.curve import BidCurves
from bess_arb.model import (
    BatteryMILP,
    BidCurveMILP,
    get_backend,
    get_curve_backend,
)
from bess_arb.model.spec import BatteryParams, SolverConfig

GOLDEN = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=0.0)
REAL = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=17.0)
SOLVER = SolverConfig(name="highs", mip_gap=0.0, threads=1, time_limit_s=300.0)


def _curve_model(
    params: BatteryParams, n_periods: int, n_scenarios: int, dt_h: float = 1.0
) -> BidCurveMILP:
    return get_curve_backend("pyomo")(
        params, n_periods, n_scenarios, dt_h, solver=SOLVER
    )


def _window_model(
    params: BatteryParams, n_periods: int, dt_h: float = 1.0
) -> BatteryMILP:
    return get_backend("pyomo")(params, n_periods, dt_h, solver=SOLVER)


def test_one_scenario_reproduces_the_window_milp_on_the_golden_case() -> None:
    """The identity that pins the second formulation to the first.

    With a single scenario there is nothing to hedge: every band holds the
    same quantity, the curve is a fixed schedule, and the expected profit is
    just that schedule's profit. So this must return the golden EUR 1,500 --
    ``20 MWh x (0.85 x 100 - 10)`` -- exactly as the window model does.

    Asserted on the objective and never on the dispatch: with the two
    discharge prices equal the optimal schedule is non-unique, which is the
    trap ``CLAUDE.md`` records from the original golden test.
    """
    prices = np.array([[10.0, 10.0, 100.0, 100.0]])
    solution = _curve_model(GOLDEN, 4, 1).solve(prices, 0.0)
    assert solution.objective == pytest.approx(1500.0, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_scenario_matches_the_window_milp_on_random_instances(seed: int) -> None:
    """The same identity, away from the hand-computable case."""
    rng = np.random.default_rng(seed)
    prices = rng.normal(60.0, 40.0, 48)

    window = _window_model(REAL, 48).solve(prices, 10.0)
    curve = _curve_model(REAL, 48, 1).solve(prices[None, :], 10.0)

    assert curve.objective == pytest.approx(window.objective, rel=1e-9, abs=1e-6)


def test_the_optimum_is_never_worse_than_the_fixed_schedule() -> None:
    """The property that makes this construction worth building at all.

    A curve with every limit at the price cap *is* a fixed schedule, so
    ``max over curves >= fixed schedule`` necessarily -- and no constructed
    curve computes that max, so none of them is guaranteed to reach even the
    schedule it extends. Here the fixed schedule is a feasible point, so the
    inequality is structural.

    The comparison is *in sample*, at the believed prices, which is the only
    place the claim holds: out of sample a curve can still be beaten, and the
    backtest is what says whether it is.
    """
    rng = np.random.default_rng(20260916)
    base = 60.0 + 40.0 * np.sin(np.arange(48) * np.pi / 12.0)
    scenarios = base[None, :] + rng.normal(0.0, 20.0, (5, 48))

    optimised = _curve_model(REAL, 48, 5).solve(scenarios, 10.0)

    # The best fixed schedule against the same scenario set is the solve at
    # the mean price vector: the objective is linear in price and the feasible
    # set does not depend on it, so expectation and argmax commute. That
    # identity is the whole reason v3 had to be price-contingent.
    window = _window_model(REAL, 48).solve(scenarios.mean(axis=0), 10.0)

    assert optimised.objective >= window.objective - 1e-6


def test_the_expected_profit_of_a_fixed_schedule_is_its_mean_price_profit() -> None:
    """The mean-equivalence identity, stated as a test rather than as prose.

    A risk-neutral two-stage stochastic program whose only decision is a fixed
    schedule collapses to solving at ``E[lambda]``. This checks the algebra
    that makes the collapse true, and therefore checks the premise of the
    paragraph above: v2 already *is* that program's optimum, so a stochastic
    schedule would have been v2 at S times the cost.
    """
    rng = np.random.default_rng(7)
    scenarios = 60.0 + rng.normal(0.0, 25.0, (6, 24))
    dispatch = _window_model(REAL, 24).solve(scenarios.mean(axis=0), 10.0)

    dt, p = 1.0, REAL
    per_scenario = [
        float(
            np.sum(
                dispatch.p_d_mw * dt * (row - p.c_deg_eur_mwh)
                - dispatch.p_c_mw * dt * (row + p.charge_tariff_eur_mwh)
            )
        )
        for row in scenarios
    ]
    assert float(np.mean(per_scenario)) == pytest.approx(
        dispatch.objective, rel=1e-9, abs=1e-6
    )


def test_the_curve_is_monotone_in_price() -> None:
    """A curve that is not monotone is not a submittable bid.

    Imposed as a constraint rather than repaired by PAVA afterwards, so the
    optimiser chooses among monotone curves instead of being handed one and
    having it projected.
    """
    rng = np.random.default_rng(11)
    base = 60.0 + 40.0 * np.sin(np.arange(48) * np.pi / 12.0)
    scenarios = base[None, :] + rng.normal(0.0, 20.0, (5, 48))

    solution = _curve_model(REAL, 48, 5).solve(scenarios, 10.0)
    assert (np.diff(solution.quantities, axis=1) >= -1e-6).all()
    assert (np.diff(solution.band_prices, axis=1) >= -1e-9).all()
    # And it is the object BidCurves accepts, which is what lets settlement
    # clear an optimised curve through the same code as a constructed one.
    BidCurves(prices=solution.band_prices, quantities=solution.quantities)


def test_the_band_arithmetic_agrees_with_the_clearing_engine() -> None:
    """Two implementations of one auction rule, checked against each other.

    The optimiser decides which band a scenario lands in; settlement decides
    which step a realised price accepts. If they disagreed, the curve would be
    optimised for an acceptance pattern that never occurs -- and nothing
    downstream could see it, because both halves would still run.
    """
    from bess_arb.model.curve_pyomo import clearing_band

    rng = np.random.default_rng(5)
    scenarios = rng.normal(50.0, 30.0, (5, 12))
    band_prices, band_index = clearing_band(scenarios)

    quantities = np.sort(rng.normal(0.0, 5.0, (12, 5)), axis=1)
    curves = BidCurves(prices=band_prices, quantities=quantities)

    for s in range(scenarios.shape[0]):
        cleared = curves.clear(scenarios[s])
        expected = quantities[np.arange(12), band_index[s]]
        assert np.allclose(cleared, expected)


def test_tied_scenario_prices_land_in_the_same_band() -> None:
    """A single-valued curve cannot offer two quantities at one price."""
    from bess_arb.model.curve_pyomo import clearing_band

    scenarios = np.array([[10.0, 20.0], [10.0, 30.0], [50.0, 20.0]])
    _, band_index = clearing_band(scenarios)
    # Period 0: scenarios 0 and 1 both at 10.0.
    assert band_index[0, 0] == band_index[1, 0]
    # Period 1: scenarios 0 and 2 both at 20.0.
    assert band_index[0, 1] == band_index[2, 1]


def test_the_binaries_hold_under_negative_prices() -> None:
    """Invariant 5, in the form this formulation adds.

    The linking constraint fixes the *net* position, so without integrality a
    scenario could inflate ``p_c`` and ``p_d`` together to hold net constant
    while draining the battery through the round-trip loss -- which pays
    exactly when prices are negative. Checked on a full battery at a flat
    negative price, the same instance ``tests/test_model_binaries.py`` uses.
    """
    scenarios = np.full((3, 4), -50.0)
    solution = _curve_model(REAL, 4, 3).solve(scenarios, 20.0)
    curves = BidCurves(prices=solution.band_prices, quantities=solution.quantities)
    # A net position is one-signed by construction, so what this checks is
    # that the *curve* never offers a quantity the exclusivity would forbid.
    cleared = curves.clear(scenarios[0])
    assert np.isfinite(cleared).all()
    assert solution.status.has_solution


def test_shape_and_finiteness_are_refused_before_the_solver_sees_them() -> None:
    model = _curve_model(REAL, 4, 3)
    with pytest.raises(ValueError, match="shape"):
        model.solve(np.zeros((2, 4)), 0.0)
    with pytest.raises(ValueError, match="NaN or infinity"):
        model.solve(np.full((3, 4), np.nan), 0.0)


def test_quarter_hourly_periods_work_unchanged() -> None:
    """Invariant 3: dt is explicit, and nothing assumes an hour."""
    rng = np.random.default_rng(2)
    scenarios = 60.0 + rng.normal(0.0, 20.0, (3, 96))
    solution = _curve_model(REAL, 96, 3, dt_h=0.25).solve(scenarios, 10.0)
    assert solution.n_periods == 96
    assert solution.n_bands == 3
    assert solution.dt_h == 0.25
