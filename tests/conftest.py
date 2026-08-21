"""Shared fixtures for the modelling-layer tests.

Every test here goes through :func:`bess_arb.model.get_backend`, never
through a backend class by name, and never imports a solver. That is not
politeness: the import boundary is a contract (CLAUDE.md), and it also means
these tests automatically cover PyOptInterface the day it is registered.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from bess_arb.model import BatteryMILP, BatteryParams, FloatArray, get_backend
from bess_arb.model.spec import Solution, SolverConfig

# The golden case's asset: the real battery with degradation switched off,
# so the optimum is hand-computable. See docs/DECISIONS.md §3.1.
GOLDEN_PARAMS = BatteryParams(
    p_max_mw=10.0,
    e_max_mwh=20.0,
    eta_rt=0.85,
    c_deg_eur_mwh=0.0,
    charge_tariff_eur_mwh=0.0,
)

ModelFactory = Callable[..., BatteryMILP]


@pytest.fixture(params=["pyomo"])
def backend_name(request: pytest.FixtureRequest) -> str:
    """Backends under test.

    Listed explicitly rather than read from ``available_backends()`` so that
    adding a backend without wiring it into the equivalence test is a
    deliberate act rather than an oversight.
    """
    return str(request.param)


@pytest.fixture
def make_model(backend_name: str) -> ModelFactory:
    """Build a backend instance without naming its class."""
    backend = get_backend(backend_name)

    def _factory(
        params: BatteryParams = GOLDEN_PARAMS,
        n_periods: int = 4,
        dt_h: float = 1.0,
        *,
        relax_binaries: bool = False,
        solver: SolverConfig | None = None,
    ) -> BatteryMILP:
        return backend(
            params,
            n_periods,
            dt_h,
            solver=solver,
            relax_binaries=relax_binaries,
        )

    return _factory


@pytest.fixture
def golden_params() -> BatteryParams:
    return GOLDEN_PARAMS


def profit_from_dispatch(
    solution: Solution, prices: FloatArray, params: BatteryParams
) -> float:
    """Recompute the objective from the dispatch vectors.

    Independent arithmetic on the solver's own answer: it catches a sign
    error or a missing ``dt`` that a plausible-looking objective value would
    hide.
    """
    dt = solution.dt_h
    revenue = float((solution.p_d_mw * prices).sum()) * dt
    energy_cost = float((solution.p_c_mw * prices).sum()) * dt
    degradation = solution.discharged_mwh * params.c_deg_eur_mwh
    tariff = solution.charged_mwh * params.charge_tariff_eur_mwh
    return revenue - energy_cost - degradation - tariff


def soc_trajectory(
    solution: Solution, params: BatteryParams, soc_initial: float
) -> FloatArray:
    """Replay the energy balance from the dispatch, independently of the model."""
    dt = solution.dt_h
    delta = params.eta_c * solution.p_c_mw * dt - solution.p_d_mw * dt / params.eta_d
    return np.asarray(soc_initial + np.cumsum(delta), dtype=np.float64)
