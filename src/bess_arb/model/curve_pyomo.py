"""v3 stage 2: the bid curve as a decision, not a construction.

The formulation, and why it exists
----------------------------------

A curve with every limit at the price cap *is* the fixed schedule, so
``max over curves >= fixed schedule`` necessarily -- but a curve *constructed*
from K independent single-scenario solves computes no such max. It is cleared
period by period, and clearing each period against its own curve can assemble
a trajectory no scenario ever endorsed, which the state of charge then has to
clip. A battery's dispatch is a trajectory; a constructed bid curve is a bag
of per-period objects.

This module makes the curve a first-stage decision of a two-stage stochastic
program, with the per-scenario dispatch as recourse. That is the one change
that can fix the clipping structurally rather than by luck::

    first stage   q[t,k]              the curve: net MW offered in price band k
    recourse      p_c[s,t], p_d[s,t]  what scenario s actually does
    linking       p_d[s,t] - p_c[s,t] = q[t, k(s,t)]

``k(s,t)`` is **data, not a decision**: given scenario ``s``'s price at
``t``, the band it clears in is fixed by the auction rule, exactly as
:meth:`bess_arb.bid.curve.BidCurves.clear` computes it. So the linking is
linear, and the whole thing is one MILP rather than anything bilinear.

Three properties follow, and they are the entire case for the module:

1. **Feasible by construction on the sampled acceptance patterns.** Each
   scenario carries its own SoC trajectory and its own power limits, so the
   curve chosen is one whose clearing the battery can actually deliver in
   every scenario it was optimised against. Clipping is not repaired
   afterwards; it is designed out.
2. **At least the fixed schedule, in sample.** Setting every band of a period
   to the same quantity reproduces a fixed schedule, which is feasible here,
   so the optimum cannot be worse than v2's on the same scenario set.
3. **Not equivalent to the mean forecast.** A risk-neutral stochastic program
   whose only decision is a fixed schedule collapses to solving at ``E[lambda]``,
   because the objective is linear in price and the feasible set does not
   depend on it -- which is why v2 already *is* that program's optimum. The
   collapse needs the decision to be price-independent. Here it is not: the
   recourse is contingent on which band cleared, so the mixture across
   scenarios does not cancel.

What it does not promise
------------------------

Feasibility is enforced on the ``S`` sampled acceptance patterns, not on all
``S**n`` of them. A realised day that clears period 3 in one scenario's band
and period 20 in another's is a pattern the optimiser never saw, and it can
still need repair. Enforcing every pattern means capping each step at what any
acceptance pattern could support, and that worst-case envelope only ever
narrows until almost nothing can be offered -- so this is the deliberate
middle, and the residual clipping is the number that says how well it held.

The invariant 2 boundary
------------------------

CLAUDE.md invariant 2 says every *policy* obtains its schedule from the same
:class:`~bess_arb.model.base.BatteryMILP`, differing only in the price vector.
That stands unchanged: the floor, the forecast policy and the oracle all still
go through the single window model, and the ladder's "% of bound" is still a
statement about information alone. This is not a policy and is not a rung of
that ladder -- it is a bidding object, scored as a bidding result. The
exemption is scoped to exactly that; see
:class:`~bess_arb.model.base.BidCurveMILP`.

*The physical model is not duplicated.* The balance, exclusivity, big-M and
objective below are the same relations
:mod:`bess_arb.model.pyomo` writes, per scenario, with the same
``dt``-explicit energy terms and the same one-binary-per-period orientation
flag. What is new is only the first-stage curve and the linking constraint.
The binaries are needed here for the reason they are needed there, and one
more: the linking constraint fixes the *net* position, so without integrality
a scenario could inflate ``p_c`` and ``p_d`` together to hold net constant
while draining the battery, which is profitable exactly when prices are
negative (CLAUDE.md invariant 5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray
from pyomo.contrib.solver.common.factory import SolverFactory

from bess_arb.model.base import SolveError
from bess_arb.model.pyomo import _finite_or_none, _normalise_status, _version_string
from bess_arb.model.spec import (
    BatteryParams,
    CurveSolution,
    FloatArray,
    SolverConfig,
)

__all__ = ["PyomoBidCurveMILP", "clearing_band"]

_SOC_TOL_MWH = 1e-6


def clearing_band(
    scenario_prices: FloatArray,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Sorted band limit prices, and which band each scenario clears in.

    Returns ``(band_prices, band_index)`` shaped ``(n_periods, S)`` and
    ``(S, n_periods)``.

    The bands are the scenario prices themselves, sorted -- the same steps
    :func:`bess_arb.bid.curve.build_curves` produces, so a curve solved here
    can be handed straight to :class:`~bess_arb.bid.curve.BidCurves` and
    cleared by the same code that clears a constructed one. That shared
    clearing path is what makes the two comparable.

    ``side="right" - 1`` is
    :meth:`bess_arb.bid.curve.BidCurves.clear`'s rule, copied deliberately
    rather than approximated: a bid at limit ``L`` is in the money when the
    clearing price *reaches* ``L``. It also settles the tie case -- two
    scenarios at the same price land in the same band and therefore receive
    the same quantity, which is what a single-valued curve requires.
    """
    prices = np.asarray(scenario_prices, dtype=np.float64)
    band_prices = np.sort(prices, axis=0).T
    n_scenarios, n_periods = prices.shape
    band_index = np.empty((n_scenarios, n_periods), dtype=np.int64)
    for t in range(n_periods):
        band_index[:, t] = (
            np.searchsorted(band_prices[t], prices[:, t], side="right") - 1
        )
    np.clip(band_index, 0, n_scenarios - 1, out=band_index)
    return band_prices, band_index


class PyomoBidCurveMILP:
    """Two-stage stochastic bid-curve optimiser, built once and re-solved.

    Same discipline as :class:`~bess_arb.model.pyomo.PyomoBatteryMILP`: the
    constructor fixes the structure for a given ``(n_periods, n_scenarios)``
    and :meth:`solve` mutates only ``Param`` values. The subtlety is that the
    *linking* changes between windows -- which band a scenario clears in
    depends on its prices -- so the band assignment is carried as a mutable
    selector ``select[s,t,k]`` rather than as a rebuilt constraint. Rebuilding
    it per window would put model construction back inside the loop, which is
    the one thing CLAUDE.md forbids here.
    """

    def __init__(
        self,
        params: BatteryParams,
        n_periods: int,
        n_scenarios: int,
        dt_h: float,
        *,
        solver: SolverConfig | None = None,
    ) -> None:
        if n_periods < 1:
            raise ValueError(f"n_periods must be at least 1, got {n_periods}")
        if n_scenarios < 1:
            raise ValueError(f"n_scenarios must be at least 1, got {n_scenarios}")
        if not dt_h > 0.0:
            raise ValueError(f"dt_h must be positive, got {dt_h}")

        self._params = params
        self._n_periods = n_periods
        self._n_scenarios = n_scenarios
        self._dt_h = dt_h
        self._solver_config = solver if solver is not None else SolverConfig()
        self._model = self._build_model()
        self._solver = self._make_solver(self._solver_config)

    # -- construction ----------------------------------------------------

    def _build_model(self) -> pyo.ConcreteModel:
        p = self._params
        dt = self._dt_h

        m = pyo.ConcreteModel(name="bid_curve_stochastic")
        m.T = pyo.RangeSet(0, self._n_periods - 1)
        m.S = pyo.RangeSet(0, self._n_scenarios - 1)
        m.K = pyo.RangeSet(0, self._n_scenarios - 1)

        m.price = pyo.Param(m.S, m.T, mutable=True, initialize=0.0, within=pyo.Reals)
        m.soc_initial = pyo.Param(mutable=True, initialize=0.0, within=pyo.Reals)
        # The band assignment, as data. One-hot over K per (s, t): a mutable
        # Param so that a new window is a value update rather than a rebuild.
        m.select = pyo.Param(
            m.S, m.T, m.K, mutable=True, initialize=0.0, within=pyo.Reals
        )

        # First stage: the curve. Free in sign -- negative is bought, positive
        # is sold, the same signed net position the constructed curve uses.
        m.q = pyo.Var(m.T, m.K, bounds=(-p.p_max_mw, p.p_max_mw))

        # Recourse: what each scenario ends up doing.
        m.p_c = pyo.Var(m.S, m.T, bounds=(0.0, p.p_max_mw))
        m.p_d = pyo.Var(m.S, m.T, bounds=(0.0, p.p_max_mw))
        m.soc = pyo.Var(m.S, m.T, bounds=(0.0, p.e_max_mwh))
        m.u = pyo.Var(m.S, m.T, domain=pyo.Binary)

        def _monotone(m: pyo.ConcreteModel, t: int, k: int) -> Any:
            """A submittable curve sells no less as the price rises.

            Skipped at the last band because there is no successor. This is
            the same condition PAVA imposes on a constructed curve; here it is
            a constraint rather than a projection, so the optimiser chooses
            among monotone curves instead of being handed a non-monotone one
            and having it repaired.
            """
            if k == self._n_scenarios - 1:
                return pyo.Constraint.Skip
            return m.q[t, k] <= m.q[t, k + 1]

        m.monotone = pyo.Constraint(m.T, m.K, rule=_monotone)

        def _link(m: pyo.ConcreteModel, s: int, t: int) -> Any:
            """Scenario ``s`` takes the quantity its cleared band offers."""
            return m.p_d[s, t] - m.p_c[s, t] == sum(
                m.select[s, t, k] * m.q[t, k] for k in m.K
            )

        m.link = pyo.Constraint(m.S, m.T, rule=_link)

        def _balance(m: pyo.ConcreteModel, s: int, t: int) -> Any:
            previous = m.soc_initial if t == 0 else m.soc[s, t - 1]
            return m.soc[s, t] == (
                previous + p.eta_c * m.p_c[s, t] * dt - m.p_d[s, t] * dt / p.eta_d
            )

        m.balance = pyo.Constraint(m.S, m.T, rule=_balance)

        def _charge_exclusivity(m: pyo.ConcreteModel, s: int, t: int) -> Any:
            return m.p_c[s, t] <= p.p_max_mw * m.u[s, t]

        def _discharge_exclusivity(m: pyo.ConcreteModel, s: int, t: int) -> Any:
            return m.p_d[s, t] <= p.p_max_mw * (1.0 - m.u[s, t])

        m.charge_exclusivity = pyo.Constraint(m.S, m.T, rule=_charge_exclusivity)
        m.discharge_exclusivity = pyo.Constraint(m.S, m.T, rule=_discharge_exclusivity)

        # Expected profit: each scenario settled at its own prices, equally
        # weighted because the scenarios are an equally-weighted sample of the
        # predictive law (they are drawn, not quadrature nodes).
        weight = 1.0 / self._n_scenarios
        m.profit = pyo.Objective(
            expr=weight
            * sum(
                m.p_d[s, t] * dt * (m.price[s, t] - p.c_deg_eur_mwh)
                - m.p_c[s, t] * dt * (m.price[s, t] + p.charge_tariff_eur_mwh)
                for s in m.S
                for t in m.T
            ),
            sense=pyo.maximize,
        )
        return m

    def _make_solver(self, config: SolverConfig) -> Any:
        solver = SolverFactory(config.name)
        if solver is None:  # pragma: no cover - depends on the environment
            raise SolveError(f"solver {config.name!r} is not registered")
        if not solver.available():  # pragma: no cover - depends on the environment
            raise SolveError(f"solver {config.name!r} is registered but not available")
        solver.config.load_solutions = True
        solver.config.raise_exception_on_nonoptimal_result = False
        if config.mip_gap is not None:
            solver.config.rel_gap = config.mip_gap
        if config.time_limit_s is not None:
            solver.config.time_limit = config.time_limit_s
        if config.threads is not None:
            solver.config.threads = config.threads
        for key, value in config.options.items():
            solver.config.solver_options[key] = value
        return solver

    # -- solving ---------------------------------------------------------

    def solve(self, scenario_prices: FloatArray, soc_initial: float) -> CurveSolution:
        """Choose the curve maximising expected profit over these scenarios."""
        prices = np.asarray(scenario_prices, dtype=np.float64)
        expected = (self._n_scenarios, self._n_periods)
        if prices.shape != expected:
            raise ValueError(
                f"scenario_prices must have shape {expected}, got {prices.shape}"
            )
        if not np.isfinite(prices).all():
            raise ValueError("scenario_prices contain NaN or infinity")

        soc_initial = self._clamp_soc(soc_initial)
        band_prices, band_index = clearing_band(prices)

        m = self._model
        m.price.store_values(
            {
                (s, t): float(prices[s, t])
                for s in range(self._n_scenarios)
                for t in range(self._n_periods)
            }
        )
        m.soc_initial.set_value(soc_initial)
        m.select.store_values(
            {
                (s, t, k): 1.0 if k == int(band_index[s, t]) else 0.0
                for s in range(self._n_scenarios)
                for t in range(self._n_periods)
                for k in range(self._n_scenarios)
            }
        )

        results = self._solver.solve(m)
        status = _normalise_status(results)
        if not status.has_solution or results.incumbent_objective is None:
            raise SolveError(
                f"bid-curve solve returned no usable solution: "
                f"status={status.value}, "
                f"termination={results.termination_condition}"
            )

        quantities = np.array(
            [
                [pyo.value(m.q[t, k]) for k in range(self._n_scenarios)]
                for t in range(self._n_periods)
            ],
            dtype=np.float64,
        )
        return CurveSolution(
            status=status,
            objective=float(results.incumbent_objective),
            band_prices=band_prices,
            quantities=quantities,
            dt_h=self._dt_h,
            objective_bound=_finite_or_none(results.objective_bound),
            solver_name=self._solver_config.name,
            solver_version=_version_string(results.solver_version),
        )

    def _clamp_soc(self, soc_initial: float) -> float:
        e_max = self._params.e_max_mwh
        if not np.isfinite(soc_initial):
            raise ValueError(f"soc_initial must be finite, got {soc_initial}")
        if not -_SOC_TOL_MWH <= soc_initial <= e_max + _SOC_TOL_MWH:
            raise ValueError(f"soc_initial must lie in [0, {e_max}], got {soc_initial}")
        return float(min(max(soc_initial, 0.0), e_max))
