"""Pyomo backend — the v1 implementation of :class:`BatteryMILP`.

The formulation, and why each piece is the way it is
----------------------------------------------------

Decision variables, per period ``t`` of length ``dt_h``:

``p_c[t] in [0, P_max]``   charging power at the connection point (MW)
``p_d[t] in [0, P_max]``   discharging power at the connection point (MW)
``soc[t] in [0, E_max]``   state of charge at the *end* of period t (MWh)
``u[t] in {0, 1}``         1 admits charging, 0 admits discharging

::

    soc[t] = soc[t-1] + eta_c*p_c[t]*dt - p_d[t]*dt/eta_d      (balance)
    p_c[t] <= P_max * u[t]                                     (exclusivity)
    p_d[t] <= P_max * (1 - u[t])
    max sum_t [ p_d[t]*dt*(lambda[t] - c_deg)
                - p_c[t]*dt*(lambda[t] + charge_tariff) ]

*Every energy term carries ``dt`` explicitly.* Periods are one hour before
1 October 2025 and a quarter of an hour after; nothing here may assume
either (CLAUDE.md invariant 3).

*One binary per period, not two.* ``u`` is an orientation flag rather than a
pair of on/off indicators with ``u_c + u_d <= 1``: idle is already
representable (``u[t] = 1`` with ``p_c[t] = 0``), so a second binary would
double the integer count and tighten nothing.

*The big-M is ``P_max``, the physical rating.* The exclusivity constraints
therefore add no slack beyond the variable bounds that are present anyway:
at ``u[t] = 1`` the charge constraint is exactly the existing upper bound
and the discharge constraint pins ``p_d[t]`` to zero. This is as tight as a
single-binary formulation gets; tightening further would need SoC-dependent
power limits, which this asset does not have.

*The binaries are not redundant.* With positive prices and ``eta_rt < 1``
exclusivity holds at the LP optimum on its own, but negative prices pay for
simultaneous charge and discharge — holding SoC flat costs ``eta_c*eta_d``
of what is charged and collects ``(1 - eta_rt)*|lambda|`` per MW of
fictitious throughput. Spain sees negative prices, so the integrality is
load-bearing; ``docs/DECISIONS.md`` §3.2 and ``tests/test_model_binaries.py``.

*Degradation is charged on discharge only*, following Xu et al.: charging
throughput would double-count the same ageing.

Build once, mutate, re-solve
----------------------------

Prices and the initial SoC are ``Param(mutable=True)``. :meth:`solve`
overwrites those and re-solves; the model is never rebuilt. At ~15,000
window solves of a few-hundred-variable model, construction dominates the
solve, which is the whole reason for the persistent interface.

``SolverFactory`` here is the one from ``pyomo.contrib.solver`` — the newer
interface registered under the name ``"highs"``, which beats
``"appsi_highs"`` on the same workload. Measured on this backend over sixty
192-period windows: 64 ms/solve re-solving against 125 ms/solve rebuilding.
The legacy ``pyomo.environ.SolverFactory("highs")`` wraps the same class but
hands back legacy result objects; this module wants the v2 :class:`Results`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)

from bess_arb.model.base import SolveError
from bess_arb.model.spec import (
    BatteryParams,
    FloatArray,
    Solution,
    SolverConfig,
    SolveStatus,
)

__all__ = ["PyomoBatteryMILP"]

# Tolerance for accepting an initial SoC that a previous window's solution
# put marginally outside [0, E_max]. Solvers return -1e-15 where they mean
# zero; that must not be mistaken for a caller passing nonsense.
_SOC_TOL_MWH = 1e-6

_TERMINATION_MAP: dict[TerminationCondition, SolveStatus] = {
    TerminationCondition.convergenceCriteriaSatisfied: SolveStatus.OPTIMAL,
    TerminationCondition.maxTimeLimit: SolveStatus.FEASIBLE,
    TerminationCondition.iterationLimit: SolveStatus.FEASIBLE,
    TerminationCondition.objectiveLimit: SolveStatus.FEASIBLE,
    TerminationCondition.interrupted: SolveStatus.FEASIBLE,
    TerminationCondition.provenInfeasible: SolveStatus.INFEASIBLE,
    TerminationCondition.locallyInfeasible: SolveStatus.INFEASIBLE,
    TerminationCondition.infeasibleOrUnbounded: SolveStatus.INFEASIBLE,
    TerminationCondition.unbounded: SolveStatus.UNBOUNDED,
}


def _finite_or_none(value: Any) -> float | None:
    """The dual bound, or ``None`` where the solver reported no usable one.

    An unsolved MILP's bound is ±inf, which is true but useless in a JSON
    artifact and would make :attr:`Solution.relative_gap` infinite rather
    than absent. Absent is the honest encoding of "not reported".
    """
    if value is None:
        return None
    bound = float(value)
    return bound if np.isfinite(bound) else None


def _version_string(value: Any) -> str | None:
    """Normalise Pyomo's version tuple to a plain string.

    Provenance for ``results/annual_bound.json``: a number produced with a
    time limit is only traceable if the version that produced it is recorded
    next to it. A string, so no solver-owned type crosses the boundary.
    """
    if value is None:
        return None
    if isinstance(value, tuple | list):
        return ".".join(str(part) for part in value)
    return str(value)


def _normalise_status(results: Results) -> SolveStatus:
    """Map Pyomo's termination codes onto the project's own enum.

    Nothing solver-specific may escape this module, so the mapping happens
    here rather than at the call site. A limit-terminated solve with an
    incumbent is FEASIBLE, not OPTIMAL: the distinction is the difference
    between a bound and an estimate, and the backtest is entitled to know.
    """
    status = _TERMINATION_MAP.get(results.termination_condition, SolveStatus.ERROR)
    if status is SolveStatus.OPTIMAL and results.solution_status not in (
        SolutionStatus.optimal,
        SolutionStatus.feasible,
    ):
        return SolveStatus.ERROR
    return status


class PyomoBatteryMILP:
    """Persistent Pyomo/HiGHS implementation of the battery MILP."""

    def __init__(
        self,
        params: BatteryParams,
        n_periods: int,
        dt_h: float,
        *,
        solver: SolverConfig | None = None,
        relax_binaries: bool = False,
    ) -> None:
        if n_periods < 1:
            raise ValueError(f"n_periods must be at least 1, got {n_periods}")
        if not dt_h > 0.0:
            raise ValueError(f"dt_h must be positive, got {dt_h}")

        self._params = params
        self._n_periods = n_periods
        self._dt_h = dt_h
        self._relax_binaries = relax_binaries
        self._solver_config = solver if solver is not None else SolverConfig()

        self._model = self._build_model()
        self._solver = self._make_solver(self._solver_config)

    # -- construction ----------------------------------------------------

    def _build_model(self) -> pyo.ConcreteModel:
        p = self._params
        dt = self._dt_h

        m = pyo.ConcreteModel(name="battery_arbitrage")
        m.T = pyo.RangeSet(0, self._n_periods - 1)

        # Mutable: these two are the only things a window changes.
        # `within=Reals` on the price since negative
        # day-ahead prices are the case the binaries exist for.
        m.price = pyo.Param(m.T, mutable=True, initialize=0.0, within=pyo.Reals)
        m.soc_initial = pyo.Param(mutable=True, initialize=0.0, within=pyo.Reals)

        m.p_c = pyo.Var(m.T, bounds=(0.0, p.p_max_mw))
        m.p_d = pyo.Var(m.T, bounds=(0.0, p.p_max_mw))
        m.soc = pyo.Var(m.T, bounds=(0.0, p.e_max_mwh))
        m.u = pyo.Var(
            m.T, domain=pyo.UnitInterval if self._relax_binaries else pyo.Binary
        )

        # The rules return `Any` because what `m.soc[t] == ...` builds is an
        # EqualityExpression / InequalityExpression — Pyomo overloads the
        # comparison operators to construct an expression tree rather than a
        # bool. `pyo.Expression` would be the wrong annotation: that is the
        # component for *named* expressions, not the type of a relational
        # one, and pyomo is untyped so nothing would catch the mistake.
        def _balance(m: pyo.ConcreteModel, t: int) -> Any:
            previous = m.soc_initial if t == 0 else m.soc[t - 1]
            return m.soc[t] == (
                previous + p.eta_c * m.p_c[t] * dt - m.p_d[t] * dt / p.eta_d
            )

        m.balance = pyo.Constraint(m.T, rule=_balance)

        def _charge_exclusivity(m: pyo.ConcreteModel, t: int) -> Any:
            return m.p_c[t] <= p.p_max_mw * m.u[t]

        def _discharge_exclusivity(m: pyo.ConcreteModel, t: int) -> Any:
            return m.p_d[t] <= p.p_max_mw * (1.0 - m.u[t])

        m.charge_exclusivity = pyo.Constraint(m.T, rule=_charge_exclusivity)
        m.discharge_exclusivity = pyo.Constraint(m.T, rule=_discharge_exclusivity)

        m.profit = pyo.Objective(
            expr=sum(
                m.p_d[t] * dt * (m.price[t] - p.c_deg_eur_mwh)
                - m.p_c[t] * dt * (m.price[t] + p.charge_tariff_eur_mwh)
                for t in m.T
            ),
            sense=pyo.maximize,
        )
        return m

    # `Any`: Pyomo ships no usable stubs, and the alternative is a fake
    # Protocol for the solver that would have to be kept in step with an
    # untyped upstream. The typed surface of this module is its own.
    def _make_solver(self, config: SolverConfig) -> Any:
        solver = SolverFactory(config.name)
        if solver is None:  # pragma: no cover - depends on the environment
            raise SolveError(
                f"solver {config.name!r} is not registered with Pyomo's "
                "contrib solver factory"
            )
        if not solver.available():  # pragma: no cover - depends on the environment
            raise SolveError(f"solver {config.name!r} is registered but not available")

        solver.config.load_solutions = True
        # Non-optimal results are normalised and reported, not raised over,
        # so that a gap-limited solve stays distinguishable from a failure.
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

    def solve(self, prices: FloatArray, soc_initial: float) -> Solution:
        prices = np.asarray(prices, dtype=float)
        if prices.shape != (self._n_periods,):
            raise ValueError(
                f"prices must have shape ({self._n_periods},), got {prices.shape}"
            )
        if not np.isfinite(prices).all():
            raise ValueError("prices contain NaN or infinity")

        soc_initial = self._clamp_soc(soc_initial)

        m = self._model
        m.price.store_values(dict(enumerate(prices.tolist())))
        m.soc_initial.set_value(soc_initial)

        results = self._solver.solve(m)
        status = _normalise_status(results)
        if not status.has_solution or results.incumbent_objective is None:
            raise SolveError(
                f"window solve returned no usable solution: status={status.value}, "
                f"termination={results.termination_condition}"
            )

        return Solution(
            status=status,
            objective=float(results.incumbent_objective),
            p_c_mw=self._values(m.p_c),
            p_d_mw=self._values(m.p_d),
            soc_mwh=self._values(m.soc),
            dt_h=self._dt_h,
            objective_bound=_finite_or_none(results.objective_bound),
            solver_name=self._solver_config.name,
            solver_version=_version_string(results.solver_version),
        )

    def _clamp_soc(self, soc_initial: float) -> float:
        """Accept a previous solution's terminal SoC, reject genuine nonsense.

        The backtest carries SoC forward between windows, and a solver that
        returns -1e-15 for an empty battery would otherwise fail the bound
        check on the next window. Anything outside tolerance is a caller
        error and is raised, not quietly squeezed into range.
        """
        e_max = self._params.e_max_mwh
        if not np.isfinite(soc_initial):
            raise ValueError(f"soc_initial must be finite, got {soc_initial}")
        if not -_SOC_TOL_MWH <= soc_initial <= e_max + _SOC_TOL_MWH:
            raise ValueError(f"soc_initial must lie in [0, {e_max}], got {soc_initial}")
        return float(min(max(soc_initial, 0.0), e_max))

    def _values(self, var: pyo.Var) -> FloatArray:
        return np.fromiter(
            (pyo.value(var[t]) for t in self._model.T),
            dtype=np.float64,
            count=self._n_periods,
        )
