"""The annual-window bound: how much of the gap is horizon, not information.

``docs/DECISIONS.md`` §2.4 asks for one extra solve — the same optimiser over
a whole year with the state of charge free across days, rather than the
rolling two-day window the policies get. Comparing it against the *rolling*
oracle over the identical days separates two things the headline ratio would
otherwise blur:

- **the value of information** — what the rolling oracle earns over a policy
  that has to forecast, which is the number the project is about;
- **the value of horizon** — what an unrestricted year earns over the rolling
  oracle, which for a 2-hour battery should be small, because carrying energy
  overnight costs the intraday cycle and the intraday cycle almost always
  pays more.

Small is the expectation, not the finding. It is measured.

Why this cannot be relaxed
--------------------------

Hourly, this is 8,760 periods: ~35,040 variables of which 8,760 are binary
(~140,160 and 35,040 quarter-hourly). That is far past a window solve, and
the obvious escape — drop the integrality and solve an LP — is forbidden
(CLAUDE.md invariant 5, §7). With negative prices the relaxation charges and
discharges at once and *scores higher than the true optimum*, so relaxing
would inflate the denominator of every ratio in the repository. A bound may
be loose; it may not be wrong in the direction that flatters the result.

What is done instead is to accept a gap and report it. The run records the
incumbent, the dual bound, the gap actually reached and the wall clock, so a
reader can see exactly how much slack the published number carries — and the
same command run with ``--solver gurobi`` on a cluster produces the reference
figure to compare it against. See ``hpc/README.md``.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bess_arb.model import get_backend
from bess_arb.model.spec import BatteryParams, SolverConfig, SolveStatus
from bess_arb.timeline import Regime, utc_index

__all__ = ["AnnualBound", "environment", "solve_annual_bound"]


@dataclass(frozen=True, slots=True)
class AnnualBound:
    """One annual-window solve, with the slack it was found with."""

    first_day: dt.date
    last_day: dt.date
    regime: str
    dt_h: float
    periods: int
    variables: int
    binaries: int
    status: SolveStatus
    objective_eur: float
    objective_bound_eur: float | None
    relative_gap: float | None
    charged_mwh: float
    discharged_mwh: float
    equivalent_cycles: float
    soc_initial_mwh: float
    soc_final_mwh: float
    wall_clock_s: float
    solver_name: str | None
    solver_version: str | None
    mip_gap_requested: float | None
    time_limit_s: float | None

    @property
    def proven_optimal(self) -> bool:
        return self.status is SolveStatus.OPTIMAL

    def as_dict(self) -> dict[str, Any]:
        return {
            "first_day": self.first_day.isoformat(),
            "last_day": self.last_day.isoformat(),
            "regime": self.regime,
            "dt_h": self.dt_h,
            "periods": self.periods,
            "variables": self.variables,
            "binaries": self.binaries,
            "status": self.status.value,
            "proven_optimal": self.proven_optimal,
            "objective_eur": round(self.objective_eur, 2),
            "objective_bound_eur": (
                None
                if self.objective_bound_eur is None
                else round(self.objective_bound_eur, 2)
            ),
            "relative_gap": (
                None if self.relative_gap is None else round(self.relative_gap, 8)
            ),
            "charged_mwh": round(self.charged_mwh, 3),
            "discharged_mwh": round(self.discharged_mwh, 3),
            "equivalent_cycles": round(self.equivalent_cycles, 3),
            "soc_initial_mwh": round(self.soc_initial_mwh, 6),
            "soc_final_mwh": round(self.soc_final_mwh, 6),
            "wall_clock_s": round(self.wall_clock_s, 3),
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "mip_gap_requested": self.mip_gap_requested,
            "time_limit_s": self.time_limit_s,
        }


def solve_annual_bound(
    prices: pd.Series,
    params: BatteryParams,
    regime: Regime,
    first_day: dt.date,
    last_day: dt.date,
    *,
    soc_initial_mwh: float,
    backend: str = "pyomo",
    solver: SolverConfig | None = None,
) -> AnnualBound:
    """Solve the whole span as one problem, SoC free across days.

    Everything except the horizon matches the rolling oracle: the same
    battery, the same degradation cost, the same opening state of charge,
    realised prices, no terminal condition. That is what makes the difference
    attributable to the horizon alone.
    """
    window = utc_index(first_day, last_day, regime)
    values = prices.reindex(window)
    if values.isna().any():
        raise KeyError(
            f"the annual window {first_day}..{last_day} is missing "
            f"{int(values.isna().sum())} of {len(window)} periods"
        )
    vector = values.to_numpy(dtype="float64")

    model = get_backend(backend)(params, len(window), regime.dt_h, solver=solver)

    started = time.perf_counter()
    solution = model.solve(vector, soc_initial_mwh)
    elapsed = time.perf_counter() - started

    discharged = solution.discharged_mwh
    return AnnualBound(
        first_day=first_day,
        last_day=last_day,
        regime=regime.name,
        dt_h=regime.dt_h,
        periods=len(window),
        # p_c, p_d and soc per period, plus the orientation binary. Stated
        # from the formulation rather than asked of the solver, because
        # asking would mean a solver object crossing the model/ boundary.
        variables=4 * len(window),
        binaries=len(window),
        status=solution.status,
        objective_eur=solution.objective,
        objective_bound_eur=solution.objective_bound,
        relative_gap=solution.relative_gap,
        charged_mwh=solution.charged_mwh,
        discharged_mwh=discharged,
        equivalent_cycles=discharged / params.e_max_mwh,
        soc_initial_mwh=soc_initial_mwh,
        soc_final_mwh=float(solution.soc_mwh[-1]),
        wall_clock_s=elapsed,
        solver_name=solution.solver_name,
        solver_version=solution.solver_version,
        mip_gap_requested=None if solver is None else solver.mip_gap,
        time_limit_s=None if solver is None else solver.time_limit_s,
    )


def environment() -> dict[str, Any]:
    """Machine and interpreter, so a wall clock means something later.

    "HiGHS reached 0.1% in 42 minutes" is only a comparison against the
    cluster run if both records say what they ran on.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
