"""The backend contract: one Protocol, one exception.

Solver-free by contract, like :mod:`bess_arb.model.spec`.

The backend is a *stateful object*, not a pure function, and that is the
whole design. Build-once-mutate-re-solve means something has to hold a live
solver model across the backtest loop; if the loop held it, the solver would
have leaked out of ``model/`` and "swap one file" would stop being true. So
the persistent model is owned inside the backend, callers pass numpy in and
get a :class:`~bess_arb.model.spec.Solution` out.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bess_arb.model.spec import BatteryParams, FloatArray, Solution, SolverConfig

__all__ = ["BatteryMILP", "SolveError"]


class SolveError(RuntimeError):
    """Raised when a solve returns no usable primal solution.

    A window that cannot be solved must stop the backtest rather than
    contribute a schedule of zeros, which would look like a legitimate
    "stay idle" decision and silently depress the result.
    """


@runtime_checkable
class BatteryMILP(Protocol):
    """What every backend implements.

    The constructor fixes the *structure* — battery, horizon length, period
    length — and builds the model once. :meth:`solve` changes only the price
    coefficients and the initial state, which is what makes a persistent
    re-solve possible. A backend that rebuilds inside :meth:`solve` satisfies
    this Protocol and defeats its purpose; see ``docs/DECISIONS.md`` §6.3.

    ``relax_binaries`` is a diagnostic, not a policy option. It exists so
    ``tests/test_model_binaries.py`` can demonstrate that the LP relaxation
    is strictly better under negative prices, from outside ``model/`` and
    without importing a solver. No policy may use it: relaxing the MILP,
    including for the bound, is forbidden (CLAUDE.md invariant 5).
    """

    def __init__(
        self,
        params: BatteryParams,
        n_periods: int,
        dt_h: float,
        *,
        solver: SolverConfig | None = None,
        relax_binaries: bool = False,
    ) -> None: ...

    def solve(self, prices: FloatArray, soc_initial: float) -> Solution:
        """Re-solve for a new price vector and initial state.

        ``prices`` is €/MWh over ``n_periods`` periods and may be negative;
        ``soc_initial`` is MWh at the start of the first period.
        """
        ...
