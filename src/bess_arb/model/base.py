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

from bess_arb.model.spec import (
    BatteryParams,
    CurveSolution,
    FloatArray,
    Solution,
    SolverConfig,
)

__all__ = ["BatteryMILP", "BidCurveMILP", "SolveError"]


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


@runtime_checkable
class BidCurveMILP(Protocol):
    """v3 stage 2: what a bid-curve optimiser implements.

    Declared beside :class:`BatteryMILP` rather than as a widening of it,
    because the two answer different questions. :meth:`BatteryMILP.solve`
    returns a *dispatch* for one price vector. This returns a *curve* — a
    first-stage decision taken before prices are known, with the dispatch as
    per-scenario recourse. Handing both back through one Protocol would make
    the return type ambiguous exactly where the distinction matters.

    The invariant 2 boundary, stated here because this is where it is easiest
    to cross by accident
    -------------------------------------------------------------------------

    CLAUDE.md invariant 2 says every **policy** obtains its schedule from the
    same :class:`BatteryMILP`, differing only in the price vector. That is
    what makes "the policy captured 84% of the bound" a statement about
    information rather than about two programs. It is unchanged: the floor,
    the forecast policy and the oracle all still go through the single window
    model, and the ladder that produces the chart is untouched.

    **This is not a policy and must never become a rung of that ladder.** It
    is a bidding object: it is scored as a bidding result against the same
    oracle bound, and it is reported separately. The scope of the exemption is
    exactly that sentence — a second formulation is permitted for the bidding
    question and for nothing else. A backend that used it to serve a policy
    would satisfy this Protocol and defeat its purpose, in the same way
    ``docs/DECISIONS.md`` §6.3 describes for a backend that rebuilds inside
    :meth:`solve`.

    The physical model is not duplicated by it: the balance, exclusivity,
    big-M and objective are the same relations, written once per scenario.
    What is genuinely new is the curve variable and the constraint linking a
    scenario's dispatch to the band its price cleared in.
    """

    def __init__(
        self,
        params: BatteryParams,
        n_periods: int,
        n_scenarios: int,
        dt_h: float,
        *,
        solver: SolverConfig | None = None,
    ) -> None: ...

    def solve(self, scenario_prices: FloatArray, soc_initial: float) -> CurveSolution:
        """Choose the curve maximising expected profit over ``scenario_prices``.

        ``scenario_prices`` is ``(n_scenarios, n_periods)`` in €/MWh and may
        be negative; ``soc_initial`` is MWh at the start of the first period
        and is shared by every scenario, because the battery has one state and
        the uncertainty is about prices rather than about where it started.
        """
        ...
