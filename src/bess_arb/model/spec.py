"""Solver-free description of the battery, the solver settings and a result.

This module imports nothing solver-related, and neither does
:mod:`bess_arb.model.base`. Together they are the whole of the modelling
layer that the rest of the package — and every backend — is allowed to see.
A raw solver status or a Pyomo/PyOptInterface object must never appear here.
See ``docs/DECISIONS.md`` §8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "BatteryParams",
    "CurveSolution",
    "FloatArray",
    "Solution",
    "SolveStatus",
    "SolverConfig",
]


class SolveStatus(Enum):
    """Project-owned normalisation of solver termination.

    Every backend maps its solver's status codes onto this enum. A raw
    solver enum escaping ``model/`` would reintroduce the coupling the
    backend abstraction exists to prevent, through a side door.
    """

    OPTIMAL = "optimal"
    """Convergence criteria satisfied — optimal within the configured gap."""

    FEASIBLE = "feasible"
    """A solution exists but the gap was not closed (time or node limit)."""

    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"

    @property
    def has_solution(self) -> bool:
        return self in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)


@dataclass(frozen=True, slots=True)
class BatteryParams:
    """Physical and economic parameters of the asset.

    Round-trip efficiency is stored, not the one-way efficiencies: they are
    derived as ``sqrt(eta_rt)`` so the symmetric split cannot drift out of
    step with the round-trip figure it is supposed to decompose.
    """

    p_max_mw: float
    e_max_mwh: float
    eta_rt: float
    c_deg_eur_mwh: float
    charge_tariff_eur_mwh: float = 0.0

    def __post_init__(self) -> None:
        if not self.p_max_mw > 0.0:
            raise ValueError(f"p_max_mw must be positive, got {self.p_max_mw}")
        if not self.e_max_mwh > 0.0:
            raise ValueError(f"e_max_mwh must be positive, got {self.e_max_mwh}")
        if not 0.0 < self.eta_rt <= 1.0:
            raise ValueError(f"eta_rt must lie in (0, 1], got {self.eta_rt}")
        if self.c_deg_eur_mwh < 0.0:
            raise ValueError(
                f"c_deg_eur_mwh must be non-negative, got {self.c_deg_eur_mwh}"
            )
        if self.charge_tariff_eur_mwh < 0.0:
            raise ValueError(
                "charge_tariff_eur_mwh must be non-negative, got "
                f"{self.charge_tariff_eur_mwh}"
            )

    @property
    def eta_c(self) -> float:
        """One-way charging efficiency, ``sqrt(eta_rt)``."""
        return math.sqrt(self.eta_rt)

    @property
    def eta_d(self) -> float:
        """One-way discharging efficiency, ``sqrt(eta_rt)``."""
        return math.sqrt(self.eta_rt)

    @property
    def duration_h(self) -> float:
        """Hours at rated power to traverse the usable energy window."""
        return self.e_max_mwh / self.p_max_mw


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """How to solve, in terms no backend-specific type appears in.

    ``name`` is passed through to whichever factory the backend uses. The
    gap and limit are held here rather than in the backend so switching
    solver never requires a code change (CLAUDE.md invariant 7).
    """

    name: str = "highs"
    mip_gap: float | None = 0.0
    time_limit_s: float | None = None
    threads: int | None = 1
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("solver name must not be empty")
        if self.mip_gap is not None and self.mip_gap < 0.0:
            raise ValueError(f"mip_gap must be non-negative, got {self.mip_gap}")
        if self.time_limit_s is not None and self.time_limit_s <= 0.0:
            raise ValueError(f"time_limit_s must be positive, got {self.time_limit_s}")
        if self.threads is not None and self.threads < 1:
            raise ValueError(f"threads must be at least 1, got {self.threads}")


@dataclass(frozen=True, slots=True)
class Solution:
    """One window's dispatch, as returned by any backend.

    ``objective`` is the solver's own objective value, not a recomputation
    from the dispatch vectors. The two agreeing is a test, not an
    assumption — see ``tests/test_model_golden.py``.

    ``objective_bound`` is the dual bound the solver finished with. A
    :attr:`SolveStatus.FEASIBLE` result is uninterpretable without it — the
    difference between "an incumbent" and "an incumbent within 0.1% of
    optimal" is the whole content of a gap-limited run, and the annual-window
    bound of ``docs/DECISIONS.md`` §2.4 is reported as exactly that. The
    provenance fields carry which solver, at which version, produced the
    number; both are plain strings, so nothing solver-typed escapes.
    """

    status: SolveStatus
    objective: float
    p_c_mw: FloatArray
    p_d_mw: FloatArray
    soc_mwh: FloatArray
    dt_h: float
    objective_bound: float | None = None
    solver_name: str | None = None
    solver_version: str | None = None

    def __post_init__(self) -> None:
        n = self.p_c_mw.shape[0]
        if self.p_d_mw.shape[0] != n or self.soc_mwh.shape[0] != n:
            raise ValueError(
                "p_c_mw, p_d_mw and soc_mwh must have equal length, got "
                f"{n}, {self.p_d_mw.shape[0]}, {self.soc_mwh.shape[0]}"
            )
        if not self.dt_h > 0.0:
            raise ValueError(f"dt_h must be positive, got {self.dt_h}")

    @property
    def n_periods(self) -> int:
        return int(self.p_c_mw.shape[0])

    @property
    def relative_gap(self) -> float | None:
        """``|bound − incumbent| / |incumbent|``, or ``None`` if unreported.

        The usual MIP relative gap. Guarded on the denominator because a
        window in which the battery correctly stays idle has objective
        exactly zero, and a division there would turn a closed gap into a
        NaN reported as a failure to converge.
        """
        if self.objective_bound is None or not math.isfinite(self.objective_bound):
            return None
        return abs(self.objective_bound - self.objective) / max(
            abs(self.objective), 1e-10
        )

    @property
    def charged_mwh(self) -> float:
        """Energy drawn from the grid — metered at the connection point."""
        return float(self.p_c_mw.sum() * self.dt_h)

    @property
    def discharged_mwh(self) -> float:
        """Energy delivered to the grid — metered at the connection point."""
        return float(self.p_d_mw.sum() * self.dt_h)

    def equivalent_cycles(self, e_max_mwh: float) -> float:
        """Discharged throughput expressed in full equivalent cycles.

        The sanity diagnostic that has to accompany every economic result:
        a 2-hour battery showing hundreds of cycles a year means ``c_deg``
        is too low and no other number on the page is worth reading.
        """
        if not e_max_mwh > 0.0:
            raise ValueError(f"e_max_mwh must be positive, got {e_max_mwh}")
        return self.discharged_mwh / e_max_mwh


@dataclass(frozen=True, slots=True)
class CurveSolution:
    """A bid curve chosen by optimisation rather than by construction.

    v3 stage 2's return type. It is deliberately *not* a
    :class:`Solution`: that object is one window's dispatch, and this one is
    a first-stage decision that has no single dispatch attached to it -- the
    dispatch is the recourse, and there is one per scenario. Returning a
    Solution here would invite exactly the confusion the type exists to
    prevent, namely reading the optimiser's expected objective as though it
    were a settled profit (``docs/DECISIONS.md`` section 4.1: a policy is
    scored on what its schedule earned at realised prices, never on the
    objective it was shown).

    ``band_prices`` is ``(n_periods, n_bands)`` ascending and ``quantities``
    the same shape, non-decreasing along axis 1 -- the two arrays
    :class:`bess_arb.bid.curve.BidCurves` is built from, so the optimised
    curve clears through the identical code path as a constructed one.
    """

    status: SolveStatus
    objective: float
    """Expected profit over the scenario set, at the *believed* prices.

    Not a profit. The scenarios are a belief; what the curve earns is what it
    earns once cleared against the prices that actually settled.
    """

    band_prices: FloatArray
    quantities: FloatArray
    dt_h: float
    objective_bound: float | None = None
    solver_name: str | None = None
    solver_version: str | None = None

    def __post_init__(self) -> None:
        if self.band_prices.shape != self.quantities.shape:
            raise ValueError(
                f"band_prices {self.band_prices.shape} and quantities "
                f"{self.quantities.shape} must have the same shape"
            )
        if self.band_prices.ndim != 2:
            raise ValueError(
                f"expected (n_periods, n_bands), got {self.band_prices.shape}"
            )
        if not self.dt_h > 0.0:
            raise ValueError(f"dt_h must be positive, got {self.dt_h}")

    @property
    def n_periods(self) -> int:
        return int(self.band_prices.shape[0])

    @property
    def n_bands(self) -> int:
        return int(self.band_prices.shape[1])
