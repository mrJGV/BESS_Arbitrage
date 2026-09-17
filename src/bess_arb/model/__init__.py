"""The modelling layer: the MILP, behind a pluggable backend.

This is the only subpackage permitted to import a solver library. The rule
is enforced twice — an ``import-linter`` contract in ``pyproject.toml`` and
``tests/test_import_boundary.py`` — so it survives a refactor of either
mechanism.

:func:`get_backend` resolves a backend by the name in ``config/params.yaml``
and imports it lazily, by string. That is deliberate: importing this package
must not drag Pyomo (or, later, PyOptInterface) into a process that only
wanted :class:`~bess_arb.model.spec.BatteryParams`.
"""

from __future__ import annotations

import importlib

from bess_arb.model.base import BatteryMILP, BidCurveMILP, SolveError
from bess_arb.model.spec import (
    BatteryParams,
    CurveSolution,
    FloatArray,
    Solution,
    SolverConfig,
    SolveStatus,
)

__all__ = [
    "BatteryMILP",
    "BatteryParams",
    "BidCurveMILP",
    "CurveSolution",
    "FloatArray",
    "Solution",
    "SolveError",
    "SolveStatus",
    "SolverConfig",
    "available_backends",
    "get_backend",
    "get_curve_backend",
]

# name -> (module, class). PyOptInterface joins this table when the port
# happens; nothing else changes, which is the point of the abstraction.
_BACKENDS: dict[str, tuple[str, str]] = {
    "pyomo": ("bess_arb.model.pyomo", "PyomoBatteryMILP"),
}

# v3 stage 2's optimiser, keyed by the same backend name. Held in a second
# table rather than as a second entry in the first: it is a different object
# answering a different question -- the window MILP returns a dispatch, this
# returns a bid curve -- and a single table would make `get_backend` ambiguous
# about which one a name resolves to. Not every backend need implement it.
_CURVE_BACKENDS: dict[str, tuple[str, str]] = {
    "pyomo": ("bess_arb.model.curve_pyomo", "PyomoBidCurveMILP"),
}


def available_backends() -> tuple[str, ...]:
    """Registered backend names, sorted. The equivalence test iterates this."""
    return tuple(sorted(_BACKENDS))


def get_backend(name: str) -> type[BatteryMILP]:
    """Resolve a backend name to its class, importing it on demand."""
    try:
        module_name, class_name = _BACKENDS[name]
    except KeyError:
        known = ", ".join(available_backends())
        raise ValueError(f"unknown backend {name!r}; known backends: {known}") from None

    module = importlib.import_module(module_name)
    backend: type[BatteryMILP] = getattr(module, class_name)
    return backend


def get_curve_backend(name: str) -> type[BidCurveMILP]:
    """Resolve a bid-curve optimiser by backend name, importing on demand.

    Separate from :func:`get_backend` because the two answer different
    questions and a backend may implement one without the other. The
    resolution and the lazy import are identical, so a backend that gains a
    curve optimiser joins one table and changes nothing else.
    """
    try:
        module_name, class_name = _CURVE_BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(_CURVE_BACKENDS))
        raise ValueError(
            f"backend {name!r} has no bid-curve optimiser; backends that do: {known}"
        ) from None

    module = importlib.import_module(module_name)
    backend: type[BidCurveMILP] = getattr(module, class_name)
    return backend
