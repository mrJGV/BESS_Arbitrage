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

from bess_arb.model.base import BatteryMILP, SolveError
from bess_arb.model.spec import (
    BatteryParams,
    FloatArray,
    Solution,
    SolverConfig,
    SolveStatus,
)

__all__ = [
    "BatteryMILP",
    "BatteryParams",
    "FloatArray",
    "Solution",
    "SolveError",
    "SolveStatus",
    "SolverConfig",
    "available_backends",
    "get_backend",
]

# name -> (module, class). PyOptInterface joins this table when the port
# happens; nothing else changes, which is the point of the abstraction.
_BACKENDS: dict[str, tuple[str, str]] = {
    "pyomo": ("bess_arb.model.pyomo", "PyomoBatteryMILP"),
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
