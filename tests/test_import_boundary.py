"""No solver library may be imported outside ``bess_arb.model``.

Enforced here *and* by the ``import-linter`` contracts in
``pyproject.toml``. Two mechanisms for one rule is deliberate: each survives
a careless refactor of the other, and the claim "swap one file to change
solver" is only true while the rule holds.

The scan is static — modules are parsed, never executed — plus one dynamic
check that importing the package's non-modelling half really does leave the
solver out of ``sys.modules``. A positive control asserts the detector can
still see an import it is supposed to catch; a boundary test that has
quietly stopped looking at anything is worse than no test.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import bess_arb

SOLVER_ROOTS = frozenset({"pyomo", "highspy", "pyoptinterface", "gurobipy"})

HEAVY_ROOTS = SOLVER_ROOTS | {"lightgbm"}
"""What must not be in ``sys.modules`` after importing the application half.

LightGBM is here for the same reason the solvers are: one module imports it,
everything else talks to an interface. The static contracts cannot check it —
the three modules that name it defer the import, and reading the source cannot
tell a deferred import from a live one — so this is the mechanism that does.
"""

PACKAGE_ROOT = Path(bess_arb.__file__ or "").resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "model"
FORECAST_ROOT = PACKAGE_ROOT / "forecast"

# Solver-free even inside model/: these are what every backend and every
# caller share, so a solver type reaching them would leak everywhere.
SOLVER_FREE_INSIDE_MODEL = ("spec.py", "base.py", "__init__.py")


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names imported by a module, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _modules_outside_model() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if MODEL_ROOT not in path.parents and path.parent != MODEL_ROOT
    )


def test_the_scan_looks_at_the_modules_it_claims_to() -> None:
    """The glob picks up new modules on its own; this pins that it really did.

    Named explicitly rather than counted, so a module that stops being
    scanned — moved, renamed, excluded by a stray path rule — fails here
    instead of quietly narrowing the boundary it is supposed to guard.
    """
    scanned = {path.name for path in _modules_outside_model()}

    assert {
        "__init__.py",
        "cli.py",
        "config.py",
        "timeline.py",
        "series.py",
        "esios.py",
        "omie.py",
        "floor.py",
        "oracle.py",
        "forecast.py",
        "features.py",
        "lgbm.py",
        "runner.py",
        "metrics.py",
        "bound.py",
        "compare.py",
        "figures.py",
    } <= scanned


def test_the_detector_catches_a_solver_import() -> None:
    """Positive control: the backend does import Pyomo, and the scan sees it."""
    roots = _imported_roots(MODEL_ROOT / "pyomo.py")

    assert "pyomo" in roots


@pytest.mark.parametrize(
    "path", _modules_outside_model(), ids=lambda path: str(path.name)
)
def test_no_solver_import_outside_the_model_package(path: Path) -> None:
    offending = _imported_roots(path) & SOLVER_ROOTS

    assert not offending, f"{path} imports {sorted(offending)}"


@pytest.mark.parametrize("name", SOLVER_FREE_INSIDE_MODEL)
def test_the_spec_and_the_protocol_are_solver_free(name: str) -> None:
    offending = _imported_roots(MODEL_ROOT / name) & SOLVER_ROOTS

    assert not offending, f"model/{name} imports {sorted(offending)}"


def test_no_lightgbm_import_outside_the_forecast_backend() -> None:
    """LightGBM lives in ``forecast/lgbm.py``, and nowhere else statically.

    The three modules that legitimately name it defer the import, so this
    checks the rest of the package the same way the solver scan does.
    ``test_importing_the_application_half_loads_nothing_heavy`` is what proves
    the deferred ones really are deferred.
    """
    deferred = {
        FORECAST_ROOT / "lgbm.py",
        FORECAST_ROOT / "__init__.py",
        PACKAGE_ROOT / "policy" / "__init__.py",
        PACKAGE_ROOT / "policy" / "forecast.py",
    }
    offenders = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
        if path not in deferred and "lightgbm" in _imported_roots(path)
    }

    assert not offenders, f"LightGBM is imported by {sorted(offenders)}"


def test_the_detector_catches_a_lightgbm_import() -> None:
    """Positive control for the check above."""
    assert "lightgbm" in _imported_roots(FORECAST_ROOT / "lgbm.py")


def test_importing_the_application_half_loads_nothing_heavy() -> None:
    """The dynamic half: a lazy import that fires anyway would pass the AST scan.

    ``get_backend`` resolves backends by string, and ``build_forecaster``
    imports its trainer inside the function body, precisely so that importing
    the config or the CLI drags in neither Pyomo nor LightGBM. This is the
    only check that can tell the difference between a deferred import and one
    that merely looks deferred — ``bess_arb.policy`` and ``bess_arb.forecast``
    are in the list because the static contracts cannot cover them.
    """
    program = (
        "import sys;"
        " import bess_arb, bess_arb.cli, bess_arb.config, bess_arb.model,"
        " bess_arb.timeline, bess_arb.series, bess_arb.data.esios,"
        " bess_arb.data.omie, bess_arb.policy, bess_arb.policy.forecast,"
        " bess_arb.forecast, bess_arb.forecast.features,"
        " bess_arb.backtest.runner,"
        " bess_arb.backtest.bound, bess_arb.backtest.metrics;"
        " loaded = sorted(m for m in sys.modules if m.split('.')[0] in"
        f" {sorted(HEAVY_ROOTS)});"
        " print(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]", result.stdout
