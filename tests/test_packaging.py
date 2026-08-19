"""Slice 0 gate: the src-layout package is importable and self-consistent.

Trivial on its face, but it is what proves the ``src/`` layout, the hatchling
build target and the editable install actually agree with each other. If this
fails, nothing downstream is worth debugging.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import bess_arb

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_package_imports_and_exposes_a_version() -> None:
    assert bess_arb.__version__


def test_version_matches_pyproject() -> None:
    """Catches the classic drift between packaging metadata and the module."""
    with PYPROJECT.open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["version"] == bess_arb.__version__


def test_package_is_installed_from_src_layout() -> None:
    """A src-layout install must resolve to src/bess_arb, not the repo root.

    Guards the reason for choosing this layout: a module named ``pyomo.py``
    inside ``model/`` is only safe while the package is imported as a package
    and never from a directory that sits on ``sys.path``.
    """
    module_file = Path(bess_arb.__file__ or "")

    assert module_file.parent.name == "bess_arb"
    assert module_file.parent.parent.name == "src"
