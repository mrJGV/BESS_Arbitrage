"""The backend registry: one name in the config, one class out.

Small, but it is what makes "swap one file to change solver" checkable
rather than aspirational. The equivalence test will iterate
:func:`available_backends` the day a second backend exists, so a backend
that is registered but does not satisfy the Protocol has to fail here first.
"""

from __future__ import annotations

import pytest

from bess_arb.config import load_config
from bess_arb.model import BatteryMILP, available_backends, get_backend


def test_the_configured_backend_is_registered() -> None:
    assert load_config().backend in available_backends()


def test_pyomo_is_the_v1_backend() -> None:
    """Pyomo first, PyOptInterface second — docs/DECISIONS.md §8."""
    assert "pyomo" in available_backends()


def test_an_unknown_backend_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="pyomo"):
        get_backend("gurobi_direct_from_a_typo")


def test_the_registered_backend_satisfies_the_protocol(backend_name: str) -> None:
    """Structural conformance, checked at runtime as well as by mypy."""
    backend = get_backend(backend_name)
    model = backend(load_config().battery, 4, 1.0)

    assert isinstance(model, BatteryMILP)
    assert callable(model.solve)
