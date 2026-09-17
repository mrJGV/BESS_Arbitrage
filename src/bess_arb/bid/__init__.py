"""Bid curves: v2.5's option-value extension to the fixed-schedule policies.

``docs/DECISIONS.md`` §2.3 (the bidding simplification). Up to v2 a policy commits a
quantity and pays whatever the period clears at, however far that is from
what it believed. This package gives every policy a price-quantity curve
instead: :mod:`bess_arb.bid.curve` defines the curve object and how it
clears, :mod:`bess_arb.bid.scenarios` traces one from K scenario solves
through the same :class:`~bess_arb.model.base.BatteryMILP` every other policy
uses.

No solver is imported here. Scenario solving is done by a callable the caller
supplies — see :func:`bess_arb.bid.scenarios.solve_curves` — so this package
knows the model exists and nothing about how it is built or pooled.
"""

from __future__ import annotations

from bess_arb.bid.curve import BidCurves, build_curves
from bess_arb.bid.deliver import Delivered, deliver
from bess_arb.bid.scenarios import (
    QUANTILE_LEVELS,
    ScenarioSolves,
    solve_curves,
    solve_scenarios,
)

__all__ = [
    "QUANTILE_LEVELS",
    "BidCurves",
    "Delivered",
    "ScenarioSolves",
    "build_curves",
    "deliver",
    "solve_curves",
    "solve_scenarios",
]
