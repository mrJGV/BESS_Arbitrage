"""Policies. A policy is a price vector and nothing else.

This is CLAUDE.md invariant 2 expressed as a package. Every policy hands the
*same* :class:`~bess_arb.model.base.BatteryMILP` instance a vector of prices
for the same window and receives a schedule back; none of them owns a
constraint, a horizon, a cadence or a terminal condition. That is what makes
"the policy captured 84% of the bound" a statement about information rather
than about two programs that happen to be in the same repository.

So the Protocol has one method, it returns an array, and there is nowhere to
put anything else. A policy that needed its own constraint set could not be
expressed here without changing this file, which is the point.

No solver is imported here: policies compute prices, the modelling layer
solves.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

import pandas as pd

from bess_arb.model.spec import FloatArray
from bess_arb.policy.floor import FloorPolicy
from bess_arb.policy.oracle import OraclePolicy

__all__ = [
    "POLICY_NAMES",
    "FloorPolicy",
    "OraclePolicy",
    "PricePolicy",
    "build_policy",
]


@runtime_checkable
class PricePolicy(Protocol):
    """What the backtest asks of a policy, in full.

    :meth:`prices_for` is handed the delivery day being decided and the UTC
    timestamps of the whole solved window — day D through the end of the
    horizon — and returns the prices the optimiser should believe. ``day`` is
    passed separately from ``window`` because it names the *decision*, and it
    is the decision that fixes the information gate: everything a policy may
    look at carries a timestamp before noon on D-1.
    """

    name: str

    def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray: ...


POLICY_NAMES: tuple[str, ...] = ("floor", "oracle")
"""Registered policies, in the order they read as a ladder.

``forecast`` joins in slice 4. Until it exists the pair is a floor and a
bound with nothing between them, which is a chart with a gap in it rather
than a result.
"""


def build_policy(name: str, prices: pd.Series) -> PricePolicy:
    """Construct a policy by the name the CLI and the config use.

    Both policies are built from the same realised price series, which is not
    a contradiction: the oracle *reads* it for the window it is deciding, the
    floor only ever reads the part of it that predates the gate.
    """
    if name == "oracle":
        return OraclePolicy(prices)
    if name == "floor":
        return FloorPolicy(prices)
    known = ", ".join(POLICY_NAMES)
    raise ValueError(f"unknown policy {name!r}; known policies: {known}")
