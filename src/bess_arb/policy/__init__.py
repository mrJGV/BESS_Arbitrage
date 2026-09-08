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
from bess_arb.policy.forecast import Forecaster, ForecastPolicy
from bess_arb.policy.oracle import OraclePolicy

__all__ = [
    "POLICY_NAMES",
    "FloorPolicy",
    "ForecastPolicy",
    "Forecaster",
    "OraclePolicy",
    "PricePolicy",
    "QuantilePolicy",
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


@runtime_checkable
class QuantilePolicy(Protocol):
    """v2.5's capability: a price *belief* at a stated quantile, not just one.

    The v2.5 bid-curve extension needs K price vectors per window, one per
    quantile level, to trace a curve through the same optimiser K times — see
    :func:`bess_arb.bid.scenarios.solve_curves`.
    This is declared separately from :class:`PricePolicy` rather than folded
    into it: a policy that cannot answer at a quantile (nothing here requires
    :class:`ForecastPolicy` to implement it yet) is still a complete
    :class:`PricePolicy`, and the fixed-schedule v1/v2 code paths must keep
    working against policies that only ever satisfy the narrower Protocol.
    """

    def prices_for_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> FloatArray: ...


POLICY_NAMES: tuple[str, ...] = ("floor", "forecast", "oracle")
"""Registered policies, in the order they read as a ladder.

No information, a point forecast, perfect foresight. That ordering is the
chart, and the middle bar only means something because the outer two are
there: ``docs/DECISIONS.md`` §2.5 — without a floor the headline percentage is
unfalsifiable.
"""


def build_policy(
    name: str,
    prices: pd.Series,
    *,
    forecaster: Forecaster | None = None,
) -> PricePolicy:
    """Construct a policy by the name the CLI and the config use.

    Every policy is built from the same realised price series, which is not a
    contradiction: the oracle *reads* it for the window it is deciding, and
    the other two only ever read the part of it that predates the gate.

    ``forecaster`` is required for the forecast policy and refused for the
    others. It is not built here because it is expensive and regime-specific,
    and because one instance must be shared across a degradation sweep — see
    :func:`bess_arb.forecast.build_forecaster`.
    """
    if name == "oracle":
        return OraclePolicy(prices)
    if name == "floor":
        return FloorPolicy(prices)
    if name == "forecast":
        if forecaster is None:
            raise ValueError(
                "the forecast policy needs a fitted forecaster; build one with "
                "bess_arb.forecast.build_forecaster(config, regime) and pass it "
                "in. It is deliberately not built here: it is regime-specific "
                "and must be shared across a c_deg sweep."
            )
        return ForecastPolicy(forecaster, FloorPolicy(prices))
    known = ", ".join(POLICY_NAMES)
    raise ValueError(f"unknown policy {name!r}; known policies: {known}")
