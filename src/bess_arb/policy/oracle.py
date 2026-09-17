"""The perfect-foresight policy: realised prices, and nothing else changed.

This is the denominator of the headline ratio, so what matters about it is
what it does *not* do. It does not get a longer horizon, a different cadence,
a relaxed constraint set or a terminal value function; it gets the same
:class:`~bess_arb.model.base.BatteryMILP` instance as the floor, on the same
window, with the same initial SoC. The single difference is that the vector
it returns is the prices that actually cleared.

``docs/DECISIONS.md`` §2.4: *"Same optimiser, same constraints, same 48-hour
window, same cadence, same SoC handling. Sole difference: realised prices
instead of forecast prices."* Implemented as eleven lines, which is the
strongest form that guarantee can take.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from bess_arb.model.spec import FloatArray

__all__ = ["OraclePolicy"]


class OraclePolicy:
    """Returns the realised price of every period in the window."""

    name = "oracle"

    def __init__(self, prices: pd.Series) -> None:
        self._prices = prices

    def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray:
        """Realised prices over ``window``.

        ``day`` is unused — perfect foresight is the one policy for which the
        gate is irrelevant — but it stays in the signature because the
        Protocol is what makes the policies interchangeable, and an oracle
        with a narrower interface would be an oracle the backtest could treat
        differently.
        """
        values = self._prices.reindex(window)
        if values.isna().any():
            missing = int(values.isna().sum())
            raise KeyError(
                f"{self.name}: {missing} of {len(window)} periods in the window "
                f"for delivery day {day} are not in the price series"
            )
        return np.asarray(values.to_numpy(dtype=np.float64))

    def prices_for_quantile(
        self, day: dt.date, window: pd.DatetimeIndex, tau: float
    ) -> FloatArray:
        """Realised prices, regardless of ``tau``.

        The oracle's bid curve is degenerate by construction: perfect
        foresight means there is no uncertainty for a quantile to describe,
        so every level believes the same thing :meth:`prices_for` already
        does. A caller building a curve for this policy needs only one
        scenario, not the full sweep — solving K identical windows would
        waste K-1 solves on a curve that was always going to collapse to one
        step. That single-step curve is also a free correctness check:
        clearing it against the realised prices must reproduce the
        fixed-schedule oracle exactly, because the curve was never anything
        but that schedule wearing one limit price.
        """
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must lie strictly between 0 and 1, got {tau}")
        return self.prices_for(day, window)

    def price_scenarios(
        self, day: dt.date, window: pd.DatetimeIndex, n_scenarios: int
    ) -> FloatArray:
        """One scenario: the realised prices, whatever ``n_scenarios`` asks for.

        v3's joint scenario source, degenerate for the same reason
        :meth:`prices_for_quantile` is. A joint distribution describes what a
        policy does not know, and this one knows everything; its predictive
        law is a point mass, so a sample of any size holds one distinct
        trajectory. Returning a single row rather than ``n_scenarios`` copies
        is not an optimisation of a special case — it is the statement that
        the special case *is* a point mass, and it keeps the oracle's curve a
        single step under v3 exactly as under v2.5.

        That makes the oracle the regression test for both extensions: under
        ``bidding="curve"`` and ``bidding="joint"`` alike, clearing this curve
        at realised prices must reproduce the fixed-schedule oracle to the
        cent. If it ever stops doing so, the clearing path is wrong, not the
        model.
        """
        if n_scenarios < 1:
            raise ValueError(f"n_scenarios must be at least 1, got {n_scenarios}")
        return self.prices_for(day, window)[None, :]
