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
