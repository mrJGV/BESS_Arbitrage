"""Tracing one window's bid curves from K scenario solves.

:mod:`bess_arb.bid.curve` describes what a curve is and how it clears. This
module is the other half: turning a policy that can answer "what is your
tau-quantile belief for this window" into the K price vectors, K solves and
one :class:`~bess_arb.bid.curve.BidCurves` per window that :func:`build_curves`
needs.

``solve`` is injected rather than built here
---------------------------------------------

CLAUDE.md invariant 2 says every policy obtains its schedule from the same
pooled :class:`~bess_arb.model.base.BatteryMILP`, and the module pool that
guarantees the model is built once per window length lives in
:mod:`bess_arb.backtest.runner`, alongside the day's running SoC. This module
has no business knowing about either: it is handed a plain
``Callable[[FloatArray], Solution]`` that already closes over the pooled
model instance and today's ``soc_initial``, and it calls it exactly K times —
once per quantile level, in ascending order. What differs between one call and
the next is only the price vector, which is the same discipline every other
policy in this project already follows.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bess_arb.bid.curve import BidCurves, build_curves
from bess_arb.model.spec import FloatArray, Solution
from bess_arb.policy import QuantilePolicy

__all__ = ["QUANTILE_LEVELS", "ScenarioSolves", "solve_curves"]

QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
"""Equally spaced, the median included, the tails excluded — deliberately.

MD §7's "deliberately boring" applies here too: five levels, symmetric around
the median, no defended constant beyond that. The median matters because most
realised prices land near the centre of the distribution, where the curve
needs its resolution; the tails are cut at 0.1/0.9 rather than carried further
out because the outer steps are clamps (any realised price beyond the last
step just takes that step's quantity, see
:meth:`bess_arb.bid.curve.BidCurves.clear`), so a more extreme quantile only
earns its solve if it changes the clamp — and the most extreme quantiles are
also the ones a quantile model fits worst, being pinned by the fewest
effective observations.
"""

_TAU_TOLERANCE = 1e-9
"""Slack on the monotone-in-tau check: numpy's interpolation carries dust."""


@dataclass(frozen=True, slots=True)
class ScenarioSolves:
    """The K solves behind one window's curve, kept for diagnostics.

    Nothing downstream of :func:`solve_curves` needs these — the curve is the
    product — but a solve's status and objective are cheap to keep and are
    exactly what a reader would ask for first if a curve looked wrong: which
    level failed to solve, and at what believed value.
    """

    quantile_levels: tuple[float, ...]
    solutions: tuple[Solution, ...]


def solve_curves(
    solve: Callable[[FloatArray], Solution],
    policy: QuantilePolicy,
    day: dt.date,
    window: pd.DatetimeIndex,
    *,
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> tuple[BidCurves, ScenarioSolves]:
    """Solve ``window`` at each quantile level and assemble its bid curves.

    ``solve`` is called once per level, in ascending order — see the module
    docstring for why it is a bare callable rather than a pool this module
    reaches into itself.
    """
    if not quantile_levels:
        raise ValueError("quantile_levels must not be empty")
    if list(quantile_levels) != sorted(quantile_levels):
        raise ValueError(f"quantile_levels must ascend, got {quantile_levels}")

    n_periods = len(window)
    n_levels = len(quantile_levels)
    scenario_prices = np.empty((n_levels, n_periods), dtype=np.float64)
    p_c_mw = np.empty((n_levels, n_periods), dtype=np.float64)
    p_d_mw = np.empty((n_levels, n_periods), dtype=np.float64)
    solutions: list[Solution] = []

    for i, tau in enumerate(quantile_levels):
        prices = policy.prices_for_quantile(day, window, tau)
        if prices.shape != (n_periods,):
            raise ValueError(
                f"quantile {tau}: expected {n_periods} prices, got {prices.shape[0]}"
            )
        solution = solve(prices)
        scenario_prices[i] = prices
        p_c_mw[i] = solution.p_c_mw
        p_d_mw[i] = solution.p_d_mw
        solutions.append(solution)

    _check_monotone_in_tau(scenario_prices, quantile_levels)
    curves = build_curves(scenario_prices, p_c_mw, p_d_mw)
    scenario_solves = ScenarioSolves(
        quantile_levels=tuple(quantile_levels), solutions=tuple(solutions)
    )
    return curves, scenario_solves


def _check_monotone_in_tau(
    scenario_prices: FloatArray, quantile_levels: tuple[float, ...]
) -> None:
    """Refuse a scenario family whose prices fall as tau rises.

    A quantile forecast is non-decreasing in tau by definition, and nothing
    downstream re-checks it: :func:`build_curves` sorts each period's K pairs
    by price before pairing them with quantities, so a family that arrives out
    of order is silently reinterpreted rather than rejected. The curve still
    builds, still clears, still settles, and is no longer the object it claims
    to be.

    This is not hypothetical. A quantile source whose *fallback threshold*
    varied with tau — the floor policy's, briefly — let different taus read
    different rungs of its climatology ladder, and a coarse bucket's 90th
    percentile came back below a fine bucket's 70th. It cost nothing at any
    layer that could see it, which is exactly why the check belongs here,
    where the family is assembled and the tau levels are still known.
    """
    falling = np.diff(scenario_prices, axis=0) < -_TAU_TOLERANCE
    if not falling.any():
        return
    scenario, period = (int(i) for i in np.argwhere(falling)[0])
    raise ValueError(
        f"scenario prices must not fall as tau rises: at period {period}, "
        f"tau={quantile_levels[scenario]} gives "
        f"{scenario_prices[scenario, period]:.4f} but "
        f"tau={quantile_levels[scenario + 1]} gives "
        f"{scenario_prices[scenario + 1, period]:.4f}. A quantile family that "
        "is not monotone in tau is not a quantile family; check whether the "
        "source lets different taus read different fallback levels."
    )
