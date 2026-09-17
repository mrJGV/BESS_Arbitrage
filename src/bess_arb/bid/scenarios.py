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

__all__ = ["QUANTILE_LEVELS", "ScenarioSolves", "solve_curves", "solve_scenarios"]

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

    solutions: tuple[Solution, ...]
    source: str = "quantile"
    """Which construction produced the family — ``"quantile"`` or ``"joint"``.

    Recorded rather than inferred, because the two are not comparable numbers
    and a reader looking at a run's diagnostics must be able to see which one
    produced it. Same reasoning as ``BacktestResult.bidding``.
    """
    quantile_levels: tuple[float, ...] | None = None
    """The tau levels swept, or ``None`` for a joint sample, which has none.

    A joint scenario carries no level: it is one draw from the predictive
    law, not the belief at a stated probability. Leaving this ``None`` rather
    than filling it with indices is the type saying so.
    """


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
        solutions=tuple(solutions),
        source="quantile",
        quantile_levels=tuple(quantile_levels),
    )
    return curves, scenario_solves


def solve_scenarios(
    solve: Callable[[FloatArray], Solution],
    scenario_prices: FloatArray,
) -> tuple[BidCurves, ScenarioSolves]:
    """Assemble one window's bid curves from a **joint** scenario sample.

    v3's entry point, and deliberately a sibling of :func:`solve_curves`
    rather than a widening of it. The two differ in one place — what the S
    price vectors mean — and that difference is the whole of v3, so it is
    expressed as two functions rather than as a flag.

    ``scenario_prices`` is ``(S, n_periods)``: row *s* is one whole trajectory
    drawn from the policy's predictive law, internally coherent, carrying
    whatever cross-period dependence the errors actually have. It arrives
    already built (see :class:`~bess_arb.scenarios.BeliefResiduals`) rather
    than being pulled out of a policy one level at a time, because there is no
    ladder to walk: a sample has no ordering to iterate in.

    **No monotone-in-tau check, and its absence is the point.** That guard
    exists because a quantile family *claims* to be ordered in tau and
    :func:`~bess_arb.bid.curve.build_curves` would silently reinterpret one
    that is not. A sample makes no such claim — row 2 is not dearer than row 1,
    and a period where it is cheaper is the sample doing its job. Running the
    guard here would reject every correct input. What replaces it is the shape
    and finiteness check below: the failure this path can actually have is a
    malformed matrix, not a mis-ordered one.
    """
    scenario_prices = np.asarray(scenario_prices, dtype=np.float64)
    if scenario_prices.ndim != 2 or scenario_prices.shape[0] < 1:
        raise ValueError(
            "scenario_prices must be (n_scenarios, n_periods) with at least "
            f"one scenario, got shape {scenario_prices.shape}"
        )
    if not np.isfinite(scenario_prices).all():
        raise ValueError(
            "scenario_prices carries a non-finite value; a residual drawn from "
            "a window with an unpriced period would do this, and adding it to "
            "a belief gives the optimiser a NaN coefficient rather than an error"
        )

    n_levels, n_periods = scenario_prices.shape
    p_c_mw = np.empty((n_levels, n_periods), dtype=np.float64)
    p_d_mw = np.empty((n_levels, n_periods), dtype=np.float64)
    solutions: list[Solution] = []

    for i in range(n_levels):
        solution = solve(scenario_prices[i])
        p_c_mw[i] = solution.p_c_mw
        p_d_mw[i] = solution.p_d_mw
        solutions.append(solution)

    curves = build_curves(scenario_prices, p_c_mw, p_d_mw)
    return curves, ScenarioSolves(solutions=tuple(solutions), source="joint")


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
