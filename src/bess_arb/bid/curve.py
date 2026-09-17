"""Bid curves: a period's net position as a function of the clearing price.

``docs/DECISIONS.md`` §2.3 records the simplification this module removes. Up
to v2 a policy committed a *quantity* — charge 10 MW at 03:00 — and paid
whatever 03:00 cleared at, however far that was from the forecast. That is
equivalent to bidding at the market's price limits, and §2.3's worked example
shows what it costs: a forecast of €20 that clears at €300 buys the energy
anyway. A real participant submits a price-quantity curve and simply drops out
of the money.

The curve is therefore a free option against forecast error, and §2.3 spends a
page on why v1 and v2 throw it away deliberately: giving one policy limit
prices while the others keep fixed schedules would introduce a second
difference and make "% of bound" uninterpretable. v2.5 gives the option to
**every** policy, which is what keeps the ratio readable.

Signed net position, not two curves
-----------------------------------

The MILP's exclusivity binary makes at most one of ``p_c``, ``p_d`` nonzero in
any single solve, so ``q = p_d - p_c`` discards nothing. It buys a great deal:
a demand curve falls with price and a supply curve rises with it, and the net
curve turns both into one condition — **q is non-decreasing in p** — which is
also the form a clearing engine consumes. Negative q is energy bought,
positive q is energy sold.

Where the steps come from
-------------------------

One MILP solve per quantile level. For level tau the policy supplies a price
vector, the same optimiser returns a dispatch, and period t contributes the
pair ``(p_tau(t), q_tau(t))``. Sorting those K pairs by price gives the steps.
Nothing here re-solves or re-formulates: CLAUDE.md invariant 2 holds, the
policies still differ only in the prices they pass, and there are now K of
those vectors instead of one.

**What the tau-sweep is, and what it is not.** Every period sits at the same
quantile level simultaneously, so the family is comonotone: it moves the price
*level* much more than the day's *shape*. MD §5.1 already states that
independent marginal quantiles cannot stand in for a joint trajectory — that is
v3's subject. For curve construction the approximation is the usual one, but it
has a visible failure mode: if the shape never changes, dispatch never changes,
and every curve collapses to a single step. So :attr:`BidCurves.step_counts` is
reported with the result rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bess_arb.model.spec import FloatArray

__all__ = ["BidCurves", "build_curves"]

_TOLERANCE = 1e-9
"""Slack for the monotonicity check: solver output carries numerical dust."""


@dataclass(frozen=True, slots=True)
class BidCurves:
    """One window's curves: ``(n_periods, n_steps)`` prices and quantities.

    ``prices`` ascends along axis 1 and ``quantities`` is non-decreasing along
    it. Both invariants are established by :func:`build_curves` and checked
    here, because everything downstream — clearing, the feasibility repair, the
    settlement — is only correct on a monotone curve.
    """

    prices: FloatArray
    quantities: FloatArray

    def __post_init__(self) -> None:
        if self.prices.shape != self.quantities.shape:
            raise ValueError(
                f"prices {self.prices.shape} and quantities "
                f"{self.quantities.shape} must have the same shape"
            )
        if self.prices.ndim != 2 or self.prices.shape[1] < 1:
            raise ValueError(
                f"curves must be (n_periods, n_steps >= 1), got {self.prices.shape}"
            )
        if np.any(np.diff(self.prices, axis=1) < 0.0):
            raise ValueError("curve prices must ascend along axis 1")
        if np.any(np.diff(self.quantities, axis=1) < -_TOLERANCE):
            raise ValueError("curve quantities must be non-decreasing along axis 1")

    @property
    def n_periods(self) -> int:
        return int(self.prices.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.prices.shape[1])

    def clear(self, realised: FloatArray) -> FloatArray:
        """Net MW accepted at ``realised``, one price per period.

        The auction convention, and it is the one that makes the oracle's
        identity check exact: a bid at limit L is accepted when the clearing
        price *reaches* L, so the quantity taken is the one attached to the
        highest bid price not exceeding the clearing price. Below the lowest
        step the curve is clamped to its first quantity — the price came in
        cheaper than any scenario expected, and the most-buying step is the
        right answer; above the highest, to its last, for the mirror reason.
        """
        if realised.shape != (self.n_periods,):
            raise ValueError(
                f"expected {self.n_periods} realised prices, got {realised.shape[0]}"
            )
        # side="right" then step back one: at a price exactly equal to a step's
        # limit the step is in the money, which is the "<= L" of a demand bid.
        index = np.array(
            [
                np.searchsorted(self.prices[t], realised[t], side="right") - 1
                for t in range(self.n_periods)
            ]
        )
        rows = np.arange(self.n_periods)
        return np.asarray(self.quantities[rows, np.clip(index, 0, self.n_steps - 1)])

    @property
    def step_counts(self) -> FloatArray:
        """Distinct quantity levels per period — the degeneracy diagnostic.

        A curve of one step is a fixed schedule wearing a limit price, so the
        mean of this array says whether the comonotone sweep in the module
        docstring actually produced a curve or only re-derived v2.
        """
        return np.asarray(
            [len(np.unique(np.round(row, 9))) for row in self.quantities],
            dtype=np.float64,
        )


def build_curves(
    scenario_prices: FloatArray,
    p_c_mw: FloatArray,
    p_d_mw: FloatArray,
) -> BidCurves:
    """Assemble one curve per period from ``K`` scenario solves.

    All three arrays are ``(n_scenarios, n_periods)``: the price vector each
    scenario was solved on, and the charge and discharge it produced.

    **The rows need not arrive in any order, and both callers rely on that.**
    Each period's K pairs are sorted by price here before they are paired with
    quantities, so a comonotone tau-ladder (v2.5) and an unordered joint sample
    (v3) build the same object from the same code. The consequence is worth
    stating because it is also a trap: a family that *claims* to be ordered in
    tau and is not will be silently reinterpreted rather than rejected, which
    is why :func:`bess_arb.bid.scenarios.solve_curves` checks that claim before
    calling this, and why
    :func:`bess_arb.bid.scenarios.solve_scenarios` deliberately does not — a
    sample makes no such claim to violate.
    """
    if not (scenario_prices.shape == p_c_mw.shape == p_d_mw.shape):
        raise ValueError(
            "scenario_prices, p_c_mw and p_d_mw must have the same shape, got "
            f"{scenario_prices.shape}, {p_c_mw.shape}, {p_d_mw.shape}"
        )
    if scenario_prices.ndim != 2 or scenario_prices.shape[0] < 1:
        raise ValueError(
            f"expected (n_scenarios >= 1, n_periods), got {scenario_prices.shape}"
        )

    # Signed net position: negative is bought, positive is sold. See the module
    # docstring on why this is one curve rather than two.
    net = np.asarray(p_d_mw - p_c_mw, dtype=np.float64).T
    prices = np.asarray(scenario_prices, dtype=np.float64).T

    order = np.argsort(prices, axis=1, kind="stable")
    rows = np.arange(prices.shape[0])[:, None]
    prices = prices[rows, order]
    net = net[rows, order]

    quantities = np.empty_like(net)
    for t in range(net.shape[0]):
        quantities[t] = _monotonise(prices[t], net[t])

    return BidCurves(prices=prices, quantities=quantities)


def _monotonise(prices: FloatArray, quantities: FloatArray) -> FloatArray:
    """Force one period's quantities to be non-decreasing in price.

    ``prices`` arrives sorted ascending and ``quantities`` is the net position
    each scenario chose at that price. The two need not come out monotone: the
    scenarios were solved independently, and a dearer price vector changes the
    *shape* of the day as well as its level, so a dearer scenario occasionally
    buys more than a cheaper one. A curve that is not monotone is not a
    submittable bid, so something has to be decided here.

    The fit is least-squares isotonic regression: pool-adjacent-violators
    (PAVA), hand-rolled on ``numpy`` rather than pulled in from
    ``scikit-learn`` for one ten-line algorithm. Each point starts as its own
    block; whenever a block's value undercuts the block to its left, the two
    are merged into their weighted mean and the check repeats leftward. The
    result is the unique non-decreasing sequence minimising squared error
    against the raw quantities — symmetric between over- and under-buying,
    unlike clamping to a running max or min.
    """
    values = [float(v) for v in quantities]
    weights: list[float] = []
    counts: list[int] = []
    pooled: list[float] = []
    for value in values:
        v, w, c = value, 1.0, 1
        while pooled and pooled[-1] > v + 1e-12:
            pv = pooled.pop()
            pw = weights.pop()
            pc = counts.pop()
            w, v = pw + w, (pv * pw + v * w) / (pw + w)
            c += pc
        pooled.append(v)
        weights.append(w)
        counts.append(c)

    out = np.empty(len(values), dtype=np.float64)
    position = 0
    for level, count in zip(pooled, counts, strict=True):
        out[position : position + count] = level
        position += count
    return out
