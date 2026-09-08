"""What the battery can actually deliver once the curve has cleared.

:mod:`bess_arb.bid.curve` produces a net position per period by reading each
curve at the price that cleared. Nothing in that step consults the state of
charge, and it cannot: the curve was submitted before the prices were known,
and a curve that only ever promised what *every* outcome could support would
give up most of the option value the extension exists to measure.

So a cleared position can be physically impossible — an offer to discharge
10 MW accepted at 19:00 by a battery holding 6 MWh, because the charging leg
the plan depended on cleared out of the money that morning. This module is
where that is resolved.

The rule, and what it costs
---------------------------

**Clip forward to what the state of charge allows, trade nothing else, and
charge no penalty.** The undelivered energy is simply not sold. That is
optimistic about one thing — in a real market a cleared bid you fail to
deliver is an imbalance you pay for — and the alternative was rejected
deliberately: an imbalance price is a parameter with no citation behind it in
this project, and the whole result would then turn on its value. The
conservative alternative, capping every step at what the worst scenario could
support, was rejected on measurement: it gives up most of the option value on
the days nothing goes wrong.

The exposure is therefore **measured rather than assumed**. :attr:`Delivered`
carries the clipped MWh, and it is reported with every v2.5 run beside the
cycle count. If it comes back a fraction of a percent of throughput, the
choice is vindicated; if it does not, that number is the finding, and an
imbalance charge is one config parameter away.

Why one forward pass is enough
------------------------------

The repair is monotone, which is what makes it terminate without iterating:

- clipping a **discharge** reduces the energy leaving the store, so every
  later SoC is *higher* than it would have been — that can only relieve a
  later discharge, never create a new shortfall;
- clipping a **charge** reduces the energy entering it, so every later SoC is
  *lower* — that can only relieve a later overfill.

Neither clip is ever upward, so no repair can invalidate a period already
passed. A single sweep in time order therefore leaves every period feasible,
and a fixed-point loop would find nothing more on a second pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bess_arb.model.spec import BatteryParams, FloatArray

__all__ = ["Delivered", "deliver"]


@dataclass(frozen=True, slots=True)
class Delivered:
    """One window's physically deliverable dispatch, and what it cost to get.

    ``p_c_mw`` and ``p_d_mw`` are the settled quantities — what the battery
    bought and sold after the repair — in the same form every other layer of
    this project uses, so :func:`bess_arb.backtest.metrics.settle_profit`
    takes them unchanged.
    """

    p_c_mw: FloatArray
    p_d_mw: FloatArray
    soc_mwh: FloatArray
    clipped_charge_mwh: float
    """Energy the curve cleared to buy that the store had no room for."""

    clipped_discharge_mwh: float
    """Energy the curve cleared to sell that the store could not supply.

    The one that matters. A charge that cannot be taken is an opportunity
    missed; a *discharge* that cannot be delivered is, in a real market, an
    imbalance position — see the module docstring on why this project does not
    price it, and why this number is therefore reported rather than buried.
    """

    @property
    def clipped_mwh(self) -> float:
        """Total cleared-but-undelivered energy, both directions."""
        return self.clipped_charge_mwh + self.clipped_discharge_mwh


def deliver(
    net_mw: FloatArray,
    soc_initial_mwh: float,
    dt_h: float,
    params: BatteryParams,
) -> Delivered:
    """Make a cleared net position physically feasible, in one forward pass.

    ``net_mw`` is signed as :meth:`bess_arb.bid.curve.BidCurves.clear` returns
    it: negative is energy bought, positive is energy sold. The power limit is
    applied here too, so a curve built from a mis-specified scenario cannot
    smuggle through a position the connection point could not carry.
    """
    if net_mw.ndim != 1:
        raise ValueError(f"net_mw must be one-dimensional, got {net_mw.shape}")
    if not dt_h > 0.0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")
    if not -1e-9 <= soc_initial_mwh <= params.e_max_mwh + 1e-9:
        raise ValueError(
            f"soc_initial_mwh {soc_initial_mwh} lies outside [0, "
            f"{params.e_max_mwh}]"
        )

    n = int(net_mw.shape[0])
    p_c = np.zeros(n, dtype=np.float64)
    p_d = np.zeros(n, dtype=np.float64)
    soc = np.empty(n, dtype=np.float64)

    clipped_charge = 0.0
    clipped_discharge = 0.0
    state = float(np.clip(soc_initial_mwh, 0.0, params.e_max_mwh))

    for t in range(n):
        wanted = float(net_mw[t])
        # The connection point binds before the store does.
        wanted = float(np.clip(wanted, -params.p_max_mw, params.p_max_mw))

        if wanted < 0.0:
            # Buying. Headroom is in stored terms, so the grid-side limit is
            # the room divided by the charging efficiency: eta_c*p_c*dt fits.
            room_mwh = params.e_max_mwh - state
            feasible_mw = room_mwh / (params.eta_c * dt_h)
            taken = min(-wanted, max(feasible_mw, 0.0))
            clipped_charge += (-wanted - taken) * dt_h
            p_c[t] = taken
            state += params.eta_c * taken * dt_h
        elif wanted > 0.0:
            # Selling. The store must supply p_d*dt/eta_d, so the deliverable
            # grid-side power is what is held, times the discharge efficiency.
            feasible_mw = state * params.eta_d / dt_h
            given = min(wanted, max(feasible_mw, 0.0))
            clipped_discharge += (wanted - given) * dt_h
            p_d[t] = given
            state -= given * dt_h / params.eta_d

        # Numerical dust only: the arithmetic above cannot leave the window,
        # but repeated addition can put a few 1e-15 outside it.
        state = float(np.clip(state, 0.0, params.e_max_mwh))
        soc[t] = state

    return Delivered(
        p_c_mw=p_c,
        p_d_mw=p_d,
        soc_mwh=soc,
        clipped_charge_mwh=clipped_charge,
        clipped_discharge_mwh=clipped_discharge,
    )
