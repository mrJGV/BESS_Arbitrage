"""Settlement and the three reported numbers.

``docs/DECISIONS.md`` §4:

===========================  ==========================================
€/MW/year                    annualised net profit ÷ P_max — headline
% of bound                   policy profit ÷ oracle profit
Equivalent cycles/year       discharged throughput ÷ E_max ÷ years
===========================  ==========================================

The third is not decoration. A linear degradation cost is algebraically a
minimum-spread threshold, so ``c_deg`` is what sets cycles per year; if a
2-hour battery reports 700 of them, ``c_deg`` is wrong and no other number on
the page is worth reading (§3.3). It is therefore computed here rather than
left to a caller to remember, and :meth:`Metrics.as_dict` cannot emit a
result without it.

Settlement, and why it is separate from the objective
-----------------------------------------------------

A policy's schedule is *decided* against the prices the policy believed and
*settled* against the prices that cleared. For the oracle those are the same
vector and the settled profit of the implemented periods is a slice of the
solver's own objective. For every other policy they are not, and the
difference is precisely the cost of imperfect information — which is the
quantity this project exists to measure. So settlement is its own function,
takes realised prices explicitly, and is never read off the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bess_arb.model.spec import BatteryParams, FloatArray, SolveStatus

if TYPE_CHECKING:  # pragma: no cover - import cycle only exists for the checker
    from bess_arb.backtest.runner import BacktestResult

__all__ = ["HOURS_PER_YEAR", "Metrics", "settle_profit", "summarise"]

HOURS_PER_YEAR = 365.25 * 24.0
"""Hours in an average Gregorian year, ``365.25 x 24 = 8766``.

Not 8,760. The frozen window spans a leap year, and annualising a multi-year
backtest on 365-day years would overstate €/MW/year by a quarter of a percent
— small, but it is a free quarter of a percent of wrongness.
"""


def settle_profit(
    p_c_mw: FloatArray,
    p_d_mw: FloatArray,
    prices: FloatArray,
    dt_h: float,
    params: BatteryParams,
) -> float:
    """Net profit of a dispatch at realised prices.

    The same expression the objective maximises — revenue less energy cost,
    degradation on discharged throughput only, tariff on charged energy —
    with the realised price vector substituted for the believed one. Written
    out rather than reused from the model so that a change to the objective
    cannot silently change the accounting too.
    """
    revenue = float((p_d_mw * (prices - params.c_deg_eur_mwh)).sum()) * dt_h
    cost = float((p_c_mw * (prices + params.charge_tariff_eur_mwh)).sum()) * dt_h
    return revenue - cost


@dataclass(frozen=True, slots=True)
class Metrics:
    """One policy's run, summarised over the days that count."""

    policy: str
    c_deg_eur_mwh: float
    days: int
    hours: float
    years: float
    profit_eur: float
    profit_eur_per_mw_year: float
    charged_mwh: float
    discharged_mwh: float
    equivalent_cycles: float
    equivalent_cycles_per_year: float
    non_optimal_windows: int
    fraction_of_bound: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "c_deg_eur_mwh": self.c_deg_eur_mwh,
            "days": self.days,
            "hours": round(self.hours, 3),
            "years": round(self.years, 6),
            "profit_eur": round(self.profit_eur, 2),
            "profit_eur_per_mw_year": round(self.profit_eur_per_mw_year, 2),
            "charged_mwh": round(self.charged_mwh, 3),
            "discharged_mwh": round(self.discharged_mwh, 3),
            "equivalent_cycles": round(self.equivalent_cycles, 3),
            "equivalent_cycles_per_year": round(self.equivalent_cycles_per_year, 2),
            "non_optimal_windows": self.non_optimal_windows,
            "fraction_of_bound": (
                None
                if self.fraction_of_bound is None
                else round(self.fraction_of_bound, 6)
            ),
        }

    def with_bound(self, bound: Metrics) -> Metrics:
        """This run expressed as a fraction of a perfect-foresight run.

        Undefined rather than infinite when the bound earns nothing: at a
        degradation cost above the market's spread the correct dispatch is to
        stay idle, every policy scores zero, and "0% of the bound" would read
        as a failure where the truthful answer is that the ratio has no
        content.
        """
        fraction = (
            None if abs(bound.profit_eur) < 1e-9 else self.profit_eur / bound.profit_eur
        )
        return Metrics(
            policy=self.policy,
            c_deg_eur_mwh=self.c_deg_eur_mwh,
            days=self.days,
            hours=self.hours,
            years=self.years,
            profit_eur=self.profit_eur,
            profit_eur_per_mw_year=self.profit_eur_per_mw_year,
            charged_mwh=self.charged_mwh,
            discharged_mwh=self.discharged_mwh,
            equivalent_cycles=self.equivalent_cycles,
            equivalent_cycles_per_year=self.equivalent_cycles_per_year,
            non_optimal_windows=self.non_optimal_windows,
            fraction_of_bound=fraction,
        )


def summarise(result: BacktestResult) -> Metrics:
    """Aggregate the days that survive the warm-up.

    Hours come from the days themselves — each one's own period count times
    the regime's Δt — so a 23- or 25-hour day contributes what it actually
    was. Multiplying days by 24 here would put the DST error back in at the
    last step, after every other module took care to keep it out.
    """
    days = result.evaluated
    if not days:
        raise ValueError(
            f"{result.policy}: no days left after discarding "
            f"{result.protocol.warmup_days} warm-up days from {len(result.days)}"
        )

    hours = float(np.sum([day.hours for day in days]))
    years = hours / HOURS_PER_YEAR
    profit = float(np.sum([day.profit_eur for day in days]))
    charged = float(np.sum([day.charged_mwh for day in days]))
    discharged = float(np.sum([day.discharged_mwh for day in days]))
    cycles = discharged / result.params.e_max_mwh

    return Metrics(
        policy=result.policy,
        c_deg_eur_mwh=result.params.c_deg_eur_mwh,
        days=len(days),
        hours=hours,
        years=years,
        profit_eur=profit,
        profit_eur_per_mw_year=profit / years / result.params.p_max_mw,
        charged_mwh=charged,
        discharged_mwh=discharged,
        equivalent_cycles=cycles,
        equivalent_cycles_per_year=cycles / years,
        non_optimal_windows=sum(
            1 for day in days if day.status is not SolveStatus.OPTIMAL
        ),
    )
