"""The three reported numbers, computed by hand and compared.

``docs/DECISIONS.md`` §4. Every assertion here is against arithmetic done on
paper rather than against a previous run, so the file is a specification of
the metrics and not a snapshot of them.

The one to be strict about is equivalent cycles per year. It is the sanity
diagnostic that says whether ``c_deg`` is plausible at all — 700 cycles on a
2-hour battery means the degradation cost is too low and no other number on
the page is worth reading (§3.3) — so the metric that carries that warning
must itself be right.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from bess_arb.backtest.metrics import HOURS_PER_YEAR, Metrics, settle_profit, summarise
from bess_arb.backtest.runner import BacktestResult, DayResult
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryParams, SolveStatus
from bess_arb.timeline import Regime

HOURLY = Regime("hourly", 1.0)
BATTERY = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=17.0)
PROTOCOL = HorizonConfig(
    window_days=2, implement_days=1, warmup_days=1, soc_initial_fraction=0.5
)
TOL = 1e-9


def _day(
    index: int,
    *,
    warmup: bool = False,
    profit: float = 1000.0,
    discharged: float = 17.0,
    hours: float = 24.0,
    status: SolveStatus = SolveStatus.OPTIMAL,
) -> DayResult:
    return DayResult(
        day=dt.date(2024, 1, 1) + dt.timedelta(days=index),
        warmup=warmup,
        periods=int(hours),
        window_periods=int(hours) * 2,
        hours=hours,
        status=status,
        profit_eur=profit,
        believed_objective_eur=profit,
        charged_mwh=20.0,
        discharged_mwh=discharged,
        soc_start_mwh=10.0,
        soc_end_mwh=10.0,
    )


def _result(days: list[DayResult], policy: str = "oracle") -> BacktestResult:
    return BacktestResult(
        policy=policy,
        regime=HOURLY,
        params=BATTERY,
        protocol=PROTOCOL,
        days=tuple(days),
        window_lengths=(48,),
    )


def test_the_warm_up_days_are_not_in_any_total() -> None:
    """One discarded day, ten kept, and the totals must show ten."""
    days = [_day(0, warmup=True)] + [_day(n) for n in range(1, 11)]

    metrics = summarise(_result(days))

    assert metrics.days == 10
    assert metrics.profit_eur == pytest.approx(10_000.0, abs=TOL)
    assert metrics.hours == pytest.approx(240.0, abs=TOL)


def test_annualisation_uses_the_hours_the_days_actually_had() -> None:
    """Not 24 times the day count.

    Ten ordinary days and one 23-hour day: 263 hours, not 264. On a
    year-long run the two DST days cancel and the error hides; the arithmetic
    still has to be right, because the same code annualises a two-month run
    where they do not.
    """
    days = [_day(n) for n in range(10)] + [_day(10, hours=23.0)]

    metrics = summarise(_result(days))

    assert metrics.hours == pytest.approx(263.0, abs=TOL)
    assert metrics.years == pytest.approx(263.0 / HOURS_PER_YEAR, abs=TOL)


def test_the_headline_is_profit_per_mw_per_year() -> None:
    """€1,000 a day, 10 MW: €36,525/MW/year to the nearest euro.

    365.25 days a year and not 365 — the frozen window spans a leap year,
    and a quarter of a percent of avoidable error in the headline is a
    quarter of a percent too much.
    """
    days = [_day(n) for n in range(20)]

    metrics = summarise(_result(days))

    expected = 1000.0 * HOURS_PER_YEAR / 24.0 / 10.0
    assert metrics.profit_eur_per_mw_year == pytest.approx(expected, abs=1e-6)
    assert metrics.profit_eur_per_mw_year == pytest.approx(36_525.0, abs=0.5)


def test_equivalent_cycles_is_throughput_over_usable_energy() -> None:
    """A full 20 MWh discharged is one cycle; 365 of them is one a day."""
    days = [_day(n, discharged=20.0) for n in range(20)]

    metrics = summarise(_result(days))

    assert metrics.equivalent_cycles == pytest.approx(20.0, abs=TOL)
    assert metrics.equivalent_cycles_per_year == pytest.approx(365.25, abs=1e-6)


def test_the_bound_fraction_is_a_ratio_of_settled_profit() -> None:
    floor = summarise(_result([_day(n, profit=800.0) for n in range(10)], "floor"))
    oracle = summarise(_result([_day(n, profit=1000.0) for n in range(10)]))

    assert floor.with_bound(oracle).fraction_of_bound == pytest.approx(0.8, abs=TOL)
    assert oracle.with_bound(oracle).fraction_of_bound == pytest.approx(1.0, abs=TOL)


def test_a_bound_that_earns_nothing_makes_the_ratio_undefined() -> None:
    """Not zero, and not an exception.

    Above the market's spread the correct dispatch is to stay idle and every
    policy scores zero. "0% of the bound" would read as a failure where the
    truthful answer is that the comparison has no content.
    """
    idle = summarise(_result([_day(n, profit=0.0, discharged=0.0) for n in range(10)]))
    floor = summarise(
        _result([_day(n, profit=0.0, discharged=0.0) for n in range(10)], "floor")
    )

    assert floor.with_bound(idle).fraction_of_bound is None


def test_a_gap_limited_window_is_counted_and_reported() -> None:
    """A run with unproven windows must say so rather than average it away."""
    days = [_day(n) for n in range(8)] + [
        _day(8, status=SolveStatus.FEASIBLE),
        _day(9, status=SolveStatus.FEASIBLE),
    ]

    assert summarise(_result(days)).non_optimal_windows == 2


def test_summarising_a_run_with_nothing_left_after_the_warm_up_fails() -> None:
    """Better than reporting a metric over zero days."""
    with pytest.raises(ValueError, match="warm-up"):
        summarise(_result([_day(0, warmup=True)]))


def test_the_reported_dictionary_always_carries_cycles_per_year() -> None:
    """§3.3 says 'on every run without exception', so there is no path that omits it."""
    metrics = summarise(_result([_day(n) for n in range(10)]))

    assert "equivalent_cycles_per_year" in metrics.as_dict()
    assert isinstance(metrics, Metrics)


@pytest.mark.parametrize(
    ("c_deg", "tariff", "expected"),
    [
        # 20 MWh in at €10, 17 MWh out at €100: the golden case.
        (0.0, 0.0, 1500.0),
        # €17/MWh on 17 MWh discharged.
        (17.0, 0.0, 1500.0 - 289.0),
        # €5/MWh on 20 MWh charged.
        (0.0, 5.0, 1500.0 - 100.0),
    ],
)
def test_settlement_charges_degradation_on_discharge_and_tariff_on_charge(
    c_deg: float, tariff: float, expected: float
) -> None:
    """The asymmetry of §3.1, stated as three numbers.

    Degradation sits on discharged throughput only — charging throughput
    would double-count the same ageing — while the network tariff sits on
    charged energy, which is where a tariff would actually be levied.
    """
    params = BatteryParams(
        p_max_mw=10.0,
        e_max_mwh=20.0,
        eta_rt=0.85,
        c_deg_eur_mwh=c_deg,
        charge_tariff_eur_mwh=tariff,
    )
    p_c = np.array([10.0, 10.0, 0.0, 0.0])
    p_d = np.array([0.0, 0.0, 8.5, 8.5])
    prices = np.array([10.0, 10.0, 100.0, 100.0])

    assert settle_profit(p_c, p_d, prices, 1.0, params) == pytest.approx(
        expected, abs=1e-9
    )


def test_settlement_carries_delta_t_explicitly() -> None:
    """The same four hours at Δt = 0.25 must settle to the same money.

    Sixteen quarter-hourly periods spanning the golden case's four hours:
    the energy is identical, so the profit is identical. Invariant 3 applied
    to the accounting rather than to the model.
    """
    p_c = np.concatenate([np.full(8, 10.0), np.zeros(8)])
    p_d = np.concatenate([np.zeros(8), np.full(8, 8.5)])
    prices = np.concatenate([np.full(8, 10.0), np.full(8, 100.0)])

    assert settle_profit(p_c, p_d, prices, 0.25, BATTERY) == pytest.approx(
        settle_profit(
            np.array([10.0, 10.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 8.5, 8.5]),
            np.array([10.0, 10.0, 100.0, 100.0]),
            1.0,
            BATTERY,
        ),
        abs=1e-9,
    )
