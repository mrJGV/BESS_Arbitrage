"""The annual-window bound, on a window small enough to prove optimal.

``docs/DECISIONS.md`` §2.4 asks for one solve over a year with the state of
charge free across days, to separate the value of *horizon* from the value of
*information*. A year of it takes minutes to hours and belongs on a cluster
or in a deliberate local run, not in the test suite — so what is tested here
is the mechanism, on ten days, where the answer is provable.

The assertion that carries the weight is the dominance one. The rolling
oracle's own dispatch is a feasible trajectory for the free-horizon problem
over the same days, from the same opening state of charge. So the annual
objective **cannot** be lower, and if it ever is, one of the two is not
solving the problem it claims to. That is a proof rather than a tolerance,
which is rare enough in a backtest to be worth spending a test on.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.backtest.bound import AnnualBound, environment, solve_annual_bound
from bess_arb.backtest.metrics import summarise
from bess_arb.backtest.runner import run_backtest
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryParams, SolverConfig, SolveStatus
from bess_arb.policy import build_policy
from bess_arb.timeline import MARKET_TZ, Regime, utc_index

HOURLY = Regime("hourly", 1.0)
BATTERY = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=17.0)
PROTOCOL = HorizonConfig(
    window_days=2, implement_days=1, warmup_days=0, soc_initial_fraction=0.5
)
SOC_INITIAL = PROTOCOL.soc_initial_mwh(BATTERY.e_max_mwh)

FIRST = dt.date(2023, 5, 1)
LAST = dt.date(2023, 5, 10)


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    """A daily shape with a slow multi-day drift on top.

    The drift is what makes the free-horizon solve able to beat the rolling
    one at all: with an identical shape every day there is nothing to gain
    from carrying energy overnight, and the test would pass on a tie without
    exercising anything.
    """
    index = utc_index(FIRST, LAST + dt.timedelta(days=1), HOURLY)
    local = index.tz_convert(MARKET_TZ)
    hour = np.asarray(local.hour, dtype=float)
    day = np.arange(len(index), dtype=float) / 24.0
    return pd.Series(
        60.0 + 40.0 * np.sin((hour - 6.0) * np.pi / 12.0) + 20.0 * np.sin(day),
        index=index,
        name="price_eur_mwh",
    )


@pytest.fixture(scope="module")
def annual(prices: pd.Series) -> AnnualBound:
    return solve_annual_bound(
        prices,
        BATTERY,
        HOURLY,
        FIRST,
        LAST,
        soc_initial_mwh=SOC_INITIAL,
        solver=SolverConfig(),
    )


def _rolling_profit(prices: pd.Series) -> float:
    return summarise(
        run_backtest(
            prices,
            build_policy("oracle", prices),
            BATTERY,
            HOURLY,
            PROTOCOL,
            first_day=FIRST,
            last_day=LAST,
        )
    ).profit_eur


def test_the_window_is_every_period_of_every_day(annual: AnnualBound) -> None:
    """Ten days, 240 hours, and the variable count the formulation implies."""
    assert annual.periods == 240
    assert annual.variables == 960
    assert annual.binaries == 240
    assert annual.dt_h == 1.0


def test_a_window_this_size_is_proven_optimal(annual: AnnualBound) -> None:
    """Ten days closes; a year is what needs the tolerance."""
    assert annual.status is SolveStatus.OPTIMAL
    assert annual.proven_optimal
    assert annual.relative_gap == pytest.approx(0.0, abs=1e-6)


def test_the_provenance_needed_to_read_the_number_later_is_recorded(
    annual: AnnualBound,
) -> None:
    """A wall clock with no solver and no version next to it means nothing."""
    record = annual.as_dict()

    assert record["solver_name"]
    assert record["solver_version"]
    assert record["wall_clock_s"] > 0.0
    assert record["first_day"] == FIRST.isoformat()
    assert record["last_day"] == LAST.isoformat()


def test_the_free_horizon_cannot_do_worse_than_the_rolling_oracle(
    prices: pd.Series, annual: AnnualBound
) -> None:
    """The proof, not a tolerance.

    The rolling oracle's implemented dispatch over these days is itself a
    feasible solution to the annual problem — same battery, same opening SoC,
    no terminal condition on either. So the annual optimum is an upper bound
    on it by construction.
    """
    settled = _rolling_profit(prices)

    assert annual.objective_eur >= settled - 1e-6


def test_the_value_of_horizon_is_small_for_a_two_hour_battery(
    prices: pd.Series, annual: AnnualBound
) -> None:
    """§2.4's `[Likely]`, measured on a case built to favour the free horizon.

    Carrying energy overnight costs the intraday cycle, and for a two-hour
    asset the intraday cycle almost always pays more. Even with a multi-day
    drift deliberately added to the prices, the free horizon should gain a
    few percent and not a multiple.
    """
    settled = _rolling_profit(prices)

    assert 0.0 <= (annual.objective_eur - settled) / settled < 0.25


def test_the_soc_is_free_across_days_and_at_the_end(annual: AnnualBound) -> None:
    """No cyclic condition, no forced unwind — that is the whole difference."""
    assert 0.0 <= annual.soc_final_mwh <= BATTERY.e_max_mwh + 1e-6
    assert annual.soc_initial_mwh == pytest.approx(SOC_INITIAL)


def test_a_window_reaching_past_the_prices_is_refused(prices: pd.Series) -> None:
    """Silently solving a shorter year would be worse than failing."""
    with pytest.raises(KeyError, match="missing"):
        solve_annual_bound(
            prices,
            BATTERY,
            HOURLY,
            FIRST,
            LAST + dt.timedelta(days=30),
            soc_initial_mwh=SOC_INITIAL,
        )


def test_the_environment_record_identifies_the_machine() -> None:
    """So "42 minutes on HiGHS" can be compared with the cluster run."""
    record = environment()

    assert record["python"]
    assert record["platform"]
    assert record["cpu_count"]
