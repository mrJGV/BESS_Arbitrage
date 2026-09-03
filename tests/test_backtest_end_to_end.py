"""The backtest against the frozen snapshot, not against synthetic prices.

What has to hold once the loop is wired up: it runs end to end on the hourly
regime, the oracle beats the floor, and equivalent cycles per year land in a
physically sane band for a 2-hour battery at ``c_deg = 17``. That last one is
``docs/DECISIONS.md`` §3.3's alarm — if it prints around 700, ``c_deg`` is
wrong and nothing else matters — so it is asserted rather than eyeballed.

The window is deliberately one that contains negative prices, so the
integrality that ``docs/DECISIONS.md`` §3.2 argues for is actually load-
bearing in this run rather than argued about in the abstract.

Bounds here are wide on purpose. A test that pinned the profit to two decimal
places would fail on any solver upgrade and would be pinning a number nobody
computed by hand — the golden test is where exact arithmetic belongs. What
these assert are the *qualitative* facts that would still have to hold if
every price in the snapshot changed.
"""

from __future__ import annotations

import datetime as dt

import pytest

from bess_arb.backtest.metrics import summarise
from bess_arb.backtest.runner import BacktestResult, run_backtest
from bess_arb.config import load_config
from bess_arb.policy import build_policy
from bess_arb.policy.floor import FloorPolicy
from bess_arb.series import load_prices, regime_of

CONFIG = load_config()
REGIME = "hourly"

pytestmark = pytest.mark.skipif(
    not CONFIG.data.parquet_path(REGIME).is_file(),
    reason="no frozen snapshot in the working tree",
)

# A quarter, which is enough for the warm-up to fall away and for the
# climatology to be answering from three prior years of the same months.
FIRST_DAY = dt.date(2024, 6, 1)
LAST_DAY = dt.date(2024, 8, 31)


@pytest.fixture(scope="module")
def runs() -> dict[str, BacktestResult]:
    prices = load_prices(CONFIG, REGIME)
    regime = regime_of(CONFIG, REGIME)
    return {
        name: run_backtest(
            prices,
            build_policy(name, prices),
            CONFIG.battery,
            regime,
            CONFIG.horizon,
            backend=CONFIG.backend,
            solver=CONFIG.solver,
            first_day=FIRST_DAY,
            last_day=LAST_DAY,
        )
        for name in ("floor", "oracle")
    }


@pytest.fixture(scope="module")
def forecast_run() -> tuple[BacktestResult, object]:
    """The third bar, on a shorter window because it costs a fit to produce.

    Twenty days rather than a quarter: the forecaster refits every thirty, so
    this pays for exactly one fit and still leaves a fortnight past the
    warm-up to measure. What is asserted below needs no more than that, and
    the economics of the full period belong to a run rather than to CI.
    """
    from bess_arb.forecast import build_forecaster

    prices = load_prices(CONFIG, REGIME)
    forecaster = build_forecaster(CONFIG, REGIME)
    policy = build_policy("forecast", prices, forecaster=forecaster)
    result = run_backtest(
        prices,
        policy,
        CONFIG.battery,
        regime_of(CONFIG, REGIME),
        CONFIG.horizon,
        backend=CONFIG.backend,
        solver=CONFIG.solver,
        first_day=FIRST_DAY,
        last_day=FIRST_DAY + dt.timedelta(days=19),
    )
    return result, policy


def test_the_run_covers_every_day_it_was_asked_for(
    runs: dict[str, BacktestResult],
) -> None:
    for name, result in runs.items():
        days = [day.day for day in result.days]
        assert days[0] == FIRST_DAY, name
        assert days[-1] == LAST_DAY, name
        assert len(days) == (LAST_DAY - FIRST_DAY).days + 1, name


def test_the_window_length_is_the_ordinary_one_off_a_transition(
    runs: dict[str, BacktestResult],
) -> None:
    """No DST day in June to August, so exactly one model is ever built."""
    for name, result in runs.items():
        assert result.window_lengths == (48,), name


def test_perfect_foresight_beats_the_climatology(
    runs: dict[str, BacktestResult],
) -> None:
    """The result the whole comparison rests on.

    If this ever failed, either the floor is reading prices it should not
    have or the oracle is not being given the realised ones — and both are
    failures that leave every other number in the repository intact and
    plausible.
    """
    floor = summarise(runs["floor"])
    oracle = summarise(runs["oracle"])
    share = floor.with_bound(oracle).fraction_of_bound

    assert oracle.profit_eur > floor.profit_eur
    assert share is not None
    assert 0.0 < share < 1.0


def test_both_policies_actually_trade(runs: dict[str, BacktestResult]) -> None:
    """A quarter of Spanish summer prices at c_deg = 17 is worth cycling for.

    Zero throughput would pass the dominance test above trivially, so it is
    ruled out separately.
    """
    for name, result in runs.items():
        metrics = summarise(result)
        assert metrics.discharged_mwh > 0.0, name
        assert metrics.profit_eur > 0.0, name


def test_equivalent_cycles_are_physically_sane(
    runs: dict[str, BacktestResult],
) -> None:
    """The §3.3 alarm, wired up.

    A 2-hour battery cannot usefully do much more than one cycle a day in a
    day-ahead market with one solar trough and one evening peak. Around 700
    would mean two full cycles every day, which is what a degradation cost
    far too low looks like. The band is wide because the point is to catch a
    wrong ``c_deg``, not to pin a result.
    """
    for name, result in runs.items():
        cycles = summarise(result).equivalent_cycles_per_year
        assert 50.0 < cycles < 700.0, f"{name}: {cycles:.1f} cycles/year"


def test_a_higher_degradation_cost_suppresses_cycling() -> None:
    """``c_deg`` is a minimum-spread threshold, on real prices.

    Not a re-run of the golden test's version of this: there the threshold
    was set above a single hand-made spread, here it has to bite across a
    whole quarter of a real market. The monotonicity is the mechanism §3.3
    describes, observed rather than asserted.
    """
    prices = load_prices(CONFIG, REGIME)
    regime = regime_of(CONFIG, REGIME)
    cycles = []
    for c_deg in (5.0, 40.0):
        result = run_backtest(
            prices,
            build_policy("oracle", prices),
            CONFIG.with_c_deg(c_deg).battery,
            regime,
            CONFIG.horizon,
            backend=CONFIG.backend,
            solver=CONFIG.solver,
            first_day=FIRST_DAY,
            last_day=FIRST_DAY + dt.timedelta(days=30),
        )
        cycles.append(summarise(result).equivalent_cycles_per_year)

    assert cycles[0] > cycles[1]


def test_the_forecast_policy_runs_on_the_snapshot_and_trades(
    forecast_run: tuple[BacktestResult, object],
) -> None:
    """The third bar exists, is solved through the same optimiser, and cycles.

    The window is inside the hourly regime, so the two-stage split does not
    apply and there is nothing here about intra-hour structure — that is the
    quarter-hourly run's business. What this pins is that the whole chain
    from the frozen Parquet through the feature table and a real LightGBM fit
    to a dispatch produces a schedule at all.
    """
    result, _ = forecast_run
    metrics = summarise(result)

    assert metrics.days > 0
    assert metrics.discharged_mwh > 0.0
    assert 50.0 < metrics.equivalent_cycles_per_year < 700.0


def test_the_forecast_policy_earns_something_and_not_more_than_foresight() -> None:
    """The two bounds a point forecast has to sit between.

    Deliberately not ``forecast > floor``. That ordering is the thing the
    project *measures*, and it genuinely changes sign: on this snapshot the
    forecast loses to the floor at ``c_deg = 5`` quarter-hourly and beats it by
    9.6 points of bound at ``c_deg = 40`` hourly. Asserting it here would pin
    one favourable cell of a table whose whole content is that the cells
    differ. What must hold in every window and at every degradation cost is
    that a policy deciding at the gate earns something, and that it does not
    out-earn a policy that already knows the answer.
    """
    prices = load_prices(CONFIG, REGIME)
    regime = regime_of(CONFIG, REGIME)
    last = FIRST_DAY + dt.timedelta(days=19)

    from bess_arb.forecast import build_forecaster

    forecaster = build_forecaster(CONFIG, REGIME)
    profits = {}
    for name in ("forecast", "oracle"):
        result = run_backtest(
            prices,
            build_policy(name, prices, forecaster=forecaster),
            CONFIG.battery,
            regime,
            CONFIG.horizon,
            backend=CONFIG.backend,
            solver=CONFIG.solver,
            first_day=FIRST_DAY,
            last_day=last,
        )
        profits[name] = summarise(result).profit_eur

    assert profits["forecast"] > 0.0
    assert profits["forecast"] < profits["oracle"]


def test_the_forecast_policy_did_not_fall_back_on_this_window(
    forecast_run: tuple[BacktestResult, object],
) -> None:
    """Mid-2024 has two and a half years behind it, so the model answers.

    The caveat checked rather than trusted, exactly as the floor's is below.
    A fallback firing here would mean the middle bar of the chart was partly
    the floor wearing another label, which is the one way this comparison can
    quietly become circular.
    """
    _, policy = forecast_run
    counts: dict[str, int] = policy.diagnostics()  # type: ignore[attr-defined]

    assert counts["forecast_days"] > 0
    assert counts["fallback_days"] == 0
    assert counts["no_history"] == 0


def test_the_floor_priced_the_run_off_populated_averages() -> None:
    """The caveat, checked rather than trusted.

    By mid-2024 the causal climatology has two and a half years behind it,
    so every period should be priced from its own month-and-hour average. A
    fallback firing here would mean the floor is thinner than the README
    will claim.
    """
    prices = load_prices(CONFIG, REGIME)
    policy = FloorPolicy(prices)
    run_backtest(
        prices,
        policy,
        CONFIG.battery,
        regime_of(CONFIG, REGIME),
        CONFIG.horizon,
        backend=CONFIG.backend,
        solver=CONFIG.solver,
        first_day=FIRST_DAY,
        last_day=FIRST_DAY + dt.timedelta(days=10),
    )

    counts = policy.diagnostics()
    assert counts["month_time"] > 0
    assert counts["time"] == counts["grand_mean"] == counts["no_history"] == 0
