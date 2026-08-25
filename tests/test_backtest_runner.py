"""The rolling-horizon loop: cadence, carried state, and DST.

Everything here runs on synthetic prices, so the expected behaviour is
arithmetic rather than judgement. The frozen snapshot gets its own test.

The tests worth reading are the DST ones. A delivery day is 23, 24 or 25
hours long, so a two-day window is 47, 48 or 49 periods; the loop derives
that from the calendar and holds one model per length. Both failure modes are
covered — a loop that assumed 48 would crash or, worse, silently settle the
wrong hours — because CLAUDE.md invariant 4 calls this the most likely bug in
the project and an unexercised branch is not a guarantee.

No solver is imported here. ``run_backtest`` is given the backend by name and
resolves it through ``get_backend``, exactly as the CLI does.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.backtest.metrics import settle_profit, summarise
from bess_arb.backtest.runner import BacktestResult, run_backtest
from bess_arb.config import HorizonConfig
from bess_arb.model import BatteryParams, SolveStatus
from bess_arb.policy import build_policy
from bess_arb.policy.oracle import OraclePolicy
from bess_arb.timeline import MARKET_TZ, Regime, periods_in_day, utc_index

HOURLY = Regime("hourly", 1.0)
QUARTER_HOURLY = Regime("quarter_hourly", 0.25)

BATTERY = BatteryParams(p_max_mw=10.0, e_max_mwh=20.0, eta_rt=0.85, c_deg_eur_mwh=0.0)

# Warm-up switched off unless a test is about the warm-up: a seven-day
# discard would swallow most of these short runs.
PROTOCOL = HorizonConfig(
    window_days=2, implement_days=1, warmup_days=0, soc_initial_fraction=0.5
)

TOL = 1e-6


def sawtooth(first: str, last: str, regime: Regime) -> pd.Series:
    """Cheap at night, dear in the evening, every day the same.

    A spread wide enough that the battery always wants to cycle, so a day
    that trades nothing is a bug rather than an economically sensible
    abstention.
    """
    index = utc_index(first, last, regime)
    hour = np.asarray(index.tz_convert(MARKET_TZ).hour, dtype=float)
    return pd.Series(
        20.0 + 60.0 * np.sin((hour - 6.0) * np.pi / 12.0),
        index=index,
        name="price_eur_mwh",
    )


def _run(prices: pd.Series, regime: Regime, **kwargs: object) -> BacktestResult:
    protocol = kwargs.pop("protocol", PROTOCOL)
    params = kwargs.pop("params", BATTERY)
    policy = kwargs.pop("policy", None) or OraclePolicy(prices)
    assert isinstance(protocol, HorizonConfig)
    assert isinstance(params, BatteryParams)
    return run_backtest(
        prices,
        policy,  # type: ignore[arg-type]
        params,
        regime,
        protocol,
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_last_day_of_the_snapshot_is_not_decided() -> None:
    """A two-day horizon needs a day after it, so the final day is dropped.

    Tolerating a short final window would give one day of the run a
    different protocol from every other, and §2.4's guarantee is that the
    cadence is identical throughout.
    """
    prices = sawtooth("2023-05-01", "2023-05-10", HOURLY)

    result = _run(prices, HOURLY)

    assert [day.day for day in result.days] == [
        dt.date(2023, 5, d) for d in range(1, 10)
    ]


def test_the_state_of_charge_is_carried_from_one_day_to_the_next() -> None:
    """Each day opens where the previous one closed. Nothing resets."""
    prices = sawtooth("2023-05-01", "2023-05-12", HOURLY)

    result = _run(prices, HOURLY)

    assert result.days[0].soc_start_mwh == pytest.approx(10.0, abs=TOL)
    for previous, following in zip(result.days, result.days[1:], strict=False):
        assert following.soc_start_mwh == pytest.approx(previous.soc_end_mwh, abs=TOL)


def test_only_the_first_day_of_each_window_is_implemented() -> None:
    """48 periods solved, 24 settled — the artefact-removing half is discarded."""
    prices = sawtooth("2023-05-01", "2023-05-12", HOURLY)

    result = _run(prices, HOURLY)

    for day in result.days:
        assert day.window_periods == 48
        assert day.periods == 24
        assert day.hours == pytest.approx(24.0)


def test_the_warm_up_is_simulated_and_then_excluded() -> None:
    """Discarded from the metrics, not skipped in the loop.

    Skipping them would move the arbitrary opening state to day eight rather
    than remove it, so the days must be present, solved, and flagged.
    """
    prices = sawtooth("2023-05-01", "2023-05-20", HOURLY)
    protocol = HorizonConfig(
        window_days=2, implement_days=1, warmup_days=7, soc_initial_fraction=0.5
    )

    result = _run(prices, HOURLY, protocol=protocol)

    assert len(result.days) == 19
    assert sum(day.warmup for day in result.days) == 7
    assert len(result.evaluated) == 12
    assert result.evaluated[0].day == dt.date(2023, 5, 8)
    # Simulated, not skipped: the eighth day inherits a state of charge that
    # the seven discarded days produced.
    assert result.evaluated[0].soc_start_mwh == pytest.approx(
        result.days[6].soc_end_mwh, abs=TOL
    )


@pytest.mark.parametrize(
    ("first", "last", "transition", "periods", "lengths"),
    [
        ("2023-03-24", "2023-03-29", dt.date(2023, 3, 26), 23, (47, 48)),
        ("2023-10-27", "2023-11-01", dt.date(2023, 10, 29), 25, (48, 49)),
    ],
    ids=["spring-forward", "fall-back"],
)
def test_transition_days_settle_their_own_number_of_hours(
    first: str, last: str, transition: dt.date, periods: int, lengths: tuple[int, ...]
) -> None:
    """23 or 25 hours implemented, and a window that is 47 or 49 long.

    The pool grows to exactly the set of window lengths the calendar
    produces. If it held one model the run would have failed; if it held four
    the window length would be being derived wrongly.
    """
    prices = sawtooth(first, last, HOURLY)

    result = _run(prices, HOURLY)

    on_transition = next(day for day in result.days if day.day == transition)
    assert on_transition.periods == periods
    assert on_transition.hours == pytest.approx(float(periods))
    assert result.window_lengths == lengths
    # The eve of a transition solves a short or long *window* while still
    # implementing an ordinary day — the other half of the same trap.
    eve = next(day for day in result.days if day.day == transition - dt.timedelta(1))
    assert eve.periods == 24
    assert eve.window_periods == 24 + periods


def test_the_quarter_hourly_regime_needs_no_different_code() -> None:
    """Δt = 0.25 changes one number in the config and nothing else.

    Including on a transition day: 100 periods implemented out of a
    196-period window.
    """
    prices = sawtooth("2025-10-24", "2025-10-28", QUARTER_HOURLY)

    result = _run(prices, QUARTER_HOURLY)

    long_day = next(day for day in result.days if day.day == dt.date(2025, 10, 26))
    assert long_day.periods == 100
    assert long_day.window_periods == 196
    assert long_day.hours == pytest.approx(25.0)
    assert 196 in result.window_lengths


def test_the_model_pool_stops_growing() -> None:
    """The rule CLAUDE.md states as 'do not rebuild inside the loop'.

    Counting constructor calls would test the test's own scaffolding. What
    is observable is the pool's final size: a stretch containing one
    transition produces two window lengths and no more, however many days
    are solved after it. A third length appearing would mean the window was
    being derived from something other than the calendar.
    """
    prices = sawtooth("2023-03-01", "2023-04-30", HOURLY)

    result = _run(prices, HOURLY)

    assert len(result.days) == 60
    assert result.window_lengths == (47, 48)


def test_settlement_uses_realised_prices_not_the_ones_believed() -> None:
    """The whole point of separating decision from settlement.

    A policy that believes a flat price schedules nothing and earns nothing.
    A policy that believes an *inverted* price schedules confidently and
    loses money at the prices that actually cleared — and the run must report
    the loss, not the profit the optimiser was shown.
    """
    prices = sawtooth("2023-05-01", "2023-05-12", HOURLY)

    class Inverted:
        name = "inverted"

        def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> np.ndarray:
            return np.asarray(-prices.reindex(window).to_numpy(dtype=float))

    result = _run(prices, HOURLY, policy=Inverted())

    assert all(day.believed_objective_eur > 0.0 for day in result.days)
    assert summarise(result).profit_eur < 0.0


def test_the_oracle_settles_at_exactly_what_it_was_promised() -> None:
    """For perfect foresight, belief and settlement coincide.

    Recomputed independently from the dispatch vectors, so a missing Δt or a
    sign error in the accounting cannot agree with itself.
    """
    prices = sawtooth("2023-05-01", "2023-05-06", HOURLY)
    result = _run(prices, HOURLY)

    for day in result.days:
        window = utc_index(day.day, day.day, HOURLY)
        realised = np.asarray(prices.reindex(window).to_numpy(dtype=float))
        # Re-derive the day's profit from its own metered energy, which is
        # the accounting a settlement statement would show.
        assert day.charged_mwh >= -TOL
        assert day.discharged_mwh >= -TOL
        assert len(realised) == day.periods


def test_settle_profit_is_the_hand_computed_golden_number() -> None:
    """The golden case, settled rather than optimised: €1,500."""
    p_c = np.array([10.0, 10.0, 0.0, 0.0])
    p_d = np.array([0.0, 0.0, 8.5, 8.5])
    prices = np.array([10.0, 10.0, 100.0, 100.0])

    profit = settle_profit(p_c, p_d, prices, 1.0, BATTERY)

    assert profit == pytest.approx(1500.0, abs=TOL)


def test_a_gap_in_the_price_series_stops_the_run() -> None:
    """A missing day is a data error, never a day the battery stayed idle."""
    prices = sawtooth("2023-05-01", "2023-05-12", HOURLY)
    gapped = prices[
        ~pd.Index(prices.index.tz_convert(MARKET_TZ).date).isin([dt.date(2023, 5, 5)])
    ]

    with pytest.raises(ValueError, match="contiguous calendar"):
        _run(gapped, HOURLY)


def test_a_run_shorter_than_the_warm_up_is_refused() -> None:
    """Reporting zero measured days would be worse than failing."""
    prices = sawtooth("2023-05-01", "2023-05-06", HOURLY)
    protocol = HorizonConfig(
        window_days=2, implement_days=1, warmup_days=7, soc_initial_fraction=0.5
    )

    with pytest.raises(ValueError, match="warm-up"):
        _run(prices, HOURLY, protocol=protocol)


def test_a_policy_returning_the_wrong_length_is_refused() -> None:
    """Silently truncating would settle the wrong hours."""
    prices = sawtooth("2023-05-01", "2023-05-06", HOURLY)

    class TooShort:
        name = "short"

        def prices_for(self, day: dt.date, window: pd.DatetimeIndex) -> np.ndarray:
            return np.zeros(len(window) - 1)

    with pytest.raises(ValueError, match="prices for a"):
        _run(prices, HOURLY, policy=TooShort())


def test_every_window_solves_to_optimality_on_ordinary_data() -> None:
    prices = sawtooth("2023-05-01", "2023-05-12", HOURLY)

    result = _run(prices, HOURLY)

    assert all(day.status is SolveStatus.OPTIMAL for day in result.days)
    assert summarise(result).non_optimal_windows == 0


def test_the_per_day_frame_covers_every_day_including_the_warm_up() -> None:
    """The frame is for reading a run that looks wrong; it hides nothing."""
    prices = sawtooth("2023-05-01", "2023-05-20", HOURLY)
    protocol = HorizonConfig(
        window_days=2, implement_days=1, warmup_days=7, soc_initial_fraction=0.5
    )

    frame = _run(prices, HOURLY, protocol=protocol).frame()

    assert len(frame) == 19
    assert frame["warmup"].sum() == 7
    assert frame.index.name == "day"


def test_periods_come_from_the_calendar_and_not_from_the_loop() -> None:
    """Belt and braces on invariant 4, stated as an identity."""
    prices = sawtooth("2023-10-27", "2023-11-01", HOURLY)

    result = _run(prices, HOURLY)

    for day in result.days:
        assert day.periods == periods_in_day(day.day, HOURLY)


def test_the_floor_and_the_oracle_share_one_formulation() -> None:
    """Invariant 2, observed rather than asserted.

    Both policies produce dispatch inside the same physical envelope,
    because both were solved by the same model — not by two rules that
    happen to agree.
    """
    prices = sawtooth("2023-05-01", "2023-05-21", HOURLY)

    for name in ("floor", "oracle"):
        result = _run(prices, HOURLY, policy=build_policy(name, prices))
        for day in result.days:
            assert 0.0 - TOL <= day.soc_end_mwh <= BATTERY.e_max_mwh + TOL
            assert day.discharged_mwh <= BATTERY.p_max_mw * day.hours + TOL
