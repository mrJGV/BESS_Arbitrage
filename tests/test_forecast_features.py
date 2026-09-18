"""What the feature table actually contains, beyond being causal.

``tests/test_no_lookahead.py`` proves nothing arrives too early. This one
proves the values are the right values: that ``price_lag1d`` really is the
price one delivery day back at the same local clock time, that a lag is
stamped at its day's publication rather than its delivery, and that the DST
days do not quietly become 24-hour days on the way through a pivot.

The distinction matters because a table that is causal and *wrong* would pass
the whole of the other file — a column of NaN is impeccably causal.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.forecast.features import EXOGENOUS_COLUMNS, build_features
from bess_arb.forecast.lgbm import ForecastSpec, PriceForecaster, _to_deviation
from bess_arb.timeline import (
    MARKET_TZ,
    Regime,
    day_bounds,
    gate_close_utc,
    periods_in_day,
    price_published_utc,
    to_market_time,
    utc_index,
)

HOURLY = Regime("hourly", 1.0)
QUARTER_HOURLY = Regime("quarter_hourly", 0.25)
LAGS = (1, 2, 3, 4, 5, 6, 7, 8)

FIRST_DAY = dt.date(2025, 1, 1)
LAST_DAY = dt.date(2026, 6, 30)

# Late enough that a year of history sits behind its gate, so the forecaster
# answers rather than declining for want of `min_train_days`.
DECISION_DAY = dt.date(2026, 6, 1)


def _prices(regime: Regime) -> pd.Series:
    """Prices that encode their own coordinates.

    ``day_number * 1000 + hour * 10 + quarter`` — so a lag feature can be
    checked by arithmetic instead of by looking it up again the same way the
    builder did, which would only prove the builder agrees with itself.
    """
    index = utc_index(FIRST_DAY, LAST_DAY, regime)
    local = to_market_time(index)
    day_number = np.array([(d - FIRST_DAY).days for d in local.date], dtype=float)
    return pd.Series(
        day_number * 1000.0
        + np.asarray(local.hour) * 10.0
        + np.asarray(local.minute) / 15.0,
        index=index,
        name="price_eur_mwh",
    )


def _exog(regime: Regime) -> pd.DataFrame:
    index = utc_index(FIRST_DAY, LAST_DAY, HOURLY)
    hours = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "demand_forecast_mw": 20000.0 + hours,
            "wind_forecast_mw": 5000.0 + hours,
            "solar_forecast_mw": 1000.0 + hours,
        },
        index=index,
    )


@pytest.fixture(params=["hourly", "quarter_hourly"])
def regime(request: pytest.FixtureRequest) -> Regime:
    return HOURLY if request.param == "hourly" else QUARTER_HOURLY


def _table(regime: Regime, lead: int = 0) -> object:
    return build_features(
        _prices(regime), _exog(regime), regime, lead_days=lead, lags_days=LAGS
    )


@pytest.mark.parametrize("lag", LAGS)
@pytest.mark.parametrize("lead", [0, 1])
def test_a_lag_is_that_many_days_back_at_the_same_clock_time(
    regime: Regime, lag: int, lead: int
) -> None:
    """``price_lag{k}d`` is D-k at the target's own local time of day.

    The encoded price makes the expected value a subtraction: k days back at
    the same clock time differs by exactly ``k * 1000``, and the anchor shift
    for the second horizon day adds another ``lead * 1000``.
    """
    table = _table(regime, lead)
    target = pd.Timestamp("2026-05-20 18:00", tz=MARKET_TZ).tz_convert("UTC")

    actual = table.values.loc[target, f"price_lag{lag}d"]  # type: ignore[attr-defined]
    here = table.target.loc[target]  # type: ignore[attr-defined]

    assert actual == pytest.approx(here - (lag + lead) * 1000.0)


@pytest.mark.parametrize("lead", [0, 1])
def test_a_lag_is_stamped_at_its_days_publication(regime: Regime, lead: int) -> None:
    """``price_lag1d`` for a day-D target reads D-1, public since 13:00 on D-2.

    Pinned as an instant rather than as "before the gate": the stamp is what
    the no-lookahead audit compares, and a stamp at the *delivery* of D-1
    would be after the gate and would wrongly flag yesterday as unknown.
    """
    table = _table(regime, lead)
    target = pd.Timestamp("2026-05-20 18:00", tz=MARKET_TZ).tz_convert("UTC")
    decision_day = dt.date(2026, 5, 20) - dt.timedelta(days=lead)

    stamp = table.available_at.loc[target, "price_lag1d"]  # type: ignore[attr-defined]

    assert stamp == price_published_utc(decision_day - dt.timedelta(days=1))
    assert stamp.tz_convert(MARKET_TZ).hour == 13
    assert stamp.tz_convert(MARKET_TZ).date() == decision_day - dt.timedelta(days=2)
    assert stamp <= gate_close_utc(decision_day)


def test_the_exogenous_block_reads_the_hour_containing_the_period(
    regime: Regime,
) -> None:
    """A quarter-hourly period takes the published value for its own hour.

    Not a resample and not an interpolation: the D+1 family is issued hourly
    and stayed hourly, so the four quarters of an hour share one number.
    """
    table = _table(regime)
    hour = pd.Timestamp("2026-05-20 18:00", tz=MARKET_TZ).tz_convert("UTC")
    within = utc_index(dt.date(2026, 5, 20), dt.date(2026, 5, 20), regime)
    within = within[(within >= hour) & (within < hour + pd.Timedelta(hours=1))]

    block = table.values.loc[within, list(EXOGENOUS_COLUMNS)]  # type: ignore[attr-defined]

    assert len(within) == round(1.0 / regime.dt_h)
    assert (block.nunique() == 1).all()
    assert block["demand_forecast_mw"].iloc[0] == pytest.approx(
        _exog(regime).loc[hour, "demand_forecast_mw"]
    )


def test_residual_demand_is_demand_less_the_two_renewables(regime: Regime) -> None:
    table = _table(regime)
    target = pd.Timestamp("2026-05-20 18:00", tz=MARKET_TZ).tz_convert("UTC")
    row = table.values.loc[target]  # type: ignore[attr-defined]

    assert row["residual_demand_mw"] == pytest.approx(
        row["demand_forecast_mw"] - row["wind_forecast_mw"] - row["solar_forecast_mw"]
    )


@pytest.mark.parametrize("day", [dt.date(2026, 3, 29), dt.date(2025, 10, 26)])
def test_a_transition_day_keeps_its_own_number_of_periods(
    regime: Regime, day: dt.date
) -> None:
    """23 and 25 hours survive the pivot the lag features are built on.

    A pivot on (day, local time of day) is where a 92- or 100-period day would
    silently become 96: the March day is missing a key and the October one
    repeats one. Both are handled, and neither is allowed to change the number
    of rows the table has for that day.
    """
    table = _table(regime)
    rows = pd.Index(to_market_time(pd.DatetimeIndex(table.values.index)).date) == day  # type: ignore[attr-defined]

    assert int(rows.sum()) == periods_in_day(day, regime)


def test_the_march_day_has_no_lag_for_the_hour_that_does_not_exist() -> None:
    """The 23-hour day: 02:00 local never happens, so nothing can lag onto it.

    NaN is the honest answer and LightGBM takes it natively. Inventing a value
    — the previous hour's, or an interpolation — would be a price that never
    cleared standing in for one that never existed.
    """
    table = build_features(
        _prices(HOURLY), _exog(HOURLY), HOURLY, lead_days=0, lags_days=LAGS
    )
    # 2026-03-29 skips 02:00, so a target two days later at 02:00 has no
    # same-clock-time price to reach back to.
    target = pd.Timestamp("2026-03-31 02:00", tz=MARKET_TZ).tz_convert("UTC")

    assert np.isnan(table.values.loc[target, "price_lag2d"])
    assert not np.isnan(table.values.loc[target, "price_lag3d"])


def test_the_deviation_stage_exists_only_when_the_grid_is_finer() -> None:
    """The intra-hour model is built for quarters and not for hours.

    On the hourly regime the deviation is identically zero; fitting a booster
    to a column of zeros would be a stage that cannot be wrong and cannot
    help.
    """
    spec = ForecastSpec(
        lags_days=LAGS,
        refit_days=30,
        min_train_days=200,
        min_deviation_days=30,
        min_residual_obs=30,
        level_params={"num_leaves": 4},
        deviation_params={"num_leaves": 4},
        level_rounds=5,
        deviation_rounds=5,
        validation_days=30,
        early_stopping_rounds=5,
        seed=1,
        threads=1,
    )

    quarter = PriceForecaster(
        _prices(QUARTER_HOURLY), _exog(QUARTER_HOURLY), QUARTER_HOURLY, HOURLY, spec
    )
    hourly = PriceForecaster(_prices(HOURLY), _exog(HOURLY), HOURLY, HOURLY, spec)

    assert quarter.two_stage
    assert not hourly.two_stage


def test_a_level_grid_finer_than_the_target_is_refused() -> None:
    """The level stage predicts an average; it cannot predict a finer one."""
    spec = ForecastSpec(
        lags_days=LAGS,
        refit_days=30,
        min_train_days=200,
        min_deviation_days=30,
        min_residual_obs=30,
        level_params={},
        deviation_params={},
        level_rounds=5,
        deviation_rounds=5,
        validation_days=30,
        early_stopping_rounds=5,
        seed=1,
        threads=1,
    )

    with pytest.raises(ValueError, match="at least as coarse"):
        PriceForecaster(_prices(HOURLY), _exog(HOURLY), HOURLY, QUARTER_HOURLY, spec)


def _window(day: dt.date) -> pd.DatetimeIndex:
    """The two horizon days the backtest solves: D and D+1."""
    start, _ = day_bounds(day, day)
    _, end = day_bounds(day + dt.timedelta(days=1), day + dt.timedelta(days=1))
    return pd.DatetimeIndex(
        pd.date_range(start, end, freq=QUARTER_HOURLY.step, inclusive="left")
    ).as_unit("ns")


# --------------------------------------------------------------------------
# The round count is chosen, not configured
# --------------------------------------------------------------------------


def _early_stopping_spec(ceiling: int, validation_days: int) -> ForecastSpec:
    return ForecastSpec(
        lags_days=LAGS,
        refit_days=30,
        min_train_days=200,
        min_deviation_days=30,
        min_residual_obs=30,
        level_params={"learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 20},
        deviation_params={
            "learning_rate": 0.1,
            "num_leaves": 8,
            "min_data_in_leaf": 20,
        },
        level_rounds=ceiling,
        deviation_rounds=ceiling,
        validation_days=validation_days,
        early_stopping_rounds=5,
        seed=1,
        threads=1,
    )


def test_the_round_count_is_chosen_by_the_fold_and_not_by_the_ceiling() -> None:
    """A generous ceiling must not be what gets fitted.

    The defect this replaced was a fixed 600 rounds where six measured cutoffs
    wanted 56-157, so the assertion is the qualitative one: strictly fewer
    rounds than the ceiling, and the fit reports that a fold chose them. The
    exact count is data-dependent by design and pinning it would be wrong for
    the same reason pinning the golden dispatch vector would be.
    """
    ceiling = 400
    forecaster = PriceForecaster(
        _prices(QUARTER_HOURLY),
        _exog(QUARTER_HOURLY),
        QUARTER_HOURLY,
        HOURLY,
        _early_stopping_spec(ceiling, validation_days=30),
    )

    assert forecaster.forecast(DECISION_DAY, _window(DECISION_DAY)) is not None

    counts = forecaster.diagnostics()
    assert counts["fits"] > 0
    assert counts["unvalidated_fits"] == 0
    assert 0 < counts["max_rounds"] < ceiling


def test_a_history_too_short_to_spare_a_fold_falls_back_to_the_ceiling() -> None:
    """And says so, rather than validating on a stub.

    The fold has to be the most recent stretch to proxy the horizon, so a
    history barely longer than the fold has nothing left to train on. Fitting
    the ceiling is the weaker answer and ``unvalidated_fits`` is what makes it
    visible; silently shrinking the fold would hide it.
    """
    forecaster = PriceForecaster(
        _prices(QUARTER_HOURLY),
        _exog(QUARTER_HOURLY),
        QUARTER_HOURLY,
        HOURLY,
        # A fold longer than the synthetic history itself, so nothing can be
        # held out however the rows are split.
        _early_stopping_spec(7, validation_days=100_000),
    )

    assert forecaster.forecast(DECISION_DAY, _window(DECISION_DAY)) is not None

    counts = forecaster.diagnostics()
    assert counts["fits"] > 0
    assert counts["unvalidated_fits"] == counts["fits"]
    assert counts["max_rounds"] == 7


def test_the_validation_fold_reads_nothing_the_gate_has_not_released() -> None:
    """The fold is carved from rows already admitted, so it adds no leak.

    Mechanically: every row a fit can see, fold included, settled at or before
    the gate. ``tests/test_no_lookahead.py`` makes the same point by
    perturbation across the whole pipeline; this one pins the specific claim
    that splitting an admissible set by time cannot admit anything new.
    """
    spec = _early_stopping_spec(50, validation_days=30)
    table = build_features(
        _prices(QUARTER_HOURLY),
        _exog(QUARTER_HOURLY),
        QUARTER_HOURLY,
        lead_days=0,
        lags_days=spec.lags_days,
    )
    gate = gate_close_utc(DECISION_DAY)
    trainable = table.trainable_before(gate)

    assert trainable.any()
    assert bool((table.settled_at[trainable.to_numpy()] <= gate).all())


# --------------------------------------------------------------------------
# The deviation stage trains only where a deviation exists
# --------------------------------------------------------------------------


def _reconstructed_then_real(boundary: dt.date) -> pd.Series:
    """Quarter-hourly prices that are flat within the hour until ``boundary``.

    The shape ``load_price_history`` produces: before the market moved to
    15-minute units it repeats each hourly price across four quarters, so the
    intra-hour deviation there is identically zero. After it, the quarters
    differ. One series with both eras, which is what the deviation stage has
    to tell apart.
    """
    index = utc_index(FIRST_DAY, LAST_DAY, QUARTER_HOURLY)
    local = to_market_time(index)
    hourly = np.asarray(local.hour, dtype=float) * 10.0
    within = np.asarray(local.minute, dtype=float) / 15.0
    reconstructed = np.asarray(pd.DatetimeIndex(local).date) < boundary
    return pd.Series(
        np.where(reconstructed, hourly, hourly + within),
        index=index,
        name="price_eur_mwh",
    )


def test_the_deviation_stage_ignores_the_reconstructed_era() -> None:
    """Trained only where intra-hour spread is not identically zero.

    Leaving the reconstructed era in makes it the overwhelming majority of the
    training set -- measured on the real snapshot, 92% of it -- and a booster
    fitted through it learns that intra-hour spread is nearly absent, which is
    the pooled arm's defect reproduced inside the stage built to avoid it.
    """
    boundary = dt.date(2026, 1, 1)
    table = _to_deviation(
        build_features(
            _reconstructed_then_real(boundary),
            _exog(QUARTER_HOURLY),
            QUARTER_HOURLY,
            lead_days=0,
            lags_days=LAGS,
        ),
        HOURLY,
    )

    trainable = table.trainable_before(gate_close_utc(LAST_DAY)).to_numpy()
    assert trainable.any()

    days = pd.DatetimeIndex(to_market_time(pd.DatetimeIndex(table.values.index))).date
    admitted = np.asarray(days)[trainable]
    assert admitted.min() >= boundary, "the reconstructed era is still being trained on"

    # And the rows are excluded from *training* only: the stage still has to
    # predict on that grid, so the features must survive intact.
    assert len(table.values) == len(_reconstructed_then_real(boundary))
    assert not table.values.loc[np.asarray(days) < boundary].empty


def test_a_series_with_no_intra_hour_structure_yields_no_deviation_stage() -> None:
    """The degenerate case falls out rather than fitting a column of zeros."""
    flat = _reconstructed_then_real(LAST_DAY + dt.timedelta(days=1))
    table = _to_deviation(
        build_features(
            flat, _exog(QUARTER_HOURLY), QUARTER_HOURLY, lead_days=0, lags_days=LAGS
        ),
        HOURLY,
    )

    assert not table.trainable_before(gate_close_utc(LAST_DAY)).any()
