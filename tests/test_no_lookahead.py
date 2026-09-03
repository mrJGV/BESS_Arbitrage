"""Invariant 1, mechanically: nothing after noon on D-1 may decide day D.

The single most valuable test in the repository, because it is the one thing
no amount of reading the code establishes. A lookahead bug does not raise, does
not print a warning and does not produce an implausible number — it produces an
*attractive* one, and the more careful the rest of the project is, the more
convincing the wrong answer looks.

Three mechanisms, deliberately unequal
--------------------------------------

1. **The declared publication instant.** Every feature column carries an
   ``available_at``, and none of them may exceed its own gate. Cheap, total,
   and exactly as trustworthy as the bookkeeping it checks — which is why it
   is not the main event.

2. **Perturbation.** Rewrite every price at or after ``gate(D)`` and rebuild.
   If the forecast for day D changes by so much as a float, something read the
   future. This one trusts nothing: not the metadata, not the lag arithmetic,
   not the training filter. It would catch a leak introduced by a refactor
   that also updated the ``available_at`` to match, which is the failure mode
   mechanism 1 is blind to by construction.

3. **Negative controls.** A test that has quietly stopped looking passes just
   as silently as the bug it was meant to catch, so each mechanism is shown
   failing on a table that really does leak — a one-day lag for the first, and
   a forecaster given tomorrow's prices for the second.

Resolutions are parametrised because a Parquet round trip hands back
microsecond timestamps while ``Timestamp.value`` is nanoseconds, and comparing
the two directly is wrong by a factor of a thousand — it puts every gate a
thousandfold too early, so the whole series appears to predate the cut. That
has already bitten this project once, in the floor policy.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bess_arb.forecast.features import (
    ALWAYS_KNOWN,
    EXOGENOUS_COLUMNS,
    FIRST_COMPLETE_LAG_DAYS,
    build_features,
)
from bess_arb.forecast.lgbm import ForecastSpec, PriceForecaster
from bess_arb.timeline import (
    MARKET_TZ,
    Regime,
    gate_close_utc,
    to_market_time,
    utc_index,
)

HOURLY = Regime("hourly", 1.0)
QUARTER_HOURLY = Regime("quarter_hourly", 0.25)

LAGS = (2, 3, 4, 5, 6, 7, 8)

# Long enough to fit, short enough to fit fast: two years of synthetic history
# either side of a March transition, so the 23-hour day is inside the window
# rather than assumed away.
FIRST_DAY = dt.date(2024, 1, 1)
LAST_DAY = dt.date(2025, 12, 31)

# A deliberately tiny booster. This suite is about *what the model saw*, not
# about how well it fits; 20 rounds on 8 leaves says as much about leakage as
# 600 on 63 and turns minutes of test time into seconds.
TINY = ForecastSpec(
    lags_days=LAGS,
    refit_days=30,
    min_train_days=200,
    min_deviation_days=30,
    level_params={"learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 20},
    deviation_params={"learning_rate": 0.1, "num_leaves": 8, "min_data_in_leaf": 20},
    level_rounds=20,
    deviation_rounds=20,
    validation_days=30,
    early_stopping_rounds=5,
    seed=20260819,
    threads=1,
)


def _prices(regime: Regime, unit: str = "ns") -> pd.Series:
    """A synthetic price series with a real daily shape and a real calendar.

    Built from the project's own :func:`utc_index`, so the DST days are the
    lengths the zone database says and not 24 periods each.
    """
    index = utc_index(FIRST_DAY, LAST_DAY, regime).as_unit(unit)
    local = to_market_time(index)
    hour = np.asarray(local.hour) + np.asarray(local.minute) / 60.0
    rng = np.random.default_rng(20260819)
    shape = 60.0 + 40.0 * np.sin((hour - 4.0) * np.pi / 12.0)
    season = 15.0 * np.sin(np.asarray(local.dayofyear) * 2.0 * np.pi / 365.25)
    return pd.Series(
        shape + season + rng.normal(0.0, 8.0, len(index)),
        index=index,
        name="price_eur_mwh",
    )


def _exog(regime: Regime, unit: str = "ns") -> pd.DataFrame:
    """Published D+1 forecasts on the hourly grid, whatever the price grid is."""
    index = utc_index(FIRST_DAY, LAST_DAY, HOURLY).as_unit(unit)
    local = to_market_time(index)
    hour = np.asarray(local.hour, dtype=float)
    rng = np.random.default_rng(4242)
    return pd.DataFrame(
        {
            "demand_forecast_mw": 26000 + 6000 * np.sin((hour - 6) * np.pi / 12),
            "wind_forecast_mw": np.abs(rng.normal(6000, 3000, len(index))),
            "solar_forecast_mw": np.clip(
                12000 * np.sin((hour - 7) * np.pi / 12), 0.0, None
            ),
        },
        index=index,
    )


@pytest.fixture(params=["hourly", "quarter_hourly"])
def regime(request: pytest.FixtureRequest) -> Regime:
    return HOURLY if request.param == "hourly" else QUARTER_HOURLY


# --------------------------------------------------------------------------
# 1. The declared publication instant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize("lead", [0, 1])
def test_no_feature_is_readable_after_its_gate(
    regime: Regime, lead: int, unit: str
) -> None:
    """Invariant 1 as one assertion, over every column and every row.

    Parametrised over the index resolution: a table built from a Parquet round
    trip carries microseconds, and a gate compared against the wrong unit is
    off by a factor of a thousand.
    """
    table = build_features(
        _prices(regime, unit),
        _exog(regime, unit),
        regime,
        lead_days=lead,
        lags_days=LAGS,
    )

    assert table.violations() == []


@pytest.mark.parametrize("lead", [0, 1])
def test_the_gate_is_local_noon_on_the_day_before(regime: Regime, lead: int) -> None:
    """The anchor really is ``gate_close_utc(D)``, D being the *decision* day.

    Without this the check above could pass by comparing every column against
    a gate that had itself drifted a day forward.
    """
    table = build_features(
        _prices(regime), _exog(regime), regime, lead_days=lead, lags_days=LAGS
    )
    target = pd.Timestamp("2025-06-15 09:00", tz=MARKET_TZ).tz_convert("UTC")
    decision_day = dt.date(2025, 6, 15) - dt.timedelta(days=lead)

    gate = table.gate.loc[target]

    assert gate == gate_close_utc(decision_day)
    assert gate.tz_convert(MARKET_TZ).hour == 12
    assert gate.tz_convert(MARKET_TZ).date() == decision_day - dt.timedelta(days=1)


def test_the_second_horizon_day_gets_no_published_forecast(regime: Regime) -> None:
    """At the gate for D, the bundle covering D+1 has not been published yet.

    The asymmetry between the two leads is a fact about the wire, not a
    modelling preference, so it is pinned rather than left to the docstring.
    """
    today = build_features(
        _prices(regime), _exog(regime), regime, lead_days=0, lags_days=LAGS
    )
    tomorrow = build_features(
        _prices(regime), _exog(regime), regime, lead_days=1, lags_days=LAGS
    )

    assert set(EXOGENOUS_COLUMNS) <= set(today.columns)
    assert not set(EXOGENOUS_COLUMNS) & set(tomorrow.columns)


def test_only_the_calendar_columns_may_skip_a_publication_instant(
    regime: Regime,
) -> None:
    """The exemption is a closed list, so nothing can claim it by accident."""
    table = build_features(
        _prices(regime), _exog(regime), regime, lead_days=0, lags_days=LAGS
    )

    undeclared = set(table.values.columns) - set(table.available_at.columns)

    assert undeclared == set(ALWAYS_KNOWN)


def test_training_rows_are_limited_to_settled_prices(regime: Regime) -> None:
    """A fit for day D may not see a price that had not cleared by ``gate(D)``."""
    table = build_features(
        _prices(regime), _exog(regime), regime, lead_days=0, lags_days=LAGS
    )
    cutoff = gate_close_utc(dt.date(2025, 3, 3))

    rows = table.trainable_before(cutoff)
    last = table.values.index[rows.to_numpy()][-1]

    assert last + regime.step <= cutoff
    assert not rows.loc[table.settled_at > cutoff].any()


# --------------------------------------------------------------------------
# 2. Perturbation: the check that trusts nothing
# --------------------------------------------------------------------------


def _forecaster(
    prices: pd.Series, exog: pd.DataFrame, regime: Regime
) -> PriceForecaster:
    return PriceForecaster(
        prices=prices, exog=exog, regime=regime, level_regime=HOURLY, spec=TINY
    )


def test_rewriting_every_price_after_the_gate_changes_no_forecast(
    regime: Regime,
) -> None:
    """The strongest form of the invariant, and the one that trusts nothing.

    Everything from ``gate(D)`` onwards is replaced with a price no market has
    ever cleared. If any part of the pipeline — a lag, a rolling window, a
    training filter, the exogenous join — reached past the gate, the forecast
    would move. It does not move at all, not to a tolerance.
    """
    day = dt.date(2025, 9, 10)
    gate = gate_close_utc(day)
    window = utc_index(day, day + dt.timedelta(days=1), regime)
    prices, exog = _prices(regime), _exog(regime)

    honest = _forecaster(prices, exog, regime).forecast(day, window)

    tampered = prices.copy()
    tampered.loc[tampered.index >= gate] = 999.0
    leaked = _forecaster(tampered, exog, regime).forecast(day, window)

    assert honest is not None and leaked is not None
    np.testing.assert_array_equal(honest, leaked)


def test_rewriting_the_exogenous_forecast_after_the_gate_changes_no_forecast(
    regime: Regime,
) -> None:
    """The same, for the published series.

    Their *valid* times run through day D, which is after the gate — that is
    the whole point of a forecast. What may not leak is the next publication:
    the bundle covering D+1 goes out on day D, and nothing about day D's
    decision may depend on it.
    """
    day = dt.date(2025, 9, 10)
    window = utc_index(day, day + dt.timedelta(days=1), regime)
    prices, exog = _prices(regime), _exog(regime)

    honest = _forecaster(prices, exog, regime).forecast(day, window)

    tampered = exog.copy()
    # Everything published after this day's bundle, i.e. from the start of
    # D+1 onwards. Day D's own values are left alone: they were on the wire.
    after = pd.Timestamp(day + dt.timedelta(days=1), tz=MARKET_TZ).tz_convert("UTC")
    tampered.loc[tampered.index >= after, :] = 1.0
    leaked = _forecaster(prices, tampered, regime).forecast(day, window)

    assert honest is not None and leaked is not None
    np.testing.assert_array_equal(honest, leaked)


def test_a_forecast_does_not_depend_on_the_order_days_are_visited(
    regime: Regime,
) -> None:
    """Refits are cached; a cache from a *later* gate would be a leak.

    The backtest walks forwards, so this is the one property a forward-only
    test can never exercise. Deciding a day after having decided a later one
    must give what deciding it first gives.
    """
    early, late = dt.date(2025, 6, 1), dt.date(2025, 11, 1)
    prices, exog = _prices(regime), _exog(regime)
    window = utc_index(early, early + dt.timedelta(days=1), regime)

    forwards = _forecaster(prices, exog, regime).forecast(early, window)

    backwards = _forecaster(prices, exog, regime)
    backwards.forecast(late, utc_index(late, late + dt.timedelta(days=1), regime))
    revisited = backwards.forecast(early, window)

    assert forwards is not None and revisited is not None
    np.testing.assert_array_equal(forwards, revisited)


# --------------------------------------------------------------------------
# 3. Negative controls — the guards must be able to fail
# --------------------------------------------------------------------------


def test_a_one_day_lag_is_refused(regime: Regime) -> None:
    """D-1 has not finished at noon on D-1. Asking for it is not a tuning choice.

    Refused at construction rather than caught downstream, so a config file
    cannot introduce the bug this whole module exists to prevent.
    """
    with pytest.raises(ValueError, match="not finished when the gate closes"):
        build_features(
            _prices(regime), _exog(regime), regime, lead_days=0, lags_days=(1, 2, 3)
        )


def test_the_publication_check_catches_a_lag_that_reaches_too_far(
    regime: Regime,
) -> None:
    """Positive control for mechanism 1.

    The builder refuses a one-day lag, so the leak is forged directly on the
    table: the ``available_at`` of a lag column is moved one day later, which
    is what a lag of ``FIRST_COMPLETE_LAG_DAYS - 1`` would really have. If
    :meth:`FeatureTable.violations` returned nothing here it would be
    returning nothing everywhere.
    """
    table = build_features(
        _prices(regime), _exog(regime), regime, lead_days=0, lags_days=LAGS
    )
    column = f"price_lag{FIRST_COMPLETE_LAG_DAYS}d"
    table.available_at[column] = table.available_at[column] + pd.Timedelta(days=1)

    found = table.violations()

    assert [v.column for v in found] == [column]
    assert found[0].rows > 0
    assert found[0].worst_lateness > pd.Timedelta(0)


def test_the_perturbation_check_catches_a_forecaster_given_the_future(
    regime: Regime,
) -> None:
    """Positive control for mechanism 2.

    A forecaster whose training filter admits rows settled *after* the gate is
    the archetypal leak, and mechanism 2 must see it. Built by moving each
    row's settlement a year earlier, so the filter lets tomorrow through while
    every ``available_at`` still reads correct — which is precisely the case
    mechanism 1 cannot detect, and why there are two mechanisms.
    """
    day = dt.date(2025, 9, 10)
    gate = gate_close_utc(day)
    window = utc_index(day, day + dt.timedelta(days=1), regime)
    prices, exog = _prices(regime), _exog(regime)

    def leaky(series: pd.Series) -> PriceForecaster:
        model = _forecaster(series, exog, regime)
        for table in (*model._level.values(), *model._deviation.values()):
            object.__setattr__(
                table, "settled_at", table.settled_at - pd.Timedelta(days=365)
            )
        return model

    honest = leaky(prices).forecast(day, window)
    tampered = prices.copy()
    tampered.loc[tampered.index >= gate] = 999.0
    leaked = leaky(tampered).forecast(day, window)

    assert honest is not None and leaked is not None
    assert not np.array_equal(honest, leaked), (
        "the perturbation check did not notice a forecaster trained on prices "
        "that had not cleared at the gate; it is not guarding anything"
    )
