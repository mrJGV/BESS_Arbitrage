"""The point forecaster: LightGBM, deliberately boring.

``docs/DECISIONS.md`` §6 lists the forecaster as "LightGBM, deliberately
boring" — the contribution is the decision layer, not this. Nothing here is
tuned, stacked or ensembled. What it does have to get right is **when** each
number was knowable — that lives next door in
:mod:`bess_arb.forecast.features` — and **which data a fit was allowed to
see**, which lives here.

Two stages, and why
-------------------

The headline regime is quarter-hourly, and the deferred question was whether
to train on the 10.5 months of quarter-hourly history or on the four years of
hourly history and disaggregate. Measured on the last six months of the
snapshot, with everything before it as training data:

===================================  =====  ======  ================
arm                                   RMSE   daily   mean intra-hour
                                             rho     spread predicted
===================================  =====  ======  ================
quarter-hourly history only          27.42   0.902        8.96
hourly history, applied to quarters  20.15   0.912        n/a
both, pooled onto the quarter grid   20.59   0.921        4.68
**level + intra-hour deviation**     19.60   0.919       10.11
===================================  =====  ======  ================

against a realised mean intra-hour spread of €12.83/MWh. The answer is
neither of the two the question offered.

**Pooling alone has a structural defect, not a small one.** Before 1 October
2025 the published series is on a 15-minute grid whose four values inside
each hour are *identical* — measured, maximum spread 0.0000 €/MWh
(``docs/DECISIONS.md`` §5.4). So 88% of a pooled training set states, correctly, that
intra-hour spread is zero, and the model learns to predict a quarter of the
spread that now exists. It is not that the pooled model is 1 €/MWh worse; it
is that it cannot represent the thing the quarter-hourly regime added.

**So the two questions are separated and each is asked of the data that can
answer it.** A *level* model predicts the hourly mean price from four and a
half years of history. A *deviation* model predicts each quarter's departure
from its own hour's mean, trained only on the period where that quantity is
not identically zero. The forecast is their sum. The level stage has the long
history it needs for seasonality; the deviation stage has an easy, mean-zero
target and a short window is enough for it.

On the hourly regime the deviation is zero by construction and the second
stage is not built at all — the same code path, with the degenerate case
falling out rather than being special-cased.

The economic difference between pooling and splitting is small (91.1% of the
bound against 90.7% over the same 196 days), which is worth saying plainly:
the split is not chosen for the 0.4 points. It is chosen because a single
pooled model systematically under-predicts a spread the market really has,
and that is a defect a reader would find.

Refits are causal
-----------------

A model used to decide day D may be fitted only on prices that had settled by
``gate(D)``. :meth:`FeatureTable.trainable_before` is that filter; this module
decides *when* to re-run it. Refitting every day would be honest and slow, so
the fit is reused for ``refit_days`` and then redone — still causal, merely
staler, and the staleness is bounded by a number in the config rather than by
whatever was convenient.

**What that bound costs is measured, not assumed.** On the 319-day
quarter-hourly run at ``c_deg`` = 17, dropping ``refit_days`` from 30 to 1
moves the policy from 87.25% to 87.44% of the rolling oracle — €111/MW/year —
for 1,238 fits in place of 40. A fifth of a point is the price of the
cadence, and it is small against the 1.6 points of margin the policy has over
the floor.

Days visited out of order refit unconditionally. The backtest walks forwards,
but the tests deliberately do not, and a cached fit from a *later* gate is
exactly the leak this whole module exists to prevent.

The round count is data, not config
-----------------------------------

Each fit holds out the most recent ``validation_days`` of its own trainable
rows and stops when validation loss stops improving; ``level_rounds`` and
``deviation_rounds`` are ceilings on that search rather than the number
fitted. The fold is carved from rows ``trainable_before`` has already
admitted, so it is behind the gate like everything else and adds no lookahead
surface — see :func:`_fit_stage`.

This replaced a fixed 600 and 400. Measured over six cutoffs from 2024-01-01
to 2026-02-01 the level stage wants 56-157 rounds, so the old value was around
four times past the point where validation loss stops improving. Over the
319-day quarter-hourly backtest the chosen counts average 128, and correcting
this moved the forecast policy from 86.8% to 87.3% of the rolling oracle at
``c_deg`` = 17 — the margin over the floor from 1.1 points to 1.6.

The count varies threefold across those cutoffs, which is why it is decided
per fit rather than written down as a better constant. A hardcoded 133/47 was
measured higher still, at 87.9%, but only in combination with the deviation
stage's unfiltered training set; alongside the filter that stage requires, it
is not. The higher number is not the one to reach for — the same reasoning
that took the two-stage split at 91.1% over pooling at 90.7% on structural
grounds rather than on the 0.4 points.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bess_arb.forecast.features import (
    TIME_OF_DAY_KEYED,
    FeatureTable,
    build_features,
)
from bess_arb.model.spec import FloatArray
from bess_arb.timeline import MARKET_TZ, Regime, gate_close_utc

__all__ = ["ForecastSpec", "PriceForecaster", "Stage"]


@dataclass(frozen=True, slots=True)
class ForecastSpec:
    """Everything the forecaster reads from ``config/params.yaml``.

    Held as one object so that no number in this module is written in code
    (CLAUDE.md invariant 7), including the LightGBM hyperparameters — they are
    numeric choices like any other and belong in the file a reader checks.
    """

    lags_days: tuple[int, ...]
    refit_days: int
    min_train_days: int
    min_deviation_days: int
    level_params: dict[str, Any]
    deviation_params: dict[str, Any]
    level_rounds: int
    deviation_rounds: int
    validation_days: int
    early_stopping_rounds: int
    seed: int
    threads: int

    def __post_init__(self) -> None:
        if self.refit_days < 1:
            raise ValueError(f"refit_days must be at least 1, got {self.refit_days}")
        if self.min_train_days < 1:
            raise ValueError(
                f"min_train_days must be at least 1, got {self.min_train_days}"
            )
        if self.min_deviation_days < 1:
            raise ValueError(
                f"min_deviation_days must be at least 1, got {self.min_deviation_days}"
            )
        for name, rounds in (
            ("level_rounds", self.level_rounds),
            ("deviation_rounds", self.deviation_rounds),
            ("early_stopping_rounds", self.early_stopping_rounds),
        ):
            if rounds < 1:
                raise ValueError(f"{name} must be at least 1, got {rounds}")
        if self.validation_days < 1:
            raise ValueError(
                f"validation_days must be at least 1, got {self.validation_days}"
            )
        if self.threads < 1:
            raise ValueError(f"threads must be at least 1, got {self.threads}")

    def booster_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Hyperparameters with the reproducibility keys pinned on top.

        ``seed``, ``deterministic`` and ``force_row_wise`` are set here rather
        than left to the YAML: invariant 8 asks for fixed seeds everywhere,
        and a run that cannot be repeated is not evidence. ``num_threads`` is
        pinned for the same reason — LightGBM's histogram construction is
        thread-count dependent without it.
        """
        return {
            **params,
            "objective": "regression",
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": self.threads,
            "verbosity": -1,
        }


@dataclass(frozen=True, slots=True)
class Stage:
    """One fitted booster, with what it was fitted on."""

    booster: lgb.Booster
    columns: tuple[str, ...]
    rows: int
    cutoff: pd.Timestamp
    """The gate the fit was allowed to see up to. Never later than the gate of
    any day this stage is then used to decide."""

    rounds: int
    """Boosting rounds actually fitted — chosen by the validation fold, or the
    ceiling where the history could not spare one."""

    validated: bool
    """Whether ``rounds`` was chosen by early stopping rather than taken from
    the ceiling. Counted in :meth:`PriceForecaster.diagnostics`, because a fit
    that fell back is a weaker fit and the ladder already reports its rungs."""

    def predict(self, values: pd.DataFrame) -> FloatArray:
        raw = self.booster.predict(values.loc[:, list(self.columns)])
        return np.asarray(raw, dtype=np.float64)


class PriceForecaster:
    """Point forecasts of day-ahead prices, for one regime.

    Construct once and reuse across a degradation sweep: the forecast does not
    depend on ``c_deg``, and refitting per sweep point would multiply the cost
    of a run by three for identical numbers.
    """

    def __init__(
        self,
        prices: pd.Series,
        exog: pd.DataFrame,
        regime: Regime,
        level_regime: Regime,
        spec: ForecastSpec,
    ) -> None:
        if level_regime.dt_h < regime.dt_h:
            raise ValueError(
                f"the level grid ({level_regime.dt_h} h) must be at least as "
                f"coarse as the target grid ({regime.dt_h} h)"
            )
        self._regime = regime
        self._level_regime = level_regime
        self._spec = spec
        self._two_stage = level_regime.dt_h > regime.dt_h

        level_prices = (
            prices.resample(level_regime.step).mean() if self._two_stage else prices
        )
        self._level = {
            lead: build_features(
                level_prices,
                exog,
                level_regime,
                lead_days=lead,
                lags_days=spec.lags_days,
            )
            for lead in (0, 1)
        }
        self._deviation = (
            {
                lead: _to_deviation(
                    build_features(
                        prices, exog, regime, lead_days=lead, lags_days=spec.lags_days
                    ),
                    level_regime,
                )
                for lead in (0, 1)
            }
            if self._two_stage
            else {}
        )

        self._level_stage: dict[int, Stage] = {}
        self._deviation_stage: dict[int, Stage] = {}
        self._fitted_gate: pd.Timestamp | None = None
        self._predictions: dict[int, dict[pd.Timestamp, float]] = {0: {}, 1: {}}
        # Keyed rather than counted, because one forecaster serves a whole
        # degradation sweep: every day is forecast three times, identically,
        # and a running total would report three times the periods and read as
        # three times the work. Keys make the tallies idempotent.
        self._rung: dict[int, dict[pd.Timestamp, str]] = {0: {}, 1: {}}
        self._declined: set[dt.date] = set()
        # One entry per fitted stage: the rounds the validation fold chose and
        # whether it got to choose at all. Reported with the result because a
        # count that sits at its ceiling means the ceiling bound the search,
        # which is the one way this can silently go back to fitting blind.
        self._fits: list[Stage] = []

    # -- public surface ----------------------------------------------------

    @property
    def two_stage(self) -> bool:
        """Whether an intra-hour deviation stage exists for this regime."""
        return self._two_stage

    def forecast(self, day: dt.date, window: pd.DatetimeIndex) -> FloatArray | None:
        """Forecast prices over ``window``, deciding at ``gate(day)``.

        Returns ``None`` when the history behind the gate is shorter than
        ``min_train_days`` — the caller decides what a policy with no usable
        model should do, because that is a policy question and not a
        modelling one.
        """
        gate = gate_close_utc(day)
        if not self._fit(gate):
            self._declined.add(day)
            return None

        stamps = _nanoseconds(window)
        out = np.full(len(stamps), np.nan)
        target_day = pd.Index(stamps.tz_convert(MARKET_TZ).date)
        for lead in (0, 1):
            rows = np.asarray(target_day == day + dt.timedelta(days=lead))
            if not rows.any():
                continue
            out[rows] = self._predict(lead, pd.DatetimeIndex(stamps[rows]))

        missing = int(np.isnan(out).sum())
        if missing:
            raise ValueError(
                f"the forecast for delivery day {day} left {missing} of "
                f"{len(window)} periods unfilled; the window reaches beyond "
                "the two horizon days the forecaster builds tables for"
            )
        return out

    def diagnostics(self) -> dict[str, int]:
        """Periods served by each rung of the ladder, reported with the result.

        ``level_only`` counts periods forecast without the intra-hour stage,
        because the quarter-hourly history behind the gate was still too short
        for one; ``no_history`` counts the days the forecaster declined
        outright. Both weaken the policy, and both weaken it in the flattering
        direction, so both are counted rather than described.

        ``mean_rounds`` and ``unvalidated_fits`` do the same for the round
        count: a mean sitting at the ceiling means the ceiling bound the
        search rather than the data ending it, and an unvalidated fit is one
        whose history could not spare a fold. Either is the failure that would
        otherwise return this stage to fitting blind.
        """
        served = [rung for rungs in self._rung.values() for rung in rungs.values()]
        rounds = [stage.rounds for stage in self._fits]
        return {
            "trained": sum(1 for rung in served if rung == "trained"),
            "level_only": sum(1 for rung in served if rung == "level_only"),
            "no_history": len(self._declined),
            "fits": len(self._fits),
            "mean_rounds": int(sum(rounds) / len(rounds)) if rounds else 0,
            "max_rounds": max(rounds, default=0),
            "unvalidated_fits": sum(1 for stage in self._fits if not stage.validated),
        }

    def predictions(self, lead_days: int = 0) -> pd.Series:
        """Every forecast made at horizon ``lead_days``, for the error table.

        Lead 0 is the one that matters: it prices the day the backtest
        actually implements. ``docs/DECISIONS.md`` §4.1 — RMSE is reported,
        and it is not the metric.
        """
        made = self._predictions[lead_days]
        index = pd.DatetimeIndex(list(made), name="datetime_utc")
        return pd.Series(
            list(made.values()), index=index, dtype="float64", name="forecast_eur_mwh"
        ).sort_index()

    # -- fitting -----------------------------------------------------------

    def _fit(self, gate: pd.Timestamp) -> bool:
        """Ensure a model exists that saw nothing after ``gate``.

        A cached fit is reused only if it is recent enough *and* was cut at or
        before this gate. The second condition is what makes an out-of-order
        visit safe; see the module docstring.
        """
        current = self._fitted_gate
        stale = (
            current is None
            or gate < current
            or (gate - current) >= pd.Timedelta(days=self._spec.refit_days)
        )
        if not stale:
            return bool(self._level_stage)

        level: dict[int, Stage] = {}
        for lead, table in self._level.items():
            rows = table.trainable_before(gate)
            if not _spans_enough(table, rows, self._spec.min_train_days):
                # Nothing is cached on this path, so the next day tries again
                # rather than waiting out a refit interval. The cold start is
                # short and the check is a boolean mask; caching it would
                # extend the fallback by up to `refit_days` for no saving
                # worth having.
                self._level_stage = {}
                self._deviation_stage = {}
                self._fitted_gate = None
                return False
            level[lead] = _fit_stage(
                table,
                rows,
                self._spec.booster_params(self._spec.level_params),
                self._spec.level_rounds,
                gate,
                self._spec.validation_days,
                self._spec.early_stopping_rounds,
            )

        deviation: dict[int, Stage] = {}
        for lead, table in self._deviation.items():
            rows = table.trainable_before(gate)
            if _spans_enough(table, rows, self._spec.min_deviation_days):
                deviation[lead] = _fit_stage(
                    table,
                    rows,
                    self._spec.booster_params(self._spec.deviation_params),
                    self._spec.deviation_rounds,
                    gate,
                    self._spec.validation_days,
                    self._spec.early_stopping_rounds,
                )

        self._fits.extend(level.values())
        self._fits.extend(deviation.values())
        self._level_stage = level
        self._deviation_stage = deviation
        self._fitted_gate = gate
        return True

    def _predict(self, lead: int, window: pd.DatetimeIndex) -> FloatArray:
        """Level plus deviation for one horizon day's periods."""
        level_table = self._level[lead]
        hours = window.floor(self._level_regime.step) if self._two_stage else window
        values = level_table.values.reindex(hours)
        if values.isna().all(axis=1).any():
            raise KeyError(
                f"the level feature table has no row for "
                f"{list(hours[values.isna().all(axis=1)][:3])}"
            )
        out = self._level_stage[lead].predict(values)

        stage = self._deviation_stage.get(lead)
        if stage is not None:
            out = out + stage.predict(self._deviation[lead].values.reindex(window))
            rung = "trained"
        else:
            rung = "level_only" if self._two_stage else "trained"

        self._predictions[lead].update(zip(window, out, strict=True))
        self._rung[lead].update(dict.fromkeys(window, rung))
        return np.asarray(out, dtype=np.float64)


def _nanoseconds(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """A UTC index at nanosecond resolution, whatever came in.

    :func:`bess_arb.timeline.utc_index` builds microsecond timestamps, and the
    feature tables are pinned to nanoseconds. Reindexing across the two is not
    wrong — pandas converts — but it converts *the 160,000-row table*, once per
    lookup, and throws away the hash engine it just built each time. Left
    alone it costs about half a second per horizon day, which is the whole
    backtest. Aligning here is the same unit trap the floor policy documents,
    arriving as a performance bug instead of a correctness one.
    """
    return pd.DatetimeIndex(index).as_unit("ns")


def _spans_enough(table: FeatureTable, rows: pd.Series, days: int) -> bool:
    """Whether the trainable rows cover at least ``days`` of calendar history.

    Measured as a span rather than a row count so the answer means the same
    thing on both grids: 5,000 rows is seven months of hourly history and
    seven weeks of quarter-hourly.
    """
    if not rows.any():
        return False
    stamps = table.values.index[rows.to_numpy()]
    return bool((stamps[-1] - stamps[0]) >= pd.Timedelta(days=days))


def _fit_stage(
    table: FeatureTable,
    rows: pd.Series,
    params: dict[str, Any],
    ceiling: int,
    cutoff: pd.Timestamp,
    validation_days: int,
    early_stopping_rounds: int,
) -> Stage:
    """One booster, with the round count decided by a held-out recent fold.

    ``ceiling`` bounds the search; it is not the number fitted. Measured over
    six cutoffs, the level stage wants 56-157 rounds where 600 used to be
    fitted unconditionally — some four times past the point where validation
    loss stops improving, which cost accuracy as well as time. The count also
    moves threefold across those cutoffs, so it is chosen per fit rather than
    written into the config as a constant.

    **The fold introduces no lookahead.** It is carved from the rows
    :meth:`FeatureTable.trainable_before` has already admitted, every one of
    which settled at or before ``cutoff``; splitting an admissible set by time
    cannot make any part of it inadmissible. What the split does cost is the
    most recent stretch of history, which is also the most informative — hence
    the fallback below rather than a fold taken at any price.
    """
    mask = rows.to_numpy()
    values = table.values.loc[mask]
    target = table.target.loc[mask]

    index = pd.DatetimeIndex(values.index)
    fold = _validation_fold(index, validation_days)
    if fold is None:
        return Stage(
            booster=lgb.train(
                params, lgb.Dataset(values, target), num_boost_round=ceiling
            ),
            columns=tuple(values.columns),
            rows=int(mask.sum()),
            cutoff=cutoff,
            rounds=ceiling,
            validated=False,
        )

    is_validation = np.asarray(index >= index.max() - fold)

    searched = lgb.train(
        params,
        lgb.Dataset(values.loc[~is_validation], target.loc[~is_validation]),
        num_boost_round=ceiling,
        valid_sets=[lgb.Dataset(values.loc[is_validation], target.loc[is_validation])],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    rounds = int(searched.best_iteration)

    # The fold decides the count and is then given back. Keeping the
    # early-stopped booster instead looked defensible on a single split -- it
    # scored 18.63 against the refit's 18.87 -- but over the rolling backtest
    # it loses 0.4 points of the bound, because every fit is then blind to its
    # own most recent 60 days, which are the most informative it has. The
    # second fit costs what it costs; at a chosen count averaging 128 rounds
    # rather than the 600 this replaced, two fits remain cheaper than one was.
    return Stage(
        booster=lgb.train(params, lgb.Dataset(values, target), num_boost_round=rounds),
        columns=tuple(values.columns),
        rows=int(mask.sum()),
        cutoff=cutoff,
        rounds=rounds,
        validated=True,
    )


def _to_deviation(table: FeatureTable, level_regime: Regime) -> FeatureTable:
    """Re-express a target-grid table as the intra-hour deviation problem.

    Target: the period's price minus its own hour's mean. The columns keyed on
    local time of day — the lagged prices — are re-expressed the same way, so
    the deviation stage is asked only about the part of the signal the level
    stage did not already carry. Everything else stays as it is: a daily or
    hourly aggregate cannot explain a *within*-hour departure directly, but it
    conditions one, which is why they are kept rather than dropped. A solar
    ramp hour has a wider intra-hour spread than a night hour, and residual
    demand is how the model can tell them apart.

    Publication instants come across untouched, and that is correct rather
    than convenient: subtracting an hour's mean of a column mixes only values
    from within that same column, all of them from one lag day, so nothing
    becomes knowable later than it already was. It is also why the deviation
    table goes through the no-lookahead test like any other.

    **``settled_at`` does move, and it has to.** A quarter's price settles when
    its own period ends, but its *deviation* is not known until the hour it
    belongs to is complete — it needs the other three quarters. So the
    settlement instant becomes the hour's end. On this snapshot the distinction
    happens to be inert, because the gate falls on a clock hour and hours are
    aligned to it, so no quarter is ever admitted while its hour is still
    running. Relying on that would be relying on a coincidence of two
    unrelated conventions; a gate at half past would break it silently.

    **The target is withheld before the deviation exists at all.** Ahead of the
    market's move to 15-minute units, :func:`bess_arb.series.load_price_history`
    reconstructs the quarter grid by repeating each hourly price across its four
    quarters — exact reconstruction, but it makes each quarter's departure from
    its own hour's mean identically zero. Left in, those rows are 92% of the
    training set (measured at the 2026-02-01 gate: 131,696 of 143,088), and a
    booster fitted through them learns that intra-hour spread is nearly absent.
    That is precisely the defect the two-stage split exists to avoid, so
    admitting it *inside* the deviation stage would reproduce the pooled arm
    this design rejected.

    They are removed by setting the target to ``NaN``, not by dropping rows:
    :meth:`FeatureTable.usable` already excludes a row with no target, so the
    existing training filter does the work, while ``values`` stays complete
    because the stage still has to *predict* on that grid.

    The boundary is read off the data rather than taken from a date in the
    config — it is the first instant the deviation is non-zero, and everything
    before it is structurally zero by the paragraph above. A row that happens
    to be flat *after* that instant is a real observation and is kept, which is
    why this is a boundary and not a row-by-row filter on the value.

    One consequence worth stating rather than discovering: ``min_deviation_days``
    now measures the span of genuine quarter-hourly history, which is what it
    was written to mean. Previously the zero era gave it a four-year span to
    look at, so it could never bind.
    """
    hour = pd.DatetimeIndex(table.values.index).floor(level_regime.step)
    values = table.values.copy()
    for column in values.columns:
        if column.startswith(TIME_OF_DAY_KEYED):
            values[column] = values[column] - values.groupby(hour)[column].transform(
                "mean"
            )
    target = table.target - table.target.groupby(hour).transform("mean")
    return FeatureTable(
        values=values,
        available_at=table.available_at,
        gate=table.gate,
        settled_at=pd.Series(
            hour + level_regime.step, index=table.values.index, name="settled_at"
        ),
        target=target.mask(_structurally_zero(target)),
        lead_days=table.lead_days,
        regime=table.regime,
    )


def _validation_fold(
    index: pd.DatetimeIndex, validation_days: int
) -> pd.Timedelta | None:
    """How much recent history to hold out, or ``None`` to fit the ceiling blind.

    The fold has to be the most *recent* stretch to proxy the horizon a fit will
    serve, so every day spent on validation is a day the model does not learn
    from — and they are the most informative days it has. On a long history that
    is a rounding error. On a short one it is the whole question.

    It became the whole question when the deviation stage stopped training on
    the reconstructed era: its genuine history runs from 2025-09-30 rather than
    2022-01-01, so a 60-day fold that was always affordable is now unaffordable
    for the first months of the backtest. Measured on the 319-day quarter-hourly
    run, six fits fell through to the ceiling and fitted 400 rounds unvalidated.

    **The fold keeps its configured length; the training side is what gives.**
    Validation is admitted as soon as what remains is at least half a fold,
    rather than a whole one. So an early fit may learn from as little as 30
    days, but the round count it chooses is still measured against the full 60
    — which is the right way round, because the round count is the only thing
    the fold is there to decide, and an estimate taken on a fold shrunk to
    match a short history would be noisy exactly when the history can least
    afford a bad one.

    In span terms the gate moves from twice the fold to one and a half times
    it: 90 days of history rather than 120. With ``min_deviation_days`` at 30
    that narrows the stretch where the stage exists but cannot be validated
    from 90 days to 60. It does not close it, and the residual is reported as
    ``unvalidated_fits`` rather than argued away.
    """
    fold = pd.Timedelta(days=validation_days)
    span = index.max() - index.min()
    return fold if span - fold >= fold / 2 else None


#: Below this, a deviation is the arithmetic of four identical prices rather
#: than a quantity the market set. Repeating a float across four quarters and
#: subtracting their mean is exact in binary floating point, so the true values
#: are 0.0; the tolerance is there so the test does not depend on that being
#: true of every pandas reduction path as well.
DEVIATION_ZERO_TOL = 1e-9


def _structurally_zero(target: pd.Series) -> NDArray[np.bool_]:
    """Rows before the intra-hour deviation became a real quantity.

    Everything up to the first non-zero deviation, because that is the era the
    quarter grid was reconstructed from hourly prices. Returns an all-true mask
    when the series is flat throughout — a regime with no intra-hour structure
    has no deviation stage to fit, and the caller's ``min_deviation_days`` check
    then declines it rather than fitting a column of zeros.
    """
    real = np.flatnonzero(np.abs(target.to_numpy()) > DEVIATION_ZERO_TOL)
    if not len(real):
        return np.ones(len(target), dtype=bool)
    return np.asarray(np.arange(len(target)) < real[0])
