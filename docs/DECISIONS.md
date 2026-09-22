# Modelling decisions

<!-- toc -->
**Contents**

- [1. Scope](#1-scope)
  - [1.1 Time granularity — two regimes](#11-time-granularity--two-regimes)
  - [1.2 Market scope — day-ahead only](#12-market-scope--day-ahead-only)
  - [1.3 Asset](#13-asset)
- [2. Horizon protocol](#2-horizon-protocol)
  - [2.1 Rolling window](#21-rolling-window)
  - [2.2 Information set — the gate at 12:00 on D−1, read on publication time](#22-information-set--the-gate-at-1200-on-d1-read-on-publication-time)
  - [2.3 Bidding: a fixed schedule, and what a curve would add](#23-bidding-a-fixed-schedule-and-what-a-curve-would-add)
  - [2.4 The perfect-foresight bound](#24-the-perfect-foresight-bound)
  - [2.5 The floor: a no-information policy](#25-the-floor-a-no-information-policy)
  - [2.6 Warm-up](#26-warm-up)
- [3. Physical and economic model](#3-physical-and-economic-model)
  - [3.1 Formulation](#31-formulation)
  - [3.2 Why the binaries are not redundant](#32-why-the-binaries-are-not-redundant)
  - [3.3 Degradation cost — the parameter that sets everything](#33-degradation-cost--the-parameter-that-sets-everything)
  - [3.4 Network tariffs on charged energy — parametrised and measured](#34-network-tariffs-on-charged-energy--parametrised-and-measured)
  - [3.5 Price-taker](#35-price-taker)
- [4. Metrics](#4-metrics)
- [5. The forecaster](#5-the-forecaster)
- [6. Bid curves and joint scenarios](#6-bid-curves-and-joint-scenarios)
- [7. Data](#7-data)
- [8. Solver and modelling stack](#8-solver-and-modelling-stack)
- [9. What was deliberately not done](#9-what-was-deliberately-not-done)

<!-- /toc -->

Why this project is built the way it is. Each choice is recorded next to the
alternative it beat, and the ones that were tried and rejected are recorded too.
Numbers are from the committed runs in `results/`; the README carries the headline.



---

## 1. Scope

### 1.1 Time granularity — two regimes

| Regime | Period | Δt | Use |
|---|---|---|---|
| Hourly | 2022-01-01 → 2025-09-30 | 1 h | Forecaster training history, context result |
| Quarter-hourly | 2025-10-01 → 2026-08-22 | 0.25 h | Headline result |

Since delivery day 1 October 2025 the European day-ahead market clears in 15-minute
market time units, 96 prices a day. Working only in hourly data would describe a
market that no longer exists; working only in quarter-hourly data leaves under a year
of history. So the forecaster trains on both and the headline is quarter-hourly.

Two consequences the code respects. Δt is a parameter in every energy balance, and a
test asserts that four hourly periods and sixteen quarter-hourly ones spanning the
same four hours give the same optimum. And the two regimes' €/MW/year figures are not
comparable: finer granularity captures more spread, so the hourly result is context.

### 1.2 Market scope — day-ahead only

Day-ahead arbitrage is one part of a battery's revenue stack; reserve and intraday
markets are not modelled. Which part is modelled is stated as part of the result, in
the README's limitations.

### 1.3 Asset

10 MW / 20 MWh, 2-hour duration, grid-connected, standalone. A duration sweep at 1, 2
and 4 hours moves the level of every bar and not the forecast's margin over the floor
(+3.5, +3.0 and +2.7 points of bound), so the asset was not resized after seeing
results.

---

## 2. Horizon protocol

Horizon, terminal condition, information set and bound are one decision, governed by
one principle:

> **The bound differs from the policy in exactly one respect: the prices it sees.**
> Any other difference — horizon, constraints, cadence, terminal condition — would
> contaminate the ratio.

### 2.1 Rolling window

**48-hour window, first 24 hours implemented, state of charge carried forward.** This
removes the end-of-horizon artefact without inventing a terminal value function.

The window is encoded in days, not hours: `window_days: 2` and `implement_days: 1`.
Delivery days are 23 or 25 hours twice a year, so a two-day window is 47, 48 or 49
periods (188, 192 or 196 quarter-hourly), derived from the calendar. A backend is
built with a fixed period count, so the backtest holds one model instance per window
length, three at most, and re-solves them; the rule protected is no rebuild inside the
loop. Padding a short window with invented periods would put fabricated prices into
the optimisation.

Rejected: a free terminal state of charge, which empties the battery every midnight; a
cyclic condition, which forces unwinding even when the next day opens expensive; a
terminal value function, which is correct and one more parameter to justify.

### 2.2 Information set — the gate at 12:00 on D−1, read on publication time

The day-ahead auction closes at 12:00 CET on D−1 and OMIE publishes the result about
an hour later. **No variable used to decide on day D may have been published after
that gate.** The rule is read on publication time for every series, because that is
when a number becomes knowable:

- A delivery day's prices are published about 13:00 local on the day before delivery.
  At the gate for D every price through D−1 is known and none of D's is. Price lags
  therefore start at one day, and a zero-day lag is refused in code.
- REE's D+1 forecasts of demand, wind and solar are published the morning before the
  gate and cover D. Nothing covers D+1 at the gate, so the second horizon day is
  forecast from the calendar and lagged prices alone, by a separate model.
- Every feature is stamped with its publication instant, and the forecaster trains
  only on rows published before the gate it is fitted at.

"12:00 CET" is implemented as 12:00 Europe/Madrid. In summer that closes the
information set an hour earlier than a fixed UTC+1 would, so anything that passes
under this reading passes under the other. The 13:00 publication hour is taken on
trust; any hour between the auction close and the next gate gives the same admissible
set.

The gate is enforced by `tests/test_no_lookahead.py` in three ways of unequal
strength: the declared publication instants of every feature; a perturbation test that
rewrites every price and every exogenous forecast published after the gate and checks
that no forecast changes; and negative controls that hand the forecaster the future and
confirm each mechanism catches it.

### 2.3 Bidding: a fixed schedule, and what a curve would add

A charge and discharge schedule is committed and settled at realised prices, which is
the same as bidding as a price-taker at the market's price limits. A real participant
submits a price-quantity curve per period.

Whether a curve earns more was measured in §6. Constructed curves
lose because they clear period by period while a battery's dispatch is a trajectory.
A curve chosen by a two-stage stochastic MILP, with enough scenarios per price band
not to overfit, bids one quantity per period on the delivery day and, once its
scenario sample is centred on the forecast, settles to the cent against the fixed
schedule solved from the same state of charge on 129 of 139 curve days. The fixed
schedule is not a conservative simplification here; it is what the optimiser chooses
when offered the alternative.

### 2.4 The perfect-foresight bound

Same optimiser, same constraints, same window, same cadence, same state of charge
handling; realised prices instead of forecast prices. This is a structural guarantee:
every policy obtains its schedule from the same `BatteryMILP` instance.

**The value of horizon is zero, proven.** One extra solve separates the value of
horizon from the value of information: a single MILP over an annual window with the
state of charge free across days. The rolling oracle's dispatch is a feasible point of
that problem, so the annual optimum is at least what it earns; the annual solve's dual
bound says what it is at most.

| | Hourly, calendar 2024 | Quarter-hourly, 1 Oct 2025 – 22 Aug 2026 |
|---|---|---|
| Binaries | 8,784 | 31,296 |
| Annual solve, incumbent | 331,669.73 € | 530,146.32 € |
| Annual solve, dual bound | **331,670.108** € | **530,190.036** € |
| Rolling 48-hour oracle, same days | **331,670.108** € | **530,190.036** € |
| Relative gap reached | 1.15e−06 | 8.25e−05 |
| Wall clock, HiGHS, laptop | 13.5 s | 49.1 s |

Ceiling and floor are the same number to the last bit of a float64, in both regimes.
The whole gap between a policy and the bound is information. Both files carry the
unrounded values, the solver version and the machine.



### 2.5 The floor: a no-information policy

"The forecast captures 87% of the bound" means nothing without knowing that a
climatology captures 85%. The floor is the baseline, and it is what makes the
headline a testable claim.

The floor is a price vector, the hour-of-day × month average of every price published
before the gate, passed to the same optimiser. It is not a separate heuristic with its
own constraints. Additionally:

- **It reads the same history as the forecaster**, both regimes from January 2022. 
- **It expands and never rolls.** A lookback length would be a tuning knob on the baseline.
- **The key is local wall-clock time**, `(month, hour, minute)` in Europe/Madrid, so
  the solar trough and the evening peak sit where they are, and the 92- and 100-period
  transition days index correctly.

The floor falls behind the bound as the degradation cost rises, from 88.9% of the bound
at `c_deg` = 5 to 80.3% at 40. A higher cost is a higher minimum spread; the oracle
drops its marginal cycles and keeps the wide-spread days, and a floor that selects on
time of day alone cannot.

### 2.6 Warm-up

The first 7 days of every run are simulated and excluded from every metric, to remove
the arbitrary initial state of charge (50%).

---

## 3. Physical and economic model

### 3.1 Formulation

| Symbol | Value | Unit |
|---|---|---|
| `P_max` | 10 | MW at the connection point, symmetric |
| `E_max` | 20 | MWh usable |
| `η_rt` | 0.85 | AC-to-AC round trip, split as √0.85 each way |
| `c_deg` | 17 (5 / 17 / 40 sensitivity) | €/MWh discharged |
| `charge_tariff` | 0 (5 / 15 sensitivity) | €/MWh charged |

Variables per period: `p_c[t], p_d[t] ≥ 0` in MW, `soc[t]` in MWh, `u[t] ∈ {0,1}`.

```
soc[t] = soc[t-1] + η_c · p_c[t] · Δt − p_d[t] · Δt / η_d
0 ≤ soc[t] ≤ E_max
p_c[t] ≤ P_max · u[t]
p_d[t] ≤ P_max · (1 − u[t])

max Σ_t [ p_d[t]·Δt·(λ[t] − c_deg) − p_c[t]·Δt·(λ[t] + charge_tariff) ]
```

The big-M on the exclusivity constraints is `P_max`, the physical rating, so it adds no
slack beyond the variable bounds. Degradation is charged on discharge only, following
Xu et al.: a discharging half-cycle causes the ageing of a full cycle of that depth,
and charged and discharged energy are nearly equal daily, so charging throughput
twice would double-count.

The golden case that every refactor must reproduce: prices `[10, 10, 100, 100]`,
Δt = 1, empty battery, `c_deg` = 0, optimum €1,500 = 20 MWh × (0.85 × 100 − 10).
The test asserts the objective, not the dispatch, because with tied prices the
optimal dispatch is not unique.

### 3.2 Why the binaries are not redundant

With positive prices and `η_rt < 1`, charge and discharge exclusivity holds at the LP
optimum on its own. With negative prices it does not: charging and discharging at once
burns energy and is paid for it. To hold the state of charge flat the relaxation
discharges 0.85 of what it charges and collects `0.15 · |λ|` per MW of fictitious
throughput, €7.50/MWh at λ = −50. Verified: with a full battery at λ = −50 the
relaxation scores 162.16 against the MILP's 150.00 and charges and discharges in the
same period. Spain sees negative prices with increasing frequency, so the binaries
stay, for the bound as well.

Every window closes at the root node in practice: presolve and the root heuristics
find and prove the integer optimum before branching, so a window solve costs an LP
plus rounding and the run's wall clock sits in model handling, not in the solver.

### 3.3 Degradation cost — the parameter that sets everything

A linear cost on discharged throughput. Central value **€17/MWh**, from cells at
about €100/kWh, 6,000 cycles to 70% end of life, and 20 MWh: €2M over 120,000 MWh of
lifetime throughput. Instead of guessing on the cell price, the result is a
three-point sensitivity, 5 / 17 / 40 €/MWh, the range published values span.

A linear throughput cost is a minimum spread threshold: the battery cycles only when
`λ_sell · η_d − λ_buy / η_c > c_deg`. So `c_deg` sets cycles per year, and cycles per
year are reported on every run as the sanity check. 422 a year for the forecast at the
central value is a little over one a day, which is what a 2-hour asset in a market with
one solar trough and one evening peak should do; 700 would say the cost is too low.

Choosing a throughput model over rainflow cycle counting is choosing the optimistic
model. Rainflow is more faithful and has no analytical form to embed in the
optimisation.

Reference: B. Xu et al., *Factoring the Cycle Aging Cost of Batteries Participating in
Electricity Markets*, IEEE Trans. Power Systems 33(2), 2018.

### 3.4 Network tariffs on charged energy — parametrised and measured

CNMC Circular 3/2020 exempts grid-connected storage from network access tolls;
whether the exemption covers charging, and how the separate *cargos* apply to storage,
is under consultation for the 2026–2031 period. The
tariff is a parameter set to zero, and its sensitivity is measured: at 5 and 15 €/MWh
every bar falls, the floor fastest, and the forecast's margin over the floor stays at
about €2,000/MW/year.

### 3.5 Price-taker

10 MW is negligible against OMIE's cleared volume. The assumption stops holding for a
fleet.

---

## 4. Metrics

| Metric | Definition | Role |
|---|---|---|
| Share of the gap | (policy − floor) ÷ (bound − floor) over the same days | Headline |
| €/MW/year | Annualised net profit ÷ 10 MW | Level |
| % of bound | Policy profit ÷ oracle profit | Context |
| Equivalent cycles/year | Discharged energy ÷ E_max ÷ years | Sanity check |

**The share of the gap leads** because the floor sits at 85% of the bound: a percentage
of bound credits the forecast with 87 points of which it earned 3. The share puts the
floor at 0 and the bound at 100. It is undefined when the gap is not positive, the same
way % of bound is undefined when the bound earns nothing.

**It comes with an interval.** A paired circular block bootstrap over delivery days:
the same days drawn for all three policies, in 7-day blocks that wrap around the run,
10,000 resamples, 95% percentile interval, fixed seed. Paired, because the day-to-day
variation the three policies share cancels in the ratio. Blocks, because a day's
profit depends on the days before it through the carried state of charge and through
weather and fuel prices that persist. The block length is a judgement, so 14- and
28-day blocks are reported beside it, and a margin is called distinguishable from zero
only when the interval clears zero at all three. The share of resamples at or below the
floor is reported as a summary of that distribution, not as a p-value.

**Decision and settlement are separate.** A schedule is decided against the prices the
policy believed and settled against the prices that cleared; the difference is the
cost of imperfect information, which is the quantity measured. Settlement never reads
the solver's objective.

**Years come from the days.** Hours are summed from each day's own period count, so
23- and 25-hour days contribute what they were.

**Forecast error is secondary.** RMSE and rank correlation are reported after the
economics. A better score that does not turn into captured spread is not an
improvement, and the runs bear that out: the ordering of periods within a day is what
arbitrage uses.

---

## 5. The forecaster

**LightGBM, two stages, two horizon days, causal refits, the floor as fallback.**
Deliberately plain: the hyperparameters were written once and left, and the value of
the project is in the optimiser, not in the forecaster.

- **Two stages.** An hourly *level* model over the whole history since 2022, plus an
  intra-hour *deviation* model over the 15-minute history since October 2025, whose
  target is each quarter's departure from its hour's mean. Neither of the obvious
  options works alone: training on 15-minute data leaves under a year, and training
  hourly and disaggregating throws away the intra-hour spread the market now clears.
- **Two horizon days, two information sets.** REE's D+1 forecasts cover D and not D+1,
  so the second day is a separate model on calendar and lagged prices alone.
- **Features** are the calendar, the price at the same local time 1 to 8 days
  earlier, daily and hourly price aggregates, and the exogenous demand, wind and solar
  forecasts, every column stamped with its publication instant.
- **Refits** every 30 days on every row published before the gate, with the round
  count chosen by early stopping on the most recent 60 days. Each stage has its own
  threshold and its own consequence: the level stage needs a year of history, and
  until it has one the policy bids the floor's prices; the deviation stage needs 30
  days of genuine 15-minute history, and until it has them the day is priced from the
  hourly level alone. Both counts are written to the run summary rather than
  described: 368 fallback days on the hourly regime, 30 level-only days on the
  headline one.
- **The fallback is the floor**, so a day the forecaster declines is a day scored as
  the baseline, never as zero.

Tried on the same evaluation days and not adoptedisted: 730- and 365-day rolling training windows, which lose 1.1
to 1.5 points of bound; a 365-day half-life decay and a shape-targeted second stage,
both within half a point, inside the paired interval's width. Four alternative seeds
put the headline share between 14.2% and 20.3%.

---

## 6. Bid curves and joint scenarios

Three constructions tested whether a price-contingent bid earns more than a fixed
schedule. The first two are in the shipped code, reachable from
`run_backtest(bidding=...)`; the shared-band variant of the third is a prototype
outside it, which reproduces the shipped formulation at one band per scenario to 1e-6.

**Curves traced from forecast quantiles.** One solve per quantile level, assembled
into a price-quantity curve per period, cleared at the realised prices, then clipped
to what the state of charge can deliver. Lost, because clearing each period against
its own curve assembles a trajectory no single scenario endorsed. Measured before the
information set was corrected and not re-run, so it carries no figure here; the joint
constructions below share its clearing path and lost by more.

**Joint scenarios from each policy's own errors.** The forecast object a battery needs
is a joint trajectory over the window, not a set of marginal quantiles, since the
decision depends on the ordering of prices within the day. Each scenario is a past
decision's whole error trajectory, recorded as the belief is formed and derived once
the window has published, so cross-period dependence is exact rather than modelled.
The pool admits only windows published before the gate, and its mean is subtracted so
the scenario set's expectation is the policy's own point forecast.

**The curve as a decision: a two-stage stochastic MILP** (`model/curve_pyomo.py`). The
first stage is the curve, a net quantity per period and price band; the recourse is
each scenario's dispatch under the physical model of §3.1; the linking constraint
reads the band each scenario's price clears in, which is data, so the program is one
MILP. It is feasible by construction on every sampled acceptance pattern and can never
be worse than the fixed schedule in sample.

**What it bought.** With one price band per scenario, each step is fitted to a single
sampled day and the curve loses 20.5%. With 20 scenarios sharing 3 bands it bids one
quantity per period on the delivery day in all but a handful of periods and comes
within half a point of the fixed schedule; with the scenario sample centred on the
forecast it settles to the cent against the fixed schedule solved from the same state
of charge on 129 of the 139 curve days, and the 0.1% that remains against the
schedule's own run is the two state-of-charge paths separating on the ten days that
differ. The forecaster's
errors are about half common-mode across the day (mean cross-period correlation 0.48):
a surprise mostly shifts the day's level, which moves the price and not the optimal
dispatch. Every stage-2 instance solves at the root node, so the formulation's cost
grows with the root LP and not with branching.

---

## 7. Data

- **Source:** the ESIOS API (Red Eléctrica de España), cross-checked against OMIE's
  published files. Four indicators: the day-ahead marginal price (600, geography
  Spain) and REE's D+1 forecasts of demand, wind and solar (1775, 1777, 1779).
- **Downloaded once and frozen.** The snapshot is committed as Parquet with a manifest
  of ranges, row counts and checksums, so a clone reproduces every result without a
  token. The loader validates and refuses; it never fills a gap.

- **Only the D+1 forecast family is usable.** ESIOS also serves rolling forecasts that
  are rewritten as the day approaches; a historical query returns the last value
  written. Measured: before a day happens the two families agree to 126 MW RMSE;
  after it, the rolling one has moved toward what occurred. Using it would hand the
  forecast policy information the bound is supposed to be alone in having.
- **Granularity is observed, not assumed.** ESIOS moved the price onto a 15-minute
  grid nine months before the market did; through that stretch the four values in
  each hour are identical and the snapshot downsamples them only because it is
  provably lossless. The regime boundary is confirmed from the data: the maximum
  spread within an hour is 0.00 €/MWh until 30 September 2025 and €65.50 on 1 October.
- **Time zone.** Stored in UTC, presented in Europe/Madrid; DST days have 23 or 25
  hours, and a test covers the transition days.

---

## 8. Solver and modelling stack

| Component | Choice |
|---|---|
| Modelling layer | Pyomo, persistent interface |
| Backend architecture | Pluggable behind a `Protocol`; no solver import outside `model/` |
| Solver | HiGHS, so anyone can clone and run; Gurobi behind a config key |
| Forecasting | LightGBM |
| Environment | `uv` with a committed lockfile; CI runs lint, types, import contracts and the tests |



---

## 9. What was deliberately not done

- No single annual MILP as the bound: it would measure the value of horizon, which is
  measured separately and is zero.
- No relaxed constraints in the bound, and no LP relaxation anywhere: with negative
  prices it inflates the denominator.
- No realised wind, solar or demand as forecaster inputs, and no rolling forecast
  series: both are lookahead.
- No lookback length on the floor, and no tuning of the forecaster on the evaluation
  days: the baseline and the result are not moved to fit each other.
- No second market before the day-ahead result is closed.
- No single degradation cost defended; a three-point sensitivity instead.
- No model rebuilt inside the backtest loop, and no solver import outside `model/`.
