# Modelling decisions

<!-- toc -->
**Contents**

- [1. Scope](#1-scope)
  - [1.1 Time granularity — hybrid](#11-time-granularity--hybrid)
  - [1.2 Market scope — day-ahead only](#12-market-scope--day-ahead-only)
  - [1.3 Asset](#13-asset)
- [2. Horizon protocol](#2-horizon-protocol)
  - [2.1 Rolling window](#21-rolling-window)
  - [2.2 Information set — the gate at 12:00 CET on D-1](#22-information-set--the-gate-at-1200-cet-on-d-1)
  - [2.3 Bidding simplification, and what it costs](#23-bidding-simplification-and-what-it-costs)
  - [2.4 The perfect-foresight bound](#24-the-perfect-foresight-bound)
  - [2.5 The floor: a no-information policy](#25-the-floor-a-no-information-policy)
  - [2.6 Warm-up](#26-warm-up)
- [3. Physical and economic model](#3-physical-and-economic-model)
  - [3.1 Formulation](#31-formulation)
  - [3.2 Why the binaries are not redundant](#32-why-the-binaries-are-not-redundant)
  - [3.3 Degradation cost — the parameter that sets everything](#33-degradation-cost--the-parameter-that-sets-everything)
  - [3.4 Network tariffs on charged energy — parametrised on purpose](#34-network-tariffs-on-charged-energy--parametrised-on-purpose)
  - [3.5 Price-taker](#35-price-taker)
- [4. Metrics](#4-metrics)
  - [4.1 What the frozen snapshot gives (v1: floor and bound only)](#41-what-the-frozen-snapshot-gives-v1-floor-and-bound-only)
- [5. Data](#5-data)
  - [5.1 The four indicators, and why these four](#51-the-four-indicators-and-why-these-four)
  - [5.2 The geography trap](#52-the-geography-trap)
  - [5.3 Two forecast families, and only one survives the gate](#53-two-forecast-families-and-only-one-survives-the-gate)
  - [5.4 Granularity is observed, never assumed](#54-granularity-is-observed-never-assumed)
- [6. Solver and modelling stack](#6-solver-and-modelling-stack)
  - [6.1 Why Pyomo first and PyOptInterface second](#61-why-pyomo-first-and-pyoptinterface-second)
  - [6.2 Backend abstraction](#62-backend-abstraction)
  - [6.3 Persistent re-solve with HiGHS — measured](#63-persistent-re-solve-with-highs--measured)
- [7. What was deliberately not done](#7-what-was-deliberately-not-done)

<!-- /toc -->

Why this project is built the way it is. Every non-obvious choice is recorded here
with its rationale, and the ones that were rejected are recorded too — a decision is
only legible next to the alternative it beat.

Confidence marks: `[Certain]` hard evidence · `[Likely]` strong inference ·
`[Guessing]` filling a gap.

---

## 1. Scope

### 1.1 Time granularity — hybrid

| Regime | Period | Δt | Use |
|---|---|---|---|
| Hourly | 2022-01-01 → 2025-09-30 | 1 h | Development, forecaster training, validation |
| Quarter-hourly | 2025-10-01 → latest | 0.25 h | Headline result |

Since 30 September 2025 (delivery day 1 October) the European day-ahead market clears
in 15-minute market time units: 96 prices per day. OMIE numbers period 1 as H1Q1,
00:00–00:15 CET. `[Certain]`

Working only in hourly data would describe a market that no longer exists. Working
only in quarter-hourly data leaves about ten months of history, not enough to train
on. Hence both.

Consequences the code must respect:

- **Δt is a parameter, never an implicit constant.** Every energy balance carries it
  explicitly. This is the most likely bug in the project, which is why there
  is a test whose only job is to assert that four hourly periods and sixteen
  quarter-hourly periods spanning the same four hours give the same optimum.
- **The two regimes' €/MW/year figures are not comparable.** Finer granularity
  captures more spread. The headline refers to the quarter-hourly regime; hourly
  results are historical context.

### 1.2 Market scope — day-ahead only

Day-ahead arbitrage is a *fraction* of a real battery's revenue stack; balancing
services usually pay more. Stating which fraction is modelled is not a disclaimer,
it is part of the result. See the limitations section of the README.

### 1.3 Asset

10 MW / 20 MWh (2-hour duration), grid-connected, standalone — not hybridised with a
renewable plant.

---

## 2. Horizon protocol

Horizon, terminal condition, information set and bound definition are one decision,
governed by a single principle:

> **The oracle must differ from the policy in exactly one respect: the prices it
> sees.** Any other difference — horizon, constraints, cadence, terminal condition —
> contaminates the ratio and makes it uninterpretable.

### 2.1 Rolling window

**48-hour window, first 24 hours implemented, SoC carried forward** (192/96 periods
quarter-hourly). This removes the end-of-horizon artefact without inventing a
terminal value function.

**Encoded in days, not hours.** `config/params.yaml` says `window_days: 2` and
`implement_days: 1`, because "48 hours" is the ordinary-day statement of the
protocol and delivery days are 23 or 25 hours twice a year. The period counts come
from `timeline.periods_in_day`, so a two-day window is 47, 48 or 49 periods (188,
192 or 196 quarter-hourly) and the loop never multiplies anything by 24.

That has one consequence in the modelling layer worth stating, because it looks
like a violation of "build the model once" and is not. A backend is constructed
with a fixed `n_periods`, so three window lengths need three model instances. The
backtest holds a pool keyed on window length; it reaches its final size of three
within the first year of a regime and every subsequent day re-solves an existing
instance. The rule being protected is *no rebuild inside the loop*, and it holds.
Padding a short window with invented periods would satisfy the letter of "one
model" while putting fabricated prices into the optimisation, which is worse.

The last day of a snapshot cannot be decided — its horizon runs past the end of
the data — so the run stops one day short rather than solving one day under a
shorter protocol. §2.4's guarantee is that the cadence is identical throughout.

Rejected:

- *Free terminal SoC worth nothing* → the optimiser empties the battery every day,
  producing a visible "midnight cliff" in any dispatch plot.
- *Cyclic (SoC_end = SoC_start)* → removes the accounting artefact but not the
  economic one: forces unwinding even when the next day opens expensive. Systematic
  downward bias.
- *Terminal value function* → correct, but one more parameter to justify.

### 2.2 Information set — the gate at 12:00 CET on D-1

The day-ahead auction closes bid submission at 12:00 CET on D-1 and publishes results
around 13:00. `[Certain]`

**At decision time none of day D's prices are known.** The agent forecasts prices for
D — and for D+1, to close the 48-hour window — optimises against that forecast, and
the schedule is settled at realised prices. The entire window is solved on forecasts.

If D's prices were assumed known, there would be no forecasting problem and no
content. **Hard rule: no variable used to decide on day D carries a timestamp later
than 12:00 CET on D-1.** Enforced by a test, not by discipline.

**"12:00 CET" resolved to local noon.** `[Certain]` on the ambiguity, `[Likely]` on the
reading. The phrase is unambiguous only in winter: Spain is on CEST from late March to
late October, so a literal fixed UTC+1 sits at 11:00 UTC all year, while noon on the
market clock is 11:00 UTC in winter and 10:00 UTC in summer. `timeline.gate_close_utc`
implements **12:00 Europe/Madrid**, for two reasons. It is what the market does — OMIE
quotes session times in peninsular local time. And it is the conservative of the two:
in summer it closes the information set an hour *earlier*, so anything that passes the
no-lookahead test under this definition passes under the other as well. One constant,
in one module.

### 2.3 Bidding simplification, and what it costs

A charge/discharge schedule is committed and settled at realised prices — equivalent
to bidding as a price-taker at the market price limits.

A real participant submits a price–quantity *curve*, not a fixed schedule, and would
never be forced to charge at €300/MWh because it would have set a limit price.
**This simplification penalises the forecast-based policy more than reality would**,
because it strips away the natural hedge against forecast error. The honest reading
is that the reported forecast-policy number is conservative.

### 2.4 The perfect-foresight bound

Same optimiser, same constraints, same 48-hour window, same cadence, same SoC
handling. **Sole difference: realised prices instead of forecast prices.**

This is a structural guarantee, not a discipline guarantee: every policy obtains its
schedule from the same `BatteryMILP` instance. If the oracle and the policy call the
same code, they cannot be given different constraints by accident.

**One extra solve:** a bound over an annual window with SoC free across days, to
separate the value of *horizon* from the value of *information*. For a 2-hour battery
inter-day arbitrage should be worth little — carrying energy overnight costs the
intraday cycle, which almost always pays more — so the gap should be small.
`[Likely]` It is measured rather than asserted.

**Measured, and it is exactly zero — proven, not estimated.** Calendar year 2024,
hourly: 8,784 periods, 35,136 variables, 8,784 binaries, `c_deg = 17`. HiGHS 1.15.1
finished in **24.5 seconds** on a laptop, an order of magnitude less trouble than the
solve was budgeted for.

Three numbers close the question between them:

| | € |
|---|---|
| Annual solve, best integer solution found | 331,669.726 |
| Annual solve, dual bound | **331,670.108** |
| Rolling 48-hour oracle, settled over the same days | **331,670.108** |

The rolling oracle's dispatch is a *feasible point* of the annual problem — same
battery, same opening SoC, and neither problem constrains the terminal SoC — so the
annual optimum is at least what it earns. The dual bound says the annual optimum is at
most €331,670.108. Ceiling and floor are the same number, so **the rolling 48-hour
horizon attains the annual optimum**. The value of horizon is zero, not merely small.
`[Certain]` for this asset, this year, this regime.

The €0.38 the CLI prints as a negative "value of horizon" is not the answer to the
question; it is the annual solve's own unclosed gap. HiGHS stopped with an incumbent
€0.38 under its dual bound because that relative gap, 1.15e-6, sits inside its
tolerance. Put plainly: **366 small exact solves found a better dispatch than one big
gap-limited solve did**, and the big solve's dual bound is what certifies it optimal.
Neither run answers the question alone, which is why both are reported.

That is what §2.4 exists to establish. The whole gap between a policy and the bound is
attributable to *information*, with no horizon effect mixed into it — an assumption
that was sitting under the headline and is now a measurement.

The cluster provision is not wasted; it is unneeded for this case. It stands for the
quarter-hourly annual solve, four times the size, and for any run that does not close
as easily.

That annual solve is ~35,000 variables and 8,760 binaries hourly (~140,000 and 35,040
quarter-hourly), so it is reported twice: a reference run on a cluster with a
commercial solver, and a portable run on HiGHS with an explicit gap tolerance. It is
*not* relaxed to an LP — that would inflate the denominator, which is the one thing a
bound must never do. `hpc/README.md` covers what runs where and why the main backtest
deliberately does not go to the cluster; `results/annual_bound.json` carries the
incumbent, the dual bound, the gap actually reached, the wall clock, the solver
version and the machine, so a reader who cannot re-run it can still see how much
slack the number carries.

Both runs are the same command with `--solver` changed, through the same
`model/pyomo.py`. That is §6.2's claim being exercised rather than asserted: **if
running on the cluster ever needs a code change, the abstraction has leaked**, and
that is a finding rather than an inconvenience.

The comparison is against the *rolling* oracle over the identical days, so it
isolates one thing. The rolling oracle's own dispatch is a feasible trajectory for
the free-horizon problem from the same opening state of charge, so the annual optimum
cannot be lower — which makes the ordering a proof rather than a tolerance, and it is
asserted as one in `tests/test_backtest_bound.py`.

### 2.5 The floor: a no-information policy

"I capture 85% of the bound" means nothing without knowing that a dumb heuristic
captures 78%. **The floor is the null hypothesis.** Without one, no result could ever
contradict the claim that the forecast is doing the work — a high percentage would be
read as skill, and a low one as a hard market, with nothing to distinguish the two.
The floor is what makes the headline a testable claim rather than a number.

The floor charges in the historically cheapest periods and discharges in the
historically most expensive, using hour-of-day × month averages, without looking at
the specific day. Implemented as a *price vector* — the climatological average —
passed to the same optimiser. It is not a separate heuristic with its own
constraints; that is what keeps it comparable.

Three implementation choices, each of which could have gone the other way:

**The average is causal.** For delivery day D it uses only prices stamped before the
gate. Invariant 1 is written on *timestamps*, so this discards the afternoon and
evening of D-1 even though those prices were published the day before and are
genuinely known at noon. The strictness costs half a day out of a multi-year average
and buys a floor that needs no carve-out in the no-lookahead test — and a test with a
carve-out guards less than it appears to.

Being causal has a real cost: the average starts empty and fills up, so early in a
run the floor decides on less. A thin floor is a *weak* floor, and a weak floor
flatters everything measured against it — the direction that favours this project's
own claim, and therefore the one to distrust. So the policy counts its own fallbacks
instead of the cost being argued about.

Measured on the frozen snapshot, and the pattern is tighter than "thin early":

| Regime | Fallback periods | Share | Days affected |
|---|---|---|---|
| Hourly | 1,128 / 65,662 | 1.7% | 35 of 1,368 |
| Quarter-hourly | 4,128 / 62,592 | 6.6% | 32 of 326 |

Every affected day is **the last day of a month or the first two of the next, during
the first pass through the calendar only**. The mechanism: the window for 31 January
reaches into February, and no February key has any history yet; by 3 February it does,
and in every later year both months are fully populated. The two lower tiers are just
the opening days — `no_history` is exactly the first window, when nothing predates the
gate, and `grand_mean` is exactly half of the second day's clock.

So the quarter-hourly figure is **not** a thinner floor. It is the same one-pass cost
over a run a quarter as long, and that regime spans about ten months so it never gets
a second pass. `[Certain]`

**The average expands; it does not roll.** No trailing window, no decay. A lookback
length would be a tuning knob on the null hypothesis, and a null hypothesis that can
be tuned is one that can be moved until the headline looks right.

**The key is local wall-clock time**, `(month, hour, minute)` in Europe/Madrid — the
solar trough and the evening peak sit at local clock times. For the hourly regime
that is exactly hour-of-day × month; for the quarter-hourly one it extends to the 96
quarters without ever indexing into the day, which is what keeps it correct on the
92- and 100-period transition days.

### 2.6 Warm-up

First 7 days of the backtest discarded, to remove the effect of the arbitrary initial
SoC. Initial SoC 50%.

---

## 3. Physical and economic model

### 3.1 Formulation

| Symbol | Value | Unit |
|---|---|---|
| `P_max` | 10 | MW, at the connection point (AC), symmetric |
| `E_max` | 20 | MWh **usable** |
| `η_rt` | 0.85 | AC-to-AC round trip |
| `η_c`, `η_d` | √0.85 ≈ 0.922 | symmetric split |
| `SoC` | 0 … 20 | MWh |
| `c_deg` | 17 (5 / 17 / 40 sensitivity) | €/MWh discharged |
| `charge_tariff` | 0 | €/MWh charged |

Variables: `p_c[t] ≥ 0`, `p_d[t] ≥ 0` (MW), `soc[t]` (MWh), `u[t] ∈ {0,1}` charging
indicator.

```
soc[t] = soc[t-1] + η_c · p_c[t] · Δt − p_d[t] · Δt / η_d
0 ≤ soc[t] ≤ E_max
p_c[t] ≤ P_max · u[t]
p_d[t] ≤ P_max · (1 − u[t])

max Σ_t [ p_d[t]·Δt·λ[t] − p_c[t]·Δt·(λ[t] + charge_tariff) − c_deg·p_d[t]·Δt ]
```

The big-M on the exclusivity constraints is `P_max`, the physical rating, so they add
no slack beyond the variable bounds already present.

Degradation is charged **on discharge only**, following Xu et al.: a discharging
half-cycle causes the same ageing as a full cycle of the same depth, and charged and
discharged energy are nearly identical daily, so counting throughput twice would
double-count. `[Certain]`

### 3.2 Why the binaries are not redundant

With positive prices and `η_rt < 1`, charge/discharge exclusivity holds automatically
at the LP relaxation optimum and the binaries are redundant. **With negative prices
it stops being true.** Charging and discharging at once burns energy and gets paid for
doing so. To hold SoC flat the relaxation must discharge `η_c·η_d = 0.85` of what it
charges, so it collects `(1 − η_rt)·|λ| = 0.15·|λ|` per MW of fictitious throughput —
€7.50/MWh at λ = −50. Spain sees negative prices with increasing frequency, so the
binaries are necessary.

Verified: with a full battery at λ = −50 throughout, the LP
relaxation scores 162.16 against the MILP's 150.00 and charges and discharges
simultaneously in one period, while the MILP does so in none.

### 3.3 Degradation cost — the parameter that sets everything

Structure: linear cost on discharged throughput. Central value **€17/MWh**, from
cells at ~€100/kWh, 6,000 cycles to 70% EOL, 20 MWh → €2M over 120,000 MWh of
lifetime throughput. `[Guessing]` on the 2026 cell price.

Published values span roughly £7/MWh in commercial revenue models to $62.5/MWh in
academic frequency-regulation examples — nearly an order of magnitude. `[Certain]` Results are presented as a three-point sensitivity:
**5 / 17 / 40 €/MWh.**

Choosing an energy-throughput model over cycle-based rainflow counting is **choosing
the optimistic model, and that is stated rather than hidden.** Rainflow counting is
physically more faithful but has no analytical form and cannot be embedded in the
optimisation directly; studies on ERCOT 2024 prices find it predicts substantially
higher degradation. `[Certain]`

Canonical reference: Bolun Xu et al., *Factoring the Cycle Aging Cost of Batteries
Participating in Electricity Markets*, arXiv:1707.04567, IEEE Trans. Power Systems
33(2), 2018.

**The part that is rarely written down:** a linear throughput cost is algebraically
equivalent to a **minimum spread threshold**. The battery only cycles when
`λ_sell·η_d − λ_buy/η_c > c_deg`. So `c_deg` is not an accounting parameter — **it is
what sets cycles per year.** Hence: equivalent cycles per year is reported on every
run. If a 2-hour battery shows 700 cycles/year, `c_deg` is too low and no other
number is worth reading.

### 3.4 Network tariffs on charged energy — parametrised on purpose

Article 2 of CNMC Circular 3/2020 exempts storage connected to the transmission or
distribution grid from network access tolls. `[Certain]` on the text. What is unclear:
whether the exemption covers charging as well as reinjection `[Guessing]`; and that
*charges* (`cargos`) are a separate instrument set by ministerial order, whose
application to storage is the murky part. CNMC opened a consultation in October 2024
on harmonising storage treatment and eliminating double charging, still open for the
2026–2031 period. `[Certain]`

Consequently, `charge_tariff_eur_mwh` defaults to 0 and is reported with sensitivity figure in order to deal with a live regulatory uncertainty.

### 3.5 Price-taker

10 MW is negligible against OMIE cleared volume. Stated as an assumption, not a fact
— it stops being valid the moment a fleet is discussed.

---

## 4. Metrics

| Metric | Definition | Role |
|---|---|---|
| €/MW/year | Annualised net profit ÷ 10 MW | Headline |
| % of bound | Policy profit ÷ oracle profit | The comparison that matters |
| Equivalent cycles/year | Discharged throughput ÷ E_max ÷ years | Sanity diagnostic |

Forecast RMSE is reported in a secondary table and **is never the headline.** A
better RMSE that does not convert into captured spread is not an improvement.

**Decision and settlement are separate.** A policy's schedule is decided against the
prices it believed and settled against the prices that cleared. For the oracle those
coincide; for every other policy they do not, and the difference *is* the cost of
imperfect information — the quantity this project exists to measure. So settlement
never reads the solver's objective, it recomputes from the dispatch at realised
prices.

**Years come from the days, not from the day count.** Hours are summed from each
day's own period count, so a 23- or 25-hour day contributes what it was, and the
divisor is 365.25 × 24. Over a full year the two transition days cancel and the error
would hide; the same code annualises two-month runs where they do not.

**"% of bound" is undefined, not zero, when the bound earns nothing.** Above the
market's spread the correct dispatch is to stay idle and every policy scores zero;
reporting 0% would read as failure where the truthful answer is that the ratio has no
content.

### 4.1 What the frozen snapshot gives (v1: floor and bound only)

The forecast policy arrives in the next slice, so this is the ladder with its middle
rung missing. Reported here because the floor's height is the point of §2.5 — it is
what any later claim has to beat.

Hourly regime, 1,361 evaluated days (2022-01-08 → 2025-09-29):

| `c_deg` | Floor €/MW/yr | Bound €/MW/yr | Floor as % of bound | Bound cycles/yr |
|---|---|---|---|---|
| 5 | 39,204 | 48,881 | 80.2% | 491.6 |
| **17** | **27,237** | **38,161** | **71.4%** | **400.4** |
| 40 | 11,005 | 23,458 | 46.9% | 245.2 |

Quarter-hourly regime, 319 evaluated days (2025-10-08 → 2026-08-22):

| `c_deg` | Floor €/MW/yr | Bound €/MW/yr | Floor as % of bound | Bound cycles/yr |
|---|---|---|---|---|
| 5 | 62,816 | 70,184 | 89.5% | 521.1 |
| **17** | **50,472** | **58,916** | **85.7%** | **426.9** |
| 40 | 34,421 | 42,301 | 81.4% | 302.8 |

Three readings, and only the first is comfortable.

**Cycles per year are physically sane.** 400 hourly and 427 quarter-hourly at
`c_deg = 17` — near one cycle a day, which is what a 2-hour asset in a market with
one solar trough and one evening peak should do. The §3.3 alarm was that ~700 would
mean `c_deg` was too low; it does not fire, and it is checked as a test rather than
read off a table.

**The floor is high, and it is meant to be.** 71–86% of perfect foresight from an
average of the last few years, with no forecast at all. That is precisely why §2.5
insists on having one: without it, a forecast policy scoring 85% would read as skill.
The number the next slice has to beat is not zero, it is 85.7%.

**The two regimes are not comparable** (§1.1), and the gap between them is not
mostly about resolution. The hourly window is 2022–2025 and carries the gas crisis;
the quarter-hourly one is a calmer market clearing at finer granularity. Only the
quarter-hourly figures describe the market as it exists.

The floor's share of the bound falls as `c_deg` rises, in **both** regimes — 80.2% to
46.9% hourly, 89.5% to 81.4% quarter-hourly. The direction is arithmetic rather than a
finding: net profit is `E · (m − c_deg)` for `E` MWh discharged at a gross margin `m`,
the floor's `m` is lower than the oracle's because it picks worse hours, and
`(m_f − c)/(m_o − c)` decreases in `c` whenever that holds.

The rates differ — 33 points against 8 — because the oracle can respond to a higher
threshold and the floor cannot. The oracle drops its marginal cycles and keeps the
wide-spread ones, so its *net* margin per MWh barely moves across the sweep; that is
`c_deg` working as the minimum-spread threshold of §3.3. The floor selects on
hour-of-day only and cannot tell an ordinary day from a wide-spread one, so the same
threshold filters its trades far less and its net margin collapses. `[Certain]` on the
direction, `[Likely]` on the attribution of the rate.

---

## 5. Data

- **Primary source:** ESIOS API (Red Eléctrica de España).
- **Cross-check:** OMIE `marginalpdbc` files, validated against ESIOS for at least
  one month.
- **Format:** Parquet, one file per grid, snapshot date in the filename.
- **Downloaded once.** After the snapshot the download code is not modified. The
  project risk is not the MILP — a 192-period problem solves in milliseconds — it is
  losing a week to file plumbing.
- **Reading it back lives elsewhere.** `src/bess_arb/series.py`, not
  `src/bess_arb/data/`. Loading a committed Parquet is ordinary application code that
  the backtest, the forecaster and the chart all need; putting it in the frozen
  package would make "not modified after the snapshot" quietly untrue. The loader
  validates and refuses — it never fills a gap — because a silently patched index
  produces a profit figure that traces back to no published price.

Series: day-ahead marginal price, plus wind, solar and demand **forecast** series as
exogenous forecaster inputs. Using *realised* wind or demand is information leakage
and destroys the backtest.

### 5.1 The four indicators, and why these four

Verified against the live catalogue on 24 August 2026 rather than taken from
memory. `[Certain]` — the numbers below were read back from the API.

| Column | Indicator | Name | Geography |
|---|---|---|---|
| `price_eur_mwh` | **600** | Precio mercado SPOT Diario | 3 (España) |
| `demand_forecast_mw` | **1775** | Previsión diaria D+1 demanda | 8741 (Península) |
| `wind_forecast_mw` | **1777** | Previsión diaria D+1 eólica | 8741 (Península) |
| `solar_forecast_mw` | **1779** | Previsión diaria D+1 fotovoltaica | 8741 (Península) |

Two things went wrong on the way to that table, and both are the kind that produce a
plausible wrong number rather than an error.

### 5.2 The geography trap

**Indicator 600 carries six geographies** — Portugal, France, Spain, Germany, Belgium
and the Netherlands — and returns them *interleaved on the same timestamps*, Portugal
first. `[Certain]` Any code that deduplicates on the timestamp alone therefore keeps
the **Portuguese** price and calls it Spanish.

The Iberian market couples the two, so they are equal whenever the interconnector is
uncongested — which is most of the time. Measured over January 2024: they differ in
**36 of 744 periods, by up to €39.07/MWh.** Frequent enough to matter for an
arbitrage result, rare enough to survive any amount of eyeballing.

So `geo_id` is a required field on every series in `config/params.yaml`, the client
refuses a multi-geography indicator that was not told which one to keep, and the
filter is applied client-side even when the server was asked to do it — a silently
ignored query parameter is the other half of the same bug.

### 5.3 Two forecast families, and only one survives the gate

A forecast is defined by its **vintage** — the moment it was made. Neither ESIOS
family carries a "made at" field; both are indexed by the *target* timestamp, so
they are indistinguishable in shape.

- **`Previsión diaria D+1`** (1775/1777/1779) — published once a day, covering D+1.
  Vintage fixed at D-1, never rewritten.
- **Rolling** (460 demand, 541 wind, 542 photovoltaic) — a live operational series,
  overwritten as the day approaches and passes. A historical query returns the last
  value written, which for a past day was written an hour or two before that hour,
  or during it.

The vintages are not documented, so they were measured. Two comparisons, using wind
(1777 against 541) and realised generation (551):

| | D+1 (1777) | Rolling (541) |
|---|---|---|
| **A future day** (2026-08-25, queried 2026-08-24) | agree with each other to RMSE **126 MW**, correlation 0.9986 | |
| **A completed day** (2026-08-22) against realised | RMSE 1106 MW | RMSE **624 MW** |
| **March 2024** against realised | RMSE 2043 MW | RMSE **896 MW** |

Before the day happens the two series are nearly the same forecast. After it happens
they are not, and the rolling one has moved toward what occurred. Same series, same
target hours — so the rolling values were rewritten in between. `[Certain]`

Note what this is *not*: an argument that 624 MW is suspiciously accurate. It is
ordinary for a 1–3 hour horizon, just as 1106 MW is ordinary for a 12–36 hour one.
Neither is anomalous for its own horizon. What identifies the vintage is that two
series describing the same quantity separate only once the target has passed.

Only the D+1 family is available at the gate. Using the rolling one would let the
forecast policy score against the perfect-foresight bound while holding information
the oracle is supposed to be alone in having, inflating the headline ratio by an
unmeasurable amount.

### 5.4 Granularity is observed, never assumed

The market moved to 15-minute MTUs on delivery day 1 October 2025. **ESIOS moved
indicator 600 onto a 15-minute grid on 1 January 2025**, nine months earlier.
`[Certain]` Through that pre-period the four values inside each hour are identical —
the hourly clearing price republished on the finer grid ahead of the go-live.

Measured: the maximum spread within an hour is exactly **0.0000 €/MWh** up to
30 September 2025, and **€65.50** on 1 October. The regime boundary in §1.1 is
therefore confirmed from the data itself and not only from the market notice, which
is a stronger thing to be able to say.

Consequently the snapshot builder downsamples that stretch back to hourly — **and
only because it is provably lossless.** It verifies that every target period is
constant across the finer ones and refuses otherwise, so this can never quietly
become an average that discards real spread.

The forecast series get the opposite treatment. They are still hourly today, after
the market moved. Forward-filling them onto the 96-period grid would be a modelling
decision belonging to the forecaster slice, and freezing it here would make it
unrevisable without re-pulling — which invariant 6 forbids. So **each file holds
exactly one grid** and the mismatch stays visible for slice 4 to resolve
deliberately.

The asymmetry is deliberate: collapsing a series whose extra resolution is provably
empty loses nothing, whereas expanding one whose extra resolution does not exist
would invent data.

**Time zone.** Everything stored in UTC, presented in `Europe/Madrid`. DST
transitions produce 23- and 25-hour days (92 and 100 quarter-hourly periods). Code
assuming 24 periods per day breaks twice a year, silently, surfacing as an
unexplained jump in profit. There is an explicit test on the transition days.

---

## 6. Solver and modelling stack

| Component | Choice |
|---|---|
| Modelling layer, v1 | Pyomo |
| Modelling layer, performance port | PyOptInterface |
| Backend architecture | Pluggable behind a `Protocol` |
| Default solver | **HiGHS** — free, so anyone can clone and run |
| Optional solver | Gurobi behind a config key |
| Data | pandas + pyarrow |
| Forecasting | LightGBM, deliberately boring |
| Environment | `uv` with committed lockfile |

### 6.1 Why Pyomo first and PyOptInterface second

**Change one thing at a time.** Developing the formulation against an unfamiliar API
at the same time makes failures unattributable: a wrong objective value could equally
be a sign error in the model or a misused library call, and the two are
indistinguishable from the symptom. Holding the modelling layer fixed at a known
quantity while the formulation is developed means every failure has one candidate
cause.

The order also converts the port from an assumption into a measurement: *model
construction dominated backtest runtime; here is the wall clock before and after.*
PyOptInterface documents itself as roughly 10× faster than Pyomo at model
construction, but that benchmark sets the solver time limit to zero to isolate
construction — it is not a claim about solving. Taking it on faith would have meant
building on a number that was never measured for this workload. Building both and
timing the full backtest replaces it with an owned figure, and that is only possible
in this order.

### 6.2 Backend abstraction

The build-once-mutate-re-solve pattern requires holding a live solver model across
the backtest loop. If the loop holds that object, the solver has leaked out of the
modelling module and "swap one file" stops being true. So **the backend is a stateful
object behind a `Protocol`, not a pure function** — the persistent model is owned
internally, callers pass numpy in and get a dataclass out.

```python
class BatteryMILP(Protocol):
    def __init__(
        self,
        params: BatteryParams,
        n_periods: int,
        dt_h: float,
        *,
        solver: SolverConfig | None = None,
        relax_binaries: bool = False,
    ) -> None: ...
    def solve(self, prices: FloatArray, soc_initial: float) -> Solution: ...
```

`SolverConfig` — name, gap, time limit, threads — is part of the solver-free
spec rather than of any backend, so the solver is chosen in
`config/params.yaml` and nowhere else.

`relax_binaries` is a diagnostic and not a policy option. It exists so the
demonstration in §3.2 can be run as a test from outside `model/`, without
importing a solver to do it. No policy may use it: relaxing the MILP,
including for the bound, is the one thing that is never done (§7).

Two rules make "swappable" true rather than aspirational:

1. **No solver library is imported anywhere outside `model/`.** Enforced by an
   `import-linter` contract *and* a test, so it survives a refactor of either.
2. **Solver status codes are normalised** into a project-owned `SolveStatus` enum. A
   raw solver enum escaping would reintroduce the coupling through a side door.

The payoff is the **backend equivalence test**: the golden case plus two random
96-period instances through every registered backend, objectives agreeing to 1e-6.
Three independent implementations of the same formulation agreeing is strong evidence
the formulation is correct — which is more than "three tools were used".

### 6.3 Persistent re-solve with HiGHS — measured

The backtest is roughly 1,700 days × 3 policies × 3 degradation values ≈ **15,000
window solves** of a model with a few hundred variables. At that scale model
*construction* dominates, not the solve. So the model is built once per regime and
each window mutates only the price coefficients and the initial SoC before
re-solving.

Verified before the backend was written (Pyomo 6.10.1, highspy 1.15.1):

- Both `SolverFactory("highs")` (the newer `pyomo.contrib.solver` interface) and
  `"appsi_highs"` are persistent and **re-solve correctly** after mutating price
  Params and `soc_initial` — checked against freshly-built models across five price
  vectors and five initial states, agreeing to 1e-6. A stale second solve was the
  failure mode that mattered, because it would have corrupted 15,000 solves while
  looking plausible.
- At the real window size (192 periods), warmed up: **48 ms/solve persistent against
  122 ms/solve rebuilding**, a 2.5× speedup. `"highs"` beats `"appsi_highs"`
  (62 ms/solve). Extrapolated: ~12 minutes for the full workload against ~31.

Re-measured on the backend itself once it existed, over sixty 192-period windows with
prices drawn from `N(60, 40)`: **64 ms/solve persistent against 125 ms rebuilding**,
so 2.0× rather than 2.5×. The spike's price vectors were structured and easy; random
ones produce near-ties that make the branch-and-bound work harder. The speedup is a
property of the instance mix as much as of the interface, and the smaller figure is
the one to plan with. Two settings were checked at the same time and kept: `threads:
1` costs nothing measurable (these models are too small to parallelise) and buys
determinism, and closing the gap exactly costs ~16% against HiGHS's 1e-4 default,
which is worth paying to keep the golden test's 1e-6 tolerance meaningful.

**The honest finding:** the persistent pattern does *not* make model construction
disappear, as expected — it buys 2–2.5×, not an order of magnitude, because Pyomo
still walks the model to detect what changed. That leaves the PyOptInterface port a real
motivation and, more usefully, a measured baseline to beat rather than a vendor
benchmark to trust.

---

## 7. What was deliberately not done

Recorded so the temptations are not reopened later:

- **No single annual MILP as the bound.** It inflates the denominator and measures the
  value of horizon, not of information.
- **No relaxed constraints in the oracle** "because it is a theoretical bound". It
  breaks comparability.
- **No second market** before the day-ahead result is closed.
- **No single degradation cost defended.**
- **No realised wind or demand as forecaster inputs.**
- **The MILP is not rebuilt inside the backtest loop.**
- **No solver import outside `model/`.**
- **The MILP is not relaxed to an LP**, including for the bound.
