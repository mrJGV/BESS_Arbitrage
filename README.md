# Battery arbitrage in the Spanish day-ahead market

<!-- toc -->
**Contents**

- [The result](#the-result)
  - [The hourly regime, as context](#the-hourly-regime-as-context)
  - [Forecast accuracy](#forecast-accuracy)
- [What was tried and not adopted](#what-was-tried-and-not-adopted)
  - [Price-quantity bid curves](#price-quantity-bid-curves)
  - [Forecaster variants](#forecaster-variants)
- [Limitations](#limitations)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [How this was built](#how-this-was-built)
- [Licence](#licence)

<!-- /toc -->

Day-ahead (OMIE) arbitrage demonstrator for a **10 MW / 20 MWh** battery.

A rolling-horizon MILP dispatches the battery against *forecast* prices under a
realistic information gate — bids close at 12:00 CET on D-1, and a delivery day's
prices are published about 13:00 on the day before, so every price through D-1 is
known when the schedule is committed and none of day D's is. The result is measured against
two reference points: a perfect-foresight upper bound and a no-information floor
policy.

Three policies, one optimiser, the only difference between them being the price vector:

| Policy | Sees | Role |
|---|---|---|
| Floor | Hour-of-day × month averages of every price published before the gate, from January 2022 | Lower reference |
| Forecast | LightGBM point forecast, gated at 12:00 CET D-1 | The actual result |
| Bound | Realised prices | Upper reference (perfect foresight) |

## The result

Quarter-hourly regime, 319 delivery days from 8 October 2025 to 22 August 2026,
degradation cost `c_deg` = 17 €/MWh:

> A 10 MW / 20 MWh battery bidding the forecast earns **€51,615/MW/year**. The
> no-information floor earns €49,842 and perfect foresight €58,916, so the
> forecast closes **19.5% of the gap between them**, with a 95% interval of
> 2.9% to 34.2%. Over these 319 days its margin over the floor is
> distinguishable from zero. At the lowest degradation cost it is not.

![Share of the floor-to-bound gap closed by the forecast, and the three policies in €/MW/year, at c_deg 5, 17 and 40](results/headline_quarter_hourly.png)

| `c_deg` €/MWh | Floor €/MW/yr | Forecast €/MW/yr | Bound €/MW/yr | % of bound, floor / forecast | Forecast's share of the gap, 95% interval | Cycles/yr, floor / forecast / bound |
|---|---|---|---|---|---|---|
| 5 | 62,385 | 62,356 | 70,184 | 88.9% / 88.8% | −0.4% (−22.3% to 17.7%) | 565 / 558 / 521 |
| **17** | **49,842** | **51,615** | **58,916** | **84.6% / 87.6%** | **19.5% (2.9% to 34.2%)** | **412 / 422 / 427** |
| 40 | 33,961 | 36,219 | 42,301 | 80.3% / 85.6% | 27.1% (11.4% to 40.9%) | 320 / 308 / 303 |

**Why the share of the gap is the headline.** The floor feeds the optimiser an
hour-of-day × month average price and already reaches 84.6% of the bound. A
forecast at 87.6% of the bound has added 3.0 points. The share of the gap puts the
floor at 0 and the bound at 100%, so it measures only what the forecast adds.

**How the interval is built.** A paired circular block bootstrap over delivery
days: every resample draws the same days for all three policies, in 7-day blocks,
10,000 times. Blocks, because a day's profit depends on the days before it through
the state of charge carried over and through weather and fuel prices that persist.
The block length is a judgement, so the run also reports 14- and 28-day blocks. At
`c_deg` = 17 they give 5.6% to 32.6% and 7.8% to 29.7%, and at `c_deg` = 40,
12.0% to 39.9% and 14.3% to 37.0%, so both margins clear zero at every block
length. At `c_deg` = 5 none does. Per-day profits are
in the committed summaries, so an interval at another block length needs no new
backtest.

**The share rises with the degradation cost**, from −0.4% to 19.5% to 27.1%, and
does the same in the hourly regime below. A linear degradation cost acts as a
minimum spread every trade must clear. The higher it is, the more the decision
becomes which days to trade, and the floor, which knows only hour of day and month,
cannot tell one day in a month from another. At `c_deg` = 5 the forecast earns no
more than the floor over these days.

**The floor is a choice, so a stronger one is reported too.** The floor above averages
every past year's same month and hour, from January 2022. A practitioner's first null is
more recent than that: the current month to date, hour by hour, which knows this month's
price level. Against that floor the forecast's share of the gap at `c_deg` = 17 is 10.3%,
with a 95% interval of −5.9% to 24.9%, so the margin is not distinguishable from zero; at
`c_deg` = 40 it is 18.3% (1.3% to 32.9%), and at 5 it is −12.3%. The 1.6 points of bound
between the two floors at `c_deg` = 17 is what knowing the current month's level is worth
over knowing the same month in past years, and it is about half of what the forecast adds.

**Cycles per year are plausible**: 422 for the forecast at the central `c_deg`, a
little over one a day for a 2-hour asset, so the degradation cost is in a sensible
range and the other numbers are worth reading.

The rolling 48-hour oracle turns out to **attain the annual-window bound
exactly**, in both regimes: the oracle's own dispatch is feasible for the annual
problem, so the annual optimum is at least what it earns, and the annual solve's
dual bound says it is at most that. In both cases the two meet to the last bit of
a float64. A two-day horizon leaves nothing on the table for a 2-hour asset, so
the whole gap between a policy and the bound is attributable to information
rather than to horizon.

The quarter-hourly solve is the one that had to be checked separately — 31,296
binaries against the hourly 8,784 — and it is also the one where the annual
solve stops short of its own optimum, at a gap of 8.2e-05. That is why the
denominator reported is the **dual bound** and not the incumbent: the incumbent
would put the value of horizon at a spurious +€43.72 instead of zero. Both runs
are committed, with solver, version, gap, wall clock and machine:
[results/annual_bound.json](results/annual_bound.json) (hourly, 13.5 s) and
[results/annual_bound_quarter_hourly.json](results/annual_bound_quarter_hourly.json)
(quarter-hourly, 49.1 s). Both files also carry the annual dual bound and the
rolling oracle's profit unrounded, so the equality can be checked rather than read.

### The hourly regime, as context

1,361 delivery days from 8 January 2022 to 29 September 2025. The level is not
comparable with the quarter-hourly regime, whose finer grid captures more spread,
and the forecast policy bids the floor's prices until early January 2023 because
the forecaster waits for a year of history. With four times as many days the
intervals are narrower, and
at `c_deg` = 17 and 40 they clear zero at 7-, 14- and 28-day blocks.

| `c_deg` €/MWh | Floor €/MW/yr | Forecast €/MW/yr | Bound €/MW/yr | Forecast's share of the gap, 95% interval | Cycles/yr, floor / forecast / bound |
|---|---|---|---|---|---|
| 5 | 39,508 | 39,697 | 48,881 | 2.0% (−6.0% to 9.7%) | 519 / 505 / 492 |
| **17** | **27,460** | **28,545** | **38,161** | **10.1% (2.9% to 17.1%)** | **392 / 402 / 400** |
| 40 | 11,460 | 14,362 | 23,458 | 24.2% (17.1% to 31.2%) | 275 / 256 / 245 |

### Forecast accuracy

Reported below the economics because a better error score that does not turn into
captured spread is not an improvement. Lead 0, the forecast that priced each
evaluated day:

| Regime | Periods | RMSE €/MWh | MAE €/MWh | Bias €/MWh | Mean daily rank correlation |
|---|---|---|---|---|---|
| Quarter-hourly | 30,624 | 17.03 | 12.12 | −0.00 | 0.929 |
| Hourly | 23,999 | 21.48 | 15.99 | +7.86 | 0.918 |

The hourly forecast over-predicts on average, in months when prices dropped to levels
its training history barely contained, February to May 2024 above all. The ordering
of hours within a day, which is what arbitrage uses, is about as good as in the
quarter-hourly regime.

Two things the quarter-hourly summary discloses about its own fits. The forecaster's
intra-hour stage needs 30 days of genuine quarter-hourly history before it can be
fitted, so the first 30 decision days, 23 of them after the warm-up, were priced
from the hourly level alone. And
4 of its 42 fits had too little history for a validation fold and used the round
ceiling instead of early stopping. Both counts are in
[results/backtest_quarter_hourly.json](results/backtest_quarter_hourly.json).

## What was tried and not adopted

### Price-quantity bid curves

A real participant submits a price-quantity curve for each period. This project
commits a fixed schedule, which is the same as bidding at the market's price
limits. Two extensions tested whether curves earn more. Forecast policy,
quarter-hourly, `c_deg` = 17, over the 167 evaluated days from October 2025 to
March 2026 — a shorter window than the table above, so the level differs:

| Bidding | €/MW/yr | Against the fixed schedule |
|---|---|---|
| Fixed schedule | 34,487 | — |
| Curves traced from forecast quantiles | 31,970 | −7.3% |
| Curves built from joint scenarios of past forecast errors | 23,935 | −30.6% |
| Curve chosen by a two-stage stochastic MILP, 5 scenarios | 27,306 | −20.8% |
| The same MILP, 20 scenarios sharing 3 price bands (prototype) | 33,889 | −1.7% |

- **A curve clears period by period, and a battery's dispatch is a trajectory.** A
  curve with several steps can clear into a sequence the state of charge cannot
  deliver. The scenario-built curves cleared energy the battery could not deliver
  equal to 86% of what it discharged, and that energy was not traded.
- **The stochastic MILP as shipped overfits.** `model/curve_pyomo.py` takes one
  price band per scenario, so in most periods each step of the curve is fitted to
  a single sampled day. With 20 scenarios sharing 3 bands the overfitting goes, and
  the optimiser bids a fixed schedule on the delivery day in all but 6 quarter-hours
  of 138 days. Offered a price-contingent curve, it declines to use one.
- **`docs/DECISIONS.md` §2.3**, frozen before these runs, argues that a curve is a
  free option against forecast error, which would make the fixed-schedule result
  conservative. These measurements do not support that: the fixed schedule is the
  best of the bidding rules measured.

The curve modes are reachable from `run_backtest(bidding=...)` and are not CLI
flags. The shared-band variant was a prototype outside the shipped code.

### Forecaster variants

These were scored on the same 319 quarter-hourly days the table reports, which is
selection on the evaluation set, so they are listed:

- a second stage targeting each day's within-day shape, with and without rescaling
  its amplitude: neither was better at every `c_deg`, and both were within half a
  point of the shipped forecaster at `c_deg` = 17;
- a 730-day and a 365-day rolling training window and a 365-day half-life decay,
  all fixed before any was run: both rolling windows removed the margin over the
  floor at `c_deg` = 17, and the decay stayed within 0.4 points of the shipped
  forecaster at every `c_deg`, better at one and worse at two. No window removed
  the hourly bias.

The shipped forecaster trains on every day published before the gate, and its
lagged prices start at the day before delivery, whose prices are published the
afternoon before the gate.

## Limitations

- **Day-ahead only.** Day-ahead arbitrage is one part of a battery's revenue
  stack. The other markets a battery can earn from — secondary reserve (aFRR)
  capacity and balancing energy, the intraday auctions and the continuous intraday
  market — are not modelled. Reserve would take power and energy headroom
  away from arbitrage in the hours it is sold. Intraday trading would let the
  battery correct its day-ahead position as forecasts improve, which cuts the cost
  of the forecast errors a fixed schedule pays in full here.
- **Fixed schedules**, not bid curves; see above.
- **Publication times are assumed, not observed.** A day's prices are read as
  known from 13:00 local on the day before delivery, OMIE's usual publication
  hour. The snapshot carries no publication timestamps, so a late publication
  would not be seen.
- **Price-taker.** 10 MW is assumed not to move the OMIE price. That stops holding
  for a fleet.
- **One year of the headline market.** 15-minute day-ahead products started on
  1 October 2025, so the headline sample is 319 days, which is why its intervals
  are wide.
- **A simple degradation and loss model.** Degradation is a linear cost on
  discharged energy, swept over 5, 17 and 40 €/MWh because published values span
  that range. Round-trip efficiency is a constant 85%, there is no calendar ageing,
  and the network tariff on charged energy is a parameter set to zero.

## Quick start

```bash
uv sync
uv run pytest          # golden test, Δt-invariance, binary necessity, DST,
                       # snapshot integrity, floor causality — no network,
                       # no ESIOS token

# Both regimes, all three c_deg values, then the chart: 48 minutes on an
# 8-thread laptop. `make backtest figures` runs the same three commands.
uv run bess-arb run --regime quarter_hourly --sweep --json results/backtest_quarter_hourly.json
uv run bess-arb run --regime hourly --sweep --json results/backtest_hourly.json
uv run bess-arb figures

uv run bess-arb bound          # the annual-window bound, DECISIONS.md §2.4
```

The frozen data snapshot is committed, so a clone reproduces every result
without an API token. See [data/README.md](data/README.md) for provenance and
`data/manifest.json` for row counts and checksums.

## Repository layout

```
config/params.yaml       every numeric parameter, plus backend/solver selection
data/                    frozen Parquet snapshot + manifest (provenance, sha256)
docs/DECISIONS.md        modelling rationale, frozen at v1
hpc/                     cluster provision for the annual bound, unused so far
results/                 annual bounds, backtest summaries with per-day profit,
                         and the headline chart
src/bess_arb/timeline.py UTC storage, Europe/Madrid delivery days, derived
                         periods-per-day, the 12:00 D-1 information gate
src/bess_arb/series.py   reading the frozen snapshot back, validated
src/bess_arb/model/      the MILP behind a pluggable backend Protocol, and the
                         two-stage bid-curve MILP
src/bess_arb/data/       one-shot ESIOS pull, OMIE cross-check, snapshot manifest
src/bess_arb/policy/     floor / forecast / oracle — price vectors, nothing more
src/bess_arb/forecast/   features with the instant each became knowable, and the
                         LightGBM fit — the only place the library is imported
src/bess_arb/bid/        bid curves traced from forecast quantiles
src/bess_arb/scenarios.py joint price scenarios from each policy's past errors
src/bess_arb/backtest/   rolling-horizon loop, metrics, the annual bound, the
                         share of the gap and its bootstrap interval
src/bess_arb/figures.py  the headline chart, drawn from a backtest summary
tests/                   the gates: golden, Δt-invariance, binaries, DST,
                         snapshot integrity, floor causality, no-lookahead,
                         import boundary
```

## How this was built

This repository was written with Claude Code, and the commits carry a
`Co-Authored-By` trailer to record that. The modelling decisions are mine (the
D-1 information gate, the binaries, the degradation cost, what counts as an
honest bound and floor) and every one of them is argued in
[docs/DECISIONS.md](docs/DECISIONS.md) next to the alternative it beat. The agent
wrote much of the code and wrote it faster than I would have; the decisions it
implements are mine to defend.

## Licence

[GNU General Public License v3.0 or later](LICENSE).

Price and forecast data are from [ESIOS](https://www.esios.ree.es/) (Red Eléctrica
de España) and [OMIE](https://www.omie.es/), and are redistributed here under their
own terms — the GPL covers the code, not the data. See
[data/README.md](data/README.md) for provenance and attribution.
