# Battery arbitrage in the Spanish day-ahead market

<!-- toc -->
**Contents**

- [The question](#the-question)
- [Quick start](#quick-start)
- [Where it stands](#where-it-stands)
- [Repository layout](#repository-layout)
- [How this was built](#how-this-was-built)
- [Licence](#licence)

<!-- /toc -->

Day-ahead (OMIE) arbitrage demonstrator for a **10 MW / 20 MWh** battery.

A rolling-horizon MILP dispatches the battery against *forecast* prices under a
realistic information gate — bids close at 12:00 CET on D-1, so none of day D's
prices are known when the schedule is committed. The result is measured against
two reference points: a perfect-foresight upper bound and a no-information floor
policy.

> **Status: work in progress.** This README is a stub; the headline number, the
> chart and the mandatory limitations section land at v2. See
> [docs/DECISIONS.md](docs/DECISIONS.md) for the modelling rationale.

## The question

> A 10 MW / 20 MWh battery in the Spanish day-ahead market captures **X €/MW/year**,
> which is **Y%** of the perfect-foresight bound and **Z%** more than a
> no-information policy.

Three policies, one optimiser, the only difference between them being the price vector:

| Policy | Sees | Role |
|---|---|---|
| Floor | Hour-of-day × month historical averages | Lower reference |
| Forecast | LightGBM point forecast, gated at 12:00 CET D-1 | The actual result |
| Bound | Realised prices | Upper reference (perfect foresight) |

## Quick start

```bash
uv sync
uv run pytest          # golden test, Δt-invariance, binary necessity, DST,
                       # snapshot integrity, floor causality — no network,
                       # no ESIOS token

uv run bess-arb run --regime quarter_hourly --sweep
uv run bess-arb bound          # the annual-window bound, DECISIONS.md §2.4
```

The frozen data snapshot is committed, so a clone reproduces every result
without an API token. See [data/README.md](data/README.md) for provenance and
`data/manifest.json` for row counts and checksums.

## Where it stands

The forecast policy is the next slice, so the ladder currently has its middle
rung missing. Quarter-hourly regime, 319 evaluated days, `c_deg = 17 €/MWh`:

| Policy | €/MW/year | % of bound | Cycles/year |
|---|---|---|---|
| Floor (hour-of-day × month averages) | 50,472 | 85.7% | 451 |
| Forecast | *slice 4* | | |
| Bound (perfect foresight) | 58,916 | 100% | 427 |

Two things to read off that table rather than past it. **The floor is high**,
which is exactly why [§2.5](docs/DECISIONS.md) insists on having one: a
forecast policy scoring 85% would otherwise read as skill. And **cycles per
year are sane** — near one a day for a 2-hour asset, so the degradation cost
is in a plausible range and the rest of the numbers are worth reading.

The annual-window bound closed in 24.5 seconds, and the rolling 48-hour oracle
turns out to **attain it exactly**: the oracle's own dispatch is feasible for
the annual problem, so the annual optimum is at least what it earns, and the
annual solve's dual bound says it is at most that. A two-day horizon leaves
nothing on the table for a 2-hour asset, so the whole gap between a policy and
the bound is attributable to information rather than to horizon. See
[results/annual_bound.json](results/annual_bound.json).

## Repository layout

```
config/params.yaml       every numeric parameter, plus backend/solver selection
data/                    frozen Parquet snapshot + manifest (provenance, sha256)
docs/DECISIONS.md        modelling rationale
hpc/                     the one solve that does not fit on a laptop
results/                 committed reference results, with solver and gap
src/bess_arb/timeline.py UTC storage, Europe/Madrid delivery days, derived
                         periods-per-day, the 12:00 D-1 information gate
src/bess_arb/series.py   reading the frozen snapshot back, validated
src/bess_arb/model/      the MILP, behind a pluggable backend Protocol
src/bess_arb/data/       one-shot ESIOS pull, OMIE cross-check, snapshot manifest
src/bess_arb/policy/     floor / forecast / oracle — price vectors, nothing more
src/bess_arb/backtest/   rolling-horizon loop, metrics, the annual bound
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
