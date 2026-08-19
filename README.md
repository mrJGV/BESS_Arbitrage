# Battery arbitrage in the Spanish day-ahead market

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
uv run pytest          # golden test, Δt-invariance, binary necessity, DST
```

## Repository layout

```
config/params.yaml     every numeric parameter, plus backend/solver selection
data/                  frozen Parquet snapshot + manifest (provenance, sha256)
docs/DECISIONS.md      modelling rationale
src/bess_arb/model/    the MILP, behind a pluggable backend Protocol
src/bess_arb/policy/   floor / forecast / oracle — price vectors, nothing more
src/bess_arb/backtest/ rolling-horizon loop and metrics
tests/                 the gates: golden, Δt-invariance, binaries, no-lookahead
```

## Licence

[GNU General Public License v3.0 or later](LICENSE).

Price and forecast data are from [ESIOS](https://www.esios.ree.es/) (Red Eléctrica
de España) and [OMIE](https://www.omie.es/), and are redistributed here under their
own terms — the GPL covers the code, not the data. See
[data/README.md](data/README.md) for provenance and attribution.
