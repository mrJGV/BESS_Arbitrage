# Frozen data snapshot — 24 August 2026

<!-- toc -->
**Contents**

- [Files](#files)
- [Series](#series)
  - [The geography is load-bearing](#the-geography-is-load-bearing)
  - [Only the D+1 forecast family is usable](#only-the-d1-forecast-family-is-usable)
  - [Granularity was observed, not assumed](#granularity-was-observed-not-assumed)
- [Daylight saving](#daylight-saving)
- [Cross-check against OMIE](#cross-check-against-omie)
- [Negative prices](#negative-prices)
- [Provenance and licence](#provenance-and-licence)

<!-- /toc -->

Day-ahead prices and day-ahead forecast series for the Spanish peninsular
system. **Downloaded once and committed**, so a clone reproduces every result
in this repository without an ESIOS token and without the network.

After this snapshot the download code is not modified. If a new series turns
out to be needed, that is a decision to take deliberately, not a patch —
CLAUDE.md invariant 6.

`manifest.json` is the machine-readable version of everything below, with
row counts and SHA-256 for every file. `tests/test_snapshot.py` checks the
files against it, so the manifest is evidence rather than description.

## Files

| File | Grid | Days covered | Rows | Size |
|---|---|---|---|---|
| `hourly_2026-08-24.parquet` | 1 h | 2022-01-01 → 2025-09-30 | 32,855 | 0.74 MB |
| `quarter_hourly_2026-08-24.parquet` | 15 min | 2025-10-01 → 2026-08-23 | 31,392 | 0.33 MB |
| `quarter_hourly_exog_hourly_2026-08-24.parquet` | 1 h | 2025-10-01 → 2026-08-23 | 7,848 | 0.16 MB |

Index is `datetime_utc`, tz-aware **UTC**, labelled at the *start* of each
period. Delivery days are delimited in **Europe/Madrid**. Row counts are the
sum of each day's own length, never 24 × days — see below.

**Why three files rather than two.** A Parquet has one index, and the
forecast series are still published hourly after the market moved to
15-minute market time units. Forward-filling them onto the 96-period grid is
a modelling decision that belongs to the forecaster, and freezing it here
would make it unrevisable without re-pulling. So each file holds exactly one
grid and the mismatch stays visible. `docs/DECISIONS.md` §5.4.

## Series

| Column | ESIOS indicator | Name | Geography | Unit |
|---|---|---|---|---|
| `price_eur_mwh` | 600 | Precio mercado SPOT Diario | 3 (España) | €/MWh |
| `demand_forecast_mw` | 1775 | Previsión diaria D+1 demanda | 8741 (Península) | MW |
| `wind_forecast_mw` | 1777 | Previsión diaria D+1 eólica | 8741 (Península) | MW |
| `solar_forecast_mw` | 1779 | Previsión diaria D+1 fotovoltaica | 8741 (Península) | MW |

Indicator IDs were verified against the live catalogue on 24 August 2026, not
taken from memory. Three things that verification turned up are worth
recording, because each produces a plausible wrong number rather than an
error.

### The geography is load-bearing

Indicator 600 carries **six** geographies — Portugal, France, Spain, Germany,
Belgium, the Netherlands — returned interleaved on the same timestamps, with
**Portugal first**. Code that deduplicates on the timestamp alone stores the
Portuguese price and calls it Spanish.

MIBEL couples the two markets, so they agree whenever the interconnector is
uncongested. Over January 2024 they differ in **36 of 744 periods, by up to
€39.07/MWh** — often enough to move an arbitrage result, rarely enough to
survive inspection.

### Only the D+1 forecast family is usable

A forecast is defined by its vintage — when it was made. Neither ESIOS family
carries a "made at" field; both are indexed by the target timestamp, so they
are indistinguishable in shape.

`Previsión diaria D+1` (used here) is published once a day covering D+1 and is
never rewritten. The rolling versions (460 demand, 541 wind, 542
photovoltaic) are live operational series, overwritten as the day approaches
and passes, so a historical query returns a value written close to or during
the hour it describes.

Measured, using wind (1777 against 541) and realised generation (551):

| | D+1 (1777) | Rolling (541) |
|---|---|---|
| A future day (2026-08-25, queried 2026-08-24) | agree to RMSE 126 MW, corr 0.9986 | |
| A completed day (2026-08-22) vs realised | RMSE 1106 MW | RMSE 624 MW |
| March 2024 vs realised | RMSE 2043 MW | RMSE 896 MW |

Before the day happens the two are nearly the same forecast; afterwards the
rolling one has moved toward what occurred. Same series, same target hours, so
the rolling values were rewritten in between.

This is not a claim that 624 MW is suspiciously accurate — it is ordinary for
a 1–3 hour horizon, as 1106 MW is for a 12–36 hour one. What identifies the
vintage is that the two separate only once the target has passed. Only the D+1
family exists at the 12:00 D-1 gate.

### Granularity was observed, not assumed

ESIOS moved indicator 600 onto a 15-minute grid on **1 January 2025**, nine
months before the market itself moved on delivery day 1 October 2025. Through
that pre-period the four values inside each hour are identical: maximum
intra-hour spread **0.0000 €/MWh** to 30 September 2025, and **€65.50** on
1 October.

So the regime boundary is confirmed by the data and not only by the market
notice. The hourly file downsamples the 2025 pre-period back to hourly, and
does so only after checking that every hour is constant across its four
quarters — a lossless collapse, which the builder refuses to perform if the
values ever vary.

## Daylight saving

Spanish delivery days are 23, 24 or 25 hours long (92, 96 or 100
quarter-hourly periods). Both directions occur inside this snapshot, so
neither branch of the handling is untested:

| Day | Hourly | Quarter-hourly |
|---|---|---|
| 2022-03-27, 2023-03-26, 2024-03-31, 2025-03-30 | 23 rows | — |
| 2022-10-30, 2023-10-29, 2024-10-27 | 25 rows | — |
| 2025-10-26 | — | 100 rows |
| 2026-03-29 | — | 92 rows |

Over a full year one short day and one long day cancel, so a snapshot shifted
by an hour still totals 24 h/day. `tests/test_snapshot.py` therefore checks
the transition days individually.

## Cross-check against OMIE

ESIOS is Red Eléctrica publishing the market result; OMIE is the market
operator publishing it directly. Agreement between them rules out a wrong
indicator, a wrong geography and an off-by-one hour in one measurement.

June 2024, hour by hour against OMIE's `marginalpdbc` files:

```
periods compared        720
maximum difference      0.00 €/MWh
periods over tolerance  0     (tolerance 0.01 €/MWh)
```

Reproduce with `bess-arb data crosscheck` (needs a token and the network).

## Negative prices

782 negative hourly periods and 2,654 negative quarter-hourly periods. This
is not trivia: `docs/DECISIONS.md` §3.2 justifies the MILP's binary variables
by the existence of negative prices, and a frozen window containing none
would make that argument about a market this snapshot does not cover.
`tests/test_snapshot.py` asserts they are present.

## Provenance and licence

- **Primary source:** [ESIOS](https://www.esios.ree.es/), Red Eléctrica de
  España. Free personal API token, requested from `consultasios@ree.es`.
- **Cross-check source:** [OMIE](https://www.omie.es/) public file access,
  `marginalpdbc` daily files. No credentials required.
- **Retrieved:** 24 August 2026.

The GPL in this repository covers the **code**. The data is redistributed
here under ESIOS's and OMIE's own terms and remains theirs; attribute them,
not this repository, if you reuse it.
