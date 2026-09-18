# Project context

<!-- toc -->
**Contents**

- [Project context](#project-context)
- [Invariants — never violate](#invariants--never-violate)
- [Modelling layer](#modelling-layer)
- [Working rules](#working-rules)
- [Git](#git)

<!-- /toc -->

Arbitrage demonstrator for a 10 MW / 20 MWh battery in the Spanish
day-ahead market (OMIE). The deliverable is a clean repository with a
three-bar chart. It is not a paper and not a platform.

Published rationale: `docs/DECISIONS.md`. **It is frozen as of the v1
close and is not updated as work proceeds** — what becomes public is
decided once, at the repo freeze.

The working record is kept outside the tracked tree. Decisions, design
and measurements are recorded there as work proceeds. Do not add those
files to the repo, do not name them, and do not describe their contents
in tracked files or commit messages.

Every document carries a generated index. After editing one, run
`make toc`; `make check` fails if an index is stale.

# Invariants — never violate

1. **No lookahead.** No variable used to decide on day D may have been
   *published* after 12:00 CET on D-1. The invariant is read on
   publication time, for every series: the REE D+1 forecasts are on the
   wire the morning before the gate, and a delivery day's prices are
   published about 13:00 on the day before delivery, so at the gate for
   D every price through D-1 is known and none of D's is. Use published
   forecast series as exogenous inputs, never realised values.

2. **One optimiser.** Every policy (floor, forecast, oracle) obtains its
   schedule from the same `BatteryMILP` instance via
   `.solve(prices, soc_initial)`. **The only thing that differs between
   policies is the price vector passed in.** Do not duplicate the
   formulation, and do not give a policy its own constraint set.
   - The oracle passes realised prices.
   - The forecast policy passes forecast prices.
   - The floor policy passes the hour-of-day × month climatological
     price vector. It is not a hand-rolled "N cheapest periods"
     heuristic; that is the *effect*, not the implementation.

3. **Explicit Δt.** Never assume one-hour periods. The model must work
   unchanged with Δt = 1 and Δt = 0.25.

4. **Never 24 periods per day.** DST transition days have 23 or 25 hours
   (92 or 100 quarter-hourly periods). Store in UTC, present in
   Europe/Madrid. Derive periods-per-day; never hardcode it.

5. **Binaries are required.** Charge/discharge exclusivity needs binary
   variables because negative prices exist. Do not relax the MILP to an
   LP "because the optimum already satisfies it" — with negative prices
   it does not. This also rules out relaxing the *bound*: inflating the
   denominator is forbidden.

6. **Frozen data.** After the initial Parquet snapshot, download code is
   not modified. If a new series is needed, ask first.

7. **Single YAML config.** No numeric parameter hardcoded in modelling
   code. `config/params.yaml` also carries `backend:` and `solver:`, so
   switching solver never requires a code change.

8. **Fixed seeds** everywhere randomness appears.

# Modelling layer

The MILP lives behind a pluggable backend. `src/bess_arb/model/spec.py`
and `src/bess_arb/model/base.py` define dataclasses, our own
`SolveStatus` enum, and the `BatteryMILP` Protocol, and import nothing
solver-related. Each backend (`model/pyomo.py`, `model/poi.py`, ...)
implements that Protocol.

**No solver library may be imported anywhere outside
`src/bess_arb/model/`.** Not in the backtest loop, not in plotting, not
in tests other than the backend equivalence test. Solver status codes are
always normalised to `SolveStatus`; a raw solver enum must never escape
`model/`. Enforced by an `import-linter` contract and by
`tests/test_import_boundary.py`.

The v1 backend is **Pyomo** (`model/pyomo.py`), using a persistent
interface so the model is built once per regime and re-solved per window.
Build this one first and get the full backtest running through it before
any other backend exists.

**Use `SolverFactory("highs")`, not `"appsi_highs"`.** Verified 19 August
2026: both are persistent and both re-solve correctly after
mutating prices and `soc_initial`, but `"highs"` is the newer
`pyomo.contrib.solver` implementation and is measurably faster
(~48 ms vs ~62 ms per 192-period solve). Prices and initial SoC are
`Param(mutable=True)`; mutating them and re-solving is verified to give
the new optimum, not a stale one.

PyOptInterface (`model/poi.py`) is a **later** performance port, not the
starting point. Do not begin it until the Pyomo backtest runs end to end
and the golden test passes. When that time comes: **use
`docs/poi_reference.py` as the authoritative API reference**, do not
write PyOptInterface calls from memory, and if a needed pattern is not
in the reference, say so and ask rather than guessing at a method name.

Every backend builds the model once per regime and then, per window,
updates only the objective price coefficients and the initial SoC bound
before re-solving. Do not rebuild the model inside the loop. For the
Pyomo backend this means a persistent/APPSI interface, not a plain
`SolverFactory` call per window.

# Working rules

- One complete, tested vertical slice before starting the next.
- **Explain the formulation, do not just emit it.** When writing or
  changing the MILP, give the modelling choice in one or two lines (why a
  binary, why that big-M, why that sign) rather than only producing code.
  Flag any constraint whose tightness you are unsure of. Put that
  rationale in the code and in the working record — not in the commit
  message.
- Golden test is mandatory: a four-period price series with two low and
  two high prices, optimal profit computed by hand and verified. It must
  pass before any refactor. Prices `[10, 10, 100, 100]`, Δt = 1,
  SoC₀ = 0, `c_deg` = 0 → **€1,500**, from
  `20 MWh × (0.85 × 100 − 10)`. Verified against HiGHS on 19 August.
  **Assert on the objective, never on the dispatch vector:** when adjacent
  prices are equal the optimal dispatch is non-unique (HiGHS returns
  `p_d = [0, 0, 7, 10]`, not the `[0, 0, 8.5, 8.5]` one might write down),
  so a test pinning the schedule is wrong, not strict.
- Backend equivalence test is mandatory once a second backend exists:
  the golden case plus two random 96-period instances through every
  registered backend, objectives agreeing to 1e-6.
- Always report equivalent cycles per year alongside any economic
  result. It is the sanity diagnostic.
- Do not delegate without asking: the MILP formulation, the backtest
  protocol, and the bound definition. Everything else (parsers, plots,
  CLI, tests, packaging) is fully delegated.

# Git

Conventional Commits with module scope (`feat(model):`, `test(model):`,
`chore(data):`, ...). **A commit message says what changed, never why** —
the reasoning belongs in the code and the documents, which already carry
it. One branch per slice, PR into `main`, **merge commit** (never squash)
with CI green. Tags `v0`/`v1`/`v2` mark the version ladder. The data
snapshot gets its own commit so the freeze point is one hash; a squash would
fold it into its slice.
