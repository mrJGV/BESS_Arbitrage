# Modelling decisions

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

That annual solve is ~35,000 variables and 8,760 binaries hourly (~140,000 and 35,040
quarter-hourly), so it is reported twice: a reference run on a cluster with a
commercial solver, and a portable run on HiGHS with an explicit gap tolerance. It is
*not* relaxed to an LP — that would inflate the denominator, which is the one thing a
bound must never do.

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

---

## 5. Data

- **Primary source:** ESIOS API (Red Eléctrica de España).
- **Cross-check:** OMIE `marginalpdbc` files, validated against ESIOS for at least
  one month.
- **Format:** Parquet, one file per regime, snapshot date in the filename.
- **Downloaded once.** After the snapshot the download code is not modified. The
  project risk is not the MILP — a 192-period problem solves in milliseconds — it is
  losing a week to file plumbing.

Series: day-ahead marginal price, plus wind, solar and demand **forecast** series as
exogenous forecaster inputs. Using *realised* wind or demand is information leakage
and destroys the backtest.

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
