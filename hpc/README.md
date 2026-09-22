# The one solve that might not have fitted on a laptop

<!-- toc -->
**Contents**

- [What runs here, and what deliberately does not](#what-runs-here-and-what-deliberately-does-not)
- [Run it twice, on purpose](#run-it-twice-on-purpose)
- [Switching solver is one flag, and that is the test](#switching-solver-is-one-flag-and-that-is-the-test)
- [What comes back](#what-comes-back)

<!-- /toc -->

Everything published in this repository runs locally on HiGHS, from a clean
clone, with no licence. This directory was the provision for the single
candidate exception, and it exists for one number.

**Status: not needed, kept anyway.** Both regimes have now been solved on the
laptop — hourly in 13.5 s, quarter-hourly in 49.1 s, against a budgeted hour —
so no published result depends on cluster access and no reference run is
scheduled. The scripts and the licence stay in place as contingency for a case
that does not exist in the frozen snapshot: a full calendar year of
quarter-hourly data, roughly 35,040 binaries against the 31,296 actually
measured. Sizing that against two solves whose wall clock grew sub-linearly is
a guess, which is the argument for keeping the provision rather than removing
it.

## What runs here, and what deliberately does not

`docs/DECISIONS.md` §2.4 asks for a bound over an **annual window with the
state of charge free across days**, to separate the value of *horizon* from
the value of *information*. Hourly that is 8,760 periods — roughly 35,000
variables of which 8,760 are binary; quarter-hourly, ~140,000 and 35,040 over a
full year. The snapshot holds 326 quarter-hourly days rather than 365, so the
solve that actually ran is 125,184 variables and 31,296 binaries.

That is not a window solve, and the obvious escape is closed. Relaxing the
integrality would let the solution charge and discharge simultaneously
wherever the price is negative, scoring *above* the true optimum — so the
denominator of every ratio in the project would be inflated by an unknown
amount (CLAUDE.md invariant 5). A bound is allowed to be loose. It is not
allowed to be wrong in the direction that flatters the result.

**The main backtest stays local.** A window is ~768 variables and solves in
tens of milliseconds; the whole three-bar chart is minutes on a laptop.
Moving it here would break the termination criterion — one command from a
clean clone — and buy nothing. The annual bound was the only candidate for
the cluster, and in the event it did not need to go either.

## Run it twice, on purpose

| Run | Where | Reports | Status |
|---|---|---|---|
| Reference | Cluster, Gurobi | The proven optimum, or the tightest gap reached | not run |
| Portable | Laptop, HiGHS | The gap and wall clock reachable with no licence | both regimes |

The comparison — *"HiGHS reached x% in n minutes; Gurobi proved optimality in
m"* — would have been a practitioner observation worth more than either number
alone. It was not bought, because the portable run answered the question on its
own: HiGHS proved the hourly case outright and left 8.2e-05 on the
quarter-hourly one, which the rolling oracle then closed from the primal side.
The repository is honest for the simpler reason that nothing published depends
on an academic licence at all.

```bash
# portable, from the repository root
uv run bess-arb bound --json results/annual_bound.json
uv run bess-arb bound --regime quarter_hourly --first-day 2025-10-01 --last-day 2026-08-22 --json results/annual_bound_quarter_hourly.json

# reference, if it is ever wanted
sbatch hpc/annual_bound.slurm
```

The quarter-hourly window is given explicitly because `bound.annual` in
`config/params.yaml` names a calendar year inside the hourly regime, and the
quarter-hourly snapshot has no calendar year in it. The last decision day is
2026-08-22 rather than the snapshot's final day, because the rolling oracle
compared against it needs D+1 prices to exist.

## Switching solver is one flag, and that is the test

`--solver gurobi` is the whole difference. Same `config/params.yaml`, same
`backend: pyomo`, same `src/bess_arb/model/pyomo.py` — the solver is named by
string and resolved inside the modelling layer, which is what
`docs/DECISIONS.md` §6.2 claims. The claim rests on the boundary being enforced
in code — the `import-linter` contract and `tests/test_import_boundary.py` — and
not on this script, which has not been run.

**If running here ever needs a code change, the abstraction has leaked** and
that is the finding, not an inconvenience to work around. Fix it in
`model/`, not with a branch on hostname.

`--threads` is passed explicitly rather than left to the solver's default:
the local runs pin one thread for determinism, and a reference run that used
an unrecorded number of cores would not be comparable with anything.

## What comes back

`results/annual_bound.json` and `results/annual_bound_quarter_hourly.json`,
both committed. Each carries the incumbent, the dual bound, the gap actually
reached, the wall clock, the solver and its version, and the machine — so a
reader who cannot re-run it can still see how much slack the published number
carries and what produced it.

The gap is not decorative on the quarter-hourly run. Its incumbent stops
€43.72 below its dual bound, so the two are different numbers meaning different
things, and the one to quote as a bound is the **dual bound** — the guarantee,
not the achievement. Quoting the incumbent would understate the true oracle
optimum and flatter every ratio taken against it.

The rolling oracle over the identical days is solved in the same command and
written into the same file, because the annual figure on its own is not the
quantity anyone wants: the difference between the two is.
