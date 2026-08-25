# The one solve that does not fit on a laptop

<!-- toc -->
**Contents**

- [What runs here, and what deliberately does not](#what-runs-here-and-what-deliberately-does-not)
- [Run it twice, on purpose](#run-it-twice-on-purpose)
- [Switching solver is one flag, and that is the test](#switching-solver-is-one-flag-and-that-is-the-test)
- [What comes back](#what-comes-back)

<!-- /toc -->

Everything published in this repository runs locally on HiGHS, from a clean
clone, with no licence. This directory is the single exception, and it exists
for one number.

## What runs here, and what deliberately does not

`docs/DECISIONS.md` §2.4 asks for a bound over an **annual window with the
state of charge free across days**, to separate the value of *horizon* from
the value of *information*. Hourly that is 8,760 periods — roughly 35,000
variables of which 8,760 are binary; quarter-hourly, ~140,000 and 35,040.

That is not a window solve, and the obvious escape is closed. Relaxing the
integrality would let the solution charge and discharge simultaneously
wherever the price is negative, scoring *above* the true optimum — so the
denominator of every ratio in the project would be inflated by an unknown
amount (CLAUDE.md invariant 5). A bound is allowed to be loose. It is not
allowed to be wrong in the direction that flatters the result.

**The main backtest stays local.** A window is ~768 variables and solves in
tens of milliseconds; the whole three-bar chart is minutes on a laptop.
Moving it here would break the termination criterion — one command from a
clean clone — and buy nothing. Only the annual bound comes to the cluster.

## Run it twice, on purpose

| Run | Where | Reports |
|---|---|---|
| Reference | Cluster, Gurobi | The proven optimum, or the tightest gap reached |
| Portable | Laptop, HiGHS | The gap and wall clock reachable with no licence |

The comparison — *"HiGHS reached x% in n minutes; Gurobi proved optimality in
m"* — is a practitioner observation worth more than either number alone, and
both runs were needed anyway. It also keeps the repository honest: nothing
published depends on an academic licence.

```bash
# portable, from the repository root
uv run bess-arb bound --json results/annual_bound.json

# reference
sbatch hpc/annual_bound.slurm
```

## Switching solver is one flag, and that is the test

`--solver gurobi` is the whole difference. Same `config/params.yaml`, same
`backend: pyomo`, same `src/bess_arb/model/pyomo.py` — the solver is named by
string and resolved inside the modelling layer, which is what
`docs/DECISIONS.md` §6.2 claims and what this script exercises.

**If running here ever needs a code change, the abstraction has leaked** and
that is the finding, not an inconvenience to work around. Fix it in
`model/`, not with a branch on hostname.

`--threads` is passed explicitly rather than left to the solver's default:
the local runs pin one thread for determinism, and a reference run that used
an unrecorded number of cores would not be comparable with anything.

## What comes back

`results/annual_bound.json`, committed. It carries the incumbent, the dual
bound, the gap actually reached, the wall clock, the solver and its version,
and the machine — so a reader who cannot re-run it can still see how much
slack the published number carries and what produced it.

The rolling oracle over the identical days is solved in the same command and
written into the same file, because the annual figure on its own is not the
quantity anyone wants: the difference between the two is.
