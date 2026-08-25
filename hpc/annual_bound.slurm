#!/bin/bash
# Reference run for the annual-window bound (docs/DECISIONS.md §2.4).
#
# Submit from the repository root:
#   sbatch hpc/annual_bound.slurm
#
# This is the ONLY part of the project that runs anywhere but a laptop, and
# it is a secondary diagnostic reported in one paragraph. The headline
# three-bar chart stays local on HiGHS — see hpc/README.md for why moving it
# here would cost the one-command reproduction and buy nothing.

#SBATCH --job-name=bess-annual-bound
#SBATCH --output=hpc/logs/annual_bound-%j.out
#SBATCH --error=hpc/logs/annual_bound-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

mkdir -p hpc/logs results

module load gurobi || echo "no gurobi module; assuming it is already on PATH"

# --locked: the cluster resolves the same dependency set the result was
# produced with, or it fails. A reference number from an unknown environment
# is not a reference number.
uv sync --all-groups --locked

# Everything that differs from the local run is on this command line: the
# solver name, the thread count and the tolerances. No code path is
# cluster-specific, and `model/pyomo.py` is byte-identical to the one HiGHS
# runs through — that is docs/DECISIONS.md §6.2's claim, tested by using it.
uv run bess-arb bound \
    --regime hourly \
    --solver gurobi \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --mip-gap 1e-4 \
    --time-limit 28000 \
    --quiet \
    --json results/annual_bound.gurobi.json

echo "done; commit results/annual_bound.gurobi.json alongside the HiGHS run"
