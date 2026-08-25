"""Command-line entry point.

Deliberately thin: every subcommand parses arguments and hands straight over
to a module that could be called from a notebook just as well. Subcommands
arrive with the slice that needs them — ``data`` with the snapshot, ``run``
and ``bound`` with the backtest, ``figures`` with the chart.

Heavy imports are local to the handler that needs them, so ``bess-arb
--version`` does not pay for pandas. No solver import belongs here at all —
see ``CLAUDE.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from bess_arb import __version__

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    import pandas as pd

    from bess_arb.backtest.metrics import Metrics
    from bess_arb.backtest.runner import DayResult
    from bess_arb.config import Config
    from bess_arb.model.spec import SolverConfig
    from bess_arb.timeline import Regime


def build_parser() -> argparse.ArgumentParser:
    # Local, like every other non-stdlib import here: building the parser
    # must not drag pandas into `bess-arb --version`.
    from bess_arb.policy import POLICY_NAMES

    parser = argparse.ArgumentParser(
        prog="bess-arb",
        description=(
            "Day-ahead arbitrage demonstrator for a 10 MW / 20 MWh battery (OMIE)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bess-arb {__version__}",
    )
    subcommands = parser.add_subparsers(dest="command")

    data = subcommands.add_parser(
        "data",
        help="inspect the ESIOS catalogue and build the frozen snapshot",
        description=(
            "Snapshot tooling. The pull is one-shot by design: after the "
            "snapshot commit this code is not modified (CLAUDE.md invariant 6)."
        ),
    )
    data_commands = data.add_subparsers(dest="data_command", required=True)

    indicators = data_commands.add_parser(
        "indicators",
        help="search the ESIOS indicator catalogue by name",
        description=(
            "Search rather than trust a remembered ID. An indicator number "
            "that looks plausible and means something else produces a "
            "snapshot no downstream test can tell is wrong."
        ),
    )
    indicators.add_argument(
        "terms",
        nargs="*",
        help="terms that must all appear in the name; accents are ignored",
    )
    indicators.add_argument(
        "--limit", type=int, default=40, help="rows to show (default: 40)"
    )

    inspect = data_commands.add_parser(
        "inspect",
        help="show metadata and a one-day sample for specific indicator IDs",
    )
    inspect.add_argument("ids", nargs="+", type=int, help="indicator IDs")

    freeze = data_commands.add_parser(
        "freeze",
        help="pull every series and write the frozen Parquet snapshot",
        description=(
            "One-shot. Writes the Parquet files, runs the OMIE cross-check "
            "and emits data/manifest.json. Refuses to overwrite an existing "
            "snapshot unless --force is given."
        ),
    )
    freeze.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing snapshot for the same date",
    )
    freeze.add_argument(
        "--skip-crosscheck",
        action="store_true",
        help="skip the OMIE comparison (it costs one request per day)",
    )

    data_commands.add_parser(
        "crosscheck",
        help="compare ESIOS prices against OMIE marginalpdbc for the configured month",
    )

    run = subcommands.add_parser(
        "run",
        help="run the rolling-horizon backtest",
        description=(
            "Every policy goes through the same optimiser on the same "
            "windows; the only difference between them is the price vector "
            "(CLAUDE.md invariant 2). Equivalent cycles per year is printed "
            "on every run without exception: it says whether c_deg is "
            "plausible, and if it is not, nothing else on the line matters."
        ),
    )
    _add_common(run)
    run.add_argument(
        "--policy",
        action="append",
        choices=POLICY_NAMES,
        help="policy to run; repeatable (default: all of them)",
    )
    run.add_argument(
        "--sweep",
        action="store_true",
        help="run every c_deg in sensitivity.c_deg_eur_mwh, not just the central one",
    )
    run.add_argument(
        "--c-deg",
        type=float,
        action="append",
        metavar="EUR_MWH",
        help="degradation cost to run; repeatable, overrides --sweep",
    )
    run.add_argument(
        "--json", type=Path, metavar="PATH", help="also write the summary as JSON"
    )

    bound = subcommands.add_parser(
        "bound",
        help="solve the annual-window bound (docs/DECISIONS.md section 2.4)",
        description=(
            "One solve over a whole year with SoC free across days, against "
            "the rolling oracle over the identical days. The difference is "
            "the value of horizon as distinct from the value of information. "
            "Large and slow, and deliberately not relaxed to an LP: that "
            "would inflate the denominator of every ratio in the project."
        ),
    )
    _add_common(bound)
    bound.add_argument(
        "--json",
        type=Path,
        default=Path("results/annual_bound.json"),
        metavar="PATH",
        help="where to write the reference result (default: %(default)s)",
    )
    bound.add_argument(
        "--skip-rolling",
        action="store_true",
        help="solve only the annual window, without the rolling oracle to compare",
    )

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options the two solving subcommands share.

    ``--solver`` and friends override the corresponding
    ``config/params.yaml`` keys and nothing else. Moving from HiGHS locally to
    Gurobi on a cluster is meant to be exactly this and no code change
    (docs/DECISIONS.md §6.2); the day it needs more, the backend abstraction
    has leaked.
    """
    parser.add_argument(
        "--config", type=Path, metavar="PATH", help="config file (default: the repo's)"
    )
    parser.add_argument(
        "--regime",
        default="hourly",
        help="market regime from the config (default: %(default)s)",
    )
    parser.add_argument("--first-day", metavar="YYYY-MM-DD", help="first decision day")
    parser.add_argument("--last-day", metavar="YYYY-MM-DD", help="last decision day")
    parser.add_argument("--backend", help="override the config's backend")
    parser.add_argument("--solver", help="override solver.name")
    parser.add_argument("--mip-gap", type=float, help="override solver.mip_gap")
    parser.add_argument(
        "--time-limit",
        type=float,
        metavar="SECONDS",
        help="override solver.time_limit_s",
    )
    parser.add_argument("--threads", type=int, help="override solver.threads")
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the per-day progress line"
    )


def _cmd_data_indicators(args: argparse.Namespace) -> int:
    import pandas as pd

    from bess_arb.data.esios import EsiosClient

    client = EsiosClient()
    matches = client.search_indicators(*args.terms)
    if matches.empty:
        print("no indicator matches those terms")
        return 1

    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(matches.head(args.limit).to_string(index=False))
    if len(matches) > args.limit:
        print(f"\n... {len(matches) - args.limit} more")
    return 0


def _cmd_data_inspect(args: argparse.Namespace) -> int:
    from bess_arb.data.esios import EsiosClient

    client = EsiosClient()
    for indicator_id in args.ids:
        meta = client.indicator_metadata(indicator_id)
        print(f"\n[{meta['id']}] {meta['name']}")
        print(f"  short_name  {meta['short_name']}")
        print(f"  tiempo      {meta['tiempo']}   magnitud {meta['magnitud']}")
        print(f"  step_type   {meta['step_type']}")
        print(f"  sample      {meta['sample_rows']} rows over one day")
        print(f"  geographies {meta['sample_geos']}")
    return 0


def _cmd_data_freeze(args: argparse.Namespace) -> int:
    from bess_arb.config import load_config
    from bess_arb.data.snapshot import (
        build_snapshot,
        crosscheck_against_omie,
        write_manifest,
    )

    config = load_config()
    data = config.data

    existing = sorted(data.directory.glob("*.parquet"))
    if existing and not args.force:
        print(
            f"{len(existing)} Parquet file(s) already in {data.directory}.\n"
            "The snapshot is frozen once by design (invariant 6); re-run with "
            "--force only if you mean to replace it."
        )
        return 1

    manifest = build_snapshot(config)
    if not args.skip_crosscheck:
        print("\ncross-check against OMIE")
        manifest["crosscheck"] = crosscheck_against_omie(config)
    write_manifest(data.manifest_path, manifest)

    print(f"\nwrote {len(manifest['files'])} file(s) and {data.manifest_path.name}")
    for entry in manifest["files"]:
        print(
            f"  {entry['path']:<48} {entry['rows']:>8} rows  "
            f"{entry['bytes'] / 1e6:>6.2f} MB"
        )
    return 0


def _cmd_data_crosscheck(args: argparse.Namespace) -> int:
    from bess_arb.config import load_config
    from bess_arb.data.snapshot import crosscheck_against_omie

    print(json.dumps(crosscheck_against_omie(load_config()), indent=2))
    return 0


def _load(
    args: argparse.Namespace,
) -> tuple[Config, pd.Series, Regime, SolverConfig]:
    """Config, prices, regime and solver settings with the overrides applied.

    Shared by both solving subcommands so they cannot drift apart on how the
    config is read.
    """
    from bess_arb.config import load_config
    from bess_arb.series import load_prices, regime_of

    config = load_config(args.config)
    overrides = {
        "name": args.solver,
        "mip_gap": args.mip_gap,
        "time_limit_s": args.time_limit,
        "threads": args.threads,
    }
    solver = dataclasses.replace(
        config.solver,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    return (
        config,
        load_prices(config, args.regime),
        regime_of(config, args.regime),
        solver,
    )


def _as_day(value: str | None) -> dt.date | None:
    return None if value is None else dt.date.fromisoformat(value)


def _progress(name: str) -> Callable[[int, int, DayResult], None]:
    def _report(position: int, total: int, day: DayResult) -> None:
        if position % 50 == 0 or position == total - 1:
            print(f"\r  {name:<9} {position + 1:>5}/{total} days", end="", flush=True)

    return _report


def _print_metrics(rows: Iterable[Metrics]) -> None:
    """One line per policy.

    ASCII only, and ``EUR`` rather than a currency symbol: this table is the
    project's main console output and Windows terminals default to cp1252,
    where anything else arrives as a replacement character.

    Cycles per year is never omitted — ``docs/DECISIONS.md`` §3.3.
    """
    header = (
        f"  {'policy':<9} {'EUR/MW/year':>12} {'% of bound':>11} "
        f"{'cycles/yr':>10} {'profit EUR':>14} {'days':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        share = (
            "n/a" if row.fraction_of_bound is None else f"{row.fraction_of_bound:.1%}"
        )
        print(
            f"  {row.policy:<9} {row.profit_eur_per_mw_year:>12,.0f} {share:>11} "
            f"{row.equivalent_cycles_per_year:>10.1f} {row.profit_eur:>14,.0f} "
            f"{row.days:>6}"
        )


def _policy_diagnostics(policy: object) -> dict[str, object]:
    """Whatever a policy chose to record about itself, if anything.

    Only the floor has anything to say so far: how many periods it had to
    price off a fallback because the month-of-year average was not yet
    populated. It is reported rather than argued about because a thin floor
    flatters everything measured against it.
    """
    report = getattr(policy, "diagnostics", None)
    if report is None:
        return {}
    counts: dict[str, int] = report()
    total = sum(counts.values())
    fallback = total - counts.get("month_time", 0)
    if fallback:
        print(
            f"    note: {fallback:,} of {total:,} periods "
            f"({fallback / total:.2%}) were priced off a fallback average"
        )
    return {"diagnostics": counts}


def _cmd_run(args: argparse.Namespace) -> int:
    from bess_arb.backtest.metrics import summarise
    from bess_arb.backtest.runner import run_backtest
    from bess_arb.policy import POLICY_NAMES, build_policy

    config, prices, regime, solver = _load(args)
    backend = args.backend or config.backend
    policies: list[str] = args.policy or list(POLICY_NAMES)

    if args.c_deg:
        sweep: list[float] = list(args.c_deg)
    elif args.sweep:
        sweep = list(config.c_deg_sensitivity)
    else:
        sweep = [config.battery.c_deg_eur_mwh]

    first, last = _as_day(args.first_day), _as_day(args.last_day)
    summaries: list[dict[str, object]] = []

    for c_deg in sweep:
        battery = config.with_c_deg(c_deg).battery
        print(
            f"\nc_deg {c_deg:g} EUR/MWh   regime {args.regime}   "
            f"backend {backend}   solver {solver.name}"
        )
        runs = {}
        built = {}
        for name in policies:
            built[name] = build_policy(name, prices)
            runs[name] = run_backtest(
                prices,
                built[name],
                battery,
                regime,
                config.horizon,
                backend=backend,
                solver=solver,
                first_day=first,
                last_day=last,
                on_day=None if args.quiet else _progress(name),
            )
            if not args.quiet:
                print()

        metrics = {name: summarise(result) for name, result in runs.items()}
        reference = metrics.get("oracle")
        if reference is not None:
            metrics = {
                name: value.with_bound(reference) for name, value in metrics.items()
            }
        _print_metrics(metrics[name] for name in policies)

        for name in policies:
            result = runs[name]
            summary: dict[str, object] = {
                **metrics[name].as_dict(),
                "regime": args.regime,
                "backend": backend,
                "solver_name": result.solver_name,
                "solver_version": result.solver_version,
                "window_lengths": list(result.window_lengths),
                "warmup_days": config.horizon.warmup_days,
                "first_day": result.days[0].day.isoformat(),
                "last_day": result.days[-1].day.isoformat(),
            }
            summary.update(_policy_diagnostics(built[name]))
            summaries.append(summary)

    if args.json is not None:
        _write_json(args.json, {"runs": summaries})
    return 0


def _cmd_bound(args: argparse.Namespace) -> int:
    from bess_arb.backtest.bound import environment, solve_annual_bound
    from bess_arb.backtest.metrics import summarise
    from bess_arb.backtest.runner import run_backtest
    from bess_arb.policy import build_policy

    config, prices, regime, window_solver = _load(args)
    backend = args.backend or config.backend
    annual = config.bound.annual
    battery = config.battery

    first = _as_day(args.first_day) or annual.first_day
    last = _as_day(args.last_day) or annual.last_day

    # The annual solve gets its own tolerances unless the caller overrode
    # them on the command line: a window closes the gap exactly, a year of
    # 8,760 binaries cannot afford to.
    annual_solver = dataclasses.replace(
        window_solver,
        mip_gap=window_solver.mip_gap if args.mip_gap is not None else annual.mip_gap,
        time_limit_s=(
            window_solver.time_limit_s
            if args.time_limit is not None
            else annual.time_limit_s
        ),
    )

    print(
        f"annual window {first} .. {last}   regime {args.regime}   "
        f"solver {annual_solver.name}   gap {annual_solver.mip_gap}   "
        f"limit {annual_solver.time_limit_s}s"
    )
    print("solving; this is not a window solve and will take a while", flush=True)

    result = solve_annual_bound(
        prices,
        battery,
        regime,
        first,
        last,
        soc_initial_mwh=config.horizon.soc_initial_mwh(battery.e_max_mwh),
        backend=backend,
        solver=annual_solver,
    )

    gap = (
        "not reported" if result.relative_gap is None else f"{result.relative_gap:.4%}"
    )
    dual = (
        "not reported"
        if result.objective_bound_eur is None
        else f"EUR {result.objective_bound_eur:,.2f}"
    )
    print(
        f"\n  status            {result.status.value}\n"
        f"  objective         EUR {result.objective_eur:,.2f}\n"
        f"  dual bound        {dual}\n"
        f"  relative gap      {gap}\n"
        f"  wall clock        {result.wall_clock_s:,.1f} s\n"
        f"  equivalent cycles {result.equivalent_cycles:,.1f}"
    )

    payload: dict[str, object] = {
        "annual_window": result.as_dict(),
        "battery": dataclasses.asdict(battery),
        "backend": backend,
        "environment": environment(),
    }

    if not args.skip_rolling:
        print("\nrolling oracle over the identical days")
        rolling = run_backtest(
            prices,
            build_policy("oracle", prices),
            battery,
            regime,
            # No warm-up: the comparison is only about horizon, so both runs
            # must cover exactly the same days.
            dataclasses.replace(config.horizon, warmup_days=0),
            backend=backend,
            solver=window_solver,
            first_day=first,
            last_day=last,
            on_day=None if args.quiet else _progress("oracle"),
        )
        metrics = summarise(rolling)
        difference = result.objective_eur - metrics.profit_eur
        share = (
            None if abs(metrics.profit_eur) < 1e-9 else difference / metrics.profit_eur
        )
        # The free horizon cannot truly be worth less than the rolling one:
        # the rolling dispatch is itself a feasible point of the annual
        # problem. So a negative difference is never the answer to the
        # question -- it is the annual solve's own unclosed gap, and the
        # rolling run has just certified how far short its incumbent
        # stopped. When the rolling profit also reaches the dual bound the
        # two sandwich the optimum and the value of horizon is exactly zero,
        # which is worth saying rather than leaving a reader to wonder
        # whether the bound is broken.
        slack = abs(result.objective_eur - (result.objective_bound_eur or 0.0))
        within_tolerance = difference < 0.0 and abs(difference) <= slack + 1e-6

        payload["rolling_oracle"] = metrics.as_dict()
        payload["value_of_horizon"] = {
            "annual_minus_rolling_eur": round(difference, 2),
            "fraction_of_rolling": None if share is None else round(share, 6),
            "within_annual_solve_gap": within_tolerance,
        }
        print(
            f"\n\n  rolling oracle    EUR {metrics.profit_eur:,.2f}\n"
            f"  value of horizon  EUR {difference:,.2f}"
            + ("" if share is None else f" ({share:+.2%} of rolling)")
        )
        if within_tolerance:
            reached_bound = (
                result.objective_bound_eur is not None
                and abs(metrics.profit_eur - result.objective_bound_eur)
                <= max(abs(result.objective_bound_eur), 1.0) * 1e-9
            )
            print(
                "                    not a loss: that shortfall is the "
                f"annual solve's own unclosed gap of EUR {slack:,.2f}"
            )
            if reached_bound:
                print(
                    "                    the rolling oracle reaches the "
                    "annual dual bound, so it attains the annual optimum "
                    "and the value of horizon is exactly zero"
                )

    _write_json(args.json, payload)
    return 0


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


_DISPATCH: dict[tuple[str, str | None], Callable[[argparse.Namespace], int]] = {
    ("data", "indicators"): _cmd_data_indicators,
    ("data", "inspect"): _cmd_data_inspect,
    ("data", "freeze"): _cmd_data_freeze,
    ("data", "crosscheck"): _cmd_data_crosscheck,
    ("run", None): _cmd_run,
    ("bound", None): _cmd_bound,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handler = _DISPATCH.get((args.command, getattr(args, "data_command", None)))
    if handler is None:  # pragma: no cover - argparse rejects this first
        parser.error(f"unknown command {args.command!r}")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
