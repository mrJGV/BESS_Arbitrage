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
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from bess_arb import __version__

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    import pandas as pd

    from bess_arb.backtest.compare import GapShare
    from bess_arb.backtest.metrics import Metrics
    from bess_arb.backtest.runner import BacktestResult, DayResult
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
        "--charge-tariff",
        type=float,
        metavar="EUR_MWH",
        help=(
            "network tariff on charged energy, overriding "
            "battery.charge_tariff_eur_mwh (docs/DECISIONS.md section 3.4)"
        ),
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

    figures = subcommands.add_parser(
        "figures",
        help="draw the headline chart from a run's JSON summary",
        description=(
            "Reads the file `run --sweep --json` wrote and solves nothing. The "
            "chart needs all three policies at every c_deg in that file."
        ),
    )
    figures.add_argument(
        "--config", type=Path, metavar="PATH", help="config file (default: the repo's)"
    )
    figures.add_argument(
        "--json",
        type=Path,
        default=Path("results/backtest_quarter_hourly.json"),
        metavar="PATH",
        help="run summary to draw from (default: %(default)s)",
    )
    figures.add_argument(
        "--out",
        type=Path,
        default=Path("results/headline_quarter_hourly.png"),
        metavar="PATH",
        help="image to write; the format follows the suffix (default: %(default)s)",
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


def _print_metrics(metrics: Mapping[str, Metrics], order: Iterable[str]) -> None:
    """One line per policy.

    ASCII only, and ``EUR`` rather than a currency symbol: this table is the
    project's main console output and Windows terminals default to cp1252,
    where anything else arrives as a replacement character.

    ``% of gap`` is the headline and comes first; it needs both references in
    the run, and reads ``n/a`` otherwise. Cycles per year is never omitted —
    ``docs/DECISIONS.md`` §3.3.
    """
    from bess_arb.backtest.compare import gap_share

    floor, oracle = metrics.get("floor"), metrics.get("oracle")
    header = (
        f"  {'policy':<9} {'EUR/MW/year':>12} {'% of gap':>9} {'% of bound':>11} "
        f"{'cycles/yr':>10} {'profit EUR':>14} {'days':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in order:
        row = metrics[name]
        share = (
            "n/a" if row.fraction_of_bound is None else f"{row.fraction_of_bound:.1%}"
        )
        closed = (
            None
            if floor is None or oracle is None
            else gap_share(row.profit_eur, floor.profit_eur, oracle.profit_eur)
        )
        gap = "n/a" if closed is None else f"{closed:.1%}"
        print(
            f"  {row.policy:<9} {row.profit_eur_per_mw_year:>12,.0f} {gap:>9} "
            f"{share:>11} {row.equivalent_cycles_per_year:>10.1f} "
            f"{row.profit_eur:>14,.0f} {row.days:>6}"
        )


def _print_comparison(comparison: GapShare) -> None:
    """The headline share with its interval, under the table it summarises."""
    if comparison.share is None:
        print(
            f"    {comparison.policy}: no gap between the floor and the bound "
            "to close at this c_deg"
        )
        return
    line = (
        f"    {comparison.policy}: share of the floor-to-bound gap "
        f"{comparison.share:.1%}"
    )
    if comparison.interval is not None and comparison.at_or_below_floor is not None:
        low, high = comparison.interval
        line += (
            f", {comparison.confidence:.0%} interval {low:.1%} to {high:.1%}\n"
            f"      ({comparison.block_days}-day blocks, "
            f"{comparison.resamples:,} resamples); at or below the floor in "
            f"{comparison.at_or_below_floor:.1%} of them"
        )
    else:
        line += (
            "\n      no interval: some resample had no gap between floor and "
            "bound, so the run is too short to say"
        )
    print(line)


def _print_sensitivity(comparison: GapShare) -> None:
    """The same interval at a longer block, one line under the headline's."""
    if comparison.interval is None or comparison.at_or_below_floor is None:
        print(f"      {comparison.block_days}-day blocks: no interval")
        return
    low, high = comparison.interval
    print(
        f"      {comparison.block_days}-day blocks: {low:.1%} to {high:.1%}, "
        f"at or below the floor in {comparison.at_or_below_floor:.1%}"
    )


def _policy_diagnostics(name: str, policy: object) -> dict[str, object]:
    """Whatever a policy chose to record about itself, if anything.

    Both informed policies record how often they had to fall back, and both
    fall back in the flattering direction — a thin floor and an untrained
    forecaster each make the bar above them look better. So the counts are
    printed with the result rather than left in the JSON for someone to find.

    The wording is chosen here, in the presentation layer, and not asked of
    the policies: what a fallback *means* differs between them, and a generic
    sentence covering both would say nothing about either.
    """
    report = getattr(policy, "diagnostics", None)
    if report is None:
        return {}
    counts: dict[str, int] = report()

    if name == "floor":
        total = sum(counts.values())
        fallback = total - counts.get("month_time", 0)
        if fallback:
            print(
                f"    floor:    {fallback:,} of {total:,} periods "
                f"({fallback / total:.2%}) were priced off a fallback average"
            )
    elif name == "forecast":
        days = counts.get("forecast_days", 0) + counts.get("fallback_days", 0)
        if counts.get("fallback_days"):
            print(
                f"    forecast: {counts['fallback_days']:,} of {days:,} days "
                f"({counts['fallback_days'] / days:.2%}) fell back to the floor "
                "for want of training history"
            )
        if counts.get("level_only"):
            served = counts["level_only"] + counts.get("trained", 0)
            print(
                f"    forecast: {counts['level_only']:,} of {served:,} periods "
                "were priced without an intra-hour stage"
            )
    return {"diagnostics": counts}


def _print_forecast_error(
    forecaster: object, prices: pd.Series, evaluated: set[dt.date]
) -> dict[str, object]:
    """The secondary error table.

    ``docs/DECISIONS.md`` §4.1: RMSE is reported and **is not the metric**. It
    goes below the economics, never above, because a forecast can be more
    accurate and worth less — what a battery needs is the day's *ordering*, so
    the rank correlation within each delivery day is printed beside it.

    Lead 0 only: that is the forecast that priced the day the backtest
    implemented. The second horizon day exists to keep the battery from
    emptying itself at midnight, and it is never settled. And over the
    ``evaluated`` days only, so the error table describes the same days the
    economics do rather than also counting the warm-up.
    """
    import numpy as np
    import pandas as pd

    made = forecaster.predictions(0)  # type: ignore[attr-defined]
    made = made[np.asarray(_delivery_day(made.index).isin(list(evaluated)))]
    actual = prices.reindex(made.index)
    both = made.notna() & actual.notna()
    made, actual = made[both], actual[both]
    if made.empty:
        return {}

    error = made.to_numpy() - actual.to_numpy()
    day = _delivery_day(actual.index)
    rho = (
        pd.DataFrame({"f": made.to_numpy(), "a": actual.to_numpy(), "d": day})
        .groupby("d")[["f", "a"]]
        .apply(lambda g: g["f"].corr(g["a"], method="spearman"))
        .mean()
    )
    summary = {
        "periods": len(made),
        "days": int(pd.Index(day).nunique()),
        "rmse_eur_mwh": round(float(np.sqrt(np.mean(error**2))), 3),
        "mae_eur_mwh": round(float(np.mean(np.abs(error))), 3),
        "bias_eur_mwh": round(float(np.mean(error)), 3),
        "mean_daily_rank_correlation": round(float(rho), 4),
    }
    print(
        f"    forecast: error over {summary['periods']:,} lead-0 periods -- "
        f"RMSE {summary['rmse_eur_mwh']:.2f}  MAE {summary['mae_eur_mwh']:.2f}  "
        f"bias {summary['bias_eur_mwh']:+.2f} EUR/MWh  "
        f"daily rank corr {summary['mean_daily_rank_correlation']:.3f}"
    )
    return {"forecast_error": summary}


def _delivery_day(index: pd.Index) -> pd.Index:
    import pandas as pd

    from bess_arb.timeline import delivery_day

    return delivery_day(pd.DatetimeIndex(index))


def _daily(result: BacktestResult) -> dict[str, object]:
    """Settled profit per evaluated delivery day, as two parallel lists.

    Written with every run so the headline's interval can be recomputed, and
    a different block length tried, from the JSON alone rather than from a
    second backtest. Columns rather than one record per day: a hourly sweep is
    over twelve thousand of them.
    """
    days = result.evaluated
    return {
        "day": [day.day.isoformat() for day in days],
        "profit_eur": [round(day.profit_eur, 2) for day in days],
    }


def _cmd_run(args: argparse.Namespace) -> int:
    from bess_arb.backtest.compare import compare_to_floor
    from bess_arb.backtest.metrics import summarise
    from bess_arb.backtest.runner import run_backtest
    from bess_arb.policy import POLICY_NAMES, build_policy
    from bess_arb.series import load_price_history

    config, prices, regime, solver = _load(args)
    if args.charge_tariff is not None:
        config = config.with_charge_tariff(args.charge_tariff)
    backend = args.backend or config.backend
    policies: list[str] = args.policy or list(POLICY_NAMES)

    # Every policy is built from the same multi-regime price history the
    # forecaster trains on, so the floor's climatology and the forecaster's
    # training set are the same prices read under the same gate. Settlement
    # and the decision calendar still come from `prices`, the regime's own
    # series: a result belongs to one market design.
    history = load_price_history(config, args.regime)

    # Once per run, not once per sweep point: the forecast does not depend on
    # c_deg, so refitting inside the loop would triple the wall clock to
    # reproduce numbers that are identical by construction.
    forecaster = None
    if "forecast" in policies:
        from bess_arb.forecast import build_forecaster

        print(f"fitting the forecaster on the {args.regime} history", flush=True)
        forecaster = build_forecaster(config, args.regime)

    if args.c_deg:
        sweep: list[float] = list(args.c_deg)
    elif args.sweep:
        sweep = list(config.c_deg_sensitivity)
    else:
        sweep = [config.battery.c_deg_eur_mwh]

    first, last = _as_day(args.first_day), _as_day(args.last_day)
    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    bootstrap = config.report.bootstrap

    for c_deg in sweep:
        battery = config.with_c_deg(c_deg).battery
        print(
            f"\nc_deg {c_deg:g} EUR/MWh   regime {args.regime}   "
            f"backend {backend}   solver {solver.name}"
        )
        runs = {}
        built = {}
        for name in policies:
            built[name] = build_policy(name, history, forecaster=forecaster)
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
        _print_metrics(metrics, policies)

        if "floor" in runs and "oracle" in runs:
            for name in policies:
                if name in ("floor", "oracle"):
                    continue
                comparison = compare_to_floor(
                    runs[name],
                    runs["floor"],
                    runs["oracle"],
                    block_days=bootstrap.block_days,
                    resamples=bootstrap.resamples,
                    confidence=bootstrap.confidence,
                    seed=config.seed,
                )
                _print_comparison(comparison)
                row = comparison.as_dict()
                row["sensitivity"] = []
                for days in bootstrap.sensitivity_block_days:
                    longer = compare_to_floor(
                        runs[name],
                        runs["floor"],
                        runs["oracle"],
                        block_days=days,
                        resamples=bootstrap.resamples,
                        confidence=bootstrap.confidence,
                        seed=config.seed,
                    )
                    _print_sensitivity(longer)
                    row["sensitivity"].append(
                        {
                            key: value
                            for key, value in longer.as_dict().items()
                            if key
                            in (
                                "block_days",
                                "share_of_gap_interval",
                                "at_or_below_floor",
                            )
                        }
                    )
                comparisons.append(row)

        for name in policies:
            result = runs[name]
            summary: dict[str, object] = {
                **metrics[name].as_dict(),
                "charge_tariff_eur_mwh": battery.charge_tariff_eur_mwh,
                "regime": args.regime,
                "backend": backend,
                "solver_name": result.solver_name,
                "solver_version": result.solver_version,
                "window_lengths": list(result.window_lengths),
                "warmup_days": config.horizon.warmup_days,
                "first_day": result.days[0].day.isoformat(),
                "last_day": result.days[-1].day.isoformat(),
            }
            summary.update(_policy_diagnostics(name, built[name]))
            if name == "forecast" and forecaster is not None:
                summary.update(
                    _print_forecast_error(
                        forecaster, prices, {day.day for day in result.evaluated}
                    )
                )
            summary["daily"] = _daily(result)
            summaries.append(summary)

    if args.json is not None:
        _write_json(args.json, {"runs": summaries, "comparisons": comparisons})
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from bess_arb.config import load_config
    from bess_arb.figures import draw_headline

    config = load_config(args.config)
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    forecast = next(run for run in payload["runs"] if run["policy"] == "forecast")
    days = forecast["daily"]["day"]
    battery = config.battery
    title = (
        f"{battery.p_max_mw:g} MW / {battery.e_max_mwh:g} MWh battery, OMIE day-ahead, "
        f"{forecast['regime'].replace('_', '-')} regime: {len(days)} days, "
        f"{days[0]} to {days[-1]}"
    )
    draw_headline(payload, args.out, title=title, central_c_deg=battery.c_deg_eur_mwh)
    print(f"wrote {args.out}")
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
            # Unrounded, so "the rolling oracle reaches the dual bound" can
            # be checked from the file to the last bit rather than to the
            # cent the summaries above are rounded to.
            "annual_objective_eur_exact": result.objective_eur,
            "annual_bound_eur_exact": result.objective_bound_eur,
            "rolling_profit_eur_exact": metrics.profit_eur,
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
    ("figures", None): _cmd_figures,
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
