"""Command-line entry point.

Deliberately thin: every subcommand parses arguments and hands straight over
to a module that could be called from a notebook just as well. Subcommands
arrive with the slice that needs them — ``data`` with the snapshot, ``run``
with the backtest, ``figures`` with the chart.

No solver import belongs here — see ``CLAUDE.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from bess_arb import __version__


def build_parser() -> argparse.ArgumentParser:
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

    return parser


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
    import json

    from bess_arb.config import load_config
    from bess_arb.data.snapshot import crosscheck_against_omie

    print(json.dumps(crosscheck_against_omie(load_config()), indent=2))
    return 0


_DISPATCH: dict[tuple[str, str | None], Callable[[argparse.Namespace], int]] = {
    ("data", "indicators"): _cmd_data_indicators,
    ("data", "inspect"): _cmd_data_inspect,
    ("data", "freeze"): _cmd_data_freeze,
    ("data", "crosscheck"): _cmd_data_crosscheck,
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
