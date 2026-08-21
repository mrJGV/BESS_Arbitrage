"""Command-line entry point.

Deliberately thin. Subcommands arrive with the slice that needs them
(``run`` with the backtest, ``figures`` with the chart); this module exists
now so the ``bess-arb`` script declared in ``pyproject.toml`` is real rather
than a dangling reference.

No solver import belongs here — see ``CLAUDE.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)

    # No subcommands yet: print help rather than silently succeeding.
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
