"""The headline chart, drawn from a run's JSON summary.

Drawn from the file ``bess-arb run --sweep --json`` writes rather than from a
backtest in memory, so redrawing costs nothing and the chart can only show
what the summary on disk says.

Two panels, both over the whole ``c_deg`` sweep:

- **left, the headline**: the forecast's share of the gap between the floor and
  the bound, with its bootstrap interval (:mod:`bess_arb.backtest.compare`).
  The floor is the zero line and the bound is 100%, off the top of the axis.
- **right, the scale**: the three policies in €/MW/year, each bar carrying its
  own cycles per year, because no economic number in this project is shown
  without them (``docs/DECISIONS.md`` §3.3). On a zero-based axis the floor and
  forecast bars are nearly the same height, which is the reason the left panel
  exists.

The whole sweep and not the central point alone: the share changes sign
between the low and the central degradation cost, and one bar would hide that.

Built on :class:`matplotlib.figure.Figure` directly, not on ``pyplot``, so
drawing needs no display, selects no backend and leaves no global figure
state behind in a test run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

__all__ = ["draw_headline"]

POLICIES = ("floor", "forecast", "oracle")
LABELS = {"floor": "Floor", "forecast": "Forecast", "oracle": "Bound"}
COLOURS = {"floor": "#9a9a9a", "forecast": "#0072b2", "oracle": "#3b3b3b"}
MINUS = "\N{MINUS SIGN}"


def draw_headline(
    payload: Mapping[str, Any],
    path: Path,
    *,
    title: str,
    central_c_deg: float | None = None,
) -> None:
    """Write the headline chart for every ``c_deg`` in ``payload`` to ``path``.

    ``central_c_deg`` is labelled as such under its tick. Raises ``ValueError``
    if a ``c_deg`` in the summary lacks one of the three policies or the
    forecast's comparison, since a chart with a missing bar is a chart that
    says something the run did not.
    """
    runs = {
        (run["policy"], float(run["c_deg_eur_mwh"])): run for run in payload["runs"]
    }
    comparisons = {
        float(row["c_deg_eur_mwh"]): row
        for row in payload.get("comparisons", [])
        if row["policy"] == "forecast"
    }
    sweep = sorted({c_deg for _, c_deg in runs})
    if not sweep:
        raise ValueError("the summary holds no runs")
    for c_deg in sweep:
        missing = [name for name in POLICIES if (name, c_deg) not in runs]
        if missing:
            raise ValueError(f"c_deg {c_deg:g}: no run for {', '.join(missing)}")
        if c_deg not in comparisons:
            raise ValueError(f"c_deg {c_deg:g}: no forecast comparison in the summary")

    figure = Figure(figsize=(11.0, 4.8), layout="constrained")
    share_axes, eur_axes = figure.subplots(1, 2, width_ratios=[1.0, 1.1])
    ticks = [
        f"{c_deg:g}" + (" (central)" if c_deg == central_c_deg else "")
        for c_deg in sweep
    ]

    _draw_share(share_axes, sweep, comparisons, ticks)
    _draw_eur(eur_axes, sweep, runs, ticks)

    figure.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def _draw_share(
    axes: Any,
    sweep: Sequence[float],
    comparisons: Mapping[float, Mapping[str, Any]],
    ticks: Sequence[str],
) -> None:
    positions = list(range(len(sweep)))
    shares = [100.0 * float(comparisons[c]["share_of_gap"]) for c in sweep]
    intervals = [_interval(comparisons[c]) for c in sweep]

    axes.bar(positions, shares, width=0.55, color=COLOURS["forecast"])
    axes.axhline(0.0, color=COLOURS["floor"], linewidth=1.2)

    lows, highs = [], []
    for position, share, interval in zip(positions, shares, intervals, strict=True):
        # A run too short to resample gets no whisker at all: a zero-width one
        # would draw "no interval" as "no uncertainty".
        if interval is None:
            low = high = share
            text = f"{_signed(share)}%\nno interval"
        else:
            low, high = interval
            axes.errorbar(
                position,
                share,
                yerr=[[share - low], [high - share]],
                fmt="none",
                ecolor="#222222",
                capsize=5,
                linewidth=1.1,
            )
            text = f"{_signed(share)}%\n[{_signed(low)}, {_signed(high)}]"
        lows.append(low)
        highs.append(high)
        # Outside the whisker, on the side the bar points to, so a negative
        # share's label does not sit on the floor line.
        below = share < 0.0
        axes.annotate(
            text,
            (position, low if below else high),
            xytext=(0, -5 if below else 5),
            textcoords="offset points",
            ha="center",
            va="top" if below else "bottom",
            fontsize=9,
        )

    bottom = min(0.0, *lows)
    top = max(10.0, *highs)
    span = top - bottom
    axes.set_ylim(bottom - 0.25 * span, top + 0.25 * span)
    axes.set_xticks(positions, ticks)
    axes.set_xlabel("degradation cost c_deg, €/MWh")
    axes.set_ylabel("% of the floor-to-bound gap")
    axes.set_title(
        "Share of the gap the forecast closes (floor = 0, bound = 100)",
        fontsize=10,
        loc="left",
    )
    confidence = comparisons[sweep[0]].get("confidence")
    block_days = comparisons[sweep[0]].get("block_days")
    if confidence is not None and any(interval is not None for interval in intervals):
        blocks = "" if block_days is None else f", {block_days}-day blocks"
        level = f"{100 * float(confidence):.0f}%"
        axes.text(
            0.99,
            0.99,
            f"whiskers: {level} paired bootstrap interval{blocks}",
            transform=axes.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )
    axes.spines[["top", "right"]].set_visible(False)


def _draw_eur(
    axes: Any,
    sweep: Sequence[float],
    runs: Mapping[tuple[str, float], Mapping[str, Any]],
    ticks: Sequence[str],
) -> None:
    width = 0.27
    for offset, name in enumerate(POLICIES):
        positions = [index + (offset - 1) * width for index in range(len(sweep))]
        values = [float(runs[(name, c)]["profit_eur_per_mw_year"]) / 1e3 for c in sweep]
        bars = axes.bar(
            positions, values, width=width, color=COLOURS[name], label=LABELS[name]
        )
        axes.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        for position, c_deg in zip(positions, sweep, strict=True):
            cycles = float(runs[(name, c_deg)]["equivalent_cycles_per_year"])
            axes.text(
                position,
                1.0,
                f"{cycles:.0f} cycles/yr",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222" if name == "floor" else "white",
            )
    axes.set_xticks(list(range(len(sweep))), ticks)
    axes.set_xlabel("degradation cost c_deg, €/MWh")
    axes.set_ylabel("thousand €/MW/year")
    axes.set_title(
        "The three policies, €/MW/year and cycles/year", fontsize=10, loc="left"
    )
    axes.legend(frameon=False, fontsize=9, loc="upper right")
    axes.margins(y=0.12)
    axes.spines[["top", "right"]].set_visible(False)


def _interval(comparison: Mapping[str, Any]) -> tuple[float, float] | None:
    """The comparison's interval in percent, or ``None`` if the run had none."""
    interval = comparison.get("share_of_gap_interval")
    if interval is None:
        return None
    return 100.0 * float(interval[0]), 100.0 * float(interval[1])


def _signed(value: float) -> str:
    """One decimal, with a typographic minus rather than a hyphen."""
    return f"{value:.1f}".replace("-", MINUS)
