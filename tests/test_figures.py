"""The headline chart draws from a run summary and refuses an incomplete one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bess_arb.figures import draw_headline

SWEEP = (5.0, 17.0, 40.0)


def _payload(*, drop: tuple[str, float] | None = None) -> dict[str, Any]:
    profits = {"floor": 50_000.0, "forecast": 51_000.0, "oracle": 59_000.0}
    runs = [
        {
            "policy": policy,
            "c_deg_eur_mwh": c_deg,
            "profit_eur_per_mw_year": profit - 500.0 * c_deg,
            "equivalent_cycles_per_year": 400.0,
            "regime": "quarter_hourly",
            "daily": {"day": ["2025-10-08", "2025-10-09"], "profit_eur": [1.0, 2.0]},
        }
        for c_deg in SWEEP
        for policy, profit in profits.items()
        if (policy, c_deg) != drop
    ]
    comparisons = [
        {
            "policy": "forecast",
            "c_deg_eur_mwh": c_deg,
            "share_of_gap": share,
            "share_of_gap_interval": [share - 0.08, share + 0.08],
            "confidence": 0.95,
        }
        for c_deg, share in zip(SWEEP, (-0.07, 0.11, 0.2), strict=True)
    ]
    return {"runs": runs, "comparisons": comparisons}


def test_the_chart_is_written(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "headline.png"

    draw_headline(_payload(), out, title="test", central_c_deg=17.0)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_run_too_short_for_an_interval_still_draws(tmp_path: Path) -> None:
    """``compare`` writes ``null`` when some resample had no gap to divide by."""
    payload = _payload()
    for row in payload["comparisons"]:
        row["share_of_gap_interval"] = None
    out = tmp_path / "short.png"

    draw_headline(payload, out, title="test")

    assert out.stat().st_size > 0


def test_a_missing_policy_is_refused_rather_than_drawn_as_a_gap(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no run for floor"):
        draw_headline(_payload(drop=("floor", 17.0)), tmp_path / "x.png", title="test")


def test_a_missing_comparison_is_refused(tmp_path: Path) -> None:
    payload = _payload()
    payload["comparisons"] = payload["comparisons"][:2]

    with pytest.raises(ValueError, match="no forecast comparison"):
        draw_headline(payload, tmp_path / "x.png", title="test")
