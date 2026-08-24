"""The frozen snapshot is what it says it is.

``data/manifest.json`` makes claims — these files, these many rows, this
hash, these indicators. This file checks every one of them against the
committed Parquet. Without it the manifest is a description; with it, it is
evidence.

Nothing here touches the network. That is the point: the snapshot is frozen,
so its properties are testable offline forever, and a reader who clones the
repository can verify the data without an ESIOS token.

The check that earns its keep is the DST one. Every transition day in the
window is looked up *in the stored data* and its period count compared with
what the calendar derives. A snapshot assembled with an off-by-one hour would
pass a row-count check on the year — one 23-hour day and one 25-hour day
cancel — and fail here.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bess_arb.config import load_config
from bess_arb.data.snapshot import sha256_of
from bess_arb.timeline import (
    Regime,
    delivery_day,
    dst_transition_days,
    periods_in_day,
    periods_per_day,
    validate_index,
)

CONFIG = load_config()
DATA_DIR = CONFIG.data.directory
MANIFEST_PATH = CONFIG.data.manifest_path

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.is_file(),
    reason="no frozen snapshot in the working tree (run `bess-arb data freeze`)",
)


def _manifest() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return loaded


def _files() -> list[dict[str, Any]]:
    """File entries, or nothing if there is no snapshot yet.

    ``parametrize`` is evaluated at collection time, before ``pytestmark``
    can skip anything, so this has to tolerate a missing manifest rather
    than raise into the collector.
    """
    if not MANIFEST_PATH.is_file():
        return []
    return list(_manifest()["files"])


def _ids(entry: dict[str, Any]) -> str:
    return str(entry.get("path", "no-snapshot"))


def _read(entry: dict[str, Any]) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / entry["path"])


def _regime_of(entry: dict[str, Any]) -> Regime:
    return Regime(str(entry["grid"]), float(entry["dt_h"]))


def test_the_manifest_exists_and_names_files_that_do_too() -> None:
    manifest = _manifest()

    assert manifest["files"], "manifest lists no files"
    for entry in manifest["files"]:
        assert (DATA_DIR / entry["path"]).is_file(), entry["path"]


@pytest.mark.parametrize("entry", _files() or [{}], ids=_ids)
def test_the_recorded_hash_matches_the_committed_file(entry: dict[str, Any]) -> None:
    """The freeze point is a hash, not a promise."""
    path = DATA_DIR / entry["path"]

    assert sha256_of(path) == entry["sha256"]
    assert path.stat().st_size == entry["bytes"]


@pytest.mark.parametrize("entry", _files() or [{}], ids=_ids)
def test_each_file_sits_on_the_grid_it_declares(entry: dict[str, Any]) -> None:
    frame = _read(entry)

    validate_index(
        pd.DatetimeIndex(frame.index),
        _regime_of(entry),
        first_day=dt.date.fromisoformat(entry["first_day"]),
        last_day=dt.date.fromisoformat(entry["last_day"]),
    )
    assert list(frame.columns) == entry["columns"]
    assert len(frame) == entry["rows"] == entry["expected_rows"]


@pytest.mark.parametrize("entry", _files() or [{}], ids=_ids)
def test_the_row_count_is_the_one_the_calendar_derives(
    entry: dict[str, Any],
) -> None:
    """Not 24 times the number of days — the sum of each day's own length."""
    expected = int(
        periods_per_day(
            dt.date.fromisoformat(entry["first_day"]),
            dt.date.fromisoformat(entry["last_day"]),
            _regime_of(entry),
        ).sum()
    )

    assert len(_read(entry)) == expected


@pytest.mark.parametrize("entry", _files() or [{}], ids=_ids)
def test_no_gaps_anywhere(entry: dict[str, Any]) -> None:
    """A NaN would be a day the API skipped and nothing noticed."""
    frame = _read(entry)

    assert not frame.isna().to_numpy().any(), frame.isna().sum().to_dict()


@pytest.mark.parametrize("entry", _files() or [{}], ids=_ids)
def test_every_dst_day_has_its_own_period_count(entry: dict[str, Any]) -> None:
    """The check a year-level row count cannot make.

    Over a full year the short March day and the long October day cancel, so
    a snapshot shifted by an hour still totals 24 h/day. Only counting the
    transition days themselves catches it.
    """
    regime = _regime_of(entry)
    first = dt.date.fromisoformat(entry["first_day"])
    last = dt.date.fromisoformat(entry["last_day"])
    transitions = dst_transition_days(first, last)
    if not transitions:
        pytest.skip("no DST transition in this file's window")

    frame = _read(entry)
    counts = pd.Series(delivery_day(pd.DatetimeIndex(frame.index))).value_counts()

    for day in transitions:
        assert counts[day] == periods_in_day(day, regime), day
        assert counts[day] != round(24.0 / regime.dt_h), (
            f"{day} has the ordinary-day period count; it is a transition day"
        )


def test_both_dst_directions_are_present_in_the_snapshot() -> None:
    """Short and long days both occur, so neither branch is untested."""
    lengths = set()
    for entry in _files():
        first = dt.date.fromisoformat(entry["first_day"])
        last = dt.date.fromisoformat(entry["last_day"])
        regime = _regime_of(entry)
        for day in dst_transition_days(first, last):
            lengths.add(periods_in_day(day, regime) / (1.0 / regime.dt_h))

    assert 23.0 in lengths and 25.0 in lengths, sorted(lengths)


def test_the_manifest_series_agree_with_the_config() -> None:
    """A quietly edited indicator ID would put config and data out of step."""
    configured = {spec.name: spec for spec in CONFIG.data.series}

    for record in _manifest()["series"]:
        spec = configured[record["column"]]
        assert record["indicator_id"] == spec.indicator_id
        assert record["geo_id"] == spec.geo_id


def test_the_price_series_names_spain() -> None:
    """Indicator 600 carries six geographies; the stored one must be España."""
    for record in _manifest()["series"]:
        if record["column"] == "price_eur_mwh":
            assert record["geo_name"] == "España"


def test_negative_prices_are_present() -> None:
    """The evidence behind the binaries, in the data rather than in an argument.

    docs/DECISIONS.md §3.2 justifies the integrality by the existence of
    negative prices. If the frozen window contained none, that argument would
    be about a market this snapshot does not cover.
    """
    negatives = {
        record["column"]: record["negative_periods"]
        for record in _manifest()["series"]
        if record["column"] == "price_eur_mwh"
    }

    assert negatives, "no price series in the manifest"
    assert sum(negatives.values()) > 0


def test_the_cross_check_against_omie_agreed() -> None:
    """Two independent publications of the same prices, period by period."""
    crosscheck = _manifest().get("crosscheck")
    assert crosscheck is not None, "snapshot was frozen without the OMIE cross-check"

    assert crosscheck["agrees"] is True
    assert crosscheck["periods_over_tolerance"] == 0
    assert crosscheck["periods"] >= 28 * 24, "less than a month compared"


def test_the_snapshot_covers_both_market_regimes() -> None:
    grids = {entry["grid"] for entry in _files()}
    regimes = {entry["regime"] for entry in _files()}

    assert regimes == {"hourly", "quarter_hourly"}
    assert {"hourly", "quarter_hourly"} <= grids


def test_the_filenames_carry_the_snapshot_date() -> None:
    stamp = _manifest()["snapshot_date"]

    for entry in _files():
        assert stamp in entry["path"], entry["path"]
        assert Path(entry["path"]).suffix == ".parquet"
