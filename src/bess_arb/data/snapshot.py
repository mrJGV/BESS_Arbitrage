"""Build the frozen snapshot, and the manifest that makes it evidence.

Run once. The output is committed; after that this code is not modified
(CLAUDE.md invariant 6). Everything here is therefore written to fail loudly
now rather than produce a plausible file: a short month, a missing day, a
series on an unexpected grid and a NaN are all errors, not warnings.

One grid per file
-----------------

A Parquet has one index, and the snapshot spans two market regimes and three
publication granularities that do not line up:

=========================  ============  =====================  ==============
Series                     to 2024-12-31 2025-01-01..2025-09-30 from 2025-10-01
=========================  ============  =====================  ==============
Day-ahead price (600)      hourly        quarter-hourly *       quarter-hourly
D+1 forecasts (1775/7/9)   hourly        hourly                 **still hourly**
=========================  ============  =====================  ==============

``*`` ESIOS moved indicator 600 onto a 15-minute grid on **1 January 2025**,
nine months before the market itself moved. Through that pre-period the four
values inside each hour are *identical* — it is the hourly clearing price
republished on the finer grid, ahead of the MTU go-live. Genuine intra-hour
variation begins on 2025-10-01 and not a day earlier: measured, the maximum
spread within an hour is exactly 0.0000 €/MWh up to 30 September 2025 and
€65.50 on 1 October. That is independent confirmation of the boundary date in
``docs/DECISIONS.md`` §1.1, arrived at from the data rather than from the
market notice.

So the hourly regime downsamples that stretch back to hourly — **and only
because it is provably lossless**. :func:`_downsample` checks that every
target period is constant across the finer ones and refuses otherwise, so
this never silently becomes an average that discards real spread.

The forecasts get the opposite treatment. The market moved to 15-minute MTUs;
REE's day-ahead forecast publications did not. The tempting fix is to
forward-fill them onto the 96-period grid, and it is the wrong one: that is a
modelling decision belonging to the forecaster slice, and freezing it here
would make it unrevisable without re-pulling — which invariant 6 forbids. So
each file holds exactly one grid, the mismatch stays visible, and slice 4 has
to bridge it deliberately instead of inheriting a choice nobody made.

The asymmetry is the point: collapsing a series whose extra resolution is
provably empty loses nothing, while expanding one whose extra resolution does
not exist would invent data.

Hence three files, no NaN padding and no duplicated rows:

``hourly_<date>.parquet``
    Hourly regime, price and all three forecasts.
``quarter_hourly_<date>.parquet``
    Quarter-hourly regime, price at 96 periods a day.
``quarter_hourly_exog_hourly_<date>.parquet``
    The same days as above, forecasts on their native hourly grid.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bess_arb.config import Config, DataConfig, RegimeWindow, SeriesSpec
from bess_arb.data.esios import EsiosClient
from bess_arb.data.omie import fetch_marginalpdbc_range
from bess_arb.timeline import (
    Regime,
    delivery_day,
    dst_transition_days,
    hours_in_day,
    periods_per_day,
    utc_index,
    validate_index,
)

__all__ = [
    "SnapshotError",
    "build_snapshot",
    "crosscheck_against_omie",
    "sha256_of",
    "write_manifest",
]


class SnapshotError(RuntimeError):
    """The snapshot could not be built to the standard the manifest claims."""


@dataclass(frozen=True, slots=True)
class _Fetched:
    """One series as pulled, with the grid it turned out to be on."""

    spec: SeriesSpec
    frame: pd.DataFrame
    regime: Regime
    indicator_name: str
    geo_name: str
    published_regime: Regime | None = None
    """Set when the series was downsampled, to the grid it arrived on."""


def _downsample(item: _Fetched, target: Regime) -> _Fetched:
    """Collapse a finer-grained series onto ``target``, if that loses nothing.

    Only valid when every target period is *constant* across the finer
    periods inside it — which is the case for indicator 600 between January
    and September 2025, where ESIOS republished the hourly clearing price on
    a 15-minute grid before the market moved. Then the collapse is exact.

    Any real variation and this raises. Averaging it away would quietly
    destroy the intra-hour spread that the quarter-hourly regime exists to
    capture, and it would do so while producing a perfectly plausible file.
    """
    values = item.frame["value"]
    step = pd.Timedelta(hours=target.dt_h)
    # Flooring is done on the UTC index, which is uniform. On a local index a
    # fall-back day would floor two distinct hours onto the same label.
    grouped = values.groupby(values.index.floor(step))
    spread = float((grouped.max() - grouped.min()).abs().max())
    published = sorted(set(observed_dt_h(item.frame).values()))

    if spread > 1e-9:
        raise SnapshotError(
            f"{item.spec.name}: refusing to downsample from dt_h {published} "
            f"to {target.name} — values vary within a target period by up to "
            f"{spread:g}. Collapsing that would discard real spread."
        )

    frame = item.frame.groupby(item.frame.index.floor(step)).first()
    frame.index = frame.index.rename(item.frame.index.name)
    return _Fetched(
        spec=item.spec,
        frame=frame,
        regime=target,
        indicator_name=item.indicator_name,
        geo_name=item.geo_name,
        published_regime=Regime(_grid_name(min(published)), min(published)),
    )


def sha256_of(path: Path) -> str:
    """Hash a file in chunks, so the manifest can be checked without trust."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_dt_h(frame: pd.DataFrame) -> dict[dt.date, float]:
    """The period length each delivery day was actually published at.

    Per *day*, not per series, because a series can change granularity in
    the middle of a window and indicator 600 does: hourly to 31 December
    2024, quarter-hourly from 1 January 2025. Anything that asks "what grid
    is this series on?" and expects one answer gets it wrong for 2025.

    Derived by dividing each day's true length — 23, 24 or 25 hours — by the
    number of rows it has, so a transition day is handled by the same
    arithmetic as any other.
    """
    index = pd.DatetimeIndex(frame.index)
    counts = pd.Series(delivery_day(index)).value_counts()
    return {day: hours_in_day(day) / int(rows) for day, rows in sorted(counts.items())}


def _fetch_series(
    client: EsiosClient,
    window: RegimeWindow,
    spec: SeriesSpec,
    *,
    progress: bool,
) -> _Fetched:
    """Pull one series and put it on the window's grid if that loses nothing."""
    if progress:
        print(
            f"  {spec.name:<20} indicator {spec.indicator_id} geo {spec.geo_id} …",
            flush=True,
        )
    frame = client.fetch_indicator(
        spec.indicator_id,
        window.first_day,
        window.last_day,
        geo_id=spec.geo_id,
    )
    if frame["value"].isna().any():
        raise SnapshotError(
            f"{spec.name}: {int(frame['value'].isna().sum())} NaN values"
        )

    meta = client.indicator_metadata(spec.indicator_id)
    geo_names = {str(name) for name in frame["geo_name"].dropna().unique()}
    item = _Fetched(
        spec=spec,
        frame=frame,
        regime=window.regime,
        indicator_name=str(meta.get("name", "")),
        geo_name=next(iter(geo_names), ""),
    )

    target = window.regime
    published = sorted(set(observed_dt_h(frame).values()))
    finest = min(published)
    coarsest = max(published)

    if coarsest > target.dt_h:
        # Coarser than the regime everywhere — a forecast series that stayed
        # hourly after the market went quarter-hourly. It keeps its own grid
        # and its own file; upsampling here would invent data.
        if finest != coarsest:
            raise SnapshotError(
                f"{spec.name}: mixed granularity {published} coarser than "
                f"the {target.name} grid; no rule covers that case"
            )
        grid = Regime(_grid_name(coarsest), coarsest)
        validate_index(
            pd.DatetimeIndex(frame.index),
            grid,
            first_day=window.first_day,
            last_day=window.last_day,
        )
        return _Fetched(
            spec=item.spec,
            frame=frame,
            regime=grid,
            indicator_name=item.indicator_name,
            geo_name=item.geo_name,
        )

    if finest < target.dt_h:
        item = _downsample(item, target)

    validate_index(
        pd.DatetimeIndex(item.frame.index),
        target,
        first_day=window.first_day,
        last_day=window.last_day,
    )
    return item


def _grid_name(dt_h: float) -> str:
    return {1.0: "hourly", 0.25: "quarter_hourly"}.get(dt_h, f"dt_h_{dt_h:g}")


def _write_parquet(
    path: Path,
    fetched: list[_Fetched],
    window: RegimeWindow,
    regime: Regime,
) -> dict[str, Any]:
    """Assemble one grid's series into a file and describe it for the manifest."""
    index = utc_index(window.first_day, window.last_day, regime)
    table = pd.DataFrame(index=index)
    for item in fetched:
        table[item.spec.name] = item.frame["value"].reindex(index)

    if table.isna().to_numpy().any():
        counts = {
            column: int(table[column].isna().sum())
            for column in table.columns
            if table[column].isna().any()
        }
        raise SnapshotError(f"{path.name}: NaN after alignment: {counts}")

    expected_rows = int(
        periods_per_day(window.first_day, window.last_day, regime).sum()
    )
    if len(table) != expected_rows:
        raise SnapshotError(
            f"{path.name}: {len(table)} rows, calendar says {expected_rows}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, engine="pyarrow", compression="zstd", index=True)

    return {
        "path": path.name,
        "regime": window.name,
        "grid": regime.name,
        "dt_h": regime.dt_h,
        "first_day": window.first_day.isoformat(),
        "last_day": window.last_day.isoformat(),
        "first_utc": str(table.index[0]),
        "last_utc": str(table.index[-1]),
        "rows": len(table),
        "expected_rows": expected_rows,
        "columns": list(table.columns),
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def _describe_series(item: _Fetched, window: RegimeWindow) -> dict[str, Any]:
    values = item.frame["value"]
    return {
        "column": item.spec.name,
        "indicator_id": item.spec.indicator_id,
        "indicator_name": item.indicator_name,
        "geo_id": item.spec.geo_id,
        "geo_name": item.geo_name,
        "regime": window.name,
        "grid": item.regime.name,
        "observed_dt_h": item.regime.dt_h,
        "published_grid": (
            item.published_regime.name if item.published_regime else item.regime.name
        ),
        "downsampled_losslessly": item.published_regime is not None,
        "rows": len(values),
        "first_utc": str(item.frame.index[0]),
        "last_utc": str(item.frame.index[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": round(float(values.mean()), 4),
        "negative_periods": int((values < 0).sum()),
    }


def build_snapshot(
    config: Config,
    *,
    client: EsiosClient | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Pull every series for every regime, write the Parquet, return the manifest."""
    data: DataConfig = config.data
    client = client if client is not None else EsiosClient()

    files: list[dict[str, Any]] = []
    series_meta: list[dict[str, Any]] = []

    for window in data.regimes:
        if progress:
            print(
                f"\n{window.name}: {window.first_day} .. {window.last_day} "
                f"(dt_h={window.regime.dt_h})",
                flush=True,
            )
        fetched = [
            _fetch_series(client, window, spec, progress=progress)
            for spec in data.series
        ]
        series_meta.extend(_describe_series(item, window) for item in fetched)

        native = [item for item in fetched if item.regime == window.regime]
        foreign = [item for item in fetched if item.regime != window.regime]

        if native:
            files.append(
                _write_parquet(
                    data.parquet_path(window.name), native, window, window.regime
                )
            )
        for grid in {item.regime for item in foreign}:
            on_grid = [item for item in foreign if item.regime == grid]
            path = data.directory / (
                f"{window.name}_exog_{grid.name}_{data.snapshot_date.isoformat()}.parquet"
            )
            files.append(_write_parquet(path, on_grid, window, grid))

    return {
        "snapshot_date": data.snapshot_date.isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "source": {
            "primary": "ESIOS API, Red Eléctrica de España (https://api.esios.ree.es)",
            "crosscheck": "OMIE marginalpdbc (https://www.omie.es)",
        },
        "timezone": {"storage": "UTC", "presentation": "Europe/Madrid"},
        "files": files,
        "series": series_meta,
        "dst_transition_days": [
            day.isoformat()
            for window in data.regimes
            for day in dst_transition_days(window.first_day, window.last_day)
        ],
    }


def crosscheck_against_omie(
    config: Config,
    *,
    client: EsiosClient | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Compare a month of ESIOS Spanish prices against OMIE's own files.

    Independent confirmation that the right indicator, the right geography
    and the right hour were stored. ESIOS is REE's publication of the market
    result; ``marginalpdbc`` is OMIE publishing it directly. If those two
    agree period by period, a whole family of quiet errors is ruled out.
    """
    data = config.data
    client = client if client is not None else EsiosClient()
    first, last = data.crosscheck_first_day, data.crosscheck_last_day

    window = next(
        (w for w in data.regimes if w.first_day <= first and last <= w.last_day),
        None,
    )
    if window is None:
        raise SnapshotError(
            f"cross-check window {first}..{last} does not sit inside one regime"
        )

    price = next(spec for spec in data.series if spec.name == "price_eur_mwh")
    if progress:
        print(f"ESIOS {price.indicator_id} geo {price.geo_id}: {first}..{last}")
    esios = client.fetch_indicator(
        price.indicator_id, first, last, geo_id=price.geo_id
    )["value"]

    if progress:
        print(f"OMIE marginalpdbc: {first}..{last} ({(last - first).days + 1} files)")
    omie = fetch_marginalpdbc_range(first, last, window.regime)["price_es_eur_mwh"]

    joined = pd.concat({"esios": esios, "omie": omie}, axis=1)
    if joined.isna().to_numpy().any():
        raise SnapshotError(
            f"cross-check: {int(joined.isna().any(axis=1).sum())} periods are "
            "present in one source and not the other"
        )

    diff = (joined["esios"] - joined["omie"]).abs()
    tolerance = data.crosscheck_tolerance_eur_mwh
    result = {
        "first_day": first.isoformat(),
        "last_day": last.isoformat(),
        "regime": window.name,
        "periods": len(joined),
        "max_abs_diff_eur_mwh": round(float(diff.max()), 6),
        "mean_abs_diff_eur_mwh": round(float(diff.mean()), 6),
        "periods_over_tolerance": int((diff > tolerance).sum()),
        "tolerance_eur_mwh": tolerance,
        "agrees": bool(diff.max() <= tolerance),
    }
    if not result["agrees"]:
        raise SnapshotError(
            f"ESIOS and OMIE disagree by up to "
            f"{result['max_abs_diff_eur_mwh']} €/MWh over {first}..{last}, "
            f"tolerance {tolerance}"
        )
    return result


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest, sorted and newline-terminated so diffs stay readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
