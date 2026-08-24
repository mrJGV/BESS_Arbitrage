"""Loader for ``config/params.yaml`` — the single source of numeric truth.

Imports nothing solver-related: it builds the solver-free dataclasses from
:mod:`bess_arb.model.spec` and hands them to whichever backend the config
names. Unknown keys are an error rather than a shrug, because a silently
ignored ``c_deg_eur_mhw`` would produce a plausible wrong number.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bess_arb.model.spec import BatteryParams, SolverConfig
from bess_arb.timeline import Regime

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "Config",
    "ConfigError",
    "DataConfig",
    "RegimeWindow",
    "SeriesSpec",
    "load_config",
]

# src/bess_arb/config.py -> src/bess_arb -> src -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "params.yaml"


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """One ESIOS series and the geography to keep from it.

    ``geo_id`` is not decoration. Indicator 600 returns six geographies
    interleaved on the same timestamps, so a series without an explicit
    geography is a series that silently stores whichever the server sent
    first.
    """

    name: str
    indicator_id: int
    geo_id: int | None = None


@dataclass(frozen=True, slots=True)
class RegimeWindow:
    """A market regime and the days the snapshot covers for it."""

    regime: Regime
    first_day: dt.date
    last_day: dt.date

    def __post_init__(self) -> None:
        if self.last_day < self.first_day:
            raise ValueError(
                f"regime {self.regime.name!r}: last_day {self.last_day} "
                f"precedes first_day {self.first_day}"
            )

    @property
    def name(self) -> str:
        return self.regime.name


@dataclass(frozen=True, slots=True)
class DataConfig:
    """The frozen snapshot: where it lives, what it covers, what is in it."""

    directory: Path
    snapshot_date: dt.date
    regimes: tuple[RegimeWindow, ...]
    series: tuple[SeriesSpec, ...]
    crosscheck_first_day: dt.date
    crosscheck_last_day: dt.date
    crosscheck_tolerance_eur_mwh: float

    def regime(self, name: str) -> RegimeWindow:
        for window in self.regimes:
            if window.name == name:
                return window
        known = ", ".join(window.name for window in self.regimes)
        raise KeyError(f"unknown regime {name!r}; known regimes: {known}")

    def parquet_path(self, regime_name: str) -> Path:
        """Snapshot filename for a regime — the freeze date is in the name.

        One file per regime, so which grid a file is on is answerable
        without opening it.
        """
        return (
            self.directory / f"{regime_name}_{self.snapshot_date.isoformat()}.parquet"
        )

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"


@dataclass(frozen=True, slots=True)
class Config:
    """Everything ``config/params.yaml`` currently declares."""

    battery: BatteryParams
    c_deg_sensitivity: tuple[float, ...]
    data: DataConfig
    backend: str
    solver: SolverConfig
    seed: int

    def with_c_deg(self, c_deg_eur_mwh: float) -> Config:
        """This config at a different degradation cost.

        The sensitivity sweep is three runs of one configuration, not three
        configurations — nothing else may drift between the points.
        """
        battery = BatteryParams(
            p_max_mw=self.battery.p_max_mw,
            e_max_mwh=self.battery.e_max_mwh,
            eta_rt=self.battery.eta_rt,
            c_deg_eur_mwh=c_deg_eur_mwh,
            charge_tariff_eur_mwh=self.battery.charge_tariff_eur_mwh,
        )
        return Config(
            battery=battery,
            c_deg_sensitivity=self.c_deg_sensitivity,
            data=self.data,
            backend=self.backend,
            solver=self.solver,
            seed=self.seed,
        )


class ConfigError(ValueError):
    """Raised when the config file is missing, malformed or has stray keys."""


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        value = raw[name]
    except KeyError:
        raise ConfigError(f"config is missing the {name!r} section") from None
    if not isinstance(value, Mapping):
        raise ConfigError(f"config section {name!r} must be a mapping")
    return value


def _reject_unknown(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in config section {name!r}: {', '.join(unknown)}"
        )


def load_config(path: Path | None = None) -> Config:
    """Read and validate the YAML config."""
    path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config file {path} does not contain a mapping")

    _reject_unknown(
        raw, {"battery", "sensitivity", "data", "backend", "solver", "seed"}, "<root>"
    )

    battery_section = _section(raw, "battery")
    _reject_unknown(
        battery_section,
        {
            "p_max_mw",
            "e_max_mwh",
            "eta_rt",
            "c_deg_eur_mwh",
            "charge_tariff_eur_mwh",
        },
        "battery",
    )
    battery = BatteryParams(
        p_max_mw=float(battery_section["p_max_mw"]),
        e_max_mwh=float(battery_section["e_max_mwh"]),
        eta_rt=float(battery_section["eta_rt"]),
        c_deg_eur_mwh=float(battery_section["c_deg_eur_mwh"]),
        charge_tariff_eur_mwh=float(battery_section.get("charge_tariff_eur_mwh", 0.0)),
    )

    sensitivity_section = _section(raw, "sensitivity")
    _reject_unknown(sensitivity_section, {"c_deg_eur_mwh"}, "sensitivity")
    sweep = sensitivity_section["c_deg_eur_mwh"]
    if not isinstance(sweep, list) or not sweep:
        raise ConfigError("sensitivity.c_deg_eur_mwh must be a non-empty list")
    c_deg_sensitivity = tuple(float(value) for value in sweep)
    if any(value < 0.0 for value in c_deg_sensitivity):
        raise ConfigError("sensitivity.c_deg_eur_mwh values must be non-negative")

    solver_section = _section(raw, "solver")
    _reject_unknown(
        solver_section,
        {"name", "mip_gap", "time_limit_s", "threads", "options"},
        "solver",
    )
    options = solver_section.get("options") or {}
    if not isinstance(options, Mapping):
        raise ConfigError("solver.options must be a mapping")
    solver = SolverConfig(
        name=str(solver_section["name"]),
        mip_gap=_optional_float(solver_section.get("mip_gap")),
        time_limit_s=_optional_float(solver_section.get("time_limit_s")),
        threads=_optional_int(solver_section.get("threads")),
        options=dict(options),
    )

    backend = raw.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ConfigError("backend must be a non-empty string")

    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("seed must be an integer")

    return Config(
        battery=battery,
        c_deg_sensitivity=c_deg_sensitivity,
        data=_load_data(_section(raw, "data"), path),
        backend=backend,
        solver=solver,
        seed=seed,
    )


def _load_data(section: Mapping[str, Any], config_path: Path) -> DataConfig:
    _reject_unknown(
        section,
        {"directory", "snapshot_date", "regimes", "series", "crosscheck"},
        "data",
    )

    directory = Path(str(section.get("directory", "data")))
    if not directory.is_absolute():
        # Relative to the repository, not to the working directory: `make
        # figures` from a subdirectory must find the same snapshot.
        directory = (config_path.resolve().parent.parent / directory).resolve()

    regimes_section = _section(section, "regimes")
    regimes: list[RegimeWindow] = []
    for name, spec in regimes_section.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"data.regimes.{name} must be a mapping")
        _reject_unknown(spec, {"dt_h", "first_day", "last_day"}, f"data.regimes.{name}")
        try:
            regimes.append(
                RegimeWindow(
                    regime=Regime(str(name), float(spec["dt_h"])),
                    first_day=_as_date(spec["first_day"], f"data.regimes.{name}"),
                    last_day=_as_date(spec["last_day"], f"data.regimes.{name}"),
                )
            )
        except KeyError as error:
            raise ConfigError(
                f"data.regimes.{name} is missing {error.args[0]!r}"
            ) from None
    if not regimes:
        raise ConfigError("data.regimes must declare at least one regime")

    series_section = _section(section, "series")
    series: list[SeriesSpec] = []
    for name, spec in series_section.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"data.series.{name} must be a mapping")
        _reject_unknown(spec, {"indicator", "geo_id"}, f"data.series.{name}")
        try:
            indicator_id = int(spec["indicator"])
        except KeyError:
            raise ConfigError(f"data.series.{name} is missing 'indicator'") from None
        series.append(
            SeriesSpec(
                name=str(name),
                indicator_id=indicator_id,
                geo_id=_optional_int(spec.get("geo_id")),
            )
        )
    if not series:
        raise ConfigError("data.series must declare at least one series")

    crosscheck = _section(section, "crosscheck")
    _reject_unknown(
        crosscheck, {"first_day", "last_day", "tolerance_eur_mwh"}, "data.crosscheck"
    )
    return DataConfig(
        directory=directory,
        snapshot_date=_as_date(section["snapshot_date"], "data"),
        regimes=tuple(regimes),
        series=tuple(series),
        crosscheck_first_day=_as_date(crosscheck["first_day"], "data.crosscheck"),
        crosscheck_last_day=_as_date(crosscheck["last_day"], "data.crosscheck"),
        crosscheck_tolerance_eur_mwh=float(crosscheck.get("tolerance_eur_mwh", 0.01)),
    )


def _as_date(value: Any, where: str) -> dt.date:
    """Accept what YAML gives for a date, refuse a timestamp.

    PyYAML turns an unquoted ``2022-01-01`` into a ``date`` already; a
    quoted one stays a string. A ``datetime`` means someone wrote a time as
    well, and a delivery day with a time on it is ambiguous about which zone
    decides — the same trap ``timeline`` refuses.
    """
    if isinstance(value, dt.datetime):
        raise ConfigError(
            f"{where}: {value!r} carries a time; a delivery day must be a plain date"
        )
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip())
        except ValueError:
            raise ConfigError(f"{where}: {value!r} is not an ISO date") from None
    raise ConfigError(f"{where}: expected a date, got {type(value).__name__}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
