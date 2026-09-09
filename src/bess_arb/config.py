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
    "AnnualBoundConfig",
    "BoundConfig",
    "Config",
    "ConfigError",
    "DataConfig",
    "ForecastConfig",
    "HorizonConfig",
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

    def exogenous_paths(self, regime_name: str) -> tuple[Path, ...]:
        """Sibling files holding this regime's series on a coarser grid.

        A regime whose own grid is finer than a published series keeps that
        series in its own file, named for the grid it is really on — the
        snapshot never resamples (see ``data/README.md``). Globbed rather than
        constructed, so the reader does not have to know which grids the pull
        happened to produce.
        """
        pattern = f"{regime_name}_exog_*_{self.snapshot_date.isoformat()}.parquet"
        return tuple(sorted(self.directory.glob(pattern)))

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"


@dataclass(frozen=True, slots=True)
class HorizonConfig:
    """The rolling-window protocol, in delivery days (DECISIONS.md §2.1, §2.6).

    Days rather than hours on purpose. "48-hour window, first 24 implemented"
    describes ordinary days; on a DST transition the same protocol solves 47
    or 49 hours, and the period counts come from
    :func:`bess_arb.timeline.periods_in_day` rather than from arithmetic here.
    """

    window_days: int = 2
    implement_days: int = 1
    warmup_days: int = 7
    soc_initial_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.window_days < 1:
            raise ValueError(f"window_days must be at least 1, got {self.window_days}")
        if not 1 <= self.implement_days <= self.window_days:
            raise ValueError(
                "implement_days must lie in [1, window_days], got "
                f"{self.implement_days} with window_days={self.window_days}"
            )
        if self.warmup_days < 0:
            raise ValueError(
                f"warmup_days must be non-negative, got {self.warmup_days}"
            )
        if not 0.0 <= self.soc_initial_fraction <= 1.0:
            raise ValueError(
                "soc_initial_fraction must lie in [0, 1], got "
                f"{self.soc_initial_fraction}"
            )

    def soc_initial_mwh(self, e_max_mwh: float) -> float:
        return self.soc_initial_fraction * e_max_mwh


@dataclass(frozen=True, slots=True)
class AnnualBoundConfig:
    """The §2.4 annual-window solve: which days, and with how much slack."""

    first_day: dt.date
    last_day: dt.date
    mip_gap: float | None
    time_limit_s: float | None

    def __post_init__(self) -> None:
        if self.last_day < self.first_day:
            raise ValueError(
                f"bound.annual: last_day {self.last_day} precedes "
                f"first_day {self.first_day}"
            )


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """The point forecaster: what it reads, how often it refits, how it fits.

    The LightGBM hyperparameters are carried through as opaque mappings rather
    than named fields. Everywhere else in this file an unknown key is an
    error, and that is right for a config whose keys the project defines; here
    the keys are LightGBM's, and re-declaring its parameter list would only
    guarantee a stale copy of it. The trade is stated rather than silent: a
    typo in ``level.num_leavs`` is caught by LightGBM at fit time, not here.
    """

    lags_days: tuple[int, ...]
    refit_days: int
    min_train_days: int
    min_deviation_days: int
    min_residual_obs: int
    level_rounds: int
    deviation_rounds: int
    validation_days: int
    early_stopping_rounds: int
    threads: int
    level: Mapping[str, Any]
    deviation: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.lags_days:
            raise ValueError("forecast.lags_days must not be empty")
        if any(lag < 1 for lag in self.lags_days):
            raise ValueError(
                f"forecast.lags_days must be positive, got {list(self.lags_days)}"
            )
        if self.validation_days < 1:
            raise ValueError(
                f"forecast.validation_days must be at least 1, got "
                f"{self.validation_days}"
            )
        if self.early_stopping_rounds < 1:
            raise ValueError(
                f"forecast.early_stopping_rounds must be at least 1, got "
                f"{self.early_stopping_rounds}"
            )
        if self.min_residual_obs < 1:
            raise ValueError(
                f"forecast.min_residual_obs must be at least 1, got "
                f"{self.min_residual_obs}"
            )


@dataclass(frozen=True, slots=True)
class BoundConfig:
    annual: AnnualBoundConfig


@dataclass(frozen=True, slots=True)
class Config:
    """Everything ``config/params.yaml`` currently declares."""

    battery: BatteryParams
    c_deg_sensitivity: tuple[float, ...]
    data: DataConfig
    horizon: HorizonConfig
    bound: BoundConfig
    forecast: ForecastConfig
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
            horizon=self.horizon,
            bound=self.bound,
            forecast=self.forecast,
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
        raw,
        {
            "battery",
            "sensitivity",
            "data",
            "horizon",
            "bound",
            "forecast",
            "backend",
            "solver",
            "seed",
        },
        "<root>",
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
        horizon=_load_horizon(_section(raw, "horizon")),
        bound=_load_bound(_section(raw, "bound")),
        forecast=_load_forecast(_section(raw, "forecast")),
        backend=backend,
        solver=solver,
        seed=seed,
    )


def _load_horizon(section: Mapping[str, Any]) -> HorizonConfig:
    _reject_unknown(
        section,
        {"window_days", "implement_days", "warmup_days", "soc_initial_fraction"},
        "horizon",
    )
    try:
        return HorizonConfig(
            window_days=int(section["window_days"]),
            implement_days=int(section["implement_days"]),
            warmup_days=int(section["warmup_days"]),
            soc_initial_fraction=float(section["soc_initial_fraction"]),
        )
    except KeyError as error:
        raise ConfigError(f"horizon is missing {error.args[0]!r}") from None
    except ValueError as error:
        raise ConfigError(f"horizon: {error}") from None


def _load_forecast(section: Mapping[str, Any]) -> ForecastConfig:
    _reject_unknown(
        section,
        {
            "lags_days",
            "refit_days",
            "min_train_days",
            "min_deviation_days",
            "min_residual_obs",
            "level_rounds",
            "deviation_rounds",
            "validation_days",
            "early_stopping_rounds",
            "threads",
            "level",
            "deviation",
        },
        "forecast",
    )
    lags = section.get("lags_days")
    if not isinstance(lags, list) or not lags:
        raise ConfigError("forecast.lags_days must be a non-empty list")
    for name in ("level", "deviation"):
        if not isinstance(section.get(name), Mapping):
            raise ConfigError(f"forecast.{name} must be a mapping")
    try:
        return ForecastConfig(
            lags_days=tuple(int(lag) for lag in lags),
            refit_days=int(section["refit_days"]),
            min_train_days=int(section["min_train_days"]),
            min_deviation_days=int(section["min_deviation_days"]),
            min_residual_obs=int(section["min_residual_obs"]),
            level_rounds=int(section["level_rounds"]),
            deviation_rounds=int(section["deviation_rounds"]),
            validation_days=int(section["validation_days"]),
            early_stopping_rounds=int(section["early_stopping_rounds"]),
            threads=int(section["threads"]),
            level=dict(section["level"]),
            deviation=dict(section["deviation"]),
        )
    except KeyError as error:
        raise ConfigError(f"forecast is missing {error.args[0]!r}") from None
    except ValueError as error:
        raise ConfigError(f"forecast: {error}") from None


def _load_bound(section: Mapping[str, Any]) -> BoundConfig:
    _reject_unknown(section, {"annual"}, "bound")
    annual = _section(section, "annual")
    _reject_unknown(
        annual, {"first_day", "last_day", "mip_gap", "time_limit_s"}, "bound.annual"
    )
    try:
        return BoundConfig(
            annual=AnnualBoundConfig(
                first_day=_as_date(annual["first_day"], "bound.annual"),
                last_day=_as_date(annual["last_day"], "bound.annual"),
                mip_gap=_optional_float(annual.get("mip_gap")),
                time_limit_s=_optional_float(annual.get("time_limit_s")),
            )
        )
    except KeyError as error:
        raise ConfigError(f"bound.annual is missing {error.args[0]!r}") from None
    except ValueError as error:
        raise ConfigError(f"bound.annual: {error}") from None


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
