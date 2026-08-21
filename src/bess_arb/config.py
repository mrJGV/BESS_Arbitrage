"""Loader for ``config/params.yaml`` — the single source of numeric truth.

Imports nothing solver-related: it builds the solver-free dataclasses from
:mod:`bess_arb.model.spec` and hands them to whichever backend the config
names. Unknown keys are an error rather than a shrug, because a silently
ignored ``c_deg_eur_mhw`` would produce a plausible wrong number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bess_arb.model.spec import BatteryParams, SolverConfig

__all__ = ["DEFAULT_CONFIG_PATH", "Config", "ConfigError", "load_config"]

# src/bess_arb/config.py -> src/bess_arb -> src -> repository root.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "params.yaml"


@dataclass(frozen=True, slots=True)
class Config:
    """Everything ``config/params.yaml`` currently declares."""

    battery: BatteryParams
    c_deg_sensitivity: tuple[float, ...]
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
        raw, {"battery", "sensitivity", "backend", "solver", "seed"}, "<root>"
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
        backend=backend,
        solver=solver,
        seed=seed,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
