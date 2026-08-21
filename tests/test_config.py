"""``config/params.yaml`` is the single source of numeric truth.

These tests pin the values the published rationale commits to, so that a
number can be changed but not *silently* changed: editing the asset or the
degradation sweep breaks a test that names the section of
``docs/DECISIONS.md`` it came from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bess_arb.config import DEFAULT_CONFIG_PATH, ConfigError, load_config

MINIMAL_CONFIG = """
battery:
  p_max_mw: 10.0
  e_max_mwh: 20.0
  eta_rt: 0.85
  c_deg_eur_mwh: 17.0
  charge_tariff_eur_mwh: 0.0
sensitivity:
  c_deg_eur_mwh: [5.0, 17.0, 40.0]
backend: pyomo
solver:
  name: highs
  mip_gap: 0.0
  time_limit_s: 300.0
  threads: 1
  options: {}
seed: 20260819
"""


def test_the_shipped_config_loads() -> None:
    assert DEFAULT_CONFIG_PATH.is_file(), DEFAULT_CONFIG_PATH
    load_config()


def test_the_asset_is_the_one_the_rationale_describes() -> None:
    """docs/DECISIONS.md §1.3 and §3.1: 10 MW / 20 MWh, 2-hour, 85% round trip."""
    battery = load_config().battery

    assert battery.p_max_mw == 10.0
    assert battery.e_max_mwh == 20.0
    assert battery.eta_rt == 0.85
    assert battery.duration_h == 2.0
    assert battery.eta_c == pytest.approx(0.9219544457292887)
    assert battery.eta_c == battery.eta_d


def test_the_degradation_sweep_is_the_published_three_point_one() -> None:
    """docs/DECISIONS.md §3.3: 5 / 17 / 40 €/MWh, central value 17."""
    config = load_config()

    assert config.c_deg_sensitivity == (5.0, 17.0, 40.0)
    assert config.battery.c_deg_eur_mwh == 17.0
    assert config.battery.c_deg_eur_mwh in config.c_deg_sensitivity


def test_the_charge_tariff_defaults_to_zero() -> None:
    """docs/DECISIONS.md §3.4: parametrised because the regulation is live."""
    assert load_config().battery.charge_tariff_eur_mwh == 0.0


def test_the_backend_and_solver_come_from_the_file() -> None:
    """Invariant 7: switching solver is a config change, never a code change."""
    config = load_config()

    assert config.backend == "pyomo"
    assert config.solver.name == "highs"
    # An exact gap: HiGHS's 1e-4 default would be €0.15 of slop on the
    # golden case's €1,500, which the 1e-6 assertion would not survive.
    assert config.solver.mip_gap == 0.0


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_with_c_deg_changes_only_the_degradation_cost(tmp_path: Path) -> None:
    """The sweep is three runs of one config, not three configs."""
    config = load_config(_write(tmp_path, MINIMAL_CONFIG))
    swept = config.with_c_deg(40.0)

    assert swept.battery.c_deg_eur_mwh == 40.0
    assert swept.battery.p_max_mw == config.battery.p_max_mw
    assert swept.battery.e_max_mwh == config.battery.e_max_mwh
    assert swept.battery.eta_rt == config.battery.eta_rt
    assert swept.battery.charge_tariff_eur_mwh == (config.battery.charge_tariff_eur_mwh)
    assert swept.solver == config.solver
    assert swept.seed == config.seed


def test_a_misspelled_key_is_an_error_not_a_shrug(tmp_path: Path) -> None:
    """``c_deg_eur_mhw`` silently ignored would produce a plausible wrong number."""
    text = MINIMAL_CONFIG.replace("c_deg_eur_mwh: 17.0", "c_deg_eur_mhw: 17.0")

    with pytest.raises(ConfigError, match="c_deg_eur_mhw"):
        load_config(_write(tmp_path, text))


def test_a_missing_section_is_an_error(tmp_path: Path) -> None:
    text = "\n".join(
        line
        for line in MINIMAL_CONFIG.splitlines()
        if not line.startswith("sensitivity:") and "[5.0, 17.0, 40.0]" not in line
    )

    with pytest.raises(ConfigError, match="sensitivity"):
        load_config(_write(tmp_path, text))


def test_a_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_physically_impossible_parameters_are_rejected(tmp_path: Path) -> None:
    """Efficiency above 1 is a typo, and the dataclass is where it stops."""
    text = MINIMAL_CONFIG.replace("eta_rt: 0.85", "eta_rt: 1.85")

    with pytest.raises(ValueError, match="eta_rt"):
        load_config(_write(tmp_path, text))
