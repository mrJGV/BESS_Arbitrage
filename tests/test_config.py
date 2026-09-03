"""``config/params.yaml`` is the single source of numeric truth.

These tests pin the values the published rationale commits to, so that a
number can be changed but not *silently* changed: editing the asset or the
degradation sweep breaks a test that names the section of
``docs/DECISIONS.md`` it came from.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from bess_arb.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from bess_arb.forecast.features import FIRST_COMPLETE_LAG_DAYS

MINIMAL_CONFIG = """
battery:
  p_max_mw: 10.0
  e_max_mwh: 20.0
  eta_rt: 0.85
  c_deg_eur_mwh: 17.0
  charge_tariff_eur_mwh: 0.0
sensitivity:
  c_deg_eur_mwh: [5.0, 17.0, 40.0]
data:
  directory: data
  snapshot_date: 2026-08-24
  regimes:
    hourly:
      dt_h: 1.0
      first_day: 2022-01-01
      last_day: 2025-09-30
    quarter_hourly:
      dt_h: 0.25
      first_day: 2025-10-01
      last_day: 2026-08-23
  series:
    price_eur_mwh:
      indicator: 600
      geo_id: 3
  crosscheck:
    first_day: 2024-06-01
    last_day: 2024-06-30
    tolerance_eur_mwh: 0.01
horizon:
  window_days: 2
  implement_days: 1
  warmup_days: 7
  soc_initial_fraction: 0.5
bound:
  annual:
    first_day: 2024-01-01
    last_day: 2024-12-31
    mip_gap: 0.001
    time_limit_s: 3600.0
forecast:
  lags_days: [2, 3]
  refit_days: 30
  min_train_days: 365
  min_deviation_days: 30
  level_rounds: 600
  deviation_rounds: 400
  validation_days: 60
  early_stopping_rounds: 50
  threads: 4
  level:
    num_leaves: 63
  deviation:
    num_leaves: 31
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


def test_the_verified_indicator_ids_are_pinned() -> None:
    """Verified against the live ESIOS catalogue on 24 August 2026.

    Pinned here so an ID can be changed but not *silently* changed. The
    snapshot is frozen under invariant 6, so a quietly edited indicator
    number would put the config and the committed Parquet out of step with
    no error anywhere.
    """
    series = {spec.name: spec for spec in load_config().data.series}

    assert series["price_eur_mwh"].indicator_id == 600
    assert series["wind_forecast_mw"].indicator_id == 1777
    assert series["solar_forecast_mw"].indicator_id == 1779
    assert series["demand_forecast_mw"].indicator_id == 1775


def test_the_price_series_names_its_geography() -> None:
    """Indicator 600 carries six geographies; España is 3.

    Without this key the loader keeps whichever the server sent first, which
    is Portugal. The two agree on most days, so the error would not announce
    itself — 36 of 744 periods differed in January 2024, by up to €39/MWh.
    """
    series = {spec.name: spec for spec in load_config().data.series}

    assert series["price_eur_mwh"].geo_id == 3


def test_the_regimes_are_the_two_the_market_has() -> None:
    """docs/DECISIONS.md §1.1: hourly to 30 September 2025, then 15-minute MTUs."""
    data = load_config().data

    hourly = data.regime("hourly")
    quarter = data.regime("quarter_hourly")

    assert hourly.regime.dt_h == 1.0
    assert quarter.regime.dt_h == 0.25
    # The regimes must abut exactly: a gap loses days, an overlap
    # double-counts them.
    assert quarter.first_day == hourly.last_day + dt.timedelta(days=1)
    assert hourly.first_day == dt.date(2022, 1, 1)


def test_the_horizon_protocol_is_the_published_one() -> None:
    """docs/DECISIONS.md §2.1 and §2.6, in days rather than in hours.

    "48-hour window, first 24 implemented" is what §2.1 says, and it is the
    ordinary-day statement of a two-day window with one day committed. On a
    transition day the same protocol solves 47 or 49 hours; writing 48 into
    the config would put invariant 4's failure in the one file where nothing
    else could catch it.
    """
    horizon = load_config().horizon

    assert horizon.window_days == 2
    assert horizon.implement_days == 1
    assert horizon.warmup_days == 7
    assert horizon.soc_initial_fraction == 0.5
    assert horizon.soc_initial_mwh(20.0) == 10.0


def test_the_annual_bound_covers_a_whole_year_inside_one_regime() -> None:
    """docs/DECISIONS.md §2.4: the days must be comparable with a rolling run."""
    config = load_config()
    annual = config.bound.annual
    hourly = config.data.regime("hourly")

    assert annual.first_day == dt.date(2024, 1, 1)
    assert annual.last_day == dt.date(2024, 12, 31)
    assert hourly.first_day <= annual.first_day
    assert annual.last_day <= hourly.last_day
    # Looser than a window solve, and stated rather than inherited: 8,760
    # binaries cannot afford the exact gap the golden test needs.
    assert annual.mip_gap is not None
    assert config.solver.mip_gap is not None
    assert annual.mip_gap > config.solver.mip_gap


def test_a_horizon_that_implements_more_than_it_solves_is_rejected(
    tmp_path: Path,
) -> None:
    """Settling days the optimiser never saw would be silent nonsense."""
    text = MINIMAL_CONFIG.replace("implement_days: 1", "implement_days: 3")

    with pytest.raises(ConfigError, match="implement_days"):
        load_config(_write(tmp_path, text))


def test_the_snapshot_filename_carries_the_freeze_date() -> None:
    data = load_config().data

    assert data.parquet_path("hourly").name == "hourly_2026-08-24.parquet"


def test_a_delivery_day_with_a_time_on_it_is_rejected(tmp_path: Path) -> None:
    """A day plus a time is ambiguous about which zone decides."""
    text = MINIMAL_CONFIG.replace(
        "first_day: 2022-01-01", "first_day: 2022-01-01 00:00:00"
    )

    with pytest.raises(ConfigError, match="carries a time"):
        load_config(_write(tmp_path, text))


def test_the_forecast_lags_start_after_the_last_complete_delivery_day() -> None:
    """The shipped config cannot ask for a lag that reaches past the gate.

    ``build_features`` refuses one, so this would surface as an exception at
    the first forecast rather than as a wrong number — but the shipped file is
    what a reader checks, and a lag of 1 sitting in it would read as endorsed.
    """
    lags = load_config().forecast.lags_days

    assert lags
    assert min(lags) >= FIRST_COMPLETE_LAG_DAYS


def test_the_forecast_hyperparameters_reach_the_loader_untouched() -> None:
    """LightGBM's own keys pass through; the project's are validated.

    The asymmetry is deliberate (see ``ForecastConfig``): re-declaring
    LightGBM's parameter list here would only guarantee a stale copy of it.
    """
    forecast = load_config().forecast

    assert forecast.level["num_leaves"] > 0
    assert forecast.deviation["num_leaves"] > 0
    assert forecast.refit_days >= 1
    assert forecast.min_train_days >= 365


def test_a_missing_forecast_key_is_named(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace("  refit_days: 30\n", "")

    with pytest.raises(ConfigError, match="refit_days"):
        load_config(_write(tmp_path, text))


def test_an_unknown_forecast_key_is_refused(tmp_path: Path) -> None:
    """A silently ignored key would produce a plausible wrong number."""
    text = MINIMAL_CONFIG.replace("  refit_days: 30\n", "  refit_dayz: 30\n")

    with pytest.raises(ConfigError, match="refit_dayz"):
        load_config(_write(tmp_path, text))


def test_a_series_without_an_indicator_is_an_error(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace("      indicator: 600\n", "")

    with pytest.raises(ConfigError, match="indicator"):
        load_config(_write(tmp_path, text))


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
    assert swept.horizon == config.horizon
    assert swept.bound == config.bound


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
