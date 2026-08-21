"""Why the binaries are not redundant — demonstrated, not asserted.

With positive prices and ``eta_rt < 1``, charge/discharge exclusivity holds
at the LP relaxation's optimum on its own and the binaries cost solve time
for nothing. **Negative prices end that.** Charging and discharging at once
burns energy and is paid for doing so: holding SoC flat requires discharging
``eta_c·eta_d = 0.85`` of what is charged, so the relaxation collects
``(1 − eta_rt)·|λ| = 0.15·|λ|`` per MW of fictitious throughput — €7.50/MWh
at λ = −50. Spain sees negative prices with increasing frequency.

The instance: a full battery (SoC₀ = 20 MWh) with λ = −50 throughout four
hours. The MILP's optimum is hand-computable. Being paid to draw and paying
to deliver, it wants to end the window as full as it can while having drawn
as much as possible, so it discharges 17 MWh in two hours to make room and
draws 20 MWh in the other two: ``50 × (20 − 17) = €150``. The 17 is
``0.85 × 20`` — the round trip again.

What is asserted about the *relaxation* is qualitative: that it is strictly
better, and that it achieves this by doing something physically impossible.
The size of the gap depends on which of ``P_max`` and the SoC bounds bind,
so a hard-coded gap value would look rigorous and be brittle. (For the
record, HiGHS returns 162.16 here.)
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ModelFactory

TOL = 1e-6
FLAT_NEGATIVE_PRICES = np.array([-50.0, -50.0, -50.0, -50.0])
FULL_BATTERY_MWH = 20.0
MILP_OPTIMUM = 150.0


def test_milp_optimum_under_flat_negative_prices(make_model: ModelFactory) -> None:
    solution = make_model().solve(FLAT_NEGATIVE_PRICES, soc_initial=FULL_BATTERY_MWH)

    assert solution.objective == pytest.approx(MILP_OPTIMUM, abs=TOL)
    assert solution.charged_mwh == pytest.approx(20.0, abs=TOL)
    assert solution.discharged_mwh == pytest.approx(17.0, abs=TOL)


def test_relaxation_is_strictly_better_than_the_milp(
    make_model: ModelFactory,
) -> None:
    """If this ever stops holding, the binaries have become removable."""
    milp = make_model().solve(FLAT_NEGATIVE_PRICES, soc_initial=FULL_BATTERY_MWH)
    relaxed = make_model(relax_binaries=True).solve(
        FLAT_NEGATIVE_PRICES, soc_initial=FULL_BATTERY_MWH
    )

    assert relaxed.objective > milp.objective + TOL


def test_relaxation_charges_and_discharges_at_once_and_the_milp_does_not(
    make_model: ModelFactory,
) -> None:
    """The mechanism, not just the symptom.

    A strictly better relaxation could in principle come from somewhere
    else; this pins it to the simultaneity that the integrality forbids.
    """
    milp = make_model().solve(FLAT_NEGATIVE_PRICES, soc_initial=FULL_BATTERY_MWH)
    relaxed = make_model(relax_binaries=True).solve(
        FLAT_NEGATIVE_PRICES, soc_initial=FULL_BATTERY_MWH
    )

    # `> TOL`, not `> 0`: the solver reports -0.0 and 1e-15 where it means
    # zero, and either would make a strict test lie.
    relaxed_simultaneous = (relaxed.p_c_mw > TOL) & (relaxed.p_d_mw > TOL)
    milp_simultaneous = (milp.p_c_mw > TOL) & (milp.p_d_mw > TOL)

    assert relaxed_simultaneous.any()
    assert not milp_simultaneous.any()


def test_relaxation_is_harmless_when_prices_are_positive(
    make_model: ModelFactory,
) -> None:
    """The other half of the claim: the binaries only bind where stated.

    On the golden case the relaxation reaches the same €1,500, which is what
    makes 'negative prices are the reason' a specific claim rather than a
    general suspicion of LPs.
    """
    prices = np.array([10.0, 10.0, 100.0, 100.0])

    milp = make_model().solve(prices, soc_initial=0.0)
    relaxed = make_model(relax_binaries=True).solve(prices, soc_initial=0.0)

    assert relaxed.objective == pytest.approx(milp.objective, abs=TOL)
