"""Tests for the Monte Carlo simulation statistics (seeded RNG)."""
import numpy as np
import pytest

from crown_card.simulations.monte_carlo import (
    simulate,
    segment_table,
    portfolio_summary,
    _annual_to_monthly_hazard,
)


@pytest.fixture(scope="module")
def result():
    return simulate(n_runs=4000)


def test_hazard_conversion_roundtrips():
    annual = 0.08
    monthly = _annual_to_monthly_hazard(annual)
    # Compounding the monthly hazard over 12 months recovers the annual.
    recovered = 1.0 - (1.0 - monthly) ** 12
    assert recovered == pytest.approx(annual)


def test_simulation_is_deterministic():
    a = simulate(n_runs=2000)
    b = simulate(n_runs=2000)
    assert np.allclose(a.clv, b.clv)


def test_result_shapes(result):
    n = result.clv.size
    assert result.defaulted.size == n
    assert result.total_loss.size == n
    assert result.months_active.size == n
    assert result.school_year.size == n


def test_defaulters_have_losses(result):
    # Every charged-off cardholder should carry a non-negative recorded loss.
    assert np.all(result.total_loss[result.defaulted] >= 0)
    # No losses attributed to cardholders who never defaulted.
    assert np.all(result.total_loss[~result.defaulted] == 0)


def test_thin_file_default_rate_higher(result):
    thin_rate = result.defaulted[result.thin_file].mean()
    scored_rate = result.defaulted[~result.thin_file].mean()
    assert thin_rate > scored_rate


def test_course_completion_reduces_defaults(result):
    completed = result.defaulted[result.course].mean()
    not_completed = result.defaulted[~result.course].mean()
    assert completed < not_completed


def test_months_active_within_horizon(result):
    horizon = result.cfg["horizon_months"]
    assert result.months_active.max() <= horizon
    assert result.months_active.min() >= 0


def test_portfolio_summary_keys(result):
    s = portfolio_summary(result)
    assert s["clv_ci95_low"] <= s["mean_clv"] <= s["clv_ci95_high"]
    assert 0.0 <= s["default_rate"] <= 1.0
    assert 0.0 <= s["annual_charge_off_rate"] <= 1.0
    assert 0.0 <= s["loss_on_volume"] <= 1.0
    assert "var_95" in s and "cvar_95" in s
    # CVaR (mean of the worst tail) is no better than the VaR threshold.
    assert s["cvar_95"] <= s["var_95"]


def test_segment_table_covers_all_segments(result):
    seg = segment_table(result)
    kinds = set(seg["segment_kind"])
    assert {"overall", "school_year", "file_type", "course"} <= kinds
    # Segment n's for a partition sum to the overall n.
    overall_n = seg.loc[seg["segment_kind"] == "overall", "n"].iloc[0]
    for kind in ["file_type", "course", "school_year"]:
        assert seg.loc[seg["segment_kind"] == kind, "n"].sum() == overall_n
