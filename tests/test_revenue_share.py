"""Tests for the shared revenue-share economics."""
import math

import pytest

from crown_card.economics import (
    Economics,
    blended_interchange_rate,
    crown_interchange,
    gross_interchange,
    monthly_interest,
    is_in_grace,
    crown_banking_revenue,
    net_cashback_cost,
    crown_monthly_revenue,
)


@pytest.fixture
def econ():
    return Economics.from_config()


def test_interchange_split_is_75_25(econ):
    spend, partner_share = 1000.0, 0.0
    total = gross_interchange(spend, partner_share, econ)
    crown = crown_interchange(spend, partner_share, econ)
    assert crown == pytest.approx(total * econ.interchange_crown_share)
    assert crown == pytest.approx(total * 0.75)
    # Bank keeps the remainder.
    bank = total - crown
    assert bank == pytest.approx(total * 0.25)


def test_partner_interchange_rate_is_richer(econ):
    all_partner = blended_interchange_rate(1.0, econ)
    no_partner = blended_interchange_rate(0.0, econ)
    assert all_partner > no_partner
    assert no_partner == pytest.approx(econ.interchange_base_rate)
    assert all_partner == pytest.approx(econ.interchange_partner_rate)


def test_blended_rate_is_linear(econ):
    mid = blended_interchange_rate(0.5, econ)
    expected = 0.5 * econ.interchange_base_rate + 0.5 * econ.interchange_partner_rate
    assert mid == pytest.approx(expected)


def test_grace_period_zero_interest(econ):
    # First `intro_apr_months` cycles carry 0% APR -> no interest.
    for m in range(1, econ.intro_apr_months + 1):
        assert is_in_grace(m, econ)
        assert monthly_interest(500.0, m, econ) == 0.0
    # First post-grace month accrues interest.
    m = econ.intro_apr_months + 1
    assert not is_in_grace(m, econ)
    assert monthly_interest(500.0, m, econ) == pytest.approx(500.0 * econ.apr / 12.0)


def test_banking_fees_split_is_25_75(econ):
    gross = 400.0
    crown = crown_banking_revenue(gross, econ)
    assert crown == pytest.approx(gross * 0.25)
    assert (gross - crown) == pytest.approx(gross * 0.75)


def test_partner_cashback_higher_but_sponsor_offset(econ):
    spend = 1000.0
    # All non-partner: base cashback fully Crown-funded.
    cost_non_partner = net_cashback_cost(spend, 0.0, econ)
    assert cost_non_partner == pytest.approx(spend * econ.base_cashback)
    # All partner: higher gross cashback, but sponsor reimburses part of it.
    cost_partner = net_cashback_cost(spend, 1.0, econ)
    expected = spend * econ.partner_cashback * (1.0 - econ.partner_sponsor_offset)
    assert cost_partner == pytest.approx(expected)


def test_monthly_revenue_breakdown_consistency(econ):
    bd = crown_monthly_revenue(
        spend=500.0,
        partner_share=0.4,
        revolving_balance=200.0,
        account_month=6,  # post-grace
        econ=econ,
        late_fee_incurred=True,
    )
    # Net equals interchange + interest + late + cashback (cashback is negative).
    recomputed = bd["interchange"] + bd["interest"] + bd["late_fee"] + bd["cashback"]
    assert bd["net"] == pytest.approx(recomputed)
    assert bd["interest"] > 0  # post-grace
    assert bd["cashback"] < 0


def test_no_interest_in_grace_via_breakdown(econ):
    bd = crown_monthly_revenue(
        spend=500.0, partner_share=0.4, revolving_balance=200.0,
        account_month=1, econ=econ,
    )
    assert bd["interest"] == 0.0
