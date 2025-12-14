"""Tests for the rule-based underwriting and limit assignment."""
import numpy as np
import pytest

from crown_card.config import economics_config, underwriting_config
from crown_card.pipeline.onboarding import (
    ApplicantRecord,
    underwrite,
    assign_starter_limit,
    step_up_schedule,
    generate_applicants,
)


@pytest.fixture
def uw():
    return underwriting_config()


def scored(**kw):
    base = dict(
        name="Test User", school_year="junior", monthly_income=800.0,
        is_thin_file=False, credit_score=680, dti=0.2, enrolled=True,
    )
    base.update(kw)
    return ApplicantRecord(**base)


def thin(**kw):
    base = dict(
        name="Test User", school_year="freshman", monthly_income=500.0,
        is_thin_file=True, credit_score=None, dti=0.2, enrolled=True,
    )
    base.update(kw)
    return ApplicantRecord(**base)


def test_low_income_declined(uw):
    d = underwrite(scored(monthly_income=100.0), uw)
    assert not d.approved
    assert d.reason == "insufficient_income"


def test_high_dti_declined(uw):
    d = underwrite(scored(dti=0.9), uw)
    assert not d.approved
    assert d.reason == "dti_too_high"


def test_low_score_scored_declined(uw):
    d = underwrite(scored(credit_score=500), uw)
    assert not d.approved
    assert d.reason == "credit_score_below_min"


def test_thin_file_not_enrolled_declined(uw):
    d = underwrite(thin(enrolled=False), uw)
    assert not d.approved
    assert d.reason == "not_enrolled"


def test_thin_file_enrolled_approved(uw):
    d = underwrite(thin(), uw)
    assert d.approved
    assert d.reason is None


def test_limit_within_starter_band(uw):
    econ = economics_config()["credit_limits"]
    for rec in [scored(), thin(), scored(monthly_income=5000), thin(monthly_income=250)]:
        limit = assign_starter_limit(rec, uw)
        assert econ["starter_min"] <= limit <= econ["starter_max"]
        assert limit % uw["limit_scoring"]["rounding"] == 0


def test_thin_file_gets_lower_limit_than_scored(uw):
    # Same income; thin-file should not exceed an otherwise-strong scored file.
    scored_limit = assign_starter_limit(scored(monthly_income=800, credit_score=740), uw)
    thin_limit = assign_starter_limit(thin(monthly_income=800), uw)
    assert thin_limit <= scored_limit


def test_step_up_schedule_monotonic_and_capped():
    econ = economics_config()
    cap = econ["credit_limits"]["absolute_cap"]
    rows = step_up_schedule(1000.0, econ)
    limits = [r["limit"] for r in rows]
    assert limits == sorted(limits)             # non-decreasing
    assert all(l <= cap for l in limits)        # capped
    assert rows[0]["effective_month"] == 0


def test_generator_is_deterministic():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = generate_applicants(n=20, rng=rng1)
    b = generate_applicants(n=20, rng=rng2)
    assert [x.name for x in a] == [x.name for x in b]
    assert [x.monthly_income for x in a] == [x.monthly_income for x in b]


def test_generator_respects_count():
    recs = generate_applicants(n=33)
    assert len(recs) == 33
