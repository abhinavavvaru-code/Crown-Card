"""Credit onboarding pipeline.

Generates synthetic student applicants, normalizes them, runs rule-based
underwriting tuned for thin-file students, assigns starter credit limits with a
tenure-based step-up schedule, and persists everything to SQLite.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Any

import numpy as np

from ..config import economics_config, underwriting_config
from .schema import (
    Applicant,
    Account,
    CreditLimitStepUp,
    CourseCompletion,
    get_sessionmaker,
    init_db,
)

FIRST_NAMES = [
    "Ava", "Liam", "Maya", "Noah", "Sofia", "Ethan", "Zoe", "Kai", "Priya",
    "Diego", "Mei", "Omar", "Nina", "Leo", "Aisha", "Jonah", "Yara", "Theo",
    "Luca", "Amara", "Ravi", "Elena", "Sam", "Hana", "Marco",
]
LAST_NAMES = [
    "Chen", "Patel", "Kim", "Garcia", "Nguyen", "Okafor", "Rossi", "Cohen",
    "Silva", "Haddad", "Park", "Mensah", "Ali", "Ivanov", "Reyes", "Tanaka",
    "Diallo", "Weber", "Santos", "Khan",
]


@dataclass
class ApplicantRecord:
    """Normalized applicant prior to persistence."""
    name: str
    school_year: str
    monthly_income: float
    is_thin_file: bool
    credit_score: int | None
    dti: float
    enrolled: bool = True


# --- Synthetic generation --------------------------------------------------

def generate_applicants(
    n: int | None = None,
    cfg: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> list[ApplicantRecord]:
    """Generate `n` synthetic student applicants from the underwriting config."""
    cfg = cfg or underwriting_config()
    gen = cfg["generator"]
    n = n or gen["n_applicants"]
    rng = rng or np.random.default_rng(cfg.get("seed", 42))

    years = gen["school_years"]
    weights = np.asarray(gen["school_year_weights"], dtype=float)
    weights = weights / weights.sum()

    records: list[ApplicantRecord] = []
    for _ in range(n):
        year = str(rng.choice(years, p=weights))

        ln = gen["income_lognormal"][year]
        income = float(rng.lognormal(mean=ln["mu"], sigma=ln["sigma"]))
        income = round(min(income, 5000.0), 2)  # cap absurd tails

        is_thin = bool(rng.random() < gen["thin_file_prob"])
        if is_thin:
            score = None
        else:
            cs = gen["credit_score"]
            score = int(np.clip(rng.normal(cs["mean"], cs["sd"]), cs["min"], cs["max"]))

        dti_cfg = gen["dti"]
        dti = float(rng.beta(dti_cfg["alpha"], dti_cfg["beta"]) * 0.6)

        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        records.append(
            ApplicantRecord(
                name=name,
                school_year=year,
                monthly_income=income,
                is_thin_file=is_thin,
                credit_score=score,
                dti=round(dti, 3),
                enrolled=True,
            )
        )
    return records


# --- Underwriting ----------------------------------------------------------

@dataclass
class Decision:
    approved: bool
    reason: str | None
    limit: float | None


def underwrite(rec: ApplicantRecord, cfg: dict[str, Any] | None = None) -> Decision:
    """Rule-based underwriting decision for a single applicant.

    Hard declines short-circuit; approved applicants receive a starter limit in
    the $300-$1,000 band.
    """
    cfg = cfg or underwriting_config()
    rules = cfg["rules"]

    if rec.monthly_income < rules["min_monthly_income"]:
        return Decision(False, "insufficient_income", None)

    if rec.dti > rules["max_dti"]:
        return Decision(False, "dti_too_high", None)

    if rec.is_thin_file:
        if rules["thin_file_requires_enrollment"] and not rec.enrolled:
            return Decision(False, "not_enrolled", None)
    else:
        if rec.credit_score is not None and rec.credit_score < rules["min_credit_score_scored"]:
            return Decision(False, "credit_score_below_min", None)

    limit = assign_starter_limit(rec, cfg)
    return Decision(True, None, limit)


def assign_starter_limit(rec: ApplicantRecord, cfg: dict[str, Any] | None = None) -> float:
    """Assign a starter limit in the configured $300-$1,000 band."""
    cfg = cfg or underwriting_config()
    econ = economics_config()
    ls = cfg["limit_scoring"]
    lo = econ["credit_limits"]["starter_min"]
    hi = econ["credit_limits"]["starter_max"]

    limit = ls["base"]
    limit += rec.monthly_income * ls["income_coefficient"]

    if rec.is_thin_file:
        limit -= ls["thin_file_penalty"]
    elif rec.credit_score is not None:
        min_score = cfg["rules"]["min_credit_score_scored"]
        limit += max(0, rec.credit_score - min_score) * ls["scored_bonus_per_point"]

    if rec.school_year == "graduate":
        limit += ls["graduate_bonus"]
    elif rec.school_year == "senior":
        limit += ls["senior_bonus"]

    limit = float(np.clip(limit, lo, hi))
    rounding = ls["rounding"]
    limit = round(limit / rounding) * rounding
    return float(np.clip(limit, lo, hi))


def step_up_schedule(starter_limit: float, cfg: dict[str, Any] | None = None) -> list[dict]:
    """Tenure-based limit step-ups for an account, capped at the absolute cap."""
    econ = cfg or economics_config()
    cl = econ["credit_limits"]
    cap = cl["absolute_cap"]
    rows = []
    for row in cl["step_up_schedule"]:
        limit = min(round(starter_limit * row["multiplier"] / 50) * 50, cap)
        rows.append({"effective_month": row["months"], "limit": float(limit)})
    return rows


# --- Persistence -----------------------------------------------------------

def run_onboarding(
    records: list[ApplicantRecord] | None = None,
    db_path=None,
    drop: bool = True,
    open_accounts: bool = True,
    max_accounts: int | None = None,
    course_completion_prob: float = 0.55,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Underwrite applicants and persist applicants + opened accounts.

    Returns a summary dict. If `records` is None, a fresh synthetic cohort is
    generated from config. `max_accounts` caps how many approved applicants are
    activated into the beta (extra approvals stay as applicants without an
    account), so a target cohort size can be enforced.
    """
    uw_cfg = underwriting_config()
    econ_cfg = economics_config()
    rng = rng or np.random.default_rng(uw_cfg.get("seed", 42))

    if records is None:
        records = generate_applicants(cfg=uw_cfg, rng=rng)

    init_db(db_path, drop=drop)
    Session = get_sessionmaker(db_path)

    approved = declined = activated = 0
    with Session() as session:
        for rec in records:
            decision = underwrite(rec, uw_cfg)
            applicant = Applicant(
                name=rec.name,
                school_year=rec.school_year,
                monthly_income=rec.monthly_income,
                is_thin_file=rec.is_thin_file,
                credit_score=rec.credit_score,
                dti=rec.dti,
                enrolled=rec.enrolled,
                approved=decision.approved,
                decline_reason=decision.reason,
                assigned_limit=decision.limit,
                applied_at=date.today(),
            )
            session.add(applicant)
            session.flush()  # assign applicant.id

            if decision.approved:
                approved += 1
            else:
                declined += 1

            cap_reached = max_accounts is not None and activated >= max_accounts
            if decision.approved and open_accounts and not cap_reached:
                activated += 1
                account = Account(
                    applicant_id=applicant.id,
                    opened_at=date.today(),
                    starter_limit=decision.limit,
                    current_limit=decision.limit,
                    is_thin_file=rec.is_thin_file,
                    school_year=rec.school_year,
                    status="active",
                )
                session.add(account)
                session.flush()

                for row in step_up_schedule(decision.limit, econ_cfg):
                    session.add(
                        CreditLimitStepUp(
                            account_id=account.id,
                            effective_month=row["effective_month"],
                            limit=row["limit"],
                        )
                    )

                completed = bool(rng.random() < course_completion_prob)
                session.add(
                    CourseCompletion(
                        account_id=account.id,
                        completed=completed,
                        completed_at=None,
                    )
                )

        session.commit()

    return {
        "total": len(records),
        "approved": approved,
        "declined": declined,
        "activated": activated,
        "approval_rate": round(approved / len(records), 3) if records else 0.0,
    }
