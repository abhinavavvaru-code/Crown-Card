"""Seed a 50-user beta cohort with 12-18 months of realistic student
transaction histories, weighted toward campus-area merchants, plus the monthly
statements that roll them up (interest, late fees, grace period).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from ..config import economics_config, underwriting_config
from ..economics import Economics, monthly_interest, is_in_grace
from .onboarding import run_onboarding, generate_applicants
from .schema import (
    Account,
    Transaction,
    Statement,
    CourseCompletion,
    get_sessionmaker,
)

# Merchant categories. Partner (campus) merchants get a higher spend weight and
# are flagged so downstream interchange/cashback math can reward them.
MERCHANT_CATEGORIES = [
    # (category, is_partner, base_weight, typical_ticket)
    ("campus_dining", True, 3.5, 12.0),
    ("campus_bookstore", True, 1.2, 45.0),
    ("campus_coffee", True, 3.0, 6.0),
    ("local_restaurant", True, 2.5, 22.0),
    ("groceries", False, 2.0, 35.0),
    ("rideshare", False, 1.5, 15.0),
    ("streaming", False, 1.0, 12.0),
    ("clothing", False, 1.0, 55.0),
    ("electronics", False, 0.5, 120.0),
    ("travel", False, 0.4, 180.0),
    ("pharmacy", False, 0.8, 18.0),
]


def _month_end(start: date, month_index: int) -> date:
    """Approximate period end for the Nth 30-day cycle after `start`."""
    return start + timedelta(days=30 * (month_index + 1))


def generate_transactions_for_account(
    account: Account,
    n_months: int,
    econ_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[Transaction], list[Statement]]:
    """Build transaction + statement rows for one account over `n_months`."""
    cats = MERCHANT_CATEGORIES
    weights = np.array([c[2] for c in cats], dtype=float)
    weights = weights / weights.sum()

    econ = Economics.from_config(econ_cfg)
    opened = account.opened_at

    # School-year scaled monthly spend target (median). Older students spend more.
    year_scale = {
        "freshman": 0.85,
        "sophomore": 0.95,
        "junior": 1.05,
        "senior": 1.15,
        "graduate": 1.35,
    }.get(account.school_year, 1.0)

    transactions: list[Transaction] = []
    statements: list[Statement] = []

    for m in range(n_months):
        cycle_month = m + 1  # 1-indexed
        period_end = _month_end(opened, m)

        # Target monthly purchase volume, lognormal around a student-scaled median.
        target = float(rng.lognormal(mean=6.0, sigma=0.35)) * year_scale
        target = float(np.clip(target, 60, min(1.6 * account.current_limit, 2500)))

        # Draw a random number of transactions and distribute spend across them.
        n_txns = int(np.clip(rng.poisson(18), 5, 60))
        purchases = 0.0
        for _ in range(n_txns):
            idx = rng.choice(len(cats), p=weights)
            category, is_partner, _w, ticket = cats[idx]
            amount = float(np.clip(rng.gamma(2.0, ticket / 2.0), 1.0, target))
            purchases += amount
            transactions.append(
                Transaction(
                    account_id=account.id,
                    posted_at=period_end - timedelta(days=int(rng.integers(0, 29))),
                    amount=round(amount, 2),
                    merchant_category=category,
                    is_partner_merchant=is_partner,
                )
            )

        # Scale the drawn purchases toward the target volume.
        if purchases > 0:
            scale = target / purchases
            for t in transactions[-n_txns:]:
                t.amount = round(t.amount * scale, 2)
            purchases = target

        # Revolving behavior: thin-file students revolve a bit more.
        revolve_rate = float(rng.beta(3.0, 4.0) if account.is_thin_file else rng.beta(2.0, 5.0))
        revolving_balance = round(min(purchases * revolve_rate, account.current_limit), 2)

        in_grace = is_in_grace(cycle_month, econ)
        interest = round(monthly_interest(revolving_balance, cycle_month, econ), 2)

        late = econ.late_fee if rng.random() < econ.late_fee_prob_monthly else 0.0

        statements.append(
            Statement(
                account_id=account.id,
                cycle_month=cycle_month,
                period_end=period_end,
                purchases=round(purchases, 2),
                revolving_balance=revolving_balance,
                interest_charged=interest,
                late_fee_charged=round(late, 2),
                in_grace=in_grace,
            )
        )

    return transactions, statements


def seed_cohort(
    db_path=None,
    n_users: int = 50,
    min_months: int = 12,
    max_months: int = 18,
    seed: int | None = None,
) -> dict[str, Any]:
    """Full seed: onboard a synthetic cohort, then attach transaction histories.

    Returns a summary dict.
    """
    uw_cfg = underwriting_config()
    econ_cfg = economics_config()
    seed = seed if seed is not None else uw_cfg.get("seed", 42)
    rng = np.random.default_rng(seed)

    # Generate enough applicants that ~n_users get approved; overshoot then trim.
    records = generate_applicants(n=int(n_users * 1.6), cfg=uw_cfg, rng=rng)

    summary = run_onboarding(
        records=records,
        db_path=db_path,
        drop=True,
        open_accounts=True,
        max_accounts=n_users,
        rng=rng,
    )

    Session = get_sessionmaker(db_path)
    total_txns = 0
    total_statements = 0
    with Session() as session:
        accounts = session.query(Account).filter(Account.status == "active").all()
        for account in accounts:
            n_months = int(rng.integers(min_months, max_months + 1))
            txns, stmts = generate_transactions_for_account(account, n_months, econ_cfg, rng)
            session.add_all(txns)
            session.add_all(stmts)
            total_txns += len(txns)
            total_statements += len(stmts)
        session.commit()

    summary.update(
        {
            "seeded_accounts": min(len(accounts), n_users),
            "transactions": total_txns,
            "statements": total_statements,
        }
    )
    return summary
