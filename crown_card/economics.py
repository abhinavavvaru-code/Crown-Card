"""Crown unit economics — the single source of truth for revenue-share math.

Crown earns two revenue streams, each split with the issuing bank:

  * Interchange on purchase volume   -> Crown 75% / bank 25%
  * Banking fees (interest + late)   -> Crown 25% / bank 75%

Partner (campus) merchants pay a richer interchange rate and offer higher
cashback, part of which is reimbursed by the partner/sponsor. The first three
statement cycles carry a 0% intro APR, so no interest accrues in that window.

Every simulation and scenario routes its money through the functions here so the
splits, the grace period, and the sponsor offset are defined exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import economics_config


@dataclass(frozen=True)
class Economics:
    interchange_base_rate: float
    interchange_partner_rate: float
    interchange_crown_share: float
    interchange_bank_share: float
    banking_crown_share: float
    banking_bank_share: float
    apr: float
    intro_apr: float
    intro_apr_months: int
    late_fee: float
    late_fee_prob_monthly: float
    base_cashback: float
    partner_cashback: float
    partner_sponsor_offset: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> "Economics":
        cfg = cfg or economics_config()
        ic = cfg["interchange"]
        bf = cfg["banking_fees"]
        rw = cfg["rewards"]
        return cls(
            interchange_base_rate=ic["base_rate"],
            interchange_partner_rate=ic["partner_rate"],
            interchange_crown_share=ic["crown_share"],
            interchange_bank_share=ic["bank_share"],
            banking_crown_share=bf["crown_share"],
            banking_bank_share=bf["bank_share"],
            apr=bf["apr"],
            intro_apr=bf["intro_apr"],
            intro_apr_months=bf["intro_apr_months"],
            late_fee=bf["late_fee"],
            late_fee_prob_monthly=bf["late_fee_prob_monthly"],
            base_cashback=rw["base_cashback"],
            partner_cashback=rw["partner_cashback"],
            partner_sponsor_offset=rw["partner_sponsor_offset"],
        )


# --- Interchange -----------------------------------------------------------

def blended_interchange_rate(partner_share: float, econ: Economics) -> float:
    """Effective interchange rate given the fraction of spend at partners."""
    partner_share = _clip01(partner_share)
    return (
        econ.interchange_base_rate * (1.0 - partner_share)
        + econ.interchange_partner_rate * partner_share
    )


def gross_interchange(spend: float, partner_share: float, econ: Economics) -> float:
    """Total interchange generated on `spend` (before the bank split)."""
    return spend * blended_interchange_rate(partner_share, econ)


def crown_interchange(spend: float, partner_share: float, econ: Economics) -> float:
    """Crown's share of interchange on `spend`."""
    return gross_interchange(spend, partner_share, econ) * econ.interchange_crown_share


# --- Interest --------------------------------------------------------------

def is_in_grace(account_month: int, econ: Economics) -> bool:
    """True while the 0% intro APR applies. `account_month` is 1-indexed."""
    return account_month <= econ.intro_apr_months


def monthly_interest(revolving_balance: float, account_month: int, econ: Economics) -> float:
    """Gross interest charged on a revolving balance in a given account month.

    Returns 0 during the intro-APR grace window.
    """
    apr = econ.intro_apr if is_in_grace(account_month, econ) else econ.apr
    return revolving_balance * apr / 12.0


# --- Banking fees (interest + late) ---------------------------------------

def crown_banking_revenue(gross_banking_fees: float, econ: Economics) -> float:
    """Crown's share of banking-fee revenue (interest + late fees)."""
    return gross_banking_fees * econ.banking_crown_share


# --- Rewards / cashback ----------------------------------------------------

def net_cashback_cost(spend: float, partner_share: float, econ: Economics) -> float:
    """Cashback cost borne by Crown after the partner sponsor offset.

    Non-partner spend earns base cashback (fully Crown-funded). Partner spend
    earns the higher partner cashback, but the sponsor reimburses a share of it.
    """
    partner_share = _clip01(partner_share)
    non_partner_spend = spend * (1.0 - partner_share)
    partner_spend = spend * partner_share

    base_cost = non_partner_spend * econ.base_cashback
    partner_gross = partner_spend * econ.partner_cashback
    partner_net = partner_gross * (1.0 - econ.partner_sponsor_offset)
    return base_cost + partner_net


# --- Aggregate monthly Crown contribution ---------------------------------

def crown_monthly_revenue(
    spend: float,
    partner_share: float,
    revolving_balance: float,
    account_month: int,
    econ: Economics,
    late_fee_incurred: bool = False,
) -> dict[str, float]:
    """Crown-side revenue components for one cardholder-month.

    Returns a breakdown dict with interchange, interest, late-fee, cashback
    (negative), and the net Crown contribution.
    """
    ic = crown_interchange(spend, partner_share, econ)

    gross_interest = monthly_interest(revolving_balance, account_month, econ)
    gross_late = econ.late_fee if late_fee_incurred else 0.0
    banking = crown_banking_revenue(gross_interest + gross_late, econ)

    cashback = net_cashback_cost(spend, partner_share, econ)

    net = ic + banking - cashback
    return {
        "interchange": ic,
        "interest": crown_banking_revenue(gross_interest, econ),
        "late_fee": crown_banking_revenue(gross_late, econ),
        "cashback": -cashback,
        "net": net,
    }


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
