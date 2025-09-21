"""Deterministic 24-month operational cash-flow projections.

Grows the cardholder base from the 50-user beta toward Crown's 1,500-cardholder
break-even target under base / bull / bear scenarios (YAML-configured), using the
shared economics module for per-cardholder Crown revenue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, economics_config, scenarios_config
from ..economics import Economics, crown_monthly_revenue


def per_cardholder_monthly_revenue(
    scenario: dict[str, Any],
    global_cfg: dict[str, Any],
    econ: Economics,
    account_month: int,
) -> float:
    """Steady-state Crown-share net revenue for one active cardholder-month.

    Uses the scenario's sponsor offset (which shifts cashback cost) via a
    per-scenario Economics override.
    """
    pc = global_cfg["per_cardholder"]
    spend = pc["avg_monthly_spend"]
    partner_share = pc["partner_spend_share"]
    # Interest accrues on the accumulated carried balance, not a single month's spend.
    revolving_balance = pc.get("avg_revolving_balance", spend * pc["revolve_rate"])

    econ_scn = Economics(**{**econ.__dict__, "partner_sponsor_offset": scenario["partner_sponsor_offset"]})
    breakdown = crown_monthly_revenue(
        spend=spend,
        partner_share=partner_share,
        revolving_balance=revolving_balance,
        account_month=account_month,
        econ=econ_scn,
        late_fee_incurred=False,
    )
    return breakdown["net"]


def project_scenario(name: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a month-by-month cash-flow DataFrame for one scenario."""
    cfg = cfg or scenarios_config()
    econ = Economics.from_config(economics_config())
    scn = cfg["scenarios"][name]

    horizon = cfg["horizon_months"]
    start = cfg["starting_cardholders"]
    ceiling = int(scn["adoption_ceiling"] * cfg["student_body"])
    growth = scn["monthly_growth_rate"]
    monthly_default_hazard = 1.0 - (1.0 - scn["default_annual"]) ** (1.0 / 12.0)
    cac = scn["cac"]

    fixed_opex = cfg["fixed_opex_monthly"]
    servicing = cfg["servicing_cost_per_cardholder"]
    pc = cfg["per_cardholder"]

    rows = []
    cardholders = float(start)
    cumulative = 0.0
    for m in range(1, horizon + 1):
        prev = cardholders
        # Logistic-style bounded growth toward the adoption ceiling.
        cardholders = min(ceiling, prev * (1.0 + growth))
        new = max(0.0, cardholders - prev)
        active = cardholders

        # Post-grace steady-state per-cardholder revenue (account_month > grace).
        rev_per = per_cardholder_monthly_revenue(scn, cfg, econ, account_month=m + 3)
        gross_revenue = active * rev_per

        # Expected charge-off losses on carried balances.
        revolving_balance = pc.get("avg_revolving_balance", pc["avg_monthly_spend"] * pc["revolve_rate"])
        loss = active * monthly_default_hazard * econ_lgd() * revolving_balance

        servicing_cost = active * servicing
        acquisition_cost = new * cac
        total_cost = servicing_cost + acquisition_cost + fixed_opex + loss

        net = gross_revenue - total_cost
        cumulative += net

        rows.append(
            {
                "scenario": name,
                "month": m,
                "cardholders": round(active, 1),
                "new_cardholders": round(new, 1),
                "revenue": round(gross_revenue, 2),
                "revenue_per_cardholder": round(rev_per, 2),
                "charge_off_loss": round(loss, 2),
                "servicing_cost": round(servicing_cost, 2),
                "acquisition_cost": round(acquisition_cost, 2),
                "fixed_opex": round(fixed_opex, 2),
                "net_cash_flow": round(net, 2),
                "cumulative_cash_flow": round(cumulative, 2),
                "breakeven_target": cfg["breakeven_target"],
            }
        )
    return pd.DataFrame(rows)


def econ_lgd() -> float:
    """Loss-given-default pulled from the simulation config (shared assumption)."""
    from ..config import simulation_config

    return simulation_config()["default"]["loss_given_default"]


def run_cash_flow(out_dir: Path | None = None, make_charts: bool = True) -> dict[str, Any]:
    """Project all scenarios, write a combined CSV, and a comparison chart."""
    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = scenarios_config()

    frames = [project_scenario(name, cfg) for name in cfg["scenarios"]]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "cash_flow_projection.csv", index=False)

    # Milestone table: month cardholders cross break-even / cumulative turns positive.
    milestones = []
    target = cfg["breakeven_target"]
    for name, grp in combined.groupby("scenario"):
        crossed = grp[grp["cardholders"] >= target]["month"]
        cf_pos = grp[grp["cumulative_cash_flow"] >= 0]["month"]
        milestones.append(
            {
                "scenario": name,
                "final_cardholders": grp["cardholders"].iloc[-1],
                "month_hit_breakeven_count": int(crossed.iloc[0]) if not crossed.empty else None,
                "month_cashflow_positive": int(cf_pos.iloc[0]) if not cf_pos.empty else None,
                "final_cumulative_cash_flow": grp["cumulative_cash_flow"].iloc[-1],
            }
        )
    milestone_df = pd.DataFrame(milestones)
    milestone_df.to_csv(out_dir / "cash_flow_milestones.csv", index=False)

    charts = []
    if make_charts:
        charts = _cash_flow_charts(combined, out_dir)

    return {
        "projection": combined,
        "milestones": milestone_df,
        "charts": [str(c) for c in charts],
        "out_dir": str(out_dir),
    }


def _cash_flow_charts(combined: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#4C6EF5", "bull": "#37B24D", "bear": "#F03E3E"}
    paths = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, grp in combined.groupby("scenario"):
        ax.plot(grp["month"], grp["cardholders"], label=name, color=colors.get(name))
    ax.axhline(combined["breakeven_target"].iloc[0], color="black", linestyle="--", label="break-even target")
    ax.set_title("Cardholder growth by scenario")
    ax.set_xlabel("month"); ax.set_ylabel("cardholders"); ax.legend()
    p = out_dir / "cardholder_growth.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, grp in combined.groupby("scenario"):
        ax.plot(grp["month"], grp["cumulative_cash_flow"], label=name, color=colors.get(name))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Cumulative cash flow by scenario")
    ax.set_xlabel("month"); ax.set_ylabel("cumulative cash flow (USD)"); ax.legend()
    p = out_dir / "cumulative_cash_flow.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    return paths
