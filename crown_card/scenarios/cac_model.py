"""CAC break-even analysis.

Derives the maximum sustainable customer-acquisition cost per segment from the
Crown-share CLV, the payback-period curve, and the sensitivity of the break-even
cardholder count to the interchange rate and the revolve rate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, economics_config, scenarios_config
from ..economics import Economics, crown_monthly_revenue
from ..simulations.monte_carlo import simulate, segment_table


def crown_clv_by_segment(n_runs: int = 8000) -> pd.DataFrame:
    """Mean Crown-share CLV per segment from the Monte Carlo engine."""
    res = simulate(n_runs=n_runs)
    seg = segment_table(res)
    return seg[["segment_kind", "segment", "n", "mean_clv", "median_clv"]]


def max_sustainable_cac(clv: float, target_ltv_cac: float = 3.0) -> float:
    """Max CAC that still clears the target LTV:CAC ratio (default 3:1)."""
    return clv / target_ltv_cac


def payback_curve(
    monthly_contribution: float,
    cac: float,
    max_months: int = 36,
) -> pd.DataFrame:
    """Cumulative Crown contribution vs. CAC over time; flags the payback month."""
    months = np.arange(1, max_months + 1)
    cumulative = monthly_contribution * months
    paid_back = cumulative >= cac
    payback_month = int(months[paid_back][0]) if paid_back.any() else None
    return pd.DataFrame(
        {
            "month": months,
            "cumulative_contribution": np.round(cumulative, 2),
            "cac": cac,
            "paid_back": paid_back,
        }
    ), payback_month


def steady_state_monthly_contribution(
    econ: Economics,
    cfg: dict[str, Any],
    interchange_scale: float = 1.0,
    revolve_scale: float = 1.0,
) -> float:
    """Per-cardholder monthly Crown contribution under scaled drivers."""
    pc = cfg["per_cardholder"]
    econ2 = Economics(
        **{
            **econ.__dict__,
            "interchange_base_rate": econ.interchange_base_rate * interchange_scale,
            "interchange_partner_rate": econ.interchange_partner_rate * interchange_scale,
        }
    )
    spend = pc["avg_monthly_spend"]
    base_balance = pc.get("avg_revolving_balance", spend * pc["revolve_rate"])
    revolving_balance = min(base_balance * revolve_scale, spend * 3)
    breakdown = crown_monthly_revenue(
        spend=spend,
        partner_share=pc["partner_spend_share"],
        revolving_balance=revolving_balance,
        account_month=12,  # post-grace steady state
        econ=econ2,
    )
    return breakdown["net"]


def breakeven_cardholders(
    monthly_contribution: float,
    fixed_opex_monthly: float,
    servicing_per_cardholder: float,
) -> float:
    """Cardholders needed for monthly contribution to cover fixed opex."""
    margin = monthly_contribution - servicing_per_cardholder
    if margin <= 0:
        return float("inf")
    return fixed_opex_monthly / margin


def breakeven_sensitivity(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Break-even cardholder count vs. interchange-rate and revolve-rate scaling."""
    cfg = cfg or scenarios_config()
    econ = Economics.from_config(economics_config())
    fixed = cfg["fixed_opex_monthly"]
    servicing = cfg["servicing_cost_per_cardholder"]

    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    rows = []
    for i_scale in scales:
        for r_scale in scales:
            contribution = steady_state_monthly_contribution(
                econ, cfg, interchange_scale=i_scale, revolve_scale=r_scale
            )
            be = breakeven_cardholders(contribution, fixed, servicing)
            rows.append(
                {
                    "interchange_scale": i_scale,
                    "revolve_scale": r_scale,
                    "monthly_contribution": round(contribution, 2),
                    "breakeven_cardholders": round(be, 0) if np.isfinite(be) else None,
                }
            )
    return pd.DataFrame(rows)


def run_cac_analysis(out_dir: Path | None = None, n_runs: int = 8000) -> dict[str, Any]:
    """Build the scenario comparison table, payback curves, and sensitivity grid."""
    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = scenarios_config()
    econ = Economics.from_config(economics_config())

    # CLV per segment -> max sustainable CAC.
    seg = crown_clv_by_segment(n_runs=n_runs)
    seg = seg.copy()
    seg["max_cac_3x"] = seg["mean_clv"].apply(lambda c: round(max_sustainable_cac(c, 3.0), 2))
    seg["max_cac_2x"] = seg["mean_clv"].apply(lambda c: round(max_sustainable_cac(c, 2.0), 2))
    seg.to_csv(out_dir / "cac_by_segment.csv", index=False)

    # Scenario comparison: each scenario's CAC vs. payback month and LTV:CAC.
    monthly_contribution = steady_state_monthly_contribution(econ, cfg)
    overall_clv = float(seg.loc[seg["segment"] == "all", "mean_clv"].iloc[0])

    comparison = []
    payback_frames = []
    for name, scn in cfg["scenarios"].items():
        cac = scn["cac"]
        curve, payback_month = payback_curve(monthly_contribution, cac)
        curve = curve.assign(scenario=name)
        payback_frames.append(curve)
        comparison.append(
            {
                "scenario": name,
                "cac": cac,
                "monthly_contribution": round(monthly_contribution, 2),
                "overall_clv": round(overall_clv, 2),
                "ltv_cac_ratio": round(overall_clv / cac, 2) if cac else None,
                "payback_month": payback_month,
                "max_sustainable_cac_3x": round(max_sustainable_cac(overall_clv, 3.0), 2),
            }
        )
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(out_dir / "cac_scenario_comparison.csv", index=False)
    pd.concat(payback_frames, ignore_index=True).to_csv(out_dir / "cac_payback_curves.csv", index=False)

    sensitivity = breakeven_sensitivity(cfg)
    sensitivity.to_csv(out_dir / "breakeven_sensitivity.csv", index=False)

    return {
        "cac_by_segment": seg,
        "scenario_comparison": comparison_df,
        "breakeven_sensitivity": sensitivity,
        "out_dir": str(out_dir),
    }
