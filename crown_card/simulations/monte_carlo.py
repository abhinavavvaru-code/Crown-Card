"""Monte Carlo portfolio simulation for Crown.

Simulates tens of thousands of synthetic cardholders month-by-month over a CLV
horizon, routing every dollar through the shared `economics` module so the
revenue-share splits and the 3-month 0% APR grace period are applied exactly as
they are everywhere else.

Stochastic inputs per cardholder:
  * monthly spend            -- lognormal, student-scaled
  * partner-merchant share   -- beta (higher interchange capture)
  * revolve rate             -- beta (thin-file revolve more)
  * default (charge-off)     -- annual hazard, reduced by course completion
  * attrition / graduation   -- geometric monthly churn by school year

Outputs (written under outputs/):
  * clv_by_segment.csv       -- CLV distribution summary per segment
  * portfolio_summary.csv    -- loss rate, VaR/CVaR, CLV confidence interval
  * clv_histogram.png, loss_distribution.png, sensitivity_tornado.png
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, economics_config, simulation_config, underwriting_config
from ..economics import Economics, blended_interchange_rate


@dataclass
class SimResult:
    clv: np.ndarray             # per-cardholder Crown CLV
    defaulted: np.ndarray       # bool mask
    total_loss: np.ndarray      # per-cardholder charge-off loss (USD)
    total_spend: np.ndarray     # per-cardholder lifetime spend
    balance_exposure: np.ndarray  # per-cardholder sum of monthly revolving balances
    months_active: np.ndarray   # tenure before churn/default/horizon
    school_year: np.ndarray     # object array
    thin_file: np.ndarray       # bool
    course: np.ndarray          # bool
    econ: Economics
    cfg: dict[str, Any]


def _annual_to_monthly_hazard(annual: float) -> float:
    return 1.0 - (1.0 - annual) ** (1.0 / 12.0)


def simulate(
    n_runs: int | None = None,
    sim_cfg: dict[str, Any] | None = None,
    econ_cfg: dict[str, Any] | None = None,
    uw_cfg: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """Run the vectorized Monte Carlo simulation. Returns raw per-cardholder arrays."""
    sim_cfg = sim_cfg or simulation_config()
    econ_cfg = econ_cfg or economics_config()
    uw_cfg = uw_cfg or underwriting_config()
    econ = Economics.from_config(econ_cfg)

    n = n_runs or sim_cfg["n_runs"]
    horizon = sim_cfg["horizon_months"]
    rng = rng or np.random.default_rng(sim_cfg["seed"])

    # --- Draw static per-cardholder attributes -----------------------------
    years = uw_cfg["generator"]["school_years"]
    yweights = np.asarray(uw_cfg["generator"]["school_year_weights"], dtype=float)
    yweights = yweights / yweights.sum()
    school_year = rng.choice(years, size=n, p=yweights)

    thin_file = rng.random(n) < uw_cfg["generator"]["thin_file_prob"]
    course = rng.random(n) < 0.55  # course-completion prevalence in the cohort

    ps = sim_cfg["partner_share"]
    partner_share = rng.beta(ps["alpha"], ps["beta"], size=n)

    rr = sim_cfg["revolve_rate"]
    revolve_rate = np.where(
        thin_file,
        rng.beta(rr["thin_file"]["alpha"], rr["thin_file"]["beta"], size=n),
        rng.beta(rr["scored"]["alpha"], rr["scored"]["beta"], size=n),
    )

    # Static per-cardholder monthly-spend median (a lognormal draw), later
    # perturbed by month-to-month noise.
    msp = sim_cfg["monthly_spend"]
    spend_median = np.clip(
        rng.lognormal(msp["lognormal"]["mu"], msp["lognormal"]["sigma"], size=n),
        msp["min"],
        msp["max"],
    )

    # Hazards
    dcfg = sim_cfg["default"]
    base_annual_default = np.where(thin_file, dcfg["annual_prob"]["thin_file"], dcfg["annual_prob"]["scored"])
    course_mult = np.where(course, dcfg["course_completion_multiplier"], 1.0)
    monthly_default_hazard = _annual_to_monthly_hazard(base_annual_default * course_mult)
    lgd = dcfg["loss_given_default"]

    churn_map = sim_cfg["attrition"]["monthly_churn"]
    monthly_churn = np.array([churn_map[y] for y in school_year], dtype=float)

    # --- State -------------------------------------------------------------
    alive = np.ones(n, dtype=bool)
    defaulted = np.zeros(n, dtype=bool)
    clv = np.zeros(n)
    total_loss = np.zeros(n)
    total_spend = np.zeros(n)
    balance_exposure = np.zeros(n)
    months_active = np.zeros(n, dtype=int)

    blended_rate = np.array([blended_interchange_rate(p, econ) for p in partner_share])

    for m in range(1, horizon + 1):
        if not alive.any():
            break

        # Monthly spend = static median * lognormal noise.
        noise = rng.lognormal(0.0, 0.25, size=n)
        spend = np.clip(spend_median * noise, msp["min"], msp["max"]) * alive
        total_spend += spend

        # Interchange (Crown share).
        interchange = spend * blended_rate * econ.interchange_crown_share

        # Interest — zero during the grace window.
        in_grace = m <= econ.intro_apr_months
        apr = econ.intro_apr if in_grace else econ.apr
        revolving_balance = np.minimum(spend * revolve_rate, spend)
        balance_exposure += revolving_balance
        gross_interest = revolving_balance * apr / 12.0

        # Late fees (bernoulli per alive cardholder).
        late_hit = (rng.random(n) < econ.late_fee_prob_monthly) & alive
        gross_late = late_hit * econ.late_fee

        banking = (gross_interest + gross_late) * econ.banking_crown_share

        # Cashback cost (Crown net of sponsor offset).
        non_partner = spend * (1.0 - partner_share)
        partner_spend = spend * partner_share
        cashback = (
            non_partner * econ.base_cashback
            + partner_spend * econ.partner_cashback * (1.0 - econ.partner_sponsor_offset)
        )

        net_month = (interchange + banking - cashback) * alive
        clv += net_month
        months_active += alive.astype(int)

        # Default events (charge-off): loss = LGD * outstanding revolving balance.
        default_event = (rng.random(n) < monthly_default_hazard) & alive
        loss = lgd * revolving_balance * default_event
        total_loss += loss
        clv -= loss
        defaulted |= default_event
        alive &= ~default_event

        # Attrition / graduation churn (independent of default).
        churn_event = (rng.random(n) < monthly_churn) & alive
        alive &= ~churn_event

    return SimResult(
        clv=clv,
        defaulted=defaulted,
        total_loss=total_loss,
        total_spend=total_spend,
        balance_exposure=balance_exposure,
        months_active=months_active,
        school_year=school_year,
        thin_file=thin_file,
        course=course,
        econ=econ,
        cfg=sim_cfg,
    )


# --- Aggregation -----------------------------------------------------------

def _describe(clv: np.ndarray) -> dict[str, float]:
    return {
        "n": int(clv.size),
        "mean_clv": float(clv.mean()) if clv.size else 0.0,
        "median_clv": float(np.median(clv)) if clv.size else 0.0,
        "std_clv": float(clv.std()) if clv.size else 0.0,
        "p05_clv": float(np.percentile(clv, 5)) if clv.size else 0.0,
        "p95_clv": float(np.percentile(clv, 95)) if clv.size else 0.0,
    }


def segment_table(res: SimResult) -> pd.DataFrame:
    """CLV distribution summary broken out by segment."""
    df = pd.DataFrame(
        {
            "clv": res.clv,
            "school_year": res.school_year,
            "file_type": np.where(res.thin_file, "thin_file", "scored"),
            "course": np.where(res.course, "completed", "not_completed"),
        }
    )
    rows = []
    rows.append({"segment_kind": "overall", "segment": "all", **_describe(df["clv"].values)})
    for col, kind in [("school_year", "school_year"), ("file_type", "file_type"), ("course", "course")]:
        for val, grp in df.groupby(col):
            rows.append({"segment_kind": kind, "segment": val, **_describe(grp["clv"].values)})
    return pd.DataFrame(rows)


def portfolio_summary(res: SimResult, confidence: float | None = None) -> dict[str, float]:
    """Portfolio-level loss rate, VaR/CVaR, and CLV confidence interval."""
    confidence = confidence or res.cfg.get("var_confidence", 0.95)
    clv = res.clv

    # VaR / CVaR on the loss (negative-tail) side of the CLV distribution.
    alpha = 1.0 - confidence
    var_threshold = np.percentile(clv, alpha * 100)   # e.g. 5th percentile
    tail = clv[clv <= var_threshold]
    cvar = float(tail.mean()) if tail.size else float(var_threshold)

    # Charge-off rate on receivables: losses per balance-month, annualized.
    balance_months = res.balance_exposure.sum()
    annual_charge_off_rate = float(res.total_loss.sum() / balance_months * 12.0) if balance_months else 0.0
    # Loss as a share of purchase volume (secondary view).
    total_spend = res.total_spend.sum()
    loss_on_volume = float(res.total_loss.sum() / total_spend) if total_spend else 0.0

    # 95% CI on mean CLV (normal approx).
    mean = float(clv.mean())
    se = float(clv.std(ddof=1) / np.sqrt(clv.size)) if clv.size > 1 else 0.0
    ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se

    return {
        "n_cardholders": int(clv.size),
        "mean_clv": mean,
        "median_clv": float(np.median(clv)),
        "clv_ci95_low": ci_lo,
        "clv_ci95_high": ci_hi,
        "default_rate": float(res.defaulted.mean()),
        "annual_charge_off_rate": annual_charge_off_rate,
        "loss_on_volume": loss_on_volume,
        f"var_{int(confidence*100)}": float(var_threshold),
        f"cvar_{int(confidence*100)}": cvar,
        "mean_months_active": float(res.months_active.mean()),
    }


# --- Sensitivity (tornado) -------------------------------------------------

def sensitivity_analysis(base_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """One-at-a-time +/-20% sensitivity of mean CLV to key drivers."""
    base_sim = base_cfg or simulation_config()
    econ_base = economics_config()

    def mean_clv(sim_cfg, econ_cfg):
        res = simulate(n_runs=3000, sim_cfg=sim_cfg, econ_cfg=econ_cfg)
        return float(res.clv.mean())

    baseline = mean_clv(base_sim, econ_base)

    import copy

    drivers = []

    def flex(label, mutate_sim=None, mutate_econ=None):
        lo_sim, hi_sim = copy.deepcopy(base_sim), copy.deepcopy(base_sim)
        lo_econ, hi_econ = copy.deepcopy(econ_base), copy.deepcopy(econ_base)
        if mutate_sim:
            mutate_sim(lo_sim, 0.8)
            mutate_sim(hi_sim, 1.2)
        if mutate_econ:
            mutate_econ(lo_econ, 0.8)
            mutate_econ(hi_econ, 1.2)
        low = mean_clv(lo_sim, lo_econ)
        high = mean_clv(hi_sim, hi_econ)
        drivers.append({"driver": label, "low": low, "high": high, "baseline": baseline})

    flex("interchange_rate", mutate_econ=lambda e, f: e["interchange"].update(
        base_rate=e["interchange"]["base_rate"] * f,
        partner_rate=e["interchange"]["partner_rate"] * f,
    ))
    flex("apr", mutate_econ=lambda e, f: e["banking_fees"].update(apr=e["banking_fees"]["apr"] * f))
    flex("monthly_spend", mutate_sim=lambda s, f: s["monthly_spend"]["lognormal"].update(
        mu=s["monthly_spend"]["lognormal"]["mu"] + np.log(f)
    ))
    flex("default_rate", mutate_sim=lambda s, f: s["default"]["annual_prob"].update(
        scored=s["default"]["annual_prob"]["scored"] * f,
        thin_file=s["default"]["annual_prob"]["thin_file"] * f,
    ))
    flex("partner_cashback", mutate_econ=lambda e, f: e["rewards"].update(
        partner_cashback=e["rewards"]["partner_cashback"] * f
    ))

    df = pd.DataFrame(drivers)
    df["swing"] = (df["high"] - df["low"]).abs()
    return df.sort_values("swing", ascending=False).reset_index(drop=True)


# --- Charts ----------------------------------------------------------------

def _save_charts(res: SimResult, tornado: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # CLV histogram.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(res.clv, bins=60, color="#4C6EF5", edgecolor="white", alpha=0.85)
    ax.axvline(res.clv.mean(), color="#E8590C", linestyle="--", label=f"mean = ${res.clv.mean():,.0f}")
    ax.set_title("Crown CLV distribution (per cardholder)")
    ax.set_xlabel("Crown-share CLV (USD)")
    ax.set_ylabel("cardholders")
    ax.legend()
    p = out_dir / "clv_histogram.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # Loss distribution (defaulted cardholders only).
    losses = res.total_loss[res.total_loss > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    if losses.size:
        ax.hist(losses, bins=40, color="#F03E3E", edgecolor="white", alpha=0.85)
    ax.set_title("Charge-off loss distribution (defaulted cardholders)")
    ax.set_xlabel("loss (USD)")
    ax.set_ylabel("cardholders")
    p = out_dir / "loss_distribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # Sensitivity tornado.
    fig, ax = plt.subplots(figsize=(8, 5))
    base = tornado["baseline"].iloc[0]
    y = np.arange(len(tornado))
    for i, row in tornado.iterrows():
        ax.barh(i, row["high"] - base, left=base, color="#37B24D", alpha=0.8)
        ax.barh(i, row["low"] - base, left=base, color="#F59F00", alpha=0.8)
    ax.axvline(base, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tornado["driver"])
    ax.set_title("Mean CLV sensitivity (+/-20%)")
    ax.set_xlabel("mean CLV (USD)")
    p = out_dir / "sensitivity_tornado.png"
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    return paths


def run_simulation(
    n_runs: int | None = None,
    out_dir: Path | None = None,
    make_charts: bool = True,
    with_sensitivity: bool = True,
) -> dict[str, Any]:
    """Full simulation run: simulate, aggregate, and write CSVs + charts."""
    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    res = simulate(n_runs=n_runs)

    seg = segment_table(res)
    seg.to_csv(out_dir / "clv_by_segment.csv", index=False)

    summary = portfolio_summary(res)
    pd.DataFrame([summary]).to_csv(out_dir / "portfolio_summary.csv", index=False)

    tornado = None
    if with_sensitivity:
        tornado = sensitivity_analysis()
        tornado.to_csv(out_dir / "sensitivity.csv", index=False)

    chart_paths = []
    if make_charts and tornado is not None:
        chart_paths = _save_charts(res, tornado, out_dir)

    return {
        "summary": summary,
        "segments": seg,
        "sensitivity": tornado,
        "charts": [str(p) for p in chart_paths],
        "out_dir": str(out_dir),
    }
