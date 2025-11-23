"""Crown CLI: python -m crown_card [seed | onboard | simulate | scenarios | all]."""
from __future__ import annotations

import click
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


@click.group()
def cli():
    """Crown data infrastructure & financial modeling backend."""


@cli.command()
@click.option("--users", default=50, show_default=True, help="Beta cohort size to seed.")
@click.option("--seed", default=None, type=int, help="RNG seed override.")
def seed(users, seed):
    """Seed the beta cohort with 12-18 months of transaction history."""
    from .pipeline.seed import seed_cohort

    summary = seed_cohort(n_users=users, seed=seed)
    click.echo("Seeded Crown beta cohort:")
    for k, v in summary.items():
        click.echo(f"  {k:>18}: {v}")


@cli.command()
@click.option("--seed", default=None, type=int, help="RNG seed override.")
def onboard(seed):
    """Generate + underwrite a synthetic applicant cohort (no transactions)."""
    import numpy as np

    from .pipeline.onboarding import run_onboarding

    rng = np.random.default_rng(seed) if seed is not None else None
    summary = run_onboarding(rng=rng)
    click.echo("Onboarding complete:")
    for k, v in summary.items():
        click.echo(f"  {k:>14}: {v}")


@cli.command()
@click.option("--runs", default=None, type=int, help="Number of Monte Carlo runs.")
@click.option("--no-charts", is_flag=True, help="Skip PNG chart generation.")
@click.option("--no-sensitivity", is_flag=True, help="Skip the sensitivity tornado.")
def simulate(runs, no_charts, no_sensitivity):
    """Run the Monte Carlo portfolio simulation."""
    from .simulations.monte_carlo import run_simulation

    out = run_simulation(
        n_runs=runs,
        make_charts=not no_charts,
        with_sensitivity=not no_sensitivity,
    )
    click.echo("Portfolio summary:")
    for k, v in out["summary"].items():
        val = f"{v:,.2f}" if isinstance(v, float) else v
        click.echo(f"  {k:>20}: {val}")
    click.echo("\nCLV by segment:")
    click.echo(out["segments"].to_string(index=False))
    click.echo(f"\nArtifacts written to {out['out_dir']}")
    for c in out["charts"]:
        click.echo(f"  chart: {c}")


@cli.command()
@click.option("--no-charts", is_flag=True, help="Skip PNG chart generation.")
def scenarios(no_charts):
    """Run deterministic cash-flow + CAC break-even scenarios."""
    from .scenarios.cash_flow import run_cash_flow
    from .scenarios.cac_model import run_cac_analysis

    cf = run_cash_flow(make_charts=not no_charts)
    click.echo("Cash-flow milestones:")
    click.echo(cf["milestones"].to_string(index=False))

    cac = run_cac_analysis()
    click.echo("\nCAC scenario comparison:")
    click.echo(cac["scenario_comparison"].to_string(index=False))
    click.echo("\nMax sustainable CAC by segment (LTV:CAC 3x):")
    seg = cac["cac_by_segment"]
    click.echo(seg[["segment_kind", "segment", "mean_clv", "max_cac_3x"]].to_string(index=False))
    click.echo(f"\nArtifacts written to {cf['out_dir']}")


@cli.command()
@click.pass_context
def all(ctx):
    """Run the full pipeline end-to-end: seed -> simulate -> scenarios."""
    ctx.invoke(seed)
    click.echo("\n" + "=" * 60 + "\n")
    ctx.invoke(simulate)
    click.echo("\n" + "=" * 60 + "\n")
    ctx.invoke(scenarios)


if __name__ == "__main__":
    cli()
