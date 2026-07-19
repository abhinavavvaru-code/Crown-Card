Crown card

Data infrastructure and financial modeling backend for **Crown**, a student
credit card startup that's goal was to partnerwith a bank issuer and ran a 50-user beta at
Columbia University.

Crown's economics in one paragraph: revenue comes from **interchange fees**
(Crown 75% / bank 25%) and **banking fees** — interest and late fees — (Crown
25% / bank 75%). New accounts get a **0% intro APR for 3 months**, low
**starter credit limits ($300–$1,000)** that step up with account tenure, and
**higher cashback at partnered campus merchants** (partly sponsor-funded). A
required **financial-education course** reduces delinquency risk.

This backend generates the beta cohort, underwrites applicants, and answers the
three questions a Crown operator cares about:

1. **What is a cardholder worth to Crown?** → Monte Carlo CLV distributions.
2. **Can we reach break-even?** → 24-month deterministic cash-flow scenarios.
3. **How much can we spend to acquire one?** → CAC break-even thresholds.

---

## Architecture

```
                         configs/*.yaml  (all parameters live here)
                economics · underwriting · simulation · scenarios
                                     │
              ┌──────────────────────┼───────────────────────────┐
              │                      │                            │
   ┌──────────▼──────────┐  ┌────────▼──────────┐   ┌─────────────▼───────────┐
   │  pipeline/          │  │  simulations/     │   │  scenarios/             │
   │                     │  │                   │   │                         │
   │  onboarding.py      │  │  monte_carlo.py   │   │  cash_flow.py           │
   │   synth applicants  │  │   10k+ NumPy runs │   │   24-mo base/bull/bear  │
   │   rule underwriting │  │   CLV / VaR /CVaR │   │  cac_model.py           │
   │   starter limits    │  │   loss rates      │   │   max CAC, payback,     │
   │  seed.py            │  │   sensitivity     │   │   break-even sensitivity│
   │   50-user cohort +  │  └────────┬──────────┘   └─────────────┬───────────┘
   │   12–18mo histories │           │                            │
   │  schema.py (SQLAlchemy)         │                            │
   └──────────┬──────────┘          │                            │
              │                      │                            │
        ┌─────▼─────┐          ┌─────▼───────────────────────────▼─────┐
        │ SQLite    │          │  outputs/  CSVs + matplotlib charts   │
        │ crown.db  │          │  clv_by_segment · portfolio_summary   │
        └───────────┘          │  cash_flow_projection · cac_* · *.png │
                               └───────────────────────────────────────┘
                                     ▲
                                     │  crown_card/economics.py
                       single source of truth for revenue-share math:
                       interchange 75/25 · banking 25/75 · grace · sponsor offset
```

Every module routes money through **`crown_card/economics.py`**, so the
revenue-share splits, the 3-month grace period, and the partner sponsor offset
are defined exactly once and are identical across the simulation and the
deterministic scenarios.

---

## Install

```bash
cd crown-card
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt                       # or: pip install -e ".[dev]"
```

Python 3.11+. Dependencies: NumPy, pandas, SciPy, matplotlib, SQLAlchemy,
PyYAML, click.

---

## Usage

The CLI has four subcommands plus an end-to-end runner:

```bash
python -m crown_card seed        # generate the 50-user beta cohort + histories -> SQLite
python -m crown_card onboard     # underwrite a fresh applicant cohort (no transactions)
python -m crown_card simulate    # Monte Carlo CLV / loss / VaR + charts
python -m crown_card scenarios   # 24-month cash flow + CAC break-even + charts
python -m crown_card all         # seed -> simulate -> scenarios, end to end
```

Useful flags:

```bash
python -m crown_card seed --users 50 --seed 123
python -m crown_card simulate --runs 20000            # more Monte Carlo runs
python -m crown_card simulate --no-charts --no-sensitivity
python -m crown_card scenarios --no-charts
```

All artifacts are written to `outputs/` (CSVs, PNG charts, and `crown.db`).

---

## What each stage produces

### 1. Credit onboarding (`pipeline/`)

- **`onboarding.py`** — synthetic student applicant generator (school year,
  income/allowance, credit score or thin-file flag, DTI), normalization,
  rule-based underwriting tuned for thin-file students, and starter-limit
  assignment in the $300–$1,000 band with a tenure step-up schedule.
- **`schema.py`** — SQLAlchemy models: `Applicant`, `Account`,
  `CreditLimitStepUp`, `Transaction` (merchant category + partner flag),
  `Statement`, `CourseCompletion`.
- **`seed.py`** — activates a 50-account beta cohort and attaches 12–18 months
  of transaction histories weighted toward campus-area merchants, rolled up into
  monthly statements (interest, late fees, grace period).

### 2. Monte Carlo simulation (`simulations/monte_carlo.py`)

10,000+ vectorized NumPy runs. Stochastic inputs: monthly spend (lognormal,
student-scaled), partner-merchant share (beta), revolve rate (beta), default
probability (higher for thin-file, reduced by course completion), and
attrition/graduation churn (geometric). Models the 3-month 0% APR grace period
and Crown's revenue-share splits.

Outputs (`outputs/`):
`clv_by_segment.csv`, `portfolio_summary.csv` (mean/median CLV, 95% CI, default
rate, annualized charge-off rate, VaR/CVaR at 95%), `sensitivity.csv`, and the
charts `clv_histogram.png`, `loss_distribution.png`, `sensitivity_tornado.png`.

### 3. Deterministic scenarios (`scenarios/`)

- **`cash_flow.py`** — 24-month base/bull/bear projections (YAML-configured):
  cardholder growth from the 50-user beta toward the 1,500 break-even target,
  adoption ceiling as a share of the ~30K student body, sponsor-offset rewards,
  servicing, and fixed opex. Charts: `cardholder_growth.png`,
  `cumulative_cash_flow.png`.
- **`cac_model.py`** — max sustainable CAC per segment (LTV:CAC 2x/3x), payback
  curves, and the sensitivity of the break-even cardholder count to interchange
  and revolve rate. Table: `cac_scenario_comparison.csv`.

---

## Configuration

All parameters live in `configs/` and are the only thing you should need to edit:

| File | Controls |
|------|----------|
| `economics.yaml`    | interchange/banking splits, APR + grace, cashback, credit-limit bands & step-ups |
| `underwriting.yaml` | applicant generator, decision rules, starter-limit scoring |
| `simulation.yaml`   | Monte Carlo runs, horizon, spend/revolve/default/churn distributions |
| `scenarios.yaml`    | student body, growth, opex, CAC, base/bull/bear knobs |

---

## Modeling notes (honest defaults)

- **Interchange ~1.6%** at ordinary merchants, **~2.4%** at partnered campus
  merchants; Crown keeps 75%. Partner spend is deliberately Crown's profit
  driver — richer interchange more than covers the higher (60% sponsor-offset)
  partner cashback, while ordinary spend is near break-even after 1% cashback.
- **APR 26.5%** post-grace (24–29% band), **0%** for the first 3 statement
  cycles. Crown keeps only 25% of interest, so interest is a thin secondary
  stream; interchange dominates (see the sensitivity tornado).
- **Charge-offs** land around **5% annualized** for the blended book (elevated
  for thin-file, ~40% lower with course completion) — in the 5–8% student band.
- **Break-even** sits near **~1,780 cardholders** at base parameters (range
  ~1,500–2,100 across interchange/revolve sensitivity), consistent with Crown's
  stated 1,500 target. At Columbia-only scale the pilot proves the model and
  runs a modest cash burn; sustained profitability implies multi-campus
  expansion — a genuine output of the model, not a tuned-in result.

Numbers above are illustrative defaults produced by the seeded RNG; change the
YAML and re-run.

---

## Tests

```bash
python -m pytest -q
```

Covers underwriting rules and limit assignment, the revenue-share / grace-period
math, and seeded-RNG simulation statistics (determinism, thin-file vs. scored
default rates, course-completion effect, VaR/CVaR ordering, segment coverage).

---

## Project layout

```
crown-card/
├── crown_card/
│   ├── __main__.py            # CLI entrypoint (click)
│   ├── config.py              # YAML loading + SQLite URL
│   ├── economics.py           # shared revenue-share math (single source of truth)
│   ├── pipeline/
│   │   ├── schema.py          # SQLAlchemy models
│   │   ├── onboarding.py      # generation + underwriting + limits
│   │   └── seed.py            # 50-user cohort + transaction histories
│   ├── simulations/
│   │   └── monte_carlo.py     # portfolio Monte Carlo + charts
│   └── scenarios/
│       ├── cash_flow.py       # 24-month base/bull/bear
│       └── cac_model.py       # CAC break-even + payback
├── configs/                   # all tunable parameters (YAML)
├── tests/                     # pytest suite
├── outputs/                   # generated CSVs, charts, crown.db
├── requirements.txt
└── pyproject.toml
```
