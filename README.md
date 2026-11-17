# Insiders selling into strength: a signal that survives its own stress tests

*A study of 9,009 officer stock sales across 48 US companies, and what happened to the share price next.*

> **Part of my analytics portfolio.** I find why the numbers aren't working and build the analysis that proves it. This is a self-directed piece: it shows how I work when no one is telling me the answer. Full writeup below; the code and 84 offline tests are in this repo.
> More work and contact: [portfolio hub](https://yllucsyeoj.github.io/portfolio) · scully.joey@gmail.com

---

## The finding in one line

When a company's own officers sell stock **after the price has run up hard**, the stock tends to **lag the market over the next 6 to 12 months**. The heaviest "sell-into-strength" cases preceded roughly a **24% shortfall versus the S&P 500 over six months**, after stripping out each company's baseline drift and the broad market.

## Why this piece is worth your time (what it demonstrates)

- **A sharp, falsifiable hypothesis**, not a fishing expedition: the signal is specifically in sales made *into strength*, not insider selling in general (most of which carries no signal).
- **Robustness decided before looking:** the pattern holds split by time period (pre-2024 vs 2024+) and by a random split of the 48 companies. It does not depend on any single stock.
- **The judgment call that matters:** a flashier earlier version of this story claimed the signal came from discretionary sales. It was a mirage, two stocks (PLTR, RBLX) wearing a trench coat, 21% of the sales producing 76% of the effect. I killed it. The finding I kept passes exactly the test the flashy one failed.

That last point is the whole job. The value is not running the regression. It is knowing which result to trust and being willing to throw away the publishable-looking one that is wrong.

## Read the full writeup

- **[`results/s4_report.md`](results/s4_report.md)** — the full study, plain-English, with figures and honest caveats.
- **[`results/s4_wls_findings.md`](results/s4_wls_findings.md)** — coefficients, confidence intervals, pre-registered robustness criteria.
- Figures: `results/figures/` · headline charts: `aapl_buyback_insider_timeline.png`, `orcl_buyback_insider_timeline.png`.

## Method (for the technically inclined)

Two-way fixed-effects panel regression (company + calendar-month) of forward excess return on a "strength" measure, standard errors clustered by company. Each company is weighted by its reliability (inverse-variance WLS) so a few volatile names cannot swing the result; equal-weight OLS is reported alongside as a conservative bound. Data is SEC Form 4 filings, Oct 2021 – Nov 2026.

Focal 6-month estimate: strength coefficient **−0.239** (*t* = −6.9, *p* < 0.001, *n* = 6,972, 47 firms).

## Reproduce it

```bash
pip install -r requirements.txt
python scripts/run_s4.py      # runs the main estimation
pytest                        # 84 offline tests
```

## Honest limits

A historical association checked for robustness, not a prediction made in advance and not a trading strategy (returns are measured versus the market, before costs). 48 large, liquid, mostly-tech names are not the whole market. Delisted companies are underrepresented. The result is clean and falsifiable, and that is all it claims to be.

---

*Built by Joey Scully. Product & Commercial Analytics consultant: I find why the numbers aren't working and build the fix. Available for remote interim and contract work.*
