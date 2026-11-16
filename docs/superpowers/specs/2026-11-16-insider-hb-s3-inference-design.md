# S3 — H-B Inference Design Spec

**Date:** 2026-11-16
**Status:** Design approved
**Author:** Joey Scully
**Parent:** `2026-11-16-insider-buybacks-design.md` §7–§8 (this spec covers S3)

---

## 1. Goal

Estimate **H-B** with honest inference on the S2 sale-event panel:

> Officer open-market sales (Form 4, Code `S`) at high trailing strength predict **negative
> forward excess returns** — and *opportunistic* sales (discretionary or recently-adopted 10b5-1)
> predict **more negative** forward returns than routine ones, with the effect **heterogeneous by
> governance tier**.

H-B0 (null): forward excess after officer sales is indistinguishable from zero (market-adjusted),
and the opportunistic−routine contrast is zero.

---

## 2. Settled Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| H-A secondary | **Out of scope this cycle** | Panel carries no buyback data; H-A needs fresh extraction (authorizations/repurchases + blackout windows). Gets its own spec→plan cycle if pursued. |
| Inference stack | **Full: block bootstrap + calendar-time portfolio + panel FE regression** | The three methods in parent §7; calendar-time is the overlap-robust gold standard and the price cache makes it feasible. |
| Regression engine | **statsmodels** (new venv dependency) | Battle-tested OLS with fixed effects and cluster-robust / HAC (Newey-West) covariance. |
| Strength-matched control | **Deferred** | `exc_h` is already market-adjusted; test-vs-zero + opportunistic/routine contrast + gov-tier heterogeneity is the core result. The "worse than other high-strength days" counterfactual is a documented stretch goal. |
| Output | **JSON results + committed markdown findings writeup** (no plotting dep) | Numbers + prose, fully reproducible; no matplotlib. |
| Clustering unit | **Firm** (primary); firm×year-month as a robustness variant | Dominant correlation is within-firm — one firm's boom-bust = many correlated events (the Oracle problem). Firm-block resampling prevents a single regime masquerading as independent observations. |

---

## 3. Inputs

- `data/panel.parquet` — the S2 sale-event panel (9011 rows × 22 cols). Columns used:
  `ticker, date, value, class, opportunistic, strength, exc_21, exc_63, exc_126, exc_252,
  gov_tier` (plus `size_tier, dual_class, founder_ceo` available as covariates).
- `data/cache/prices/<TICKER>.parquet` — daily close series per firm + `SPY.parquet`
  (51 files present). Used only by the calendar-time engine.

`exc_h` is firm-minus-SPY forward excess return over `h` trading days, already computed in S1.
Right-censored events (horizon past available prices) carry `NaN` and are dropped **per horizon**.

---

## 4. Architecture

Five units — each one engine or one concern, all pure except the CLI.

| File | Responsibility | Deps |
|------|---------------|------|
| `src/insider_hb/estimands.py` | Point estimates from the panel: pooled mean `exc_h` (equal- & value-weighted), opportunistic−routine contrast, per-gov-tier means. No inference. | pandas |
| `src/insider_hb/bootstrap.py` | Cluster block bootstrap: resample firm clusters (variant: firm×year-month) with replacement, apply an estimand function, return the bootstrap distribution → percentile CI + two-sided p-value. | numpy |
| `src/insider_hb/calendar_time.py` | Rebuild a daily equal-weight portfolio from the price cache: on each date hold firms with an active post-sale window `(sale_date, sale_date + h]`; compute the portfolio's daily excess vs SPY; aggregate → mean daily excess + Newey-West t. Each date counts once (overlap-robust). | pandas, numpy, statsmodels |
| `src/insider_hb/regression.py` | Panel FE OLS via statsmodels: `exc_h ~ opportunistic + strength + log_value` with firm + year-month fixed effects and cluster-robust SEs (by firm). Returns a tidy coefficient table. | statsmodels |
| `scripts/run_inference.py` | CLI: load panel (+ price cache), run all three engines across `h ∈ {21,63,126,252}`, write results. | — |

### Data flow

```
panel.parquet ─┬─> estimands ──> bootstrap ──> {estimate, CI, p} per horizon
               └─> regression ──> coef table (opportunistic/strength/log_value + FE)
price cache  ───> calendar_time ──> daily portfolio excess + NW t  (cross-check)
                                        │
                      run_inference.py ─┴─> results/s3_hb_inference.json
                                          + results/s3_hb_findings.md
```

---

## 5. Method Detail

### 5.1 Estimands (`estimands.py`)
Pure functions over a panel DataFrame, each returning a float or small dict:
- `pooled_mean(panel, h, weight="equal")` — mean of `exc_{h}` (equal) or value-weighted by `value`.
- `opportunistic_contrast(panel, h)` — `mean(exc_h | opportunistic) − mean(exc_h | routine)`.
- `gov_tier_means(panel, h)` — dict `{tier: mean(exc_h)}` for concentrated/mixed/dispersed.
All drop `NaN` in `exc_{h}` first; a group with zero non-NaN rows yields `NaN` (not an error).

### 5.2 Block bootstrap (`bootstrap.py`)
- `bootstrap_statistic(panel, stat_fn, *, cluster="ticker", n=2000, seed=0)`:
  1. Enumerate unique clusters (firm tickers, or firm×`YYYY-MM` strings for the variant).
  2. For each of `n` draws: sample clusters with replacement, concatenate their rows, apply `stat_fn`.
  3. Return `{estimate, ci_lo, ci_hi, p_value, n_boot}` where `estimate` = `stat_fn` on the full panel,
     CI = 2.5/97.5 percentiles of the bootstrap distribution, `p_value` = two-sided via the
     proportion of bootstrap replicates on the opposite side of zero (×2, clipped to [0,1]).
- Deterministic under fixed `seed` (numpy `default_rng`).

### 5.3 Calendar-time portfolio (`calendar_time.py`)
- `daily_portfolio_excess(panel, prices, spy, h)`:
  1. Build a long-format membership table: each sale `(ticker, date)` contributes active membership
     on trading days in `(date, date + h]`.
  2. For each calendar trading day, the portfolio's excess return = equal-weight mean across member
     firms of `(firm_daily_return − spy_daily_return)`.
  3. Return the daily excess-return series.
- `calendar_time_test(daily_excess)`: regress the daily series on a constant with
  **Newey-West (HAC)** SEs (`maxlags` ≈ `h`), return `{mean_daily_excess, t_stat, p_value, n_days}`.
  Each calendar date counts once → immune to the overlapping-window inflation.
- Prices are reconstructed from `data/cache/prices/*.parquet`; the function takes already-loaded
  price frames (the CLI does the disk read) so the unit is pure/testable.

### 5.4 Panel FE regression (`regression.py`)
- `fe_regression(panel, h)`:
  - Dependent: `exc_{h}`. Regressors: `opportunistic` (bool→int), `strength`, `log_value = log1p(value)`.
  - Fixed effects: firm (`ticker`) and year-month (`date.to_period("M")`) via dummy encoding
    (drop-first) — absorbs firm-level and calendar-level means.
  - Cluster-robust SEs by firm (`cov_type="cluster"`, `cov_kwds={"groups": ticker_codes}`).
  - Return a tidy table (coef, SE, t, p, ci) for the non-FE regressors only.

### 5.5 CLI (`scripts/run_inference.py`)
- Loads `data/panel.parquet`; loads price cache lazily (only for calendar-time).
- Runs, for each `h ∈ {21,63,126,252}`:
  - bootstrap of `pooled_mean` (equal + value-weighted), `opportunistic_contrast`, each `gov_tier` mean;
  - calendar-time test;
  - FE regression.
- Writes `results/s3_hb_inference.json` (all numbers, machine-readable) and
  `results/s3_hb_findings.md` (human writeup: tables per engine + a short interpretation
  paragraph that states the H-B verdict and the opportunistic/gov-tier reads).
- Prints a one-line-per-horizon summary to stderr; `--bootstrap-n`, `--seed`, `--panel`, `--out-dir`
  flags with sensible defaults.

---

## 6. Output Schema (`s3_hb_inference.json`)

```json
{
  "meta": {"n_events": 9011, "bootstrap_n": 2000, "seed": 0, "panel_sha": "..."},
  "horizons": {
    "21": {
      "pooled_mean_equal": {"estimate": -0.01, "ci_lo": ..., "ci_hi": ..., "p_value": ...},
      "pooled_mean_value": {...},
      "opportunistic_contrast": {"estimate": ..., "ci_lo": ..., "ci_hi": ..., "p_value": ...},
      "gov_tier_means": {"concentrated": {...}, "mixed": {...}, "dispersed": {...}},
      "calendar_time": {"mean_daily_excess": ..., "t_stat": ..., "p_value": ..., "n_days": ...},
      "regression": [{"term": "opportunistic", "coef": ..., "se": ..., "t": ..., "p": ..., "ci_lo": ..., "ci_hi": ...}, ...]
    },
    "63": {...}, "126": {...}, "252": {...}
  }
}
```

---

## 7. Testing

Offline-by-default preserved (`addopts = -m 'not live'`). All S3 tests are offline — they use
synthetic panels/prices with known answers, no network.

- `test_estimands.py` — constructed panel where the opportunistic−routine gap is exactly a known
  value; equal vs value weighting differ predictably; empty-group → `NaN`.
- `test_bootstrap.py` — fixed seed reproducibility; a panel with zero cross-cluster variance gives a
  degenerate (zero-width) CI; p-value of an obviously-nonzero estimate is ~0; cluster count honored.
- `test_calendar_time.py` — a hand-built 2-firm overlap where the daily portfolio excess is
  computable by hand; membership window is half-open `(date, date+h]`; NW test returns finite stats.
- `test_regression.py` — a synthetic panel where `opportunistic` has a known linear effect recovers
  that coefficient (within tolerance); FE absorb an injected firm-level mean shift.
- `test_run_inference.py` — end-to-end on a tiny synthetic panel + synthetic prices: the CLI core
  produces the full JSON schema with all horizons and engines populated.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Overlapping windows inflate significance | Firm-cluster block bootstrap + calendar-time portfolio (each date once) + HAC SEs. Three independent angles must agree. |
| Signal is regime-driven (few boom-bust names) | Report gov-tier heterogeneity and firm-clustered CIs, never a bare pooled mean. Firm-block bootstrap down-weights any single firm. |
| Thin cells at long horizons (right-censoring) | Drop `NaN` per horizon; report `n_valid` (events) per horizon and `n_boot` (replicates) per estimate; expect wider CIs at 252d. |
| statsmodels FE via dummies is memory-heavy with 50 firms × many months | 50 + ~N-month dummies is small (hundreds of columns, 9k rows) — fine for OLS. If it ever isn't, within-transform demeaning is the fallback (not needed now). |
| Strength ≠ signal (deferred control) | Documented limitation; `strength` enters the regression as a covariate so the opportunistic coefficient is already conditional on it. |

---

## 9. Out of Scope (YAGNI)

- H-A (buyback → selling) — separate cycle.
- Strength-matched non-event control group.
- Factor models beyond market adjustment (size/sector/factor).
- Charts / PNG artifacts (writeup tables suffice; revisit if needed).
- Re-fetching any data — S3 consumes S2 artifacts only.
