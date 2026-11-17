# S4: Variance-Robust (WLS) Strength Inference — Design

**Stage:** S4 (follows S3 inference; builds on the banked 50-firm sale-event panel)
**Date:** 2026-11-16
**Hypothesis under test:** H-B `strength` channel — officer open-market sales (Form 4 Code `S`) at high trailing-price strength predict *negative* forward excess returns (firm − SPY), conditional on firm + calendar-month fixed effects.

---

## 1. Motivation

S3's pooled panel FE-OLS found `strength` only marginally negative (@126d coef −0.172, p=.073) and attributed the significant signal to an `opportunistic`/`discretionary` flag. Post-S3 exploration (sidecar, hypothesis-generation) showed:

- The `discretionary` → underperformance effect is **not generalizable** — ~75% of it is two firms (PLTR, RBLX); it dies on a firm-holdout and when those firms are removed (@126d p=.43).
- `strength` is the **robust survivor** — negative in every split, but *masked* in equal-weight OLS by a handful of hyper-variance firms (INTC is actually #1 by forward-return volatility, not PLTR/RBLX — so hand-dropping names would be gerrymandering).
- A **variance-robust estimator** recovers `strength` cleanly on all 50 firms with zero exclusions:
  - Inverse-variance WLS: @63d −0.184 (p<.001), @126d −0.239 (p<.001).
  - Per-firm z-scored DV: @126d −1.04σ (p=.001).
  - Replicates under a disjoint time split (WLS @126d: pre-2024 −0.283 p=.010; 2024+ −0.265 p<.001).

S4 productionizes this: a locked, leakage-free WLS estimator, a pre-registered holdout protocol, and a committed writeup.

## 2. Honest scope / what this is and is not

**Burned-in caveat — the panel is already spent.** We looked at the 2024+ window during exploration, so the time-holdout in S4 is a **replication / stability** check on data we have already seen — not a pristine out-of-sample test. S4's deliverables are:

1. A **defensible, reproducible estimator** (inverse-variance WLS) that is not dominated by high-variance firms, reported alongside the equal-weight OLS bound and a z-scored corroboration.
2. A **locked split protocol** (fixed cutoff, fixed seed, leakage-free weight construction) that becomes a *genuinely naive* test the moment the panel is extended (new filings, broader universe) — the infrastructure, ready to run clean.

S4 does **not**: re-run the bootstrap or calendar-time portfolio (they stay as S3 — neither is strength-focused); construct a tradeable portfolio/backtest with costs (out of scope); add new firms or refetch any data. **Offline only — consumes banked artifacts (`data/panel.parquet`, `data/cache/prices/*`).**

## 3. Settled decisions

| Decision | Choice |
|----------|--------|
| Primary holdout | **Time split** — train `date < 2024-01-01`, test `date >= 2024-01-01`. Firm split reported as secondary robustness. |
| Regressors | `strength` (focal) + `discretionary` + `opp_plan` + `log_value`, with firm + year-month fixed effects. |
| Estimators reported | (1) inverse-variance WLS [primary], (2) equal-weight OLS [conservative bound = S3 number], (3) per-firm z-scored DV [corroboration]. |
| Horizons | 21, 63, 126, 252 trading days (focal: 63 & 126). |
| SEs | Firm-clustered (cluster on `ticker`), consistent with S3. |
| Weight leakage | Time-holdout weights estimated on **train window only**, frozen, applied to test. Full-sample row uses full-sample variances (labelled in-sample). Pooled-variance fallback for firms absent from the estimation subset. |
| Firm split | Deterministic: `np.random.default_rng(7).permutation(sorted(unique tickers))`, first half vs second half. |
| Output | `results/s4_wls_inference.json` + `results/s4_wls_findings.md` (both git-tracked; `results/` is not gitignored). |
| Scope exclusions | No bootstrap/calendar-time re-run; no backtest; no new data. |

## 4. Architecture

Five units; each independently testable.

### 4.1 `src/insider_hb/wls.py`

Pure estimation functions over a panel DataFrame.

- `firm_variance(panel, h, *, firms=None) -> dict[str, float]`
  Per-firm variance of `exc_{h}` (NaN-dropped). If `firms` is given, restrict the estimation rows to that firm set (used to compute train-only weights). Firms with < 2 valid observations are omitted from the returned dict (caller applies pooled fallback).

- `inverse_variance_weights(panel, h, variances) -> pd.Series`
  Row-aligned weights = `1 / variances[ticker]`. For any row whose `ticker` is absent from `variances` (or has non-positive/NaN variance), fall back to `1 / pooled_var`, where `pooled_var` is the mean of the finite values in `variances`. Returns a float Series aligned to `panel.index`; no infs/NaNs.

- `wls_fe_regression(panel, h, *, weights=None, regressors=REGRESSORS, dv_col=None) -> list[dict]`
  Firm + year-month FE regression on `regressors`. The dependent variable is `exc_{h}` by default, or the column named by `dv_col` when given (the z-scored estimator passes `dv_col="exc_{h}_z"`). When `weights is None` → equal-weight OLS (the S3 bound). When `weights` is a Series → `sm.WLS`. Firm-clustered SEs via integer codes. Mirrors S3 `fe_regression` output: one dict per term in `regressors` with `{term, coef, se, t, p, ci_lo, ci_hi, n, n_firms}`. Rows with NaN in the DV or any regressor are dropped; the regressor columns (`discretionary`, `opp_plan`, `log_value`, `strength`) are assumed already derived on `panel` by the caller. Weights, when passed, are row-aligned to `panel.index` and subset alongside the dropped rows. Design matrix uses `drop_first=True` dummies + an added constant.

- `firm_zscore_dv(panel, h) -> pd.Series`
  Per-firm standardized DV: `(exc_{h} - firm_mean) / firm_std`, row-aligned, infs→NaN. Firms with < 2 valid obs yield NaN for their rows. Used as the DV for the z-scored corroboration estimator (fed through `wls_fe_regression` with `weights=None`).

Module constant: `REGRESSORS = ["strength", "discretionary", "opp_plan", "log_value"]`.

The caller is responsible for deriving `discretionary`, `opp_plan`, and `log_value` columns from the panel (same definitions as S3 exploration: `discretionary = class == "discretionary"`, `opp_plan = class == "opportunistic_plan"`, `log_value = log1p(value where value>0)`). These derivations live in the orchestrator, not in `wls.py`, to keep the estimator pure.

### 4.2 `src/insider_hb/holdout.py`

- `time_split(panel, cutoff="2024-01-01") -> tuple[pd.DataFrame, pd.DataFrame]`
  `(train, test)` = `(panel[date < cutoff], panel[date >= cutoff])`. `date` coerced via `pd.to_datetime`.

- `firm_split(panel, seed=7) -> tuple[pd.DataFrame, pd.DataFrame]`
  Deterministic halves by firm: permute `sorted(panel["ticker"].unique())` with `np.random.default_rng(seed)`, split at the midpoint, return the two sub-panels (firm-disjoint).

### 4.3 `scripts/run_s4.py`

- `run_s4(panel, *, horizons=(21,63,126,252), time_cutoff="2024-01-01", firm_seed=7) -> dict`
  PURE core. Derives the regressor columns, then per horizon emits:
  - `full_sample`: `{wls: <strength row>, ols: <strength row>, zscore: <strength row>, wls_table: [all REGRESSORS rows]}` — full-sample variances for WLS weights.
  - `time_holdout`: weights frozen from **train** firm-variance; `{train: <strength wls row>, test: <strength wls row>}`.
  - `firm_holdout`: `{half_a: <strength wls row>, half_b: <strength wls row>}` — each half uses its own (in-sample) variances; this is a secondary sign-consistency check, not a frozen-weight test.
  - `verdict`: see §5.
  Returns `{meta: {panel_sha, n_rows, horizons, time_cutoff, firm_seed, generated}, horizons: {str(h): {...}}, verdict: {...}}`.

- `findings_markdown(res) -> str` — human-readable writeup: headline verdict, the three-estimator table at focal horizons, the time-holdout replication, the discretionary-does-not-survive note, and the honest spent-panel caveat.

- `main()` — flags `--panel`, `--out-dir`, `--time-cutoff`, `--firm-seed`; CWD-independent `src` path (same pattern as S3 `run_inference.py`); computes `panel_sha` (first 12 hex of sha256 of the parquet bytes); writes JSON + markdown.

### 4.4 Tests

`tests/test_wls.py`, `tests/test_holdout.py`, `tests/test_run_s4.py` — all offline, synthetic fixtures with known answers.

## 5. Verdict logic

`strength` is reported **robust** at a horizon iff:
1. `full_sample.wls` strength coef < 0 **and** p < 0.05, **and**
2. `time_holdout.test` strength coef < 0 **and** p < 0.05, **and**
3. `firm_holdout.half_a` and `half_b` strength coefs are **both < 0** (sign-consistent; significance not required of each half given reduced n).

`verdict` dict: per-horizon boolean `robust` plus the three underlying checks, and an overall `focal_robust` = robust at 126d. Markdown states the verdict in plain language and never overclaims beyond the spent-panel caveat.

## 6. Data flow

```
panel.parquet ──▶ run_s4 derives discretionary/opp_plan/log_value
                     │
       ┌─────────────┼───────────────────────────┐
   full_sample    time_holdout                firm_holdout
   (3 estimators) (train→freeze weights→test)  (half_a / half_b, secondary)
       └─────────────┴───────────────────────────┘
                     ▼
            verdict  ──▶  s4_wls_inference.json
                     └──▶ s4_wls_findings.md
```

## 7. Testing strategy

- **WLS recovers a masked effect (core value prop):** synthetic panel of N low-variance firms with a true `strength` slope of β, plus 2 injected hyper-variance firms with no slope. Assert equal-weight OLS strength is non-significant (swamped) while inverse-variance WLS recovers β within tolerance and is significant. This is the whole point of S4, as a test.
- **Leakage guard:** compute train-frozen weights; mutate the **test** DV arbitrarily; assert the frozen weights are byte-identical (weights provably independent of test data).
- **Pooled fallback:** a firm present only in test (absent from train weights) receives the pooled-variance weight, not NaN/inf; regression runs.
- **Deterministic splits:** `firm_split(seed=7)` returns firm-disjoint halves; repeated calls identical; `time_split` partitions on the cutoff with no overlap.
- **Estimator output shape:** `wls_fe_regression` returns one row per regressor with the full stat field set; `weights=None` path equals plain OLS.
- **z-scored DV:** per-firm standardized values have ≈0 mean / ≈1 std within firms with ≥2 obs; <2-obs firms → NaN.
- **Verdict truth table:** fabricate result dicts hitting each branch (all-pass → robust; test-window fails → not robust; firm sign flip → not robust).
- **JSON-serializability:** `json.dumps(run_s4(...))` succeeds (no numpy scalars leak).
- Full suite stays offline; `-m "not live"` default.

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Reader treats the time-holdout as a clean OOS test | Caveat burned into §2, verdict markdown, and the findings writeup. |
| Inverse-variance weighting shifts the estimand toward low-vol megacaps | Explicitly reported: WLS is primary, equal-weight OLS printed alongside as the conservative bound; writeup names the tradeoff. |
| Weight leakage from test into train | Train-only frozen weights + a dedicated leakage-guard test. |
| 50-firm (≈48 valid) ceiling still binds all inference | Stated; S4 does not claim to escape it — only to estimate efficiently within it. |
| Month-FE differ across train/test windows | Expected and correct — FE are nuisance; each subset is re-estimated independently. |

## 9. Non-goals (explicit)

Bootstrap / calendar-time re-run; tradeable backtest with costs; new firms or refetch; escaping the firm-count ceiling; factor adjustment beyond market (SPY) — all deferred to later cycles.
