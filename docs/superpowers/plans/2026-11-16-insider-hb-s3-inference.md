# S3 — H-B Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate H-B on the S2 sale-event panel with three independent inference engines (firm-clustered block bootstrap, calendar-time portfolio, panel fixed-effects regression) and emit a machine-readable results JSON plus a committed markdown findings writeup.

**Architecture:** Four pure analysis modules under `src/insider_hb/` — `estimands` (point estimates), `bootstrap` (cluster resampling), `calendar_time` (daily overlap-robust portfolio), `regression` (statsmodels FE OLS) — orchestrated by one CLI, `scripts/run_inference.py`. All analysis is pure and offline-testable on synthetic fixtures; the CLI does the only disk I/O (reads `data/panel.parquet` + `data/cache/prices/*.parquet`, writes `results/`).

**Tech Stack:** Python 3.11, pandas, numpy, statsmodels (new), pytest. Package layout `src/insider_hb/`, `pythonpath=["src"]`, offline-by-default (`addopts = -m 'not live'`). Run tests with `.venv/bin/pytest`.

---

## Spec

Source spec: `docs/superpowers/specs/2026-11-16-insider-hb-s3-inference-design.md`.

Key facts the implementer needs:
- `data/panel.parquet` columns: `ticker, date, owner, role, shares, price, value, aff10b5one, plan_adoption, adoption_gap_days, class, opportunistic, strength, exc_21, exc_63, exc_126, exc_252, gov_tier, size_tier, dual_class, founder_ceo, insider_own_pct`. One row per officer Code-S sale. `exc_{h}` is firm-minus-SPY forward excess return at `h` trading days, **already market-adjusted**; right-censored events carry `NaN`. `opportunistic` is bool. `date` is datetime64. `gov_tier` ∈ {concentrated, mixed, dispersed}.
- Price cache: `data/cache/prices/<TICKER>.parquet`, read as `pd.read_parquet(path)["value"]` → a tz-naive daily close Series. `SPY.parquet` is present. 50 firm files + SPY.
- Horizons: `h ∈ {21, 63, 126, 252}` (import `DEFAULT_HORIZONS` from `insider_hb.features`).
- `results/` is git-tracked (not in `.gitignore`); `data/` is ignored.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/insider_hb/estimands.py` (create) | Pure point estimates over a panel: `pooled_mean`, `opportunistic_contrast`, `gov_tier_means`, `n_valid`. |
| `src/insider_hb/bootstrap.py` (create) | `bootstrap_statistic` — cluster block bootstrap returning estimate/CI/p. |
| `src/insider_hb/calendar_time.py` (create) | `daily_portfolio_excess` + `calendar_time_test` — overlap-robust daily portfolio with Newey-West SEs. |
| `src/insider_hb/regression.py` (create) | `fe_regression` — statsmodels FE OLS with firm-clustered SEs. |
| `scripts/run_inference.py` (create) | CLI orchestrator: load artifacts, run all engines across horizons, write `results/s3_hb_inference.json` + `results/s3_hb_findings.md`. |
| `tests/test_estimands.py` (create) | Unit tests, synthetic panel, known answers. |
| `tests/test_bootstrap.py` (create) | Unit tests: determinism, degenerate CI, p-value, cluster handling. |
| `tests/test_calendar_time.py` (create) | Unit tests: hand-built 2-firm overlap, half-open window, finite NW stats. |
| `tests/test_regression.py` (create) | Unit tests: recover a known opportunistic coefficient; FE absorb a firm mean shift. |
| `tests/test_run_inference.py` (create) | End-to-end on a tiny synthetic panel + prices: full JSON schema populated. |

---

## Task 1: Add statsmodels dependency

**Files:**
- Modify: (none — venv-only, matching the project's existing manual-install convention)
- Test: `tests/test_regression.py` (import check only here; real tests in Task 5)

- [ ] **Step 1: Install statsmodels into the venv**

Run: `.venv/bin/pip install statsmodels`
Expected: installs statsmodels and its deps (scipy, patsy) successfully.

- [ ] **Step 2: Verify the import works**

Run: `.venv/bin/python -c "import statsmodels.api as sm; print(sm.__name__)"`
Expected: prints `statsmodels.api` with no traceback.

- [ ] **Step 3: Confirm the existing suite still passes**

Run: `.venv/bin/pytest -q`
Expected: all currently-passing offline tests still pass (41 passed, some deselected).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: add statsmodels for S3 inference"
```

(No file changes expected; if `pip` wrote nothing tracked, this commit may be empty — that's fine, skip it and move on.)

---

## Task 2: Point estimates (`estimands.py`)

**Files:**
- Create: `src/insider_hb/estimands.py`
- Test: `tests/test_estimands.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_estimands.py
import numpy as np
import pandas as pd
from insider_hb.estimands import (
    pooled_mean, opportunistic_contrast, gov_tier_means, n_valid,
)


def _panel():
    return pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2022-01-03", "2022-02-03", "2022-01-10", "2022-03-01"]),
        "value": [100.0, 300.0, 100.0, 100.0],
        "opportunistic": [True, True, False, False],
        "gov_tier": ["concentrated", "concentrated", "dispersed", "mixed"],
        "strength": [0.9, 0.8, 0.7, 0.6],
        "exc_21": [-0.02, -0.10, 0.04, np.nan],
    })


def test_pooled_mean_equal():
    # nanmean of [-0.02, -0.10, 0.04] = -0.02666...
    assert abs(pooled_mean(_panel(), 21) - (-0.08 / 3)) < 1e-9


def test_pooled_mean_value_weighted():
    # (100*-0.02 + 300*-0.10 + 100*0.04) / 500 = (-0.02 - 0.30 + 0.04)/... = -0.28/500
    assert abs(pooled_mean(_panel(), 21, weight="value") - (-0.28 / 5)) < 1e-9


def test_opportunistic_contrast():
    # opp mean = mean(-0.02, -0.10) = -0.06 ; routine mean = mean(0.04) = 0.04
    assert abs(opportunistic_contrast(_panel(), 21) - (-0.06 - 0.04)) < 1e-9


def test_gov_tier_means():
    out = gov_tier_means(_panel(), 21)
    assert abs(out["concentrated"] - (-0.06)) < 1e-9
    assert abs(out["dispersed"] - 0.04) < 1e-9
    assert np.isnan(out["mixed"])  # its only row is NaN at exc_21


def test_n_valid_drops_nan():
    assert n_valid(_panel(), 21) == 3


def test_empty_group_is_nan_not_error():
    only_opp = _panel()[_panel()["opportunistic"]]
    assert np.isnan(opportunistic_contrast(only_opp, 21))  # no routine rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_estimands.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.estimands'`.

- [ ] **Step 3: Implement `estimands.py`**

```python
# src/insider_hb/estimands.py
"""Pure point estimates over the sale-event panel. No inference, no I/O."""
from __future__ import annotations

import pandas as pd

GOV_TIERS = ("concentrated", "mixed", "dispersed")


def n_valid(panel: pd.DataFrame, h: int) -> int:
    """Count rows with a non-NaN forward excess return at horizon `h`."""
    return int(panel[f"exc_{h}"].notna().sum())


def pooled_mean(panel: pd.DataFrame, h: int, weight: str = "equal") -> float:
    """Mean forward excess return at horizon `h`. `weight` is 'equal' or 'value'."""
    col = f"exc_{h}"
    df = panel[[col, "value"]].dropna(subset=[col])
    if df.empty:
        return float("nan")
    if weight == "value":
        w = df["value"]
        total = float(w.sum())
        if total == 0:
            return float("nan")
        return float((df[col] * w).sum() / total)
    if weight != "equal":
        raise ValueError(f"unknown weight {weight!r}")
    return float(df[col].mean())


def opportunistic_contrast(panel: pd.DataFrame, h: int) -> float:
    """mean(exc_h | opportunistic) - mean(exc_h | routine). NaN if either side empty."""
    col = f"exc_{h}"
    df = panel.dropna(subset=[col])
    opp = df.loc[df["opportunistic"].astype(bool), col]
    rou = df.loc[~df["opportunistic"].astype(bool), col]
    if opp.empty or rou.empty:
        return float("nan")
    return float(opp.mean() - rou.mean())


def gov_tier_means(panel: pd.DataFrame, h: int) -> dict[str, float]:
    """Mean exc_h per governance tier. Empty tier -> NaN (not an error)."""
    col = f"exc_{h}"
    df = panel.dropna(subset=[col])
    out: dict[str, float] = {}
    for tier in GOV_TIERS:
        sub = df.loc[df["gov_tier"] == tier, col]
        out[tier] = float(sub.mean()) if len(sub) else float("nan")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_estimands.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/estimands.py tests/test_estimands.py
git commit -m "feat: panel point estimates (pooled mean, opportunistic contrast, gov-tier)"
```

---

## Task 3: Cluster block bootstrap (`bootstrap.py`)

**Files:**
- Create: `src/insider_hb/bootstrap.py`
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bootstrap.py
import numpy as np
import pandas as pd
import pytest
from insider_hb.bootstrap import bootstrap_statistic


def _mean_exc21(df):
    s = df["exc_21"].dropna()
    return float(s.mean()) if len(s) else float("nan")


def _panel(n_firms=8, per=6, shift=-0.05, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        for j in range(per):
            rows.append({
                "ticker": f"F{f}",
                "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=30 * j),
                "exc_21": shift + rng.normal(0, 0.02),
            })
    return pd.DataFrame(rows)


def test_deterministic_under_seed():
    p = _panel()
    a = bootstrap_statistic(p, _mean_exc21, n=200, seed=7)
    b = bootstrap_statistic(p, _mean_exc21, n=200, seed=7)
    assert a == b


def test_estimate_is_statistic_on_full_panel():
    p = _panel()
    out = bootstrap_statistic(p, _mean_exc21, n=200, seed=0)
    assert abs(out["estimate"] - _mean_exc21(p)) < 1e-12


def test_negative_effect_has_small_p_and_ci_below_zero():
    p = _panel(shift=-0.05)
    out = bootstrap_statistic(p, _mean_exc21, n=1000, seed=0)
    assert out["estimate"] < 0
    assert out["ci_hi"] < 0        # whole 95% CI below zero
    assert out["p_value"] < 0.05


def test_zero_variance_gives_degenerate_ci():
    # every cluster identical -> resampling can't move the statistic
    p = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "date": pd.to_datetime(["2022-01-01"] * 4),
        "exc_21": [-0.03, -0.03, -0.03, -0.03],
    })
    out = bootstrap_statistic(p, _mean_exc21, n=200, seed=0)
    assert abs(out["ci_lo"] - (-0.03)) < 1e-9
    assert abs(out["ci_hi"] - (-0.03)) < 1e-9


def test_firm_month_cluster_runs():
    p = _panel()
    out = bootstrap_statistic(p, _mean_exc21, cluster="firm_month", n=100, seed=0)
    assert np.isfinite(out["estimate"])


def test_unknown_cluster_raises():
    with pytest.raises(ValueError, match="cluster"):
        bootstrap_statistic(_panel(), _mean_exc21, cluster="sector", n=10)


def test_empty_panel_returns_nan_without_crashing():
    empty = pd.DataFrame({"ticker": [], "date": [], "exc_21": []})
    out = bootstrap_statistic(empty, _mean_exc21, n=50, seed=0)
    assert out["n_boot"] == 0
    assert np.isnan(out["ci_lo"]) and np.isnan(out["ci_hi"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bootstrap.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.bootstrap'`.

- [ ] **Step 3: Implement `bootstrap.py`**

```python
# src/insider_hb/bootstrap.py
"""Cluster block bootstrap for panel statistics. Pure (numpy RNG, no I/O).

Resamples whole clusters (firms, or firm x calendar-month) with replacement so
that within-cluster correlation (one firm's boom-bust = many correlated events)
cannot masquerade as independent observations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _cluster_keys(panel: pd.DataFrame, cluster: str) -> pd.Series:
    if cluster == "ticker":
        return panel["ticker"].astype(str)
    if cluster == "firm_month":
        month = pd.to_datetime(panel["date"]).dt.to_period("M").astype(str)
        return panel["ticker"].astype(str) + "_" + month
    raise ValueError(f"unknown cluster {cluster!r}")


def bootstrap_statistic(
    panel: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "ticker",
    n: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap `stat_fn` by resampling clusters with replacement.

    Returns {estimate, ci_lo, ci_hi, p_value, n_boot}:
      - estimate: stat_fn on the full panel
      - ci_lo/ci_hi: 2.5/97.5 percentiles of the bootstrap distribution
      - p_value: two-sided, 2 x fraction of replicates on the opposite side of
        zero from the estimate, clipped to [0, 1]
      - n_boot: number of non-NaN bootstrap replicates
    """
    estimate = stat_fn(panel)
    keys = _cluster_keys(panel, cluster)
    groups = [panel.loc[idx] for idx in panel.groupby(keys).groups.values()]
    k = len(groups)
    if k == 0:
        return {"estimate": float(estimate), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    reps = np.empty(n)
    for i in range(n):
        picks = rng.integers(0, k, size=k)
        sample = pd.concat([groups[j] for j in picks], ignore_index=True)
        reps[i] = stat_fn(sample)
    reps = reps[~np.isnan(reps)]
    if len(reps) == 0:
        return {"estimate": float(estimate), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value": float("nan"), "n_boot": 0}
    ci_lo, ci_hi = np.percentile(reps, [2.5, 97.5])
    if np.isnan(estimate) or estimate == 0:
        p = 1.0
    else:
        frac_opposite = float(np.mean(reps * np.sign(estimate) <= 0))
        p = min(1.0, 2 * frac_opposite)
    return {
        "estimate": float(estimate),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p),
        "n_boot": int(len(reps)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bootstrap.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: firm-cluster block bootstrap for panel statistics"
```

---

## Task 4: Calendar-time portfolio (`calendar_time.py`)

**Files:**
- Create: `src/insider_hb/calendar_time.py`
- Test: `tests/test_calendar_time.py`

**Context:** This is the overlap-robust cross-check. Each sale `(ticker, date)` contributes membership
on the `h` trading days strictly after the sale (half-open window `(date, date+h]`). On each calendar
day the portfolio excess return is the equal-weight mean across member firms of
`(firm_daily_return − spy_daily_return)`. Aggregating to one number per date, then a constant-only
regression with Newey-West (HAC) SEs, makes every calendar date count once.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calendar_time.py
import numpy as np
import pandas as pd
from insider_hb.calendar_time import daily_portfolio_excess, calendar_time_test


def _prices():
    idx = pd.bdate_range("2022-01-03", periods=6)   # Mon Jan 3 .. Mon Jan 10
    # firm A: +1% per day; firm B: +3% per day; SPY: +1% per day
    a = pd.Series(100.0 * 1.01 ** np.arange(6), index=idx, name="A")
    b = pd.Series(100.0 * 1.03 ** np.arange(6), index=idx, name="B")
    spy = pd.Series(100.0 * 1.01 ** np.arange(6), index=idx, name="SPY")
    return {"A": a, "B": b}, spy


def test_half_open_window_and_equal_weight():
    prices, spy = _prices()
    idx = prices["A"].index
    # A sells on day0; B sells on day0. Window h=2 -> days 1 and 2 (strictly after).
    panel = pd.DataFrame({"ticker": ["A", "B"], "date": [idx[0], idx[0]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=2)
    # day0 excluded (strictly after sale); days 1 and 2 present
    assert list(daily.index) == [idx[1], idx[2]]
    # excess is the ARITHMETIC return difference: A = 0.01-0.01 = 0; B = 0.03-0.01 = 0.02
    # equal-weight mean = (0 + 0.02) / 2 = 0.01
    assert abs(daily.iloc[0] - 0.01) < 1e-9


def test_window_length_capped_at_h_trading_days():
    prices, spy = _prices()
    idx = prices["A"].index
    panel = pd.DataFrame({"ticker": ["B"], "date": [idx[0]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=3)
    assert len(daily) == 3  # exactly h trading days after the sale


def test_unknown_ticker_skipped():
    prices, spy = _prices()
    idx = prices["A"].index
    panel = pd.DataFrame({"ticker": ["ZZZ"], "date": [idx[0]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=2)
    assert daily.empty


def test_calendar_time_test_returns_finite_stats():
    prices, spy = _prices()
    idx = prices["A"].index
    panel = pd.DataFrame({"ticker": ["B", "B"], "date": [idx[0], idx[1]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=3)
    res = calendar_time_test(daily, maxlags=3)
    assert res["n_days"] == len(daily)
    assert np.isfinite(res["mean_daily_excess"])
    assert np.isfinite(res["t_stat"])
    assert res["mean_daily_excess"] > 0  # B outperforms SPY


def test_calendar_time_test_handles_too_few_days():
    res = calendar_time_test(pd.Series([0.01]), maxlags=3)
    assert res["n_days"] == 1
    assert np.isnan(res["t_stat"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_calendar_time.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.calendar_time'`.

- [ ] **Step 3: Implement `calendar_time.py`**

```python
# src/insider_hb/calendar_time.py
"""Overlap-robust calendar-time portfolio. Pure: callers load prices from disk.

A sale (ticker, date) is "active" on the h trading days strictly after it. On each
calendar day the portfolio excess return is the equal-weight mean across active
firms of (firm_daily_return - spy_daily_return). One value per date -> overlapping
forward windows cannot inflate the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def daily_portfolio_excess(
    panel: pd.DataFrame,
    prices: dict[str, pd.Series],
    spy: pd.Series,
    h: int,
) -> pd.Series:
    """Equal-weight daily excess-return series for the post-sale portfolio."""
    spy_ret = spy.sort_index().pct_change()
    firm_ret = {t: s.sort_index().pct_change() for t, s in prices.items()}

    contributions: dict[pd.Timestamp, list[float]] = {}
    for ticker, sale in zip(panel["ticker"], pd.to_datetime(panel["date"])):
        fr = firm_ret.get(ticker)
        if fr is None:
            continue
        window = fr.index[fr.index > sale][:h]   # half-open (sale, sale+h]
        for day in window:
            if day not in spy_ret.index:
                continue
            er = fr.loc[day] - spy_ret.loc[day]
            if pd.notna(er):
                contributions.setdefault(day, []).append(float(er))

    if not contributions:
        return pd.Series(dtype=float, name="port_excess")
    days = sorted(contributions)
    vals = [float(np.mean(contributions[d])) for d in days]
    return pd.Series(vals, index=pd.DatetimeIndex(days), name="port_excess")


def calendar_time_test(daily_excess: pd.Series, maxlags: int) -> dict:
    """Constant-only OLS with Newey-West (HAC) SEs on the daily excess series."""
    y = daily_excess.dropna().to_numpy()
    if len(y) < 2:
        return {"mean_daily_excess": float("nan"), "t_stat": float("nan"),
                "p_value": float("nan"), "n_days": int(len(y))}
    x = np.ones((len(y), 1))
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, maxlags)})
    return {
        "mean_daily_excess": float(fit.params[0]),
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "n_days": int(len(y)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_calendar_time.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/calendar_time.py tests/test_calendar_time.py
git commit -m "feat: overlap-robust calendar-time portfolio with Newey-West SEs"
```

---

## Task 5: Panel fixed-effects regression (`regression.py`)

**Files:**
- Create: `src/insider_hb/regression.py`
- Test: `tests/test_regression.py`

**Context:** `exc_h ~ opportunistic + strength + log_value` with firm (`ticker`) and year-month
fixed effects via drop-first dummies, and cluster-robust SEs by firm. Returns a tidy table for the
three non-FE regressors only. The opportunistic coefficient — conditional on strength and size — is
the headline number.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_regression.py
import numpy as np
import pandas as pd
from insider_hb.regression import fe_regression


def _synthetic(beta_opp=-0.08, seed=0):
    """Panel where opportunistic lowers exc by beta_opp, plus firm-level shifts.

    Eight firms (→ eight clusters for stable cluster-robust SEs) and genuinely
    varying strength/value (so neither collides with the intercept).
    """
    rng = np.random.default_rng(seed)
    shifts = {f"F{i}": float(rng.normal(0, 0.05)) for i in range(8)}
    rows = []
    for firm, shift in shifts.items():
        for m in range(1, 7):
            for opp in (True, False):
                exc = shift + beta_opp * opp + rng.normal(0, 0.005)
                rows.append({
                    "ticker": firm,
                    "date": pd.Timestamp(f"2022-{m:02d}-05"),
                    "value": float(rng.integers(100_000, 10_000_000)),
                    "opportunistic": opp,
                    "strength": float(rng.uniform(0.3, 0.95)),
                    "exc_63": exc,
                })
    return pd.DataFrame(rows)


def test_recovers_opportunistic_coefficient():
    table = fe_regression(_synthetic(beta_opp=-0.08), 63)
    opp = next(r for r in table if r["term"] == "opportunistic")
    assert abs(opp["coef"] - (-0.08)) < 0.01
    assert opp["ci_hi"] < 0            # significantly negative


def test_returns_only_the_three_regressors():
    table = fe_regression(_synthetic(), 63)
    assert sorted(r["term"] for r in table) == ["log_value", "opportunistic", "strength"]


def test_each_row_has_full_stat_fields():
    row = fe_regression(_synthetic(), 63)[0]
    assert set(row) == {"term", "coef", "se", "t", "p", "ci_lo", "ci_hi"}


def test_drops_nan_rows():
    df = _synthetic()
    df.loc[df.index[:5], "exc_63"] = np.nan
    table = fe_regression(df, 63)        # must not raise
    assert np.isfinite(table[0]["coef"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_regression.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.regression'`.

- [ ] **Step 3: Implement `regression.py`**

```python
# src/insider_hb/regression.py
"""Panel fixed-effects OLS for H-B. statsmodels with firm-clustered SEs.

exc_h ~ opportunistic + strength + log_value, with firm and year-month fixed
effects (drop-first dummies). SEs cluster-robust by firm. Returns a tidy table
of the three non-FE regressors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

TERMS = ["opportunistic", "strength", "log_value"]


def fe_regression(panel: pd.DataFrame, h: int) -> list[dict]:
    col = f"exc_{h}"
    df = panel.dropna(subset=[col, "strength", "value"]).copy()
    df["opportunistic"] = df["opportunistic"].astype(bool).astype(float)
    df["strength"] = df["strength"].astype(float)
    df["log_value"] = np.log1p(df["value"].astype(float))
    df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    base = df[TERMS].astype(float)
    firm_d = pd.get_dummies(df["ticker"], prefix="f", drop_first=True, dtype=float)
    ym_d = pd.get_dummies(df["ym"], prefix="m", drop_first=True, dtype=float)
    x = sm.add_constant(pd.concat([base, firm_d, ym_d], axis=1), has_constant="add")
    y = df[col].astype(float)

    groups = pd.Categorical(df["ticker"]).codes
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})

    ci = fit.conf_int()
    out: list[dict] = []
    for term in TERMS:
        lo, hi = ci.loc[term]
        out.append({
            "term": term,
            "coef": float(fit.params[term]),
            "se": float(fit.bse[term]),
            "t": float(fit.tvalues[term]),
            "p": float(fit.pvalues[term]),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_regression.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/regression.py tests/test_regression.py
git commit -m "feat: panel FE regression with firm-clustered SEs"
```

---

## Task 6: CLI orchestrator (`scripts/run_inference.py`)

**Files:**
- Create: `scripts/run_inference.py`
- Test: `tests/test_run_inference.py`

**Context:** Mirrors `scripts/build_panel.py` structure (argparse, `sys.path.insert(0, "src")`, a pure
core function plus a thin `main`). The testable core, `run_inference(panel, prices, spy, *, horizons,
bootstrap_n, seed)`, takes already-loaded data and returns the results dict. `main` does disk I/O:
loads `data/panel.parquet`, loads the price cache, calls the core, writes `results/s3_hb_inference.json`
and `results/s3_hb_findings.md`.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_run_inference.py
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "run_inference", Path(__file__).resolve().parents[1] / "scripts" / "run_inference.py"
)
run_inference_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_inference_mod)
run_inference = run_inference_mod.run_inference
findings_markdown = run_inference_mod.findings_markdown


def _panel():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=400)
    rows = []
    for firm, tier in [("A", "concentrated"), ("B", "dispersed")]:
        for k in range(20):
            sale = idx[10 + 5 * k]
            opp = k % 2 == 0
            rows.append({
                "ticker": firm, "date": sale, "value": 1e6,
                "opportunistic": opp, "gov_tier": tier, "strength": 0.8,
                "exc_21": -0.02 + rng.normal(0, 0.01),
                "exc_63": -0.03 + rng.normal(0, 0.01),
                "exc_126": -0.04 + rng.normal(0, 0.01),
                "exc_252": -0.05 + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def _prices():
    idx = pd.bdate_range("2022-01-03", periods=400)
    a = pd.Series(100.0 * 0.999 ** np.arange(400), index=idx, name="A")
    b = pd.Series(100.0 * 0.998 ** np.arange(400), index=idx, name="B")
    spy = pd.Series(100.0 * 1.0005 ** np.arange(400), index=idx, name="SPY")
    return {"A": a, "B": b}, spy


def test_run_inference_full_schema():
    prices, spy = _prices()
    res = run_inference(_panel(), prices, spy, horizons=(21, 63, 126, 252),
                        bootstrap_n=200, seed=0)
    assert res["meta"]["bootstrap_n"] == 200
    for h in (21, 63, 126, 252):
        block = res["horizons"][str(h)]
        assert set(block) >= {
            "pooled_mean_equal", "pooled_mean_value", "opportunistic_contrast",
            "gov_tier_means", "calendar_time", "regression", "n_valid",
        }
        assert set(block["pooled_mean_equal"]) == {
            "estimate", "ci_lo", "ci_hi", "p_value", "n_boot",
        }
        assert set(block["gov_tier_means"]) == {"concentrated", "mixed", "dispersed"}
        assert block["calendar_time"]["n_days"] > 0
        assert [r["term"] for r in block["regression"]] == [
            "opportunistic", "strength", "log_value",
        ]


def test_findings_markdown_mentions_verdict():
    prices, spy = _prices()
    res = run_inference(_panel(), prices, spy, horizons=(21, 63),
                        bootstrap_n=100, seed=0)
    md = findings_markdown(res)
    assert "# S3" in md
    assert "H-B" in md
    assert "opportunistic" in md.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_inference.py -q`
Expected: FAIL (module file doesn't exist yet → import error inside the test).

- [ ] **Step 3: Implement `scripts/run_inference.py`**

```python
# scripts/run_inference.py
"""Run the S3 H-B inference on the sale-event panel.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_inference.py \
        [--panel data/panel.parquet] [--cache data/cache/prices] \
        [--out-dir results] [--bootstrap-n 2000] [--seed 0]

Loads the panel + price cache, runs three inference engines (firm-cluster block
bootstrap, calendar-time portfolio, panel FE regression) across horizons, and
writes results/s3_hb_inference.json + results/s3_hb_findings.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from insider_hb.estimands import (
    pooled_mean, opportunistic_contrast, gov_tier_means, n_valid,
)
from insider_hb.bootstrap import bootstrap_statistic
from insider_hb.calendar_time import daily_portfolio_excess, calendar_time_test
from insider_hb.regression import fe_regression
from insider_hb.features import DEFAULT_HORIZONS


def run_inference(panel, prices, spy, *, horizons=DEFAULT_HORIZONS,
                  bootstrap_n=2000, seed=0) -> dict:
    """Pure core: already-loaded data in, results dict out."""
    out_horizons: dict[str, dict] = {}
    for h in horizons:
        boot = lambda fn: bootstrap_statistic(panel, fn, cluster="ticker",
                                              n=bootstrap_n, seed=seed)
        gov = gov_tier_means(panel, h)
        gov_boot = {}
        for tier in gov:
            gov_boot[tier] = bootstrap_statistic(
                panel[panel["gov_tier"] == tier],
                lambda df, _h=h: pooled_mean(df, _h),
                cluster="ticker", n=bootstrap_n, seed=seed,
            )
        daily = daily_portfolio_excess(panel, prices, spy, h)
        out_horizons[str(h)] = {
            "n_valid": n_valid(panel, h),
            "pooled_mean_equal": boot(lambda df, _h=h: pooled_mean(df, _h)),
            "pooled_mean_value": boot(lambda df, _h=h: pooled_mean(df, _h, weight="value")),
            "opportunistic_contrast": boot(lambda df, _h=h: opportunistic_contrast(df, _h)),
            "gov_tier_means": gov_boot,
            "calendar_time": calendar_time_test(daily, maxlags=h),
            "regression": fe_regression(panel, h),
        }
    return {
        "meta": {
            "n_events": int(len(panel)),
            "bootstrap_n": int(bootstrap_n),
            "seed": int(seed),
        },
        "horizons": out_horizons,
    }


def _fmt(x) -> str:
    return "n/a" if x != x else f"{x:+.4f}"   # x!=x catches NaN


def findings_markdown(res: dict) -> str:
    lines = ["# S3 — H-B Inference Findings", ""]
    lines.append(f"Events: {res['meta']['n_events']} · "
                 f"bootstrap n={res['meta']['bootstrap_n']} · seed={res['meta']['seed']}")
    lines.append("")
    lines.append("## Pooled post-sale forward excess (firm-clustered bootstrap)")
    lines.append("")
    lines.append("| h | n | equal-wt | 95% CI | p | value-wt | opp−routine | CI | p |")
    lines.append("|---|---|----------|--------|---|----------|-------------|----|---|")
    for h, b in res["horizons"].items():
        eq, vw, oc = b["pooled_mean_equal"], b["pooled_mean_value"], b["opportunistic_contrast"]
        lines.append(
            f"| {h} | {b['n_valid']} | {_fmt(eq['estimate'])} | "
            f"[{_fmt(eq['ci_lo'])}, {_fmt(eq['ci_hi'])}] | {eq['p_value']:.3f} | "
            f"{_fmt(vw['estimate'])} | {_fmt(oc['estimate'])} | "
            f"[{_fmt(oc['ci_lo'])}, {_fmt(oc['ci_hi'])}] | {oc['p_value']:.3f} |"
        )
    lines += ["", "## Governance-tier pooled means (bootstrap)", "",
              "| h | concentrated | mixed | dispersed |", "|---|---|---|---|"]
    for h, b in res["horizons"].items():
        g = b["gov_tier_means"]
        lines.append(f"| {h} | {_fmt(g['concentrated']['estimate'])} | "
                     f"{_fmt(g['mixed']['estimate'])} | {_fmt(g['dispersed']['estimate'])} |")
    lines += ["", "## Calendar-time portfolio (Newey-West)", "",
              "| h | mean daily excess | t | p | n_days |", "|---|---|---|---|---|"]
    for h, b in res["horizons"].items():
        c = b["calendar_time"]
        lines.append(f"| {h} | {_fmt(c['mean_daily_excess'])} | {c['t_stat']:.2f} | "
                     f"{c['p_value']:.3f} | {c['n_days']} |")
    lines += ["", "## Panel FE regression (firm-clustered SEs)", ""]
    for h, b in res["horizons"].items():
        lines += [f"### h = {h}", "", "| term | coef | se | t | p | 95% CI |",
                  "|------|------|----|---|---|--------|"]
        for r in b["regression"]:
            lines.append(f"| {r['term']} | {_fmt(r['coef'])} | {r['se']:.4f} | "
                         f"{r['t']:.2f} | {r['p']:.3f} | "
                         f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] |")
        lines.append("")
    lines += ["## Verdict", "", _verdict(res)]
    return "\n".join(lines) + "\n"


def _verdict(res: dict) -> str:
    h = "252" if "252" in res["horizons"] else list(res["horizons"])[-1]
    b = res["horizons"][h]
    eq, oc = b["pooled_mean_equal"], b["opportunistic_contrast"]
    neg = eq["ci_hi"] < 0
    opp_worse = oc["estimate"] < 0 and oc["ci_hi"] < 0
    parts = [
        f"At h={h}: pooled post-sale excess {_fmt(eq['estimate'])} "
        f"(95% CI [{_fmt(eq['ci_lo'])}, {_fmt(eq['ci_hi'])}]) — "
        + ("H-B supported: forward excess is significantly negative."
           if neg else "H-B not supported at conventional significance."),
        "Opportunistic sales predict "
        + ("significantly more negative" if opp_worse else "no significantly different")
        + f" forward returns than routine ones (contrast {_fmt(oc['estimate'])}).",
    ]
    return " ".join(parts)


def _load_prices(cache_dir: Path, tickers) -> dict:
    out = {}
    for t in tickers:
        p = cache_dir / f"{t}.parquet"
        if p.exists():
            out[t] = pd.read_parquet(p)["value"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel.parquet")
    ap.add_argument("--cache", default="data/cache/prices")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--bootstrap-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    cache = Path(args.cache)
    prices = _load_prices(cache, sorted(panel["ticker"].unique()))
    spy = pd.read_parquet(cache / "SPY.parquet")["value"]

    res = run_inference(panel, prices, spy, bootstrap_n=args.bootstrap_n, seed=args.seed)
    res["meta"]["panel_sha"] = hashlib.sha256(
        Path(args.panel).read_bytes()).hexdigest()[:12]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s3_hb_inference.json").write_text(json.dumps(res, indent=2))
    (out / "s3_hb_findings.md").write_text(findings_markdown(res))
    print(json.dumps({"n_events": res["meta"]["n_events"],
                      "wrote": [str(out / "s3_hb_inference.json"),
                                str(out / "s3_hb_findings.md")]}), file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_run_inference.py -q`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all offline tests pass (prior suite + the new S3 tests), live deselected.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_inference.py tests/test_run_inference.py
git commit -m "feat: S3 inference CLI orchestrator + findings writeup"
```

---

## Task 7: Production run on the real panel + findings

**Files:**
- Create: `results/s3_hb_inference.json`, `results/s3_hb_findings.md` (generated)

**Context:** This is the real H-B test — the deliverable. Runs offline against the existing
`data/panel.parquet` and `data/cache/prices/*.parquet`; no network. `results/` is git-tracked.

- [ ] **Step 1: Run the inference on the real panel**

Run: `PYTHONPATH=src .venv/bin/python scripts/run_inference.py --bootstrap-n 2000 --seed 0`
Expected: a JSON line on stderr reporting `n_events` (≈9011) and the two written paths; no traceback.

- [ ] **Step 2: Sanity-check the outputs**

Run: `.venv/bin/python -c "import json; d=json.load(open('results/s3_hb_inference.json')); print(sorted(d['horizons'])); print(d['horizons']['252']['pooled_mean_equal']); print(d['horizons']['252']['opportunistic_contrast'])"`
Expected: horizons `['126','21','252','63']`; the 252d pooled mean and opportunistic contrast print as finite numbers with CIs. Confirm `results/s3_hb_findings.md` renders readable tables and a Verdict line.

- [ ] **Step 3: Commit the results**

```bash
git add results/s3_hb_inference.json results/s3_hb_findings.md
git commit -m "results: S3 H-B inference on the 50-firm panel"
```

---

## Notes for the executor

- **Offline always:** every S3 test and the production run are offline. Do not set `EDGAR_IDENTITY`; do not fetch. If a test hits the network, it's a bug in the test.
- **basedpyright noise:** missing stubs for pandas/numpy/statsmodels and unresolved `insider_hb.*` imports are pre-existing environment noise. Tests are ground truth; ignore the diagnostics.
- **statsmodels HAC/cluster:** `cov_type="HAC"` needs `cov_kwds={"maxlags": ...}`; `cov_type="cluster"` needs `cov_kwds={"groups": <int codes array>}`. Pass integer group codes (from `pd.Categorical(...).codes`), not string labels.
- **Lambda default-arg trick:** the CLI binds the loop variable `h` into each bootstrap `stat_fn` via a default argument (`lambda df, _h=h: ...`). Keep it — a bare closure over `h` would capture the last horizon only.
- **DRY:** reuse `DEFAULT_HORIZONS` from `insider_hb.features`; do not redefine the horizon tuple.
```
