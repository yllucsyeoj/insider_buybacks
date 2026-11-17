# S4 WLS Strength Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a variance-robust (inverse-variance WLS) estimator for the H-B `strength` channel, with equal-weight OLS and per-firm z-scored corroboration, a pre-registered leakage-free time/firm holdout, and a committed writeup.

**Architecture:** Two pure estimation modules (`wls.py` estimators, `holdout.py` splits) consumed by a `run_s4.py` orchestrator that emits JSON + markdown. Mirrors the S3 `regression.py` / `run_inference.py` structure (firm+month FE dummies, firm-clustered SEs, importlib-loaded script tests). Offline throughout.

**Tech Stack:** Python, pandas, numpy, statsmodels (`sm.OLS`/`sm.WLS` with `cov_type="cluster"`). Tests via pytest (`.venv/bin/pytest`, `pythonpath=["src"]`, `-m "not live"` default).

**Reference:** spec at `docs/superpowers/specs/2026-11-16-insider-hb-s4-wls-design.md`. Existing patterns: `src/insider_hb/regression.py`, `scripts/run_inference.py`, `tests/test_regression.py`, `tests/test_run_inference.py`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/insider_hb/wls.py` | `firm_variance`, `inverse_variance_weights`, `wls_fe_regression`, `firm_zscore_dv`, `REGRESSORS` |
| `src/insider_hb/holdout.py` | `time_split`, `firm_split` |
| `scripts/run_s4.py` | `run_s4` (pure core), `findings_markdown`, `_verdict`, `main` |
| `tests/test_wls.py` | Tasks 1–2 + frozen-weight leakage guard |
| `tests/test_holdout.py` | Task 3 |
| `tests/test_run_s4.py` | Tasks 4–5 (importlib-loaded script) |

---

### Task 1: Firm variance + inverse-variance weights

**Files:**
- Create: `src/insider_hb/wls.py`
- Test: `tests/test_wls.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wls.py
import numpy as np
import pandas as pd
from insider_hb.wls import firm_variance, inverse_variance_weights


def _panel():
    # firm A: low variance; firm B: high variance; firm C: single obs (dropped from variances)
    return pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B", "B", "C"],
        "exc_126": [0.01, 0.02, 0.00, 0.5, -0.4, 0.3, 0.1],
    })


def test_firm_variance_per_firm_and_drops_singletons():
    v = firm_variance(_panel(), 126)
    assert set(v) == {"A", "B"}                 # C has <2 obs -> omitted
    assert v["B"] > v["A"]                       # B is the high-variance firm
    assert abs(v["A"] - np.var([0.01, 0.02, 0.00], ddof=1)) < 1e-12


def test_firm_variance_restricted_to_firms():
    v = firm_variance(_panel(), 126, firms=["A"])
    assert set(v) == {"A"}


def test_inverse_variance_weights_and_pooled_fallback():
    v = firm_variance(_panel(), 126)
    w = inverse_variance_weights(_panel(), 126, v)
    assert list(w.index) == list(_panel().index)   # aligned to panel index
    assert np.isfinite(w).all()                     # no inf/nan anywhere
    # A rows get 1/var_A
    assert abs(w.iloc[0] - 1.0 / v["A"]) < 1e-9
    # C absent from variances -> pooled = mean of finite variance values
    pooled = float(np.mean([v["A"], v["B"]]))
    assert abs(w.iloc[6] - 1.0 / pooled) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wls.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insider_hb.wls'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/insider_hb/wls.py
"""Variance-robust (WLS) estimators for the H-B strength channel.

Inverse-variance weighting stops a handful of hyper-variance firms from
dominating an equal-weight OLS fit (the S3 artifact). All functions are pure:
they take an already-derived panel (regressor columns present) and return
estimates. Firm-clustered SEs via statsmodels, consistent with S3 regression.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

REGRESSORS = ["strength", "discretionary", "opp_plan", "log_value"]


def firm_variance(panel: pd.DataFrame, h: int, *, firms=None) -> dict[str, float]:
    """Per-firm variance (ddof=1) of exc_h. Firms with <2 valid obs omitted."""
    col = f"exc_{h}"
    df = panel.dropna(subset=[col])
    if firms is not None:
        df = df[df["ticker"].isin(list(firms))]
    g = df.groupby("ticker")[col]
    var, cnt = g.var(), g.count()
    return {t: float(var[t]) for t in var.index
            if cnt[t] >= 2 and np.isfinite(var[t])}


def inverse_variance_weights(panel: pd.DataFrame, h: int,
                             variances: dict[str, float]) -> pd.Series:
    """Row weights = 1/variances[ticker], pooled-variance fallback for missing
    firms. Pooled = mean of the finite positive variance values. Aligned to
    panel.index; never inf/NaN."""
    finite = [v for v in variances.values() if np.isfinite(v) and v > 0]
    pooled = float(np.mean(finite)) if finite else 1.0

    def w(t):
        v = variances.get(t)
        if v is None or not np.isfinite(v) or v <= 0:
            v = pooled
        return 1.0 / v

    return panel["ticker"].map(w).astype(float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_wls.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/wls.py tests/test_wls.py
git commit -m "feat: S4 firm variance + inverse-variance weights"
```

---

### Task 2: WLS/OLS FE regression + per-firm z-scored DV

**Files:**
- Modify: `src/insider_hb/wls.py`
- Test: `tests/test_wls.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wls.py`:

```python
from insider_hb.wls import wls_fe_regression, firm_zscore_dv, REGRESSORS


def _reg_panel(beta=-0.10, seed=0):
    """6 low-variance firms with a true strength slope `beta`, plus 2
    hyper-variance firms with NO slope. Equal-weight OLS is swamped by the
    noisy firms; inverse-variance WLS recovers `beta`. This is S4's whole point.
    """
    rng = np.random.default_rng(seed)
    rows = []
    months = [f"2022-{m:02d}-05" for m in range(1, 11)]
    for i in range(6):                       # low-variance firms
        shift = rng.normal(0, 0.02)
        for m in months:
            s = rng.uniform(0.2, 0.95)
            rows.append({"ticker": f"L{i}", "date": m, "value": 1e6,
                         "strength": s, "discretionary": rng.integers(0, 2),
                         "opp_plan": 0.0, "log_value": np.log1p(1e6),
                         "exc_126": shift + beta * s + rng.normal(0, 0.01)})
    for j in range(2):                       # hyper-variance firms, no slope
        for m in months * 10:                # many noisy obs
            s = rng.uniform(0.2, 0.95)
            rows.append({"ticker": f"H{j}", "date": m, "value": 1e6,
                         "strength": s, "discretionary": rng.integers(0, 2),
                         "opp_plan": 0.0, "log_value": np.log1p(1e6),
                         "exc_126": rng.normal(0, 2.0)})
    return pd.DataFrame(rows)


def _strength(rows):
    return next(r for r in rows if r["term"] == "strength")


def test_ols_is_swamped_but_wls_recovers_the_slope():
    from insider_hb.wls import firm_variance, inverse_variance_weights
    df = _reg_panel(beta=-0.10, seed=1)
    ols = _strength(wls_fe_regression(df, 126, weights=None))
    var = firm_variance(df, 126)
    w = inverse_variance_weights(df, 126, var)
    wls = _strength(wls_fe_regression(df, 126, weights=w))
    assert ols["p"] > 0.05                       # equal-weight OLS: not significant
    assert wls["coef"] < 0 and wls["p"] < 0.05   # WLS: significant negative
    assert abs(wls["coef"] - (-0.10)) < 0.05     # recovers the true slope


def test_regression_row_shape_all_terms():
    df = _reg_panel(seed=2)
    rows = wls_fe_regression(df, 126, weights=None)
    assert [r["term"] for r in rows] == REGRESSORS
    assert set(rows[0]) == {"term", "coef", "se", "t", "p", "ci_lo", "ci_hi", "n", "n_firms"}


def test_firm_zscore_dv_and_dv_col_path():
    df = _reg_panel(seed=3)
    df["exc_126_z"] = firm_zscore_dv(df, 126)
    # within-firm standardized: mean ~0, std ~1 for a multi-obs firm
    g = df.groupby("ticker")["exc_126_z"]
    assert abs(float(g.mean().mean())) < 1e-6
    assert abs(float(g.std().dropna().mean()) - 1.0) < 1e-6
    rows = wls_fe_regression(df, 126, weights=None, dv_col="exc_126_z")
    assert _strength(rows)["coef"] < 0           # negative slope survives standardization


def test_firm_zscore_singleton_is_nan():
    df = pd.DataFrame({"ticker": ["A", "A", "Z"], "exc_126": [0.1, 0.2, 0.3]})
    z = firm_zscore_dv(df, 126)
    assert np.isnan(z.iloc[2])                    # single-obs firm -> NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wls.py -v`
Expected: FAIL — `ImportError: cannot import name 'wls_fe_regression'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/insider_hb/wls.py`:

```python
def wls_fe_regression(panel: pd.DataFrame, h: int, *, weights=None,
                      regressors=REGRESSORS, dv_col=None) -> list[dict]:
    """Firm + year-month FE regression. DV is exc_h, or `dv_col` if given.
    weights=None -> equal-weight OLS (the S3 bound); a Series -> sm.WLS.
    Firm-clustered SEs. Regressor columns assumed already derived on `panel`.
    Returns one dict per regressor: term/coef/se/t/p/ci_lo/ci_hi/n/n_firms."""
    dv = dv_col or f"exc_{h}"
    cols = [dv] + list(regressors)
    df = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).copy()
    df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    base = df[list(regressors)].astype(float)
    firm_d = pd.get_dummies(df["ticker"], prefix="f", drop_first=True, dtype=float)
    ym_d = pd.get_dummies(df["ym"], prefix="m", drop_first=True, dtype=float)
    x = sm.add_constant(pd.concat([base, firm_d, ym_d], axis=1), has_constant="add")
    y = df[dv].astype(float)
    groups = pd.Categorical(df["ticker"]).codes

    if weights is None:
        fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        w = pd.Series(weights).reindex(df.index).astype(float)
        fit = sm.WLS(y, x, weights=w.values).fit(
            cov_type="cluster", cov_kwds={"groups": groups})

    ci = fit.conf_int()
    out: list[dict] = []
    for term in regressors:
        lo, hi = ci.loc[term]
        out.append({
            "term": term, "coef": float(fit.params[term]), "se": float(fit.bse[term]),
            "t": float(fit.tvalues[term]), "p": float(fit.pvalues[term]),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(df)), "n_firms": int(df["ticker"].nunique()),
        })
    return out


def firm_zscore_dv(panel: pd.DataFrame, h: int) -> pd.Series:
    """Per-firm standardized DV: (exc_h - firm_mean) / firm_std. Single-obs
    firms (std NaN) yield NaN. Aligned to panel.index; infs -> NaN."""
    col = f"exc_{h}"
    g = panel.groupby("ticker")[col]
    z = (panel[col] - g.transform("mean")) / g.transform("std")
    return z.replace([np.inf, -np.inf], np.nan)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_wls.py -v`
Expected: PASS (7 tests total). If `test_ols_is_swamped_but_wls_recovers_the_slope` is marginal, it is seeded (`seed=1`) and the noise scale (2.0 vs signal 0.1) is chosen to swamp OLS reliably; do not weaken the assertions.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/wls.py tests/test_wls.py
git commit -m "feat: S4 WLS/OLS FE regression + per-firm z-scored DV"
```

---

### Task 3: Holdout splits (time + firm)

**Files:**
- Create: `src/insider_hb/holdout.py`
- Test: `tests/test_holdout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_holdout.py
import pandas as pd
from insider_hb.holdout import time_split, firm_split


def _panel():
    return pd.DataFrame({
        "ticker": ["A", "A", "B", "B", "C", "D"],
        "date": pd.to_datetime(["2022-06-01", "2024-03-01", "2023-01-01",
                                "2025-07-01", "2024-01-01", "2021-12-31"]),
        "exc_126": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })


def test_time_split_partitions_on_cutoff():
    train, test = time_split(_panel(), cutoff="2024-01-01")
    assert (train["date"] < pd.Timestamp("2024-01-01")).all()
    assert (test["date"] >= pd.Timestamp("2024-01-01")).all()
    assert len(train) + len(test) == 6          # exact partition, no loss
    assert len(train) == 3 and len(test) == 3    # 2024-01-01 falls in test


def test_firm_split_is_firm_disjoint_and_deterministic():
    a1, b1 = firm_split(_panel(), seed=7)
    a2, b2 = firm_split(_panel(), seed=7)
    firms_a, firms_b = set(a1["ticker"]), set(b1["ticker"])
    assert firms_a.isdisjoint(firms_b)                       # no shared firms
    assert firms_a | firms_b == {"A", "B", "C", "D"}         # covers all firms
    assert set(a2["ticker"]) == firms_a and set(b2["ticker"]) == firms_b  # deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_holdout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insider_hb.holdout'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/insider_hb/holdout.py
"""Pre-registered holdout splits for S4. Split constants live in the spec;
callers pass them explicitly. Deterministic firm split via a fixed seed."""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(panel: pd.DataFrame, cutoff: str = "2024-01-01"):
    """(train, test) = (date < cutoff, date >= cutoff)."""
    d = pd.to_datetime(panel["date"])
    cut = pd.Timestamp(cutoff)
    return panel[d < cut].copy(), panel[d >= cut].copy()


def firm_split(panel: pd.DataFrame, seed: int = 7):
    """Firm-disjoint halves: permute sorted unique tickers with a fixed seed,
    split at the midpoint. Deterministic for a given (firm set, seed)."""
    firms = sorted(panel["ticker"].unique())
    perm = np.random.default_rng(seed).permutation(firms)
    mid = len(perm) // 2
    a, b = set(perm[:mid]), set(perm[mid:])
    return panel[panel["ticker"].isin(a)].copy(), panel[panel["ticker"].isin(b)].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_holdout.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/holdout.py tests/test_holdout.py
git commit -m "feat: S4 pre-registered time + firm holdout splits"
```

---

### Task 4: Orchestrator core + verdict + frozen-weight leakage guard

**Files:**
- Create: `scripts/run_s4.py`
- Modify: `tests/test_wls.py` (leakage guard), create `tests/test_run_s4.py`

- [ ] **Step 1: Write the failing leakage-guard test (test_wls.py)**

Append to `tests/test_wls.py`:

```python
def test_time_holdout_weights_are_frozen_from_train():
    """Weights applied to the test fit must use TRAIN firm variances, never
    test ones — the core leakage guarantee."""
    from insider_hb.wls import firm_variance, inverse_variance_weights
    train = pd.DataFrame({"ticker": ["A"] * 4 + ["B"] * 4,
                          "exc_126": [0.01, 0.02, 0.00, 0.03, 0.5, -0.4, 0.3, -0.2]})
    test = pd.DataFrame({"ticker": ["A", "A", "B", "B"],
                         "exc_126": [0.9, -0.9, 0.001, 0.002]})  # A & B variances flipped vs train
    train_var = firm_variance(train, 126)
    test_var = firm_variance(test, 126)
    assert train_var["A"] != test_var["A"]                 # the two differ (precondition)
    w_test = inverse_variance_weights(test, 126, train_var)
    # every A row in test weighted by 1/train_var[A], NOT 1/test_var[A]
    assert abs(w_test.iloc[0] - 1.0 / train_var["A"]) < 1e-9
    assert abs(w_test.iloc[0] - 1.0 / test_var["A"]) > 1e-6
```

- [ ] **Step 2: Write the failing core test (test_run_s4.py)**

```python
# tests/test_run_s4.py
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "run_s4", Path(__file__).resolve().parents[1] / "scripts" / "run_s4.py")
run_s4_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_s4_mod)
run_s4 = run_s4_mod.run_s4
findings_markdown = run_s4_mod.findings_markdown


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2022-01-03", periods=60).append(
        pd.bdate_range("2024-02-01", periods=60))        # straddles the 2024 cutoff
    classes = ["routine_plan", "discretionary", "opportunistic_plan"]
    for i in range(8):
        shift = rng.normal(0, 0.02)
        for d in dates:
            s = rng.uniform(0.2, 0.95)
            rows.append({
                "ticker": f"F{i}", "date": d, "value": float(rng.integers(1e5, 1e7)),
                "class": classes[rng.integers(0, 3)], "strength": s,
                "exc_21": shift - 0.05 * s + rng.normal(0, 0.01),
                "exc_63": shift - 0.08 * s + rng.normal(0, 0.01),
                "exc_126": shift - 0.10 * s + rng.normal(0, 0.01),
                "exc_252": shift - 0.12 * s + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def test_run_s4_full_schema_and_serializable():
    res = run_s4(_panel(), horizons=(63, 126))
    json.dumps(res)                                         # no numpy scalars leak
    assert set(res) >= {"meta", "horizons", "verdict"}
    for h in ("63", "126"):
        blk = res["horizons"][h]
        assert set(blk) == {"full_sample", "time_holdout", "firm_holdout"}
        fs = blk["full_sample"]
        assert set(fs) == {"wls", "ols", "zscore", "wls_table"}
        assert set(fs["wls"]) >= {"coef", "p", "n", "n_firms"}
        assert [r["term"] for r in fs["wls_table"]] == run_s4_mod.REGRESSORS
        assert set(blk["time_holdout"]) == {"train", "test"}
        assert set(blk["firm_holdout"]) == {"half_a", "half_b"}
    assert set(res["verdict"]) >= {"by_horizon", "focal_robust"}


def test_verdict_truth_table():
    _v = run_s4_mod._verdict
    def blk(full_c, full_p, test_c, test_p, ha_c, hb_c):
        return {
            "full_sample": {"wls": {"coef": full_c, "p": full_p}},
            "time_holdout": {"test": {"coef": test_c, "p": test_p}},
            "firm_holdout": {"half_a": {"coef": ha_c}, "half_b": {"coef": hb_c}},
        }
    allpass = _v({"126": blk(-0.2, 0.01, -0.2, 0.02, -0.1, -0.1)})
    assert allpass["by_horizon"]["126"]["robust"] is True
    assert allpass["focal_robust"] is True
    testfail = _v({"126": blk(-0.2, 0.01, -0.2, 0.40, -0.1, -0.1)})
    assert testfail["by_horizon"]["126"]["robust"] is False
    signflip = _v({"126": blk(-0.2, 0.01, -0.2, 0.02, -0.1, +0.1)})
    assert signflip["by_horizon"]["126"]["robust"] is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_wls.py::test_time_holdout_weights_are_frozen_from_train tests/test_run_s4.py -v`
Expected: FAIL — leakage test passes trivially only if functions exist (they do, so it should PASS); `test_run_s4.py` FAILs with `FileNotFoundError`/`ModuleNotFoundError` for `scripts/run_s4.py`.
(If the leakage test errors on import, that is also a fail — fine.)

- [ ] **Step 4: Write minimal implementation**

```python
# scripts/run_s4.py
"""Run the S4 variance-robust (WLS) strength inference on the sale-event panel.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_s4.py \
        [--panel data/panel.parquet] [--out-dir results] \
        [--time-cutoff 2024-01-01] [--firm-seed 7]

Three estimators per horizon (inverse-variance WLS [primary], equal-weight OLS
[conservative bound], per-firm z-scored DV [corroboration]); a leakage-free
time holdout (weights frozen from the train window) plus a secondary firm-half
split; a verdict; and results/s4_wls_inference.json + results/s4_wls_findings.md.

CAVEAT: the panel is already spent (2024+ was examined during exploration), so
the time holdout is a replication/stability check, not a pristine out-of-sample
test. See the spec §2.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insider_hb.wls import (
    REGRESSORS, firm_variance, inverse_variance_weights, wls_fe_regression,
    firm_zscore_dv,
)
from insider_hb.holdout import time_split, firm_split

DEFAULT_HORIZONS = (21, 63, 126, 252)


def _derive(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the S4 regressor columns; drop non-positive value (log1p guard)."""
    df = panel[panel["value"] > 0].copy()
    df["discretionary"] = (df["class"] == "discretionary").astype(float)
    df["opp_plan"] = (df["class"] == "opportunistic_plan").astype(float)
    df["log_value"] = np.log1p(df["value"].astype(float))
    df["strength"] = df["strength"].astype(float)
    return df


def _strength(rows: list[dict]) -> dict:
    return next(r for r in rows if r["term"] == "strength")


def _wls_strength(df, h):
    var = firm_variance(df, h)
    w = inverse_variance_weights(df, h, var)
    return _strength(wls_fe_regression(df, h, weights=w))


def run_s4(panel, *, horizons=DEFAULT_HORIZONS,
           time_cutoff="2024-01-01", firm_seed=7) -> dict:
    """Pure core: derived panel in, results dict out."""
    df = _derive(panel)
    out: dict[str, dict] = {}
    for h in horizons:
        # full sample (in-sample variances; labelled as such in the writeup)
        full_var = firm_variance(df, h)
        w_full = inverse_variance_weights(df, h, full_var)
        wls_table = wls_fe_regression(df, h, weights=w_full)
        df[f"exc_{h}_z"] = firm_zscore_dv(df, h)
        full_sample = {
            "wls": _strength(wls_table),
            "ols": _strength(wls_fe_regression(df, h, weights=None)),
            "zscore": _strength(wls_fe_regression(df, h, weights=None,
                                                  dv_col=f"exc_{h}_z")),
            "wls_table": wls_table,
        }
        # time holdout: weights FROZEN from train, applied to both fits
        train, test = time_split(df, cutoff=time_cutoff)
        train_var = firm_variance(train, h)
        time_holdout = {
            "train": _strength(wls_fe_regression(
                train, h, weights=inverse_variance_weights(train, h, train_var))),
            "test": _strength(wls_fe_regression(
                test, h, weights=inverse_variance_weights(test, h, train_var))),
        }
        # firm holdout (secondary): each half uses its own in-sample variances
        half_a, half_b = firm_split(df, seed=firm_seed)
        firm_holdout = {"half_a": _wls_strength(half_a, h),
                        "half_b": _wls_strength(half_b, h)}
        out[str(h)] = {"full_sample": full_sample,
                       "time_holdout": time_holdout,
                       "firm_holdout": firm_holdout}
    return {
        "meta": {
            "n_rows": int(len(df)),
            "horizons": list(horizons),
            "time_cutoff": time_cutoff,
            "firm_seed": int(firm_seed),
            "generated": _dt.date.today().isoformat(),
        },
        "horizons": out,
        "verdict": _verdict(out),
    }


def _sig_neg(row: dict) -> bool:
    return row["coef"] < 0 and row["p"] < 0.05


def _verdict(horizons_out: dict) -> dict:
    by_h = {}
    for h, blk in horizons_out.items():
        c1 = _sig_neg(blk["full_sample"]["wls"])
        c2 = _sig_neg(blk["time_holdout"]["test"])
        c3 = (blk["firm_holdout"]["half_a"]["coef"] < 0
              and blk["firm_holdout"]["half_b"]["coef"] < 0)
        by_h[h] = {"robust": bool(c1 and c2 and c3),
                   "full_sample_sig_neg": bool(c1),
                   "test_window_sig_neg": bool(c2),
                   "firm_halves_sign_consistent": bool(c3)}
    return {"by_horizon": by_h,
            "focal_robust": bool(by_h.get("126", {}).get("robust", False))}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_wls.py tests/test_run_s4.py -v`
Expected: PASS (test_wls.py leakage test + the two run_s4 tests). `findings_markdown` import in test_run_s4.py will fail until Task 5 — so run only the two named tests here:
Run: `.venv/bin/pytest "tests/test_run_s4.py::test_run_s4_full_schema_and_serializable" "tests/test_run_s4.py::test_verdict_truth_table" tests/test_wls.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_s4.py tests/test_run_s4.py tests/test_wls.py
git commit -m "feat: S4 orchestrator core + verdict + frozen-weight leakage guard"
```

---

### Task 5: Findings writeup + CLI + production run

**Files:**
- Modify: `scripts/run_s4.py`
- Modify: `tests/test_run_s4.py`
- Create: `results/s4_wls_inference.json`, `results/s4_wls_findings.md` (generated)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_s4.py`:

```python
def test_findings_markdown_has_estimators_verdict_and_caveat():
    res = run_s4(_panel(), horizons=(63, 126))
    md = findings_markdown(res)
    assert "# S4" in md
    assert "WLS" in md and "OLS" in md
    assert "strength" in md.lower()
    assert "spent" in md.lower() or "replication" in md.lower()   # honest caveat present
    assert "Verdict" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_s4.py::test_findings_markdown_has_estimators_verdict_and_caveat -v`
Expected: FAIL — `AttributeError: module 'run_s4' has no attribute 'findings_markdown'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/run_s4.py` (before `main`):

```python
def _fmt(x) -> str:
    return "n/a" if x != x else f"{x:+.4f}"   # x!=x catches NaN


def _row(label, r) -> str:
    return (f"| {label} | {_fmt(r['coef'])} | {r['se']:.4f} | {r['t']:.2f} | "
            f"{r['p']:.3f} | [{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | "
            f"{r['n']} | {r['n_firms']} |")


def findings_markdown(res: dict) -> str:
    m = res["meta"]
    lines = ["# S4 — Variance-Robust (WLS) Strength Inference", "",
             f"Rows: {m['n_rows']} · time cutoff: {m['time_cutoff']} · "
             f"firm seed: {m['firm_seed']} · generated: {m['generated']}", "",
             "> **Caveat:** the panel is already spent (2024+ was examined during "
             "exploration). The time holdout below is a replication/stability check, "
             "not a pristine out-of-sample test. WLS shifts the estimand toward "
             "low-variance firms; equal-weight OLS is printed alongside as the "
             "conservative bound.", ""]

    lines += ["## Strength coefficient by estimator (full sample)", "",
              "| h | WLS (primary) | OLS (bound) | z-scored DV |",
              "|---|---|---|---|"]
    for h, blk in res["horizons"].items():
        fs = blk["full_sample"]
        lines.append(
            f"| {h} | {_fmt(fs['wls']['coef'])} (p={fs['wls']['p']:.3f}) | "
            f"{_fmt(fs['ols']['coef'])} (p={fs['ols']['p']:.3f}) | "
            f"{_fmt(fs['zscore']['coef'])} (p={fs['zscore']['p']:.3f}) |")

    lines += ["", "## Time holdout — WLS strength (weights frozen from train)", "",
              "| h | train (pre-cutoff) | test (post-cutoff) |", "|---|---|---|"]
    for h, blk in res["horizons"].items():
        t = blk["time_holdout"]
        lines.append(f"| {h} | {_fmt(t['train']['coef'])} (p={t['train']['p']:.3f}) | "
                     f"{_fmt(t['test']['coef'])} (p={t['test']['p']:.3f}) |")

    lines += ["", "## Firm holdout — WLS strength (secondary, sign check)", "",
              "| h | half A | half B |", "|---|---|---|"]
    for h, blk in res["horizons"].items():
        f = blk["firm_holdout"]
        lines.append(f"| {h} | {_fmt(f['half_a']['coef'])} (p={f['half_a']['p']:.3f}) | "
                     f"{_fmt(f['half_b']['coef'])} (p={f['half_b']['p']:.3f}) |")

    lines += ["", "## Full WLS regressor table", ""]
    for h, blk in res["horizons"].items():
        lines += [f"### h = {h}", "",
                  "| term | coef | se | t | p | 95% CI | n | n_firms |",
                  "|------|------|----|---|---|--------|---|---------|"]
        for r in blk["full_sample"]["wls_table"]:
            lines.append(_row(r["term"], r))
        lines.append("")

    v = res["verdict"]
    lines += ["## Verdict", ""]
    for h, vb in v["by_horizon"].items():
        verdict = ("ROBUST" if vb["robust"] else "not robust")
        lines.append(
            f"- **h={h}: strength {verdict}** — full-sample sig-neg: "
            f"{vb['full_sample_sig_neg']}, test-window sig-neg: "
            f"{vb['test_window_sig_neg']}, firm-halves sign-consistent: "
            f"{vb['firm_halves_sign_consistent']}.")
    lines += ["",
              ("At the focal 126-day horizon, the variance-robust WLS estimate of "
               "sell-into-strength → forward excess is "
               + ("robust by the pre-registered criteria."
                  if v["focal_robust"] else
                  "not robust by the pre-registered criteria."))]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel.parquet")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--time-cutoff", default="2024-01-01")
    ap.add_argument("--firm-seed", type=int, default=7)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    res = run_s4(panel, time_cutoff=args.time_cutoff, firm_seed=args.firm_seed)
    res["meta"]["panel_sha"] = hashlib.sha256(
        Path(args.panel).read_bytes()).hexdigest()[:12]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s4_wls_inference.json").write_text(json.dumps(res, indent=2))
    (out / "s4_wls_findings.md").write_text(findings_markdown(res))
    print(json.dumps({"n_rows": res["meta"]["n_rows"],
                      "focal_robust": res["verdict"]["focal_robust"],
                      "wrote": [str(out / "s4_wls_inference.json"),
                                str(out / "s4_wls_findings.md")]}), file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_run_s4.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full offline suite**

Run: `.venv/bin/pytest`
Expected: all previous S0–S3 tests PASS + the new S4 tests PASS; `live` tests deselected.

- [ ] **Step 6: Produce the production run**

Run: `PYTHONPATH=src .venv/bin/python scripts/run_s4.py`
Expected: writes `results/s4_wls_inference.json` + `results/s4_wls_findings.md`; stderr prints `focal_robust` and the two paths. Sanity-check: full-sample WLS strength @126d should be ≈ −0.24 (p<.01) and OLS ≈ −0.17 (p≈.07), matching the exploration.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_s4.py tests/test_run_s4.py results/s4_wls_inference.json results/s4_wls_findings.md
git commit -m "feat: S4 WLS findings writeup + CLI + production run"
```

---

## Self-Review Notes

- **Spec coverage:** §4.1 wls.py → Tasks 1–2; §4.2 holdout.py → Task 3; §4.3 run_s4 → Tasks 4–5; §5 verdict → Task 4; §7 testing (WLS-recovers-masked-effect, leakage guard, pooled fallback, deterministic splits, output shape, z-score, verdict truth table, JSON) → covered across Tasks 1–5; §2 caveat → asserted in Task 5 markdown test; scope exclusions honoured (no bootstrap/calendar-time, no new data).
- **Leakage:** train-frozen weights implemented in `run_s4` (train_var computed once, applied to both train and test fits) and guarded by `test_time_holdout_weights_are_frozen_from_train`.
- **Type consistency:** `REGRESSORS`, `wls_fe_regression(..., weights=, regressors=, dv_col=)`, `firm_variance(..., firms=)`, `inverse_variance_weights(panel, h, variances)`, `_strength`, `_verdict`, `findings_markdown` names are used identically across tasks.
