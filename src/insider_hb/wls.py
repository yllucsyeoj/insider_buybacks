"""Variance-robust (WLS) estimators for the H-B strength channel.

Inverse-variance weighting stops a handful of hyper-variance firms from
dominating an equal-weight OLS fit (the S3 artifact). All functions are pure:
they take an already-derived panel (regressor columns present) and return
estimates. Firm-clustered SEs via statsmodels, consistent with S3 regression.py.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

REGRESSORS = ["strength", "discretionary", "opp_plan", "log_value"]


def firm_variance(panel: pd.DataFrame, h: int, *,
                  firms: Iterable[str] | None = None) -> dict[str, float]:
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
        if w.isna().any():  # NaN weights silently corrupt sm.WLS -> fail loud
            raise ValueError(
                f"weights missing for {int(w.isna().sum())} of {len(df)} rows")
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
