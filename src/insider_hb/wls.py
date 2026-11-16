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
