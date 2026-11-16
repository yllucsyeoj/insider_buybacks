"""Panel fixed-effects OLS for H-B. statsmodels with firm-clustered SEs.

exc_h ~ opportunistic + strength + log_value, with firm and year-month fixed
effects (drop-first dummies). SEs cluster-robust by firm. Returns a tidy table
of the three non-FE regressors.

Scale/caveats: dense `get_dummies` + OLS is fine for the ~50-firm panel
(hundreds of dummy columns on ~9k rows); a much larger universe would want a
within-transform (e.g. linearmodels.PanelOLS). Cluster-robust inference with
~50 firm clusters is near the conventional floor — read the SEs with that in
mind, especially on sliced sub-samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

TERMS = ["opportunistic", "strength", "log_value"]


def fe_regression(panel: pd.DataFrame, h: int) -> list[dict]:
    col = f"exc_{h}"
    df = panel.dropna(subset=[col, "strength", "value"]).copy()
    df = df[df["value"] > 0].copy()  # log1p undefined/contaminating for non-positive value
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
