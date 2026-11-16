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
