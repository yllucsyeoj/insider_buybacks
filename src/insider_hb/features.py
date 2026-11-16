"""Price-derived features: trailing strength and forward excess returns. Pure."""
from __future__ import annotations

import pandas as pd

DEFAULT_HORIZONS = (21, 63, 126, 252)


def trailing_strength(close: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Fraction of the trailing `window` closes strictly below the current close.

    ~1.0 means the price is near the top of its trailing range (strength);
    ~0.0 means near the bottom.
    """
    return close.rolling(window, min_periods=min_periods).apply(
        lambda w: (w.iloc[-1] > w).mean(), raw=False
    )


def forward_excess_returns(
    firm_close: pd.Series,
    bench_close: pd.Series,
    horizons=DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Firm forward return minus benchmark forward return, per horizon (trading days).

    Aligns on the intersection of the two date indexes.
    """
    df = pd.DataFrame({"firm": firm_close, "bench": bench_close}).dropna()
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        fr = df["firm"].shift(-h) / df["firm"] - 1
        br = df["bench"].shift(-h) / df["bench"] - 1
        out[f"exc_{h}"] = fr - br
    return out
