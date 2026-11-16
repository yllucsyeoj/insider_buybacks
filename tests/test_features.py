import numpy as np
import pandas as pd
from insider_hb.features import trailing_strength, forward_excess_returns

def test_trailing_strength_monotonic_series():
    # strictly increasing -> each day is the max of its trailing window -> strength ~1
    idx = pd.bdate_range("2020-01-01", periods=120)
    close = pd.Series(np.arange(1, 121, dtype=float), index=idx)
    s = trailing_strength(close, window=20, min_periods=20)
    # fraction of the 20-day window strictly below the last point = 19/20
    assert abs(s.iloc[-1] - 19 / 20) < 1e-9
    assert s.iloc[:19].isna().all()  # min_periods not met

def test_forward_excess_returns():
    idx = pd.bdate_range("2020-01-01", periods=5)
    firm = pd.Series([100, 110, 121, 133.1, 146.41], index=idx)   # +10%/day
    bench = pd.Series([100, 105, 110.25, 115.76, 121.55], index=idx)  # +5%/day
    out = forward_excess_returns(firm, bench, horizons=(1,))
    # day0 -> day1: firm +10%, bench +5% -> excess +5%
    assert abs(out["exc_1"].iloc[0] - 0.05) < 1e-6
    # last row has no +1 forward value
    assert np.isnan(out["exc_1"].iloc[-1])

def test_forward_excess_aligns_on_common_dates():
    idx = pd.bdate_range("2020-01-01", periods=4)
    firm = pd.Series([10, 11, 12, 13], index=idx, dtype=float)
    bench = pd.Series([10, 10, 10], index=idx[:3], dtype=float)  # shorter
    out = forward_excess_returns(firm, bench, horizons=(1,))
    assert len(out) == 3  # intersection only
