import numpy as np
import pandas as pd
from insider_hb.analysis import hb_summary

def test_hb_summary_basic():
    # events with strength + exc columns; baseline frame of all days
    events = pd.DataFrame({
        "value": [100.0, 300.0],
        "strength": [0.8, 0.9],
        "exc_21": [-0.02, -0.10],
        "exc_252": [-0.05, -0.25],
    })
    exc_all = pd.DataFrame({
        "exc_21": [0.0, 0.01, 0.02, np.nan],
        "exc_252": [0.03, 0.05, np.nan, np.nan],
    })
    out = hb_summary(events, exc_all, horizons=(21, 252))
    assert abs(out["strength_sale"] - 0.85) < 1e-9
    # baseline = nanmean of exc_all
    assert abs(out["base"][21] - 0.01) < 1e-9
    assert abs(out["base"][252] - 0.04) < 1e-9
    # equal-weight sale mean
    assert abs(out["sale_eq"][21] - (-0.06)) < 1e-9
    # value-weighted: (100*-0.02 + 300*-0.10)/400 = -0.08
    assert abs(out["sale_vw"][21] - (-0.08)) < 1e-9
