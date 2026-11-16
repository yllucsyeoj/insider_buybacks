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
    # (100*-0.02 + 300*-0.10 + 100*0.04) / 500 = -0.28/5 (scaled by /100)
    assert abs(pooled_mean(_panel(), 21, weight="value") - (-0.28 / 5)) < 1e-9


def test_pooled_mean_all_nan_returns_nan():
    p = _panel().assign(exc_21=float("nan"))
    assert np.isnan(pooled_mean(p, 21))
    assert np.isnan(pooled_mean(p, 21, weight="value"))


def test_pooled_mean_equal_without_value_column():
    p = _panel().drop(columns=["value"])
    assert abs(pooled_mean(p, 21) - (-0.08 / 3)) < 1e-9


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
