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
