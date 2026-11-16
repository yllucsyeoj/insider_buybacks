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


def test_wls_raises_on_incomplete_weights():
    import pytest
    df = _reg_panel(seed=4)
    w = pd.Series(1.0, index=df.index[:-5])       # weights miss some rows -> NaN on reindex
    with pytest.raises(ValueError, match="weights missing"):
        wls_fe_regression(df, 126, weights=w)
