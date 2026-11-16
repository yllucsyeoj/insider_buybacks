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
