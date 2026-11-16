import numpy as np
import pandas as pd
from insider_hb.regression import fe_regression


def _synthetic(beta_opp=-0.08, seed=0):
    """Panel where opportunistic lowers exc by beta_opp, plus firm-level shifts.

    Eight firms (-> eight clusters for stable cluster-robust SEs) and genuinely
    varying strength/value (so neither collides with the intercept).
    """
    rng = np.random.default_rng(seed)
    shifts = {f"F{i}": float(rng.normal(0, 0.05)) for i in range(8)}
    rows = []
    for firm, shift in shifts.items():
        for m in range(1, 7):
            for opp in (True, False):
                exc = shift + beta_opp * opp + rng.normal(0, 0.005)
                rows.append({
                    "ticker": firm,
                    "date": pd.Timestamp(f"2022-{m:02d}-05"),
                    "value": float(rng.integers(100_000, 10_000_000)),
                    "opportunistic": opp,
                    "strength": float(rng.uniform(0.3, 0.95)),
                    "exc_63": exc,
                })
    return pd.DataFrame(rows)


def test_recovers_opportunistic_coefficient():
    table = fe_regression(_synthetic(beta_opp=-0.08), 63)
    opp = next(r for r in table if r["term"] == "opportunistic")
    assert abs(opp["coef"] - (-0.08)) < 0.01
    assert opp["ci_hi"] < 0            # significantly negative


def test_returns_only_the_three_regressors():
    table = fe_regression(_synthetic(), 63)
    assert sorted(r["term"] for r in table) == ["log_value", "opportunistic", "strength"]


def test_each_row_has_full_stat_fields():
    row = fe_regression(_synthetic(), 63)[0]
    assert set(row) == {"term", "coef", "se", "t", "p", "ci_lo", "ci_hi"}


def test_drops_nan_rows():
    df = _synthetic()
    df.loc[df.index[:5], "exc_63"] = np.nan
    table = fe_regression(df, 63)        # must not raise
    opp = next(r for r in table if r["term"] == "opportunistic")
    assert abs(opp["coef"] - (-0.08)) < 0.015   # still recovers the effect


def test_non_positive_value_rows_excluded():
    df = _synthetic()
    df.loc[df.index[:4], "value"] = 0.0        # $0 rows must not contaminate log_value
    df.loc[df.index[4:8], "value"] = -5.0      # negative must not leak NaN into the design
    table = fe_regression(df, 63)              # must not raise
    opp = next(r for r in table if r["term"] == "opportunistic")
    assert np.isfinite(opp["coef"])
