"""Run the S4 variance-robust (WLS) strength inference on the sale-event panel.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_s4.py \
        [--panel data/panel.parquet] [--out-dir results] \
        [--time-cutoff 2024-01-01] [--firm-seed 7]

Three estimators per horizon (inverse-variance WLS [primary], equal-weight OLS
[conservative bound], per-firm z-scored DV [corroboration]); a leakage-free
time holdout (weights frozen from the train window) plus a secondary firm-half
split; a verdict; and results/s4_wls_inference.json + results/s4_wls_findings.md.

CAVEAT: the panel is already spent (2024+ was examined during exploration), so
the time holdout is a replication/stability check, not a pristine out-of-sample
test. See the spec section 2.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insider_hb.wls import (
    REGRESSORS, firm_variance, inverse_variance_weights, wls_fe_regression,
    firm_zscore_dv,
)
from insider_hb.holdout import time_split, firm_split

DEFAULT_HORIZONS = (21, 63, 126, 252)


def _derive(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the S4 regressor columns; drop non-positive value (log1p guard)."""
    df = panel[panel["value"] > 0].copy()
    df["discretionary"] = (df["class"] == "discretionary").astype(float)
    df["opp_plan"] = (df["class"] == "opportunistic_plan").astype(float)
    df["log_value"] = np.log1p(df["value"].astype(float))
    df["strength"] = df["strength"].astype(float)
    return df


def _strength(rows: list) -> dict:
    return next(r for r in rows if r["term"] == "strength")


def _wls_strength(df, h):
    var = firm_variance(df, h)
    w = inverse_variance_weights(df, h, var)
    return _strength(wls_fe_regression(df, h, weights=w))


def run_s4(panel, *, horizons=DEFAULT_HORIZONS,
           time_cutoff="2024-01-01", firm_seed=7) -> dict:
    """Pure core: derived panel in, results dict out."""
    df = _derive(panel)
    out: dict = {}
    for h in horizons:
        # full sample (in-sample variances; labelled as such in the writeup)
        full_var = firm_variance(df, h)
        w_full = inverse_variance_weights(df, h, full_var)
        wls_table = wls_fe_regression(df, h, weights=w_full)
        df[f"exc_{h}_z"] = firm_zscore_dv(df, h)
        full_sample = {
            "wls": _strength(wls_table),
            "ols": _strength(wls_fe_regression(df, h, weights=None)),
            "zscore": _strength(wls_fe_regression(df, h, weights=None,
                                                  dv_col=f"exc_{h}_z")),
            "wls_table": wls_table,
        }
        # time holdout: weights FROZEN from train, applied to both fits
        train, test = time_split(df, cutoff=time_cutoff)
        train_var = firm_variance(train, h)
        time_holdout = {
            "train": _strength(wls_fe_regression(
                train, h, weights=inverse_variance_weights(train, h, train_var))),
            "test": _strength(wls_fe_regression(
                test, h, weights=inverse_variance_weights(test, h, train_var))),
        }
        # firm holdout (secondary): each half uses its own in-sample variances
        half_a, half_b = firm_split(df, seed=firm_seed)
        firm_holdout = {"half_a": _wls_strength(half_a, h),
                        "half_b": _wls_strength(half_b, h)}
        out[str(h)] = {"full_sample": full_sample,
                       "time_holdout": time_holdout,
                       "firm_holdout": firm_holdout}
    return {
        "meta": {
            "n_rows": int(len(df)),
            "horizons": list(horizons),
            "time_cutoff": time_cutoff,
            "firm_seed": int(firm_seed),
            "generated": _dt.date.today().isoformat(),
        },
        "horizons": out,
        "verdict": _verdict(out),
    }


def _sig_neg(row: dict) -> bool:
    return row["coef"] < 0 and row["p"] < 0.05


def _verdict(horizons_out: dict) -> dict:
    by_h = {}
    for h, blk in horizons_out.items():
        c1 = _sig_neg(blk["full_sample"]["wls"])
        c2 = _sig_neg(blk["time_holdout"]["test"])
        c3 = (blk["firm_holdout"]["half_a"]["coef"] < 0
              and blk["firm_holdout"]["half_b"]["coef"] < 0)
        by_h[h] = {"robust": bool(c1 and c2 and c3),
                   "full_sample_sig_neg": bool(c1),
                   "test_window_sig_neg": bool(c2),
                   "firm_halves_sign_consistent": bool(c3)}
    return {"by_horizon": by_h,
            "focal_robust": bool(by_h.get("126", {}).get("robust", False))}
