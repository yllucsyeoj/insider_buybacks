"""Descriptive H-B summary: post-sale forward excess returns vs baseline. Pure.

NOTE: this is descriptive only. The forward-return windows overlap and the
sample is one firm, so these numbers carry NO honest p-value. Proper inference
(calendar-time portfolios / block bootstrap on a panel) is S3, not S1.
"""
from __future__ import annotations

import pandas as pd

from .features import DEFAULT_HORIZONS


def hb_summary(events: pd.DataFrame, exc_all: pd.DataFrame, horizons=DEFAULT_HORIZONS) -> dict:
    w = events["value"] / events["value"].sum()
    base = {h: float(exc_all[f"exc_{h}"].mean()) for h in horizons}
    sale_eq = {h: float(events[f"exc_{h}"].mean()) for h in horizons}
    sale_vw = {h: float((events[f"exc_{h}"] * w).sum()) for h in horizons}
    return {
        "n_events": int(len(events)),
        "strength_sale": float(events["strength"].mean()),
        "base": base,
        "sale_eq": sale_eq,
        "sale_vw": sale_vw,
    }
