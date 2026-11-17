"""Pre-registered holdout splits for S4. Split constants live in the spec;
callers pass them explicitly. Deterministic firm split via a fixed seed."""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(panel: pd.DataFrame,
               cutoff: str = "2024-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) = (date < cutoff, date >= cutoff)."""
    d = pd.to_datetime(panel["date"])
    cut = pd.Timestamp(cutoff)
    return panel[d < cut].copy(), panel[d >= cut].copy()


def firm_split(panel: pd.DataFrame,
               seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Firm-disjoint halves: permute sorted unique tickers with a fixed seed,
    split at the midpoint. Deterministic for a given (firm set, seed)."""
    firms = sorted(panel["ticker"].unique())
    perm = np.random.default_rng(seed).permutation(firms)
    mid = len(perm) // 2
    a, b = set(perm[:mid]), set(perm[mid:])
    return panel[panel["ticker"].isin(a)].copy(), panel[panel["ticker"].isin(b)].copy()
