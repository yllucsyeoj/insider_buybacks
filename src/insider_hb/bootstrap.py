"""Cluster block bootstrap for panel statistics. Pure (numpy RNG, no I/O).

Resamples whole clusters (firms, or firm x calendar-month) with replacement so
that within-cluster correlation (one firm's boom-bust = many correlated events)
cannot masquerade as independent observations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _cluster_keys(panel: pd.DataFrame, cluster: str) -> pd.Series:
    if cluster == "ticker":
        return panel["ticker"].astype(str)
    if cluster == "firm_month":
        month = pd.to_datetime(panel["date"]).dt.to_period("M").astype(str)
        return panel["ticker"].astype(str) + "_" + month
    raise ValueError(f"unknown cluster {cluster!r}")


def bootstrap_statistic(
    panel: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "ticker",
    n: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap `stat_fn` by resampling clusters with replacement.

    Returns {estimate, ci_lo, ci_hi, p_value, n_boot}:
      - estimate: stat_fn on the full panel
      - ci_lo/ci_hi: 2.5/97.5 percentiles of the bootstrap distribution
      - p_value: two-sided, 2 x fraction of replicates on the opposite side of
        zero from the estimate, clipped to [0, 1]. Replicates exactly at zero
        count as "opposite" (conservative by at most 1/n_boot).
      - n_boot: number of non-NaN bootstrap replicates

    Intended scale: tens of clusters, n in the low thousands (the 50-firm panel
    runs in a few seconds). `firm_month` clustering on a long panel yields
    hundreds of clusters; still fine, but each replicate concatenates all of
    them, so cost grows with cluster count.
    """
    estimate = stat_fn(panel)
    keys = _cluster_keys(panel, cluster)
    groups = [panel.loc[idx] for idx in panel.groupby(keys).groups.values()]
    k = len(groups)
    if k == 0:
        return {"estimate": float(estimate), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    reps = np.empty(n)
    for i in range(n):
        picks = rng.integers(0, k, size=k)
        sample = pd.concat([groups[j] for j in picks], ignore_index=True)
        reps[i] = stat_fn(sample)
    reps = reps[~np.isnan(reps)]
    if len(reps) == 0:
        return {"estimate": float(estimate), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value": float("nan"), "n_boot": 0}
    ci_lo, ci_hi = np.percentile(reps, [2.5, 97.5])
    if np.isnan(estimate) or estimate == 0:
        p = 1.0  # sign is 0 -> formula undefined; an exact-zero estimate cannot reject H0
    else:
        frac_opposite = float(np.mean(reps * np.sign(estimate) <= 0))
        p = min(1.0, 2 * frac_opposite)
    return {
        "estimate": float(estimate),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p),
        "n_boot": int(len(reps)),
    }
