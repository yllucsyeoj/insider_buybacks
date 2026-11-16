"""Pure point estimates over the sale-event panel. No inference, no I/O."""
from __future__ import annotations

import pandas as pd

GOV_TIERS = ("concentrated", "mixed", "dispersed")


def n_valid(panel: pd.DataFrame, h: int) -> int:
    """Count rows with a non-NaN forward excess return at horizon `h`."""
    return int(panel[f"exc_{h}"].notna().sum())


def pooled_mean(panel: pd.DataFrame, h: int, weight: str = "equal") -> float:
    """Mean forward excess return at horizon `h`. `weight` is 'equal' or 'value'."""
    col = f"exc_{h}"
    df = panel[[col, "value"]].dropna(subset=[col])
    if df.empty:
        return float("nan")
    if weight == "value":
        w = df["value"]
        total = float(w.sum())
        if total == 0:
            return float("nan")
        return float((df[col] * w).sum() / total)
    if weight != "equal":
        raise ValueError(f"unknown weight {weight!r}")
    return float(df[col].mean())


def opportunistic_contrast(panel: pd.DataFrame, h: int) -> float:
    """mean(exc_h | opportunistic) - mean(exc_h | routine). NaN if either side empty."""
    col = f"exc_{h}"
    df = panel.dropna(subset=[col])
    opp = df.loc[df["opportunistic"].astype(bool), col]
    rou = df.loc[~df["opportunistic"].astype(bool), col]
    if opp.empty or rou.empty:
        return float("nan")
    return float(opp.mean() - rou.mean())


def gov_tier_means(panel: pd.DataFrame, h: int) -> dict[str, float]:
    """Mean exc_h per governance tier. Empty tier -> NaN (not an error)."""
    col = f"exc_{h}"
    df = panel.dropna(subset=[col])
    out: dict[str, float] = {}
    for tier in GOV_TIERS:
        sub = df.loc[df["gov_tier"] == tier, col]
        out[tier] = float(sub.mean()) if len(sub) else float("nan")
    return out
