"""Pure assembly of per-firm sale-event tables into the cross-firm panel.

No network, no disk — orchestration lives in scripts/build_panel.py.
"""
from __future__ import annotations

import pandas as pd

from .universe import Firm

GOV_COLS = ["gov_tier", "size_tier", "dual_class", "founder_ceo", "insider_own_pct"]

# S1 event columns (minus the `td` helper) followed by governance columns.
PANEL_COLUMNS = [
    "ticker", "date", "owner", "role", "shares", "price", "value",
    "aff10b5one", "plan_adoption", "adoption_gap_days", "class", "opportunistic",
    "strength", "exc_21", "exc_63", "exc_126", "exc_252",
] + GOV_COLS


def attach_governance(
    events: pd.DataFrame, firm: Firm, insider_own_pct: float | None
) -> pd.DataFrame:
    """Drop the `td` helper and append the firm's governance columns to every row."""
    out = events.drop(columns=["td"], errors="ignore").copy()
    out["gov_tier"] = firm.gov_tier
    out["size_tier"] = firm.size_tier
    out["dual_class"] = firm.dual_class
    out["founder_ceo"] = firm.founder_ceo
    out["insider_own_pct"] = insider_own_pct
    return out


def assemble_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-firm frames into one panel with a stable column order."""
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    panel = pd.concat(non_empty, ignore_index=True)
    return panel.reindex(columns=PANEL_COLUMNS)
