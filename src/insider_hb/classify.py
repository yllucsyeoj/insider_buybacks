"""Classify officer sales by 10b5-1 status and plan-adoption timing. Pure.

Opportunistic = discretionary (no plan) OR a 10b5-1 plan adopted close to the
90-day cooling-off floor (adopted recently relative to the trade). Routine = a
seasoned 10b5-1 plan. Plans with unknown adoption date are treated as routine
(conservative: we do not accuse without evidence).
"""
from __future__ import annotations

import pandas as pd

COOLING_OFF_DAYS = 90
OPPORTUNISTIC_WINDOW = 150  # plan adopted < this many days before trade -> opportunistic


def adoption_gap_days(trade_date: str, plan_adoption_date: str | None) -> int | None:
    if not plan_adoption_date:
        return None
    return int((pd.Timestamp(trade_date) - pd.Timestamp(plan_adoption_date)).days)


def classify_sale(aff10b5one: bool, gap_days: int | None) -> str:
    if not aff10b5one:
        return "discretionary"
    if gap_days is not None and gap_days < OPPORTUNISTIC_WINDOW:
        return "opportunistic_plan"
    return "routine_plan"


def is_opportunistic(sale_class: str) -> bool:
    return sale_class in ("discretionary", "opportunistic_plan")
