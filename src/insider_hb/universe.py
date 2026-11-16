"""Load + validate the curated firm universe. Pure, no network."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

GOV_TIERS = {"concentrated", "mixed", "dispersed"}
SIZE_TIERS = {"mega", "large", "mid", "small"}
_REQUIRED = ["ticker", "gov_tier", "size_tier", "dual_class", "founder_ceo", "notes"]


@dataclass(frozen=True)
class Firm:
    ticker: str
    gov_tier: str
    size_tier: str
    dual_class: bool
    founder_ceo: bool
    notes: str


def _parse_bool(val: str, field: str, ticker: str) -> bool:
    v = val.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"{field} must be true/false for {ticker}, got {val!r}")


def load_universe(path: str | Path) -> list[Firm]:
    firms: list[Firm] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"universe.csv missing columns: {missing}")
        for row in reader:
            ticker = row["ticker"].strip()
            if not ticker:
                raise ValueError(f"empty ticker in row: {row}")
            gov = row["gov_tier"].strip()
            size = row["size_tier"].strip()
            if gov not in GOV_TIERS:
                raise ValueError(f"gov_tier {gov!r} for {ticker} not in {GOV_TIERS}")
            if size not in SIZE_TIERS:
                raise ValueError(f"size_tier {size!r} for {ticker} not in {SIZE_TIERS}")
            firms.append(
                Firm(
                    ticker=ticker,
                    gov_tier=gov,
                    size_tier=size,
                    dual_class=_parse_bool(row["dual_class"], "dual_class", ticker),
                    founder_ceo=_parse_bool(row["founder_ceo"], "founder_ceo", ticker),
                    notes=row["notes"].strip(),
                )
            )
    return firms
