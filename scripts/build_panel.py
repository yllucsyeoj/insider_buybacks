"""Build the cross-firm sale-event panel.

Usage:
    EDGAR_IDENTITY="Name email" PYTHONPATH=src .venv/bin/python scripts/build_panel.py [--force] [--limit N]

Resumable: each firm's events persist to data/cache/panel/{ticker}_events.parquet
and completed firms are skipped unless --force. Writes data/panel.parquet + prints
a JSON coverage summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from insider_hb.universe import load_universe
from insider_hb.cache import get_or_fetch_form4_xml, get_or_fetch_prices
from insider_hb.fetch import fetch_insider_own_pct
from insider_hb.pipeline import build_ticker_data
from insider_hb.panel import attach_governance, assemble_panel, PANEL_COLUMNS

UNIVERSE = Path("data/universe.csv")
PANEL_OUT = Path("data/panel.parquet")
PER_FIRM_DIR = Path("data/cache/panel")
RETRY_DELAY_S = 5.0


def _build_one(firm, limit: int) -> pd.DataFrame:
    td = build_ticker_data(
        firm.ticker,
        limit=limit,
        form4_fetcher=get_or_fetch_form4_xml,
        price_fetcher=get_or_fetch_prices,
    )
    pct = fetch_insider_own_pct(firm.ticker)
    return attach_governance(td.events, firm, pct)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild all firms")
    ap.add_argument("--limit", type=int, default=200, help="Form 4 filings per firm")
    args = ap.parse_args()

    PER_FIRM_DIR.mkdir(parents=True, exist_ok=True)
    firms = load_universe(UNIVERSE)
    failed: list[str] = []

    for firm in firms:
        out_path = PER_FIRM_DIR / f"{firm.ticker}_events.parquet"
        if out_path.exists() and not args.force:
            continue
        try:
            events = _build_one(firm, args.limit)
        except Exception:
            time.sleep(RETRY_DELAY_S)
            try:
                events = _build_one(firm, args.limit)
            except Exception as e:  # noqa: BLE001
                failed.append(firm.ticker)
                sys.stderr.write(f"FAILED {firm.ticker}: {e}\n")
                continue
        events.to_parquet(out_path)

    frames = [pd.read_parquet(p) for p in sorted(PER_FIRM_DIR.glob("*_events.parquet"))]
    panel = assemble_panel(frames) if frames else pd.DataFrame(columns=PANEL_COLUMNS)
    panel.to_parquet(PANEL_OUT)

    horizons = ["exc_21", "exc_63", "exc_126", "exc_252"]
    summary = {
        "n_firms": len(firms),
        "n_firms_failed": len(failed),
        "failed": failed,
        "n_events": int(len(panel)),
        "events_by_gov_tier": (
            panel["gov_tier"].value_counts().to_dict() if not panel.empty else {}
        ),
        "horizon_coverage": {
            h: (float(panel[h].notna().mean()) if not panel.empty else 0.0)
            for h in horizons
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
