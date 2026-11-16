"""CLI: build and persist the sale-event table for one ticker, print H-B summary.

Usage: python scripts/build_events.py AAPL [SPY]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from insider_hb.pipeline import build_ticker_data
from insider_hb.analysis import hb_summary


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/build_events.py TICKER [BENCHMARK]")
        raise SystemExit(2)
    ticker = sys.argv[1].upper()
    bench = sys.argv[2].upper() if len(sys.argv) > 2 else "SPY"

    td = build_ticker_data(ticker, benchmark=bench)
    Path("data").mkdir(exist_ok=True)
    out = Path("data") / f"{ticker}_sale_events.parquet"
    td.events.to_parquet(out)

    if td.events.empty or "strength" not in td.events.columns:
        print(f"{ticker}: 0 officer Code-S sales -> {out} (no summary)")
        return

    summary = hb_summary(td.events.dropna(subset=["strength"]), td.price_features)
    print(f"{ticker}: {len(td.events)} officer Code-S sales -> {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
