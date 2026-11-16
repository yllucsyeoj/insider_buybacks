"""Network I/O adapters over edgartools (Form 4) and yfinance (prices).

Kept thin and side-effect-isolated so the pure modules stay testable offline.
"""
from __future__ import annotations

import os

import pandas as pd
import yfinance as yf
from edgar import Company, set_identity


def ensure_identity() -> None:
    ident = os.environ.get("EDGAR_IDENTITY")
    if not ident:
        raise RuntimeError(
            "Set EDGAR_IDENTITY='Your Name your@email' before fetching from EDGAR"
        )
    set_identity(ident)


def fetch_form4_xml(ticker: str, limit: int = 200) -> list[tuple[str, str, str]]:
    """Return [(accession, filing_date, xml_text)] for the latest `limit` Form 4s."""
    ensure_identity()
    co = Company(ticker)
    out: list[tuple[str, str, str]] = []
    for f in co.get_filings(form="4").head(limit):
        xdoc = next(
            (a for a in f.attachments if str(a.document).lower().endswith(".xml")), None
        )
        if xdoc is None:
            continue
        out.append((f.accession_no, str(f.filing_date), xdoc.text()))
    return out


def fetch_prices(ticker: str, period: str = "6y") -> pd.Series:
    """Return a tz-naive daily adjusted-close Series named by ticker."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s
