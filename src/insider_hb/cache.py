"""Disk cache over the Form 4 / price fetchers. Keeps the cold panel build
re-runnable and resilient to EDGAR throttling. Pure I/O, no business logic."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .fetch import fetch_form4_xml, fetch_prices

CACHE_DIR = Path("data/cache")


def _fresh(fetched_iso: str, max_age_days: int) -> bool:
    try:
        age = datetime.now() - datetime.fromisoformat(fetched_iso)
    except (ValueError, TypeError):
        return False
    return age.days < max_age_days


def get_or_fetch_form4_xml(
    ticker: str,
    limit: int = 200,
    *,
    fetcher: Callable[[str, int], list[tuple[str, str, str]]] = fetch_form4_xml,
    cache_dir: Path = CACHE_DIR,
    max_age_days: int = 7,
) -> list[tuple[str, str, str]]:
    """Return [(accession, filing_date, xml)], served from disk when fresh."""
    d = Path(cache_dir) / "form4"
    d.mkdir(parents=True, exist_ok=True)
    manifest = d / f"{ticker}.json"

    if manifest.exists():
        man = json.loads(manifest.read_text())
        if _fresh(man.get("fetched", ""), max_age_days):
            out = []
            for acc, fdate in man["accessions"]:
                xml = (d / f"{acc}.xml").read_text()
                out.append((acc, fdate, xml))
            return out

    rows = fetcher(ticker, limit)
    for acc, _fdate, xml in rows:
        (d / f"{acc}.xml").write_text(xml)
    manifest.write_text(
        json.dumps(
            {
                "fetched": datetime.now().isoformat(),
                "accessions": [[acc, fdate] for acc, fdate, _ in rows],
            }
        )
    )
    return rows


def get_or_fetch_prices(
    ticker: str,
    *,
    fetcher: Callable[[str], pd.Series] = fetch_prices,
    cache_dir: Path = CACHE_DIR,
    max_age_days: int = 7,
) -> pd.Series:
    """Return a price Series, served from disk when fresh."""
    d = Path(cache_dir) / "prices"
    d.mkdir(parents=True, exist_ok=True)
    data_path = d / f"{ticker}.parquet"
    meta_path = d / f"{ticker}.meta.json"

    if data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if _fresh(meta.get("fetched", ""), max_age_days):
            s = pd.read_parquet(data_path)["value"]
            s.name = ticker
            return s

    s = fetcher(ticker)
    s.to_frame("value").to_parquet(data_path)
    meta_path.write_text(json.dumps({"fetched": datetime.now().isoformat()}))
    return s
