import os
import pytest
import pandas as pd

from insider_hb.universe import Firm
from insider_hb.cache import get_or_fetch_form4_xml, get_or_fetch_prices
from insider_hb.fetch import fetch_insider_own_pct
from insider_hb.pipeline import build_ticker_data
from insider_hb.panel import attach_governance, assemble_panel, PANEL_COLUMNS

pytestmark = pytest.mark.live

FIRMS = [
    Firm("AAPL", "dispersed", "mega", False, False, "recon baseline"),
    Firm("ORCL", "concentrated", "mega", False, False, "recon case"),
]


@pytest.mark.skipif(not os.environ.get("EDGAR_IDENTITY"), reason="EDGAR_IDENTITY unset")
def test_two_firm_panel_end_to_end(tmp_path):
    frames = []
    for firm in FIRMS:
        td = build_ticker_data(
            firm.ticker,
            limit=60,
            form4_fetcher=lambda t, l, cd=tmp_path: get_or_fetch_form4_xml(t, l, cache_dir=cd),
            price_fetcher=lambda t, cd=tmp_path: get_or_fetch_prices(t, cache_dir=cd),
        )
        frames.append(attach_governance(td.events, firm, fetch_insider_own_pct(firm.ticker)))

    panel = assemble_panel(frames)
    assert list(panel.columns) == PANEL_COLUMNS
    assert set(panel["ticker"]) == {"AAPL", "ORCL"}
    assert len(panel) > 0
    assert panel["strength"].dropna().between(0, 1).all()
