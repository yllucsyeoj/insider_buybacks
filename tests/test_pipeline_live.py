import os
import pytest
import pandas as pd

pytestmark = pytest.mark.live

@pytest.fixture(autouse=True)
def _identity():
    os.environ.setdefault("EDGAR_IDENTITY", "Joey Scully scully.joey@gmail.com")

def test_fetch_form4_xml_smoke():
    from insider_hb.fetch import fetch_form4_xml
    rows = fetch_form4_xml("AAPL", limit=5)
    assert len(rows) >= 1
    acc, date, xml = rows[0]
    assert "<ownershipDocument" in xml

def test_fetch_prices_smoke():
    from insider_hb.fetch import fetch_prices
    s = fetch_prices("AAPL", period="1y")
    assert len(s) > 100
    assert s.index.tz is None

def test_build_ticker_data_aapl():
    from insider_hb.pipeline import build_ticker_data
    td = build_ticker_data("AAPL", limit=150)
    ev = td.events
    assert not ev.empty
    # expected columns present
    for col in ["owner", "role", "value", "class", "opportunistic", "strength", "exc_252"]:
        assert col in ev.columns
    # officers sell into strength: mean trailing percentile should be high
    assert ev["strength"].dropna().mean() > 0.55
    # at least one CEO/CFO sale parsed
    assert ev["role"].isin(["CEO", "CFO", "COO", "CAO"]).any()

def test_build_ticker_data_orcl_catz_present():
    from insider_hb.pipeline import build_ticker_data
    td = build_ticker_data("ORCL", limit=200)
    ev = td.events
    assert ev["owner"].str.upper().str.contains("CATZ").any()
    # Catz's largest sales should be flagged 10b5-1
    catz = ev[ev["owner"].str.upper().str.contains("CATZ")]
    assert catz["aff10b5one"].any()
