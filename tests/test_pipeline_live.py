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
