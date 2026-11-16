import json
from datetime import datetime, timedelta
import pandas as pd
from insider_hb import cache

def test_form4_miss_then_hit(tmp_path):
    calls = {"n": 0}
    def fake(ticker, limit):
        calls["n"] += 1
        return [("acc-1", "2025-01-01", "<x>1</x>"), ("acc-2", "2025-01-02", "<x>2</x>")]

    out1 = cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path)
    out2 = cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path)
    assert calls["n"] == 1                       # second call served from disk
    assert out1 == out2
    assert out2[0] == ("acc-1", "2025-01-01", "<x>1</x>")
    assert (tmp_path / "form4" / "acc-1.xml").exists()

def test_form4_refetch_when_stale(tmp_path):
    calls = {"n": 0}
    def fake(ticker, limit):
        calls["n"] += 1
        return [("acc-1", "2025-01-01", "<x>1</x>")]

    cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path)
    man = tmp_path / "form4" / "TST.json"
    data = json.loads(man.read_text())
    data["fetched"] = (datetime.now() - timedelta(days=30)).isoformat()
    man.write_text(json.dumps(data))
    cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path, max_age_days=7)
    assert calls["n"] == 2                        # stale manifest triggered refetch

def test_prices_miss_then_hit(tmp_path):
    calls = {"n": 0}
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-01-01", "2025-01-02"]), name="TST")
    def fake(ticker):
        calls["n"] += 1
        return s

    out1 = cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path)
    out2 = cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path)
    assert calls["n"] == 1
    pd.testing.assert_series_equal(out1, out2)
    assert out2.name == "TST"

def test_prices_refetch_when_stale(tmp_path):
    calls = {"n": 0}
    s = pd.Series([1.0], index=pd.to_datetime(["2025-01-01"]), name="TST")
    def fake(ticker):
        calls["n"] += 1
        return s

    cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path)
    meta = tmp_path / "prices" / "TST.meta.json"
    meta.write_text(json.dumps({"fetched": (datetime.now() - timedelta(days=30)).isoformat()}))
    cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path, max_age_days=7)
    assert calls["n"] == 2
