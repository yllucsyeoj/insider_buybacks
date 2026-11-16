from pathlib import Path
import pytest
from insider_hb.universe import Firm, load_universe

def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "u.csv"
    p.write_text("ticker,gov_tier,size_tier,dual_class,founder_ceo,notes\n" + body)
    return p

def test_loads_valid_rows(tmp_path):
    p = _write(tmp_path, "AAPL,dispersed,mega,false,false,widely held\nMETA,concentrated,mega,true,true,founder\n")
    firms = load_universe(p)
    assert firms == [
        Firm("AAPL", "dispersed", "mega", False, False, "widely held"),
        Firm("META", "concentrated", "mega", True, True, "founder"),
    ]

def test_bad_gov_tier_raises(tmp_path):
    p = _write(tmp_path, "AAPL,sideways,mega,false,false,x\n")
    with pytest.raises(ValueError, match="gov_tier"):
        load_universe(p)

def test_bad_size_tier_raises(tmp_path):
    p = _write(tmp_path, "AAPL,dispersed,huge,false,false,x\n")
    with pytest.raises(ValueError, match="size_tier"):
        load_universe(p)

def test_bad_bool_raises(tmp_path):
    p = _write(tmp_path, "AAPL,dispersed,mega,yes,false,x\n")
    with pytest.raises(ValueError, match="dual_class"):
        load_universe(p)

def test_empty_ticker_raises(tmp_path):
    p = _write(tmp_path, ",dispersed,mega,false,false,x\n")
    with pytest.raises(ValueError, match="empty ticker"):
        load_universe(p)

def test_firm_is_frozen():
    f = Firm("AAPL", "dispersed", "mega", False, False, "x")
    with pytest.raises(Exception):
        f.ticker = "MSFT"  # type: ignore[misc]

def test_curated_universe_file_is_valid():
    firms = load_universe(Path("data/universe.csv"))
    assert len(firms) >= 40
    tiers = {f.gov_tier for f in firms}
    assert {"concentrated", "mixed", "dispersed"} <= tiers
