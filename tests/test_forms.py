from pathlib import Path
from insider_hb.forms import parse_form4

FIX = Path(__file__).parent / "fixtures"

def _load(name):
    return (FIX / name).read_text()

def test_parse_10b51_sale_fields():
    f = parse_form4(_load("form4_catz_10b51.xml"), accession="0001127602-25-018029")
    assert f.accession == "0001127602-25-018029"
    assert f.issuer_symbol == "ORCL"
    assert f.owner_name == "CATZ SAFRA"
    assert f.is_officer is True
    assert f.is_director is True
    assert f.is_ten_percent is False
    assert f.officer_title == "Chief Executive Officer"
    assert f.aff10b5one is True
    assert f.plan_adoption_date == "2024-09-25"
    assert len(f.transactions) == 1
    t = f.transactions[0]
    assert t.code == "S"
    assert t.date == "2025-06-24"
    assert t.shares == 1000000.0
    assert t.price == 200.50
    assert t.acquired_disposed == "D"

def test_parse_discretionary_sale():
    f = parse_form4(_load("form4_discretionary.xml"))
    assert f.accession is None
    assert f.owner_name == "MAESTRI LUCA"
    assert f.is_officer is True
    assert f.is_director is False
    assert f.aff10b5one is False
    assert f.plan_adoption_date is None
    assert f.transactions[0].shares == 50000.0
