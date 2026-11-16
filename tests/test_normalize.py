from insider_hb.normalize import normalize_title

def test_ceo_variants_collapse():
    assert normalize_title("CEO") == "CEO"
    assert normalize_title("Chief Executive Officer") == "CEO"

def test_cfo_double_space_and_prefix():
    assert normalize_title("Senior Vice President,  CFO") == "CFO"
    assert normalize_title("EVP, Principal Financial Offcr") == "CFO"

def test_other_roles():
    assert normalize_title("COO") == "COO"
    assert normalize_title("Chief Technology Officer") == "CTO"
    assert normalize_title("Principal Accounting Officer") == "CAO"
    assert normalize_title("Executive Chairman") == "Executive Chair"
    assert normalize_title("EVP, Chief Legal Officer") == "GC"

def test_unknown_officer_and_empty():
    assert normalize_title("Senior Vice President") == "Officer"
    assert normalize_title("") == ""
