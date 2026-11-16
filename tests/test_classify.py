from insider_hb.classify import adoption_gap_days, classify_sale, is_opportunistic

def test_adoption_gap():
    assert adoption_gap_days("2025-06-24", "2024-09-25") == 272
    assert adoption_gap_days("2025-06-24", None) is None

def test_classify_discretionary_when_not_10b51():
    assert classify_sale(aff10b5one=False, gap_days=None) == "discretionary"
    assert classify_sale(aff10b5one=False, gap_days=500) == "discretionary"

def test_classify_plan_by_adoption_gap():
    # adopted just above the 90d cooling-off floor -> opportunistic
    assert classify_sale(aff10b5one=True, gap_days=100) == "opportunistic_plan"
    # long-seasoned plan -> routine
    assert classify_sale(aff10b5one=True, gap_days=272) == "routine_plan"
    # 10b5-1 flagged but adoption date unknown -> routine (conservative)
    assert classify_sale(aff10b5one=True, gap_days=None) == "routine_plan"

def test_is_opportunistic():
    assert is_opportunistic("discretionary") is True
    assert is_opportunistic("opportunistic_plan") is True
    assert is_opportunistic("routine_plan") is False
