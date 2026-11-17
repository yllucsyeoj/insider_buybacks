import pandas as pd
from insider_hb.holdout import time_split, firm_split


def _panel():
    return pd.DataFrame({
        "ticker": ["A", "A", "B", "B", "C", "D"],
        "date": pd.to_datetime(["2022-06-01", "2024-03-01", "2023-01-01",
                                "2025-07-01", "2024-01-01", "2021-12-31"]),
        "exc_126": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })


def test_time_split_partitions_on_cutoff():
    train, test = time_split(_panel(), cutoff="2024-01-01")
    assert (train["date"] < pd.Timestamp("2024-01-01")).all()
    assert (test["date"] >= pd.Timestamp("2024-01-01")).all()
    assert len(train) + len(test) == 6          # exact partition, no loss
    assert len(train) == 3 and len(test) == 3    # 2024-01-01 falls in test


def test_firm_split_is_firm_disjoint_and_deterministic():
    a1, b1 = firm_split(_panel(), seed=7)
    a2, b2 = firm_split(_panel(), seed=7)
    firms_a, firms_b = set(a1["ticker"]), set(b1["ticker"])
    assert firms_a.isdisjoint(firms_b)                       # no shared firms
    assert firms_a | firms_b == {"A", "B", "C", "D"}         # covers all firms
    assert set(a2["ticker"]) == firms_a and set(b2["ticker"]) == firms_b  # deterministic
    # golden value: pin the pre-registered seed=7 partition (guards against an
    # RNG-behavior change silently re-partitioning the holdout)
    assert firms_a == {"A", "C"} and firms_b == {"B", "D"}
