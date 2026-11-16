import pandas as pd
from insider_hb.pipeline import _snap


def _idx():
    return pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=5))  # Mon..Fri


def test_snap_exact_trading_day():
    idx = _idx()
    assert _snap(idx, pd.Timestamp("2020-01-08")) == pd.Timestamp("2020-01-08")


def test_snap_weekend_moves_forward():
    idx = _idx()
    # Saturday 2020-01-11 -> next trading day is not in this 5-day index (ends Fri 10th) -> beyond range -> NaT
    assert pd.isna(_snap(idx, pd.Timestamp("2020-01-11")))


def test_snap_midweek_gap_moves_forward():
    # index missing Wednesday; a Wednesday date snaps forward to Thursday
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07"),
                            pd.Timestamp("2020-01-09"), pd.Timestamp("2020-01-10")])
    assert _snap(idx, pd.Timestamp("2020-01-08")) == pd.Timestamp("2020-01-09")


def test_snap_before_range_snaps_to_first():
    idx = _idx()
    assert _snap(idx, pd.Timestamp("2019-12-01")) == pd.Timestamp("2020-01-06")


def test_snap_beyond_range_is_nat():
    idx = _idx()
    assert pd.isna(_snap(idx, pd.Timestamp("2025-01-01")))
