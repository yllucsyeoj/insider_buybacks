import numpy as np
import pandas as pd
from insider_hb.calendar_time import daily_portfolio_excess, calendar_time_test


def _prices():
    idx = pd.bdate_range("2022-01-03", periods=6)   # Mon Jan 3 .. Mon Jan 10
    # firm A: +1% per day; firm B: +3% per day; SPY: +1% per day
    a = pd.Series(100.0 * 1.01 ** np.arange(6), index=idx, name="A")
    b = pd.Series(100.0 * 1.03 ** np.arange(6), index=idx, name="B")
    spy = pd.Series(100.0 * 1.01 ** np.arange(6), index=idx, name="SPY")
    return {"A": a, "B": b}, spy


def test_half_open_window_and_equal_weight():
    prices, spy = _prices()
    idx = prices["A"].index
    # A sells on day0; B sells on day0. Window h=2 -> days 1 and 2 (strictly after).
    panel = pd.DataFrame({"ticker": ["A", "B"], "date": [idx[0], idx[0]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=2)
    # day0 excluded (strictly after sale); days 1 and 2 present
    assert list(daily.index) == [idx[1], idx[2]]
    # excess is the ARITHMETIC return difference: A = 0.01-0.01 = 0; B = 0.03-0.01 = 0.02
    # equal-weight mean = (0 + 0.02) / 2 = 0.01
    assert abs(daily.iloc[0] - 0.01) < 1e-9


def test_window_length_capped_at_h_trading_days():
    prices, spy = _prices()
    idx = prices["A"].index
    panel = pd.DataFrame({"ticker": ["B"], "date": [idx[0]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=3)
    assert len(daily) == 3  # exactly h trading days after the sale


def test_unknown_ticker_skipped():
    prices, spy = _prices()
    idx = prices["A"].index
    panel = pd.DataFrame({"ticker": ["ZZZ"], "date": [idx[0]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=2)
    assert daily.empty


def test_empty_panel_returns_empty_series():
    prices, spy = _prices()
    panel = pd.DataFrame({"ticker": [], "date": []})
    daily = daily_portfolio_excess(panel, prices, spy, h=2)
    assert daily.empty


def test_missing_spy_day_is_skipped():
    prices, spy = _prices()
    idx = prices["A"].index
    spy = spy.drop(idx[1])          # SPY has no data on the first post-sale day
    panel = pd.DataFrame({"ticker": ["B"], "date": [idx[0]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=2)
    # day1 dropped (no SPY); day2 present
    assert idx[1] not in daily.index
    assert idx[2] in daily.index


def test_calendar_time_test_returns_finite_stats():
    prices, spy = _prices()
    idx = prices["A"].index
    panel = pd.DataFrame({"ticker": ["B", "B"], "date": [idx[0], idx[1]]})
    daily = daily_portfolio_excess(panel, prices, spy, h=3)
    res = calendar_time_test(daily, maxlags=3)
    assert res["n_days"] == len(daily)
    assert np.isfinite(res["mean_daily_excess"])
    assert np.isfinite(res["t_stat"])
    assert res["mean_daily_excess"] > 0  # B outperforms SPY


def test_calendar_time_test_handles_too_few_days():
    res = calendar_time_test(pd.Series([0.01]), maxlags=3)
    assert res["n_days"] == 1
    assert np.isnan(res["t_stat"])
