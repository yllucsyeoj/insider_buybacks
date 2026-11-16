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


from insider_hb.pipeline import build_ticker_data


def test_build_ticker_data_uses_injected_fetchers():
    xml = (
        '<ownershipDocument><issuer><issuerCik>1</issuerCik>'
        '<issuerTradingSymbol>TST</issuerTradingSymbol></issuer>'
        '<reportingOwner><reportingOwnerId><rptOwnerName>Doe John</rptOwnerName></reportingOwnerId>'
        '<reportingOwnerRelationship><isOfficer>1</isOfficer>'
        '<officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>'
        '</reportingOwner><aff10b5One>false</aff10b5One>'
        '<nonDerivativeTable><nonDerivativeTransaction>'
        '<transactionDate><value>2022-01-03</value></transactionDate>'
        '<transactionCoding><transactionCode>S</transactionCode></transactionCoding>'
        '<transactionAmounts><transactionShares><value>100</value></transactionShares>'
        '<transactionPricePerShare><value>10</value></transactionPricePerShare>'
        '<transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>'
        '</transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>'
    )
    idx = pd.bdate_range("2021-01-01", periods=400)
    firm = pd.Series(range(1, 401), index=idx, dtype=float, name="TST")
    bench = pd.Series([100.0] * 400, index=idx, name="SPY")

    def fake_form4(ticker, limit):
        return [("acc-1", "2022-01-03", xml)]

    def fake_prices(ticker):
        return firm if ticker == "TST" else bench

    td = build_ticker_data(
        "TST", form4_fetcher=fake_form4, price_fetcher=fake_prices
    )
    assert len(td.events) == 1
    assert td.events.iloc[0]["class"] == "discretionary"
    assert "strength" in td.events.columns
