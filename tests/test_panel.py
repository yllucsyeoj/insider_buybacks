import numpy as np
import pandas as pd
from insider_hb.universe import Firm
from insider_hb.pipeline import build_ticker_data
from insider_hb.panel import attach_governance, assemble_panel, PANEL_COLUMNS, GOV_COLS

def _events():
    return pd.DataFrame(
        {
            "ticker": ["TST", "TST"],
            "date": pd.to_datetime(["2022-01-03", "2022-02-03"]),
            "owner": ["Doe John", "Doe John"],
            "role": ["CEO", "CEO"],
            "shares": [100.0, 200.0],
            "price": [10.0, 11.0],
            "value": [1000.0, 2200.0],
            "aff10b5one": [False, True],
            "plan_adoption": [None, "2021-12-01"],
            "adoption_gap_days": [None, 64],
            "class": ["discretionary", "opportunistic_plan"],
            "opportunistic": [True, True],
            "td": pd.to_datetime(["2022-01-03", "2022-02-03"]),
            "strength": [0.9, 0.8],
            "exc_21": [-0.01, -0.02],
            "exc_63": [-0.03, np.nan],
            "exc_126": [np.nan, np.nan],
            "exc_252": [np.nan, np.nan],
        }
    )

FIRM = Firm("TST", "concentrated", "mid", True, True, "test")

def test_attach_governance_adds_columns_and_drops_td():
    out = attach_governance(_events(), FIRM, insider_own_pct=0.25)
    assert "td" not in out.columns
    assert list(out["gov_tier"].unique()) == ["concentrated"]
    assert list(out["size_tier"].unique()) == ["mid"]
    assert bool(out["dual_class"].iloc[0]) is True
    assert bool(out["founder_ceo"].iloc[0]) is True
    assert out["insider_own_pct"].iloc[0] == 0.25

def test_attach_governance_handles_none_pct():
    out = attach_governance(_events(), FIRM, insider_own_pct=None)
    assert out["insider_own_pct"].isna().all()

def test_attach_governance_empty_events():
    empty = pd.DataFrame(columns=_events().columns)
    out = attach_governance(empty, FIRM, insider_own_pct=0.1)
    assert out.empty
    assert all(c in out.columns for c in GOV_COLS)

def test_assemble_panel_concats_and_orders_columns():
    a = attach_governance(_events(), FIRM, 0.25)
    b = attach_governance(_events(), Firm("OOO", "dispersed", "mega", False, False, "x"), 0.01)
    panel = assemble_panel([a, b])
    assert len(panel) == 4
    assert list(panel.columns) == PANEL_COLUMNS
    assert set(panel["gov_tier"]) == {"concentrated", "dispersed"}

def test_assemble_panel_skips_empty_frames():
    a = attach_governance(_events(), FIRM, 0.25)
    empty = attach_governance(pd.DataFrame(columns=_events().columns), FIRM, None)
    panel = assemble_panel([a, empty])
    assert len(panel) == 2

def test_assemble_panel_all_empty_returns_schema_only():
    panel = assemble_panel([])
    assert panel.empty
    assert list(panel.columns) == PANEL_COLUMNS

def test_panel_columns_match_pipeline_output():
    """Guard against silent column drift: the S1 portion of PANEL_COLUMNS must
    equal what build_ticker_data actually emits (minus the `td` helper). If this
    fails, the pipeline schema changed and PANEL_COLUMNS/reindex would silently
    NaN-fill or drop data in the cold build."""
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
    td = build_ticker_data(
        "TST",
        form4_fetcher=lambda t, l: [("acc-1", "2022-01-03", xml)],
        price_fetcher=lambda t: firm if t == "TST" else bench,
    )
    emitted = [c for c in td.events.columns if c != "td"]
    expected = [c for c in PANEL_COLUMNS if c not in GOV_COLS]
    assert emitted == expected
