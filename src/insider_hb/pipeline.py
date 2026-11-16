"""Compose fetch + pure features into a per-ticker sale-event table."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .fetch import fetch_form4_xml, fetch_prices
from .forms import parse_form4
from .normalize import normalize_title
from .classify import adoption_gap_days, classify_sale, is_opportunistic
from .features import trailing_strength, forward_excess_returns, DEFAULT_HORIZONS


@dataclass
class TickerData:
    ticker: str
    events: pd.DataFrame          # one row per officer Code-S transaction, with features
    price_features: pd.DataFrame  # daily strength + exc_* for the firm (baseline)


def _snap(idx: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    pos = idx.searchsorted(d)
    if pos == len(idx):
        return pd.NaT
    return idx[pos]


def build_ticker_data(
    ticker: str,
    benchmark: str = "SPY",
    limit: int = 200,
    horizons=DEFAULT_HORIZONS,
) -> TickerData:
    # --- officer Code-S sale events ---
    rows: list[dict] = []
    for acc, _fdate, xml in fetch_form4_xml(ticker, limit):
        f4 = parse_form4(xml, acc)
        if not f4.is_officer:
            continue
        for t in f4.transactions:
            if t.code != "S":
                continue
            gap = adoption_gap_days(t.date, f4.plan_adoption_date)
            sale_class = classify_sale(f4.aff10b5one, gap)
            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp(t.date),
                    "owner": f4.owner_name,
                    "role": normalize_title(f4.officer_title),
                    "shares": t.shares,
                    "price": t.price,
                    "value": t.shares * t.price,
                    "aff10b5one": f4.aff10b5one,
                    "plan_adoption": f4.plan_adoption_date,
                    "adoption_gap_days": gap,
                    "class": sale_class,
                    "opportunistic": is_opportunistic(sale_class),
                }
            )
    events = pd.DataFrame(rows)

    # --- daily price features ---
    firm = fetch_prices(ticker)
    bench = fetch_prices(benchmark)
    strength = trailing_strength(firm).rename("strength")
    exc = forward_excess_returns(firm, bench, horizons)
    price_features = pd.concat([strength, exc], axis=1)

    # --- attach features to events (snap each event to a trading day) ---
    if not events.empty:
        idx = price_features.index
        events["td"] = events["date"].map(lambda d: _snap(idx, d))
        events = events.join(price_features, on="td")

    return TickerData(ticker=ticker, events=events, price_features=price_features)
