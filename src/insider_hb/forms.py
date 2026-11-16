"""Parse SEC Form 4 ownership XML into a typed structure. Pure, no network.

SEC ownership documents are namespace-free, so ElementTree simple paths work.
"""
from __future__ import annotations

import re
import datetime as dt
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class Txn:
    date: str
    code: str
    shares: float
    price: float
    acquired_disposed: str | None


@dataclass
class Form4:
    accession: str | None
    issuer_cik: str
    issuer_symbol: str
    owner_name: str
    is_officer: bool
    is_director: bool
    is_ten_percent: bool
    officer_title: str
    aff10b5one: bool
    plan_adoption_date: str | None
    transactions: list[Txn] = field(default_factory=list)


def _text(el: ET.Element, path: str, default: str | None = None) -> str | None:
    found = el.find(path)
    if found is not None and found.text is not None:
        return found.text.strip()
    return default


def _flag(root, path) -> bool:
    v = (_text(root, path, "0") or "0").strip().lower()
    return v in ("1", "true")


def _adopt_iso(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return dt.datetime.strptime(raw, "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


def parse_form4(xml: str, accession: str | None = None) -> Form4:
    root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)

    aff = _flag(root, "aff10b5One")

    adopt_raw = None
    for fn in root.findall("footnotes/footnote"):
        txt = fn.text or ""
        m = re.search(r"10b5-1[^<]*?adopted[^<]*?on\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", txt)
        if m:
            adopt_raw = m.group(1)
            break

    txns: list[Txn] = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(t, "transactionCoding/transactionCode")
        if code is None:
            continue
        shares = _text(t, "transactionAmounts/transactionShares/value", "0") or "0"
        price = _text(t, "transactionAmounts/transactionPricePerShare/value", "0") or "0"
        txns.append(
            Txn(
                date=_text(t, "transactionDate/value"),
                code=code,
                shares=float(shares),
                price=float(price),
                acquired_disposed=_text(t, "transactionAmounts/transactionAcquiredDisposedCode/value"),
            )
        )

    rel = "reportingOwner/reportingOwnerRelationship"
    return Form4(
        accession=accession,
        issuer_cik=_text(root, "issuer/issuerCik", "") or "",
        issuer_symbol=_text(root, "issuer/issuerTradingSymbol", "") or "",
        owner_name=_text(root, "reportingOwner/reportingOwnerId/rptOwnerName", "") or "",
        is_officer=_flag(root, f"{rel}/isOfficer"),
        is_director=_flag(root, f"{rel}/isDirector"),
        is_ten_percent=_flag(root, f"{rel}/isTenPercentOwner"),
        officer_title=_text(root, f"{rel}/officerTitle", "") or "",
        aff10b5one=aff,
        plan_adoption_date=_adopt_iso(adopt_raw),
        transactions=txns,
    )
