# Insider H-B — S1 Single-Firm Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, tested per-ticker pipeline that turns a stock ticker into a table of officer open-market sale events enriched with 10b5-1 status, plan-adoption timing, trailing-price strength, and forward excess returns — validated on AAPL and ORCL.

**Architecture:** Pure logic (XML parsing, title normalization, sale classification, price features) lives in small single-responsibility modules unit-tested with fixtures and synthetic data — no network. Thin I/O adapters (`fetch.py`) wrap `edgartools` (Form 4) and `yfinance` (prices) and are covered by live smoke/integration tests marked `@pytest.mark.live`. A `pipeline.py` orchestrator composes them into a sale-event feature table. This mirrors — but does not depend on — the sidecar `sec` extension used during reconnaissance.

**Tech Stack:** Python 3.13, `edgartools`, `yfinance`, `pandas`, `numpy`, `pyarrow`, `pytest`.

**Scope (from spec §8, S1 only):** H-B features — Form 4 (Code S, isOfficer, 10b5-1 + adoption date), price/benchmark, strength + forward-excess features. Buyback/repurchase (H-A) extraction is explicitly out of scope for S1.

---

## File Structure

```
insider_buybacks/
├── requirements.txt                 # pinned deps
├── pyproject.toml                   # package + pytest config
├── .gitignore
├── src/insider_hb/
│   ├── __init__.py
│   ├── forms.py                     # parse Form 4 XML -> Form4 dataclass (pure)
│   ├── normalize.py                 # officer-title normalization (pure)
│   ├── classify.py                  # 10b5-1 adoption gap + sale classification (pure)
│   ├── features.py                  # trailing strength + forward excess returns (pure)
│   ├── fetch.py                     # edgartools + yfinance I/O adapters (network)
│   ├── pipeline.py                  # orchestrator: ticker -> TickerData bundle (network)
│   └── analysis.py                  # H-B summary: baseline vs post-sale (pure)
├── tests/
│   ├── fixtures/
│   │   ├── form4_catz_10b51.xml     # officer, Code S, 10b5-1, adoption footnote
│   │   └── form4_discretionary.xml  # officer, Code S, no 10b5-1
│   ├── test_forms.py
│   ├── test_normalize.py
│   ├── test_classify.py
│   ├── test_features.py
│   ├── test_analysis.py
│   └── test_pipeline_live.py        # @pytest.mark.live (network)
└── scripts/
    └── build_events.py              # CLI: ticker -> data/<ticker>_sale_events.parquet
```

Each module has one responsibility; pure modules are importable and testable without network or secrets.

---

## Task 0: Scaffold repo, venv, deps, package skeleton

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.gitignore`, `src/insider_hb/__init__.py`

- [ ] **Step 1: Initialize git**

Run:
```bash
cd /Users/joey/Projects/insider_buybacks
git init
```
Expected: `Initialized empty Git repository ...`

- [ ] **Step 2: Create `.gitignore`**

Create `.gitignore`:
```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/
.env
.sidecar/
*.egg-info/
```

Note: `.env` holds live API keys — it must stay ignored.

- [ ] **Step 3: Create `requirements.txt`**

Create `requirements.txt`:
```text
edgartools>=2.0
yfinance>=0.2.40
pandas>=2.0
numpy>=1.26
pyarrow>=15
pytest>=8.0
```

- [ ] **Step 4: Create `pyproject.toml`**

Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "insider_hb"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = [
    "live: tests that hit the network (SEC EDGAR, Yahoo Finance); require EDGAR_IDENTITY",
]
addopts = "-m 'not live'"
```

`addopts = "-m 'not live'"` means `pytest` runs only offline unit tests by default; live tests run explicitly with `pytest -m live`.

- [ ] **Step 5: Create venv and install**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Expected: installs complete without error.

- [ ] **Step 6: Create package init**

Create `src/insider_hb/__init__.py`:
```python
"""Insider H-B single-firm pipeline."""
```

- [ ] **Step 7: Verify pytest collects (no tests yet)**

Run:
```bash
.venv/bin/pytest -q
```
Expected: `no tests ran` (exit code 5) — confirms config loads.

- [ ] **Step 8: Commit**

```bash
git add .gitignore requirements.txt pyproject.toml src/insider_hb/__init__.py
git commit -m "chore: scaffold insider_hb package, venv, pytest config"
```

---

## Task 1: Form 4 XML parser (`forms.py`)

**Files:**
- Create: `tests/fixtures/form4_catz_10b51.xml`, `tests/fixtures/form4_discretionary.xml`, `tests/test_forms.py`, `src/insider_hb/forms.py`

- [ ] **Step 1: Create fixture `tests/fixtures/form4_catz_10b51.xml`**

```xml
<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2025-06-24</periodOfReport>
  <aff10b5One>true</aff10b5One>
  <issuer>
    <issuerCik>0001341439</issuerCik>
    <issuerName>ORACLE CORP</issuerName>
    <issuerTradingSymbol>ORCL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001134287</rptOwnerCik>
      <rptOwnerName>CATZ SAFRA</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2025-06-24</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>S</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000000</value></transactionShares>
        <transactionPricePerShare><value>200.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>3000000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes>
    <footnote id="F1">This transaction was made pursuant to a Rule 10b5-1 trading plan adopted by the reporting person on September 25, 2024.</footnote>
  </footnotes>
</ownershipDocument>
```

- [ ] **Step 2: Create fixture `tests/fixtures/form4_discretionary.xml`**

```xml
<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2024-11-10</periodOfReport>
  <aff10b5One>false</aff10b5One>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>Apple Inc.</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214128</rptOwnerCik>
      <rptOwnerName>MAESTRI LUCA</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Senior Vice President, CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-11-10</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>S</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50000</value></transactionShares>
        <transactionPricePerShare><value>225.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
```

- [ ] **Step 3: Write failing tests `tests/test_forms.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_forms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.forms'`

- [ ] **Step 5: Implement `src/insider_hb/forms.py`**

```python
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


def _text(el, path, default=None):
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

    aff_el = root.find("aff10b5One")
    aff = (aff_el.text or "").strip().lower() in ("1", "true") if aff_el is not None else False

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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_forms.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures src/insider_hb/forms.py tests/test_forms.py
git commit -m "feat: Form 4 XML parser with 10b5-1 flag and adoption date"
```

---

## Task 2: Officer-title normalization (`normalize.py`)

**Files:**
- Create: `tests/test_normalize.py`, `src/insider_hb/normalize.py`

- [ ] **Step 1: Write failing tests `tests/test_normalize.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.normalize'`

- [ ] **Step 3: Implement `src/insider_hb/normalize.py`**

```python
"""Canonicalize free-text Form 4 officer titles into coarse role labels. Pure."""
from __future__ import annotations

import re

# Order matters: more specific patterns first.
_ROLE_MAP: list[tuple[str, str]] = [
    (r"chief executive|\bceo\b", "CEO"),
    (r"chief financial|principal financial|\bcfo\b", "CFO"),
    (r"chief operating|\bcoo\b", "COO"),
    (r"chief technology|\bcto\b", "CTO"),
    (r"principal accounting|chief accounting", "CAO"),
    (r"general counsel|chief legal", "GC"),
    (r"executive chair", "Executive Chair"),
    (r"\bpresident\b", "President"),
]


def normalize_title(title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip()).lower()
    if not t:
        return ""
    for pattern, label in _ROLE_MAP:
        if re.search(pattern, t):
            return label
    return "Officer"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_normalize.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/normalize.py tests/test_normalize.py
git commit -m "feat: officer title normalization"
```

---

## Task 3: Sale classification (`classify.py`)

**Files:**
- Create: `tests/test_classify.py`, `src/insider_hb/classify.py`

- [ ] **Step 1: Write failing tests `tests/test_classify.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.classify'`

- [ ] **Step 3: Implement `src/insider_hb/classify.py`**

```python
"""Classify officer sales by 10b5-1 status and plan-adoption timing. Pure.

Opportunistic = discretionary (no plan) OR a 10b5-1 plan adopted close to the
90-day cooling-off floor (adopted recently relative to the trade). Routine = a
seasoned 10b5-1 plan. Plans with unknown adoption date are treated as routine
(conservative: we do not accuse without evidence).
"""
from __future__ import annotations

import pandas as pd

COOLING_OFF_DAYS = 90
OPPORTUNISTIC_WINDOW = 150  # plan adopted < this many days before trade -> opportunistic


def adoption_gap_days(trade_date: str, plan_adoption_date: str | None) -> int | None:
    if not plan_adoption_date:
        return None
    return int((pd.Timestamp(trade_date) - pd.Timestamp(plan_adoption_date)).days)


def classify_sale(aff10b5one: bool, gap_days: int | None) -> str:
    if not aff10b5one:
        return "discretionary"
    if gap_days is not None and gap_days < OPPORTUNISTIC_WINDOW:
        return "opportunistic_plan"
    return "routine_plan"


def is_opportunistic(sale_class: str) -> bool:
    return sale_class in ("discretionary", "opportunistic_plan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_classify.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/classify.py tests/test_classify.py
git commit -m "feat: 10b5-1 adoption-gap sale classification"
```

---

## Task 4: Price features (`features.py`)

**Files:**
- Create: `tests/test_features.py`, `src/insider_hb/features.py`

- [ ] **Step 1: Write failing tests `tests/test_features.py`**

```python
import numpy as np
import pandas as pd
from insider_hb.features import trailing_strength, forward_excess_returns

def test_trailing_strength_monotonic_series():
    # strictly increasing -> each day is the max of its trailing window -> strength ~1
    idx = pd.bdate_range("2020-01-01", periods=120)
    close = pd.Series(np.arange(1, 121, dtype=float), index=idx)
    s = trailing_strength(close, window=20, min_periods=20)
    # fraction of the 20-day window strictly below the last point = 19/20
    assert abs(s.iloc[-1] - 19 / 20) < 1e-9
    assert s.iloc[:19].isna().all()  # min_periods not met

def test_forward_excess_returns():
    idx = pd.bdate_range("2020-01-01", periods=5)
    firm = pd.Series([100, 110, 121, 133.1, 146.41], index=idx)   # +10%/day
    bench = pd.Series([100, 105, 110.25, 115.76, 121.55], index=idx)  # +5%/day
    out = forward_excess_returns(firm, bench, horizons=(1,))
    # day0 -> day1: firm +10%, bench +5% -> excess +5%
    assert abs(out["exc_1"].iloc[0] - 0.05) < 1e-6
    # last row has no +1 forward value
    assert np.isnan(out["exc_1"].iloc[-1])

def test_forward_excess_aligns_on_common_dates():
    idx = pd.bdate_range("2020-01-01", periods=4)
    firm = pd.Series([10, 11, 12, 13], index=idx, dtype=float)
    bench = pd.Series([10, 10, 10], index=idx[:3], dtype=float)  # shorter
    out = forward_excess_returns(firm, bench, horizons=(1,))
    assert len(out) == 3  # intersection only
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.features'`

- [ ] **Step 3: Implement `src/insider_hb/features.py`**

```python
"""Price-derived features: trailing strength and forward excess returns. Pure."""
from __future__ import annotations

import pandas as pd

DEFAULT_HORIZONS = (21, 63, 126, 252)


def trailing_strength(close: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Fraction of the trailing `window` closes strictly below the current close.

    ~1.0 means the price is near the top of its trailing range (strength);
    ~0.0 means near the bottom.
    """
    return close.rolling(window, min_periods=min_periods).apply(
        lambda w: (w.iloc[-1] > w).mean(), raw=False
    )


def forward_excess_returns(
    firm_close: pd.Series,
    bench_close: pd.Series,
    horizons=DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Firm forward return minus benchmark forward return, per horizon (trading days).

    Aligns on the intersection of the two date indexes.
    """
    df = pd.DataFrame({"firm": firm_close, "bench": bench_close}).dropna()
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        fr = df["firm"].shift(-h) / df["firm"] - 1
        br = df["bench"].shift(-h) / df["bench"] - 1
        out[f"exc_{h}"] = fr - br
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_features.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/features.py tests/test_features.py
git commit -m "feat: trailing strength + forward excess return features"
```

---

## Task 5: H-B summary (`analysis.py`)

**Files:**
- Create: `tests/test_analysis.py`, `src/insider_hb/analysis.py`

- [ ] **Step 1: Write failing tests `tests/test_analysis.py`**

```python
import numpy as np
import pandas as pd
from insider_hb.analysis import hb_summary

def test_hb_summary_basic():
    # events with strength + exc columns; baseline frame of all days
    events = pd.DataFrame({
        "value": [100.0, 300.0],
        "strength": [0.8, 0.9],
        "exc_21": [-0.02, -0.10],
        "exc_252": [-0.05, -0.25],
    })
    exc_all = pd.DataFrame({
        "exc_21": [0.0, 0.01, 0.02, np.nan],
        "exc_252": [0.03, 0.05, np.nan, np.nan],
    })
    out = hb_summary(events, exc_all, horizons=(21, 252))
    assert abs(out["strength_sale"] - 0.85) < 1e-9
    # baseline = nanmean of exc_all
    assert abs(out["base"][21] - 0.01) < 1e-9
    assert abs(out["base"][252] - 0.04) < 1e-9
    # equal-weight sale mean
    assert abs(out["sale_eq"][21] - (-0.06)) < 1e-9
    # value-weighted: (100*-0.02 + 300*-0.10)/400 = -0.08
    assert abs(out["sale_vw"][21] - (-0.08)) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'insider_hb.analysis'`

- [ ] **Step 3: Implement `src/insider_hb/analysis.py`**

```python
"""Descriptive H-B summary: post-sale forward excess returns vs baseline. Pure.

NOTE: this is descriptive only. The forward-return windows overlap and the
sample is one firm, so these numbers carry NO honest p-value. Proper inference
(calendar-time portfolios / block bootstrap on a panel) is S3, not S1.
"""
from __future__ import annotations

import pandas as pd

from .features import DEFAULT_HORIZONS


def hb_summary(events: pd.DataFrame, exc_all: pd.DataFrame, horizons=DEFAULT_HORIZONS) -> dict:
    w = events["value"] / events["value"].sum()
    base = {h: float(exc_all[f"exc_{h}"].mean()) for h in horizons}
    sale_eq = {h: float(events[f"exc_{h}"].mean()) for h in horizons}
    sale_vw = {h: float((events[f"exc_{h}"] * w).sum()) for h in horizons}
    return {
        "n_events": int(len(events)),
        "strength_sale": float(events["strength"].mean()),
        "base": base,
        "sale_eq": sale_eq,
        "sale_vw": sale_vw,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_analysis.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/analysis.py tests/test_analysis.py
git commit -m "feat: descriptive H-B summary (baseline vs post-sale)"
```

---

## Task 6: I/O adapters (`fetch.py`)

**Files:**
- Create: `src/insider_hb/fetch.py`, and the live portion of `tests/test_pipeline_live.py`

- [ ] **Step 1: Implement `src/insider_hb/fetch.py`**

```python
"""Network I/O adapters over edgartools (Form 4) and yfinance (prices).

Kept thin and side-effect-isolated so the pure modules stay testable offline.
"""
from __future__ import annotations

import os

import pandas as pd
import yfinance as yf
from edgar import Company, set_identity


def ensure_identity() -> None:
    ident = os.environ.get("EDGAR_IDENTITY")
    if not ident:
        raise RuntimeError(
            "Set EDGAR_IDENTITY='Your Name your@email' before fetching from EDGAR"
        )
    set_identity(ident)


def fetch_form4_xml(ticker: str, limit: int = 200) -> list[tuple[str, str, str]]:
    """Return [(accession, filing_date, xml_text)] for the latest `limit` Form 4s."""
    ensure_identity()
    co = Company(ticker)
    out: list[tuple[str, str, str]] = []
    for f in co.get_filings(form="4").head(limit):
        xdoc = next(
            (a for a in f.attachments if str(a.document).lower().endswith(".xml")), None
        )
        if xdoc is None:
            continue
        out.append((f.accession_no, str(f.filing_date), xdoc.text()))
    return out


def fetch_prices(ticker: str, period: str = "6y") -> pd.Series:
    """Return a tz-naive daily adjusted-close Series named by ticker."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s
```

- [ ] **Step 2: Write a live smoke test in `tests/test_pipeline_live.py`**

```python
import os
import pytest
import pandas as pd

pytestmark = pytest.mark.live

@pytest.fixture(autouse=True)
def _identity():
    os.environ.setdefault("EDGAR_IDENTITY", "Joey Scully scully.joey@gmail.com")

def test_fetch_form4_xml_smoke():
    from insider_hb.fetch import fetch_form4_xml
    rows = fetch_form4_xml("AAPL", limit=5)
    assert len(rows) >= 1
    acc, date, xml = rows[0]
    assert "<ownershipDocument" in xml

def test_fetch_prices_smoke():
    from insider_hb.fetch import fetch_prices
    s = fetch_prices("AAPL", period="1y")
    assert len(s) > 100
    assert s.index.tz is None
```

- [ ] **Step 3: Run the live smoke tests**

Run: `.venv/bin/pytest -m live tests/test_pipeline_live.py -v`
Expected: 2 passed (requires internet + EDGAR_IDENTITY)

- [ ] **Step 4: Confirm offline suite still green**

Run: `.venv/bin/pytest -q`
Expected: all non-live tests pass; live tests deselected.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/fetch.py tests/test_pipeline_live.py
git commit -m "feat: edgartools + yfinance I/O adapters with live smoke tests"
```

---

## Task 7: Pipeline orchestrator (`pipeline.py`)

**Files:**
- Create: `src/insider_hb/pipeline.py`, extend `tests/test_pipeline_live.py`

- [ ] **Step 1: Implement `src/insider_hb/pipeline.py`**

```python
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
    return idx[min(pos, len(idx) - 1)]


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
```

- [ ] **Step 2: Add a live integration test to `tests/test_pipeline_live.py`**

```python
def test_build_ticker_data_aapl():
    from insider_hb.pipeline import build_ticker_data
    td = build_ticker_data("AAPL", limit=150)
    ev = td.events
    assert not ev.empty
    # expected columns present
    for col in ["owner", "role", "value", "class", "opportunistic", "strength", "exc_252"]:
        assert col in ev.columns
    # officers sell into strength: mean trailing percentile should be high
    assert ev["strength"].dropna().mean() > 0.55
    # at least one CEO/CFO sale parsed
    assert ev["role"].isin(["CEO", "CFO", "COO", "CAO"]).any()

def test_build_ticker_data_orcl_catz_present():
    from insider_hb.pipeline import build_ticker_data
    td = build_ticker_data("ORCL", limit=200)
    ev = td.events
    assert ev["owner"].str.upper().str.contains("CATZ").any()
    # Catz's largest sales should be flagged 10b5-1
    catz = ev[ev["owner"].str.upper().str.contains("CATZ")]
    assert catz["aff10b5one"].any()
```

- [ ] **Step 3: Run the live integration tests**

Run: `.venv/bin/pytest -m live tests/test_pipeline_live.py -v`
Expected: all live tests pass (network; ~1–2 min for the Form 4 fetches).

- [ ] **Step 4: Confirm offline suite still green**

Run: `.venv/bin/pytest -q`
Expected: offline tests pass; live deselected.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/pipeline.py tests/test_pipeline_live.py
git commit -m "feat: per-ticker pipeline composing fetch + features into sale events"
```

---

## Task 8: CLI + validation on AAPL & ORCL

**Files:**
- Create: `scripts/build_events.py`

- [ ] **Step 1: Implement `scripts/build_events.py`**

```python
"""CLI: build and persist the sale-event table for one ticker, print H-B summary.

Usage: python scripts/build_events.py AAPL [SPY]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from insider_hb.pipeline import build_ticker_data
from insider_hb.analysis import hb_summary


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/build_events.py TICKER [BENCHMARK]")
        raise SystemExit(2)
    ticker = sys.argv[1].upper()
    bench = sys.argv[2].upper() if len(sys.argv) > 2 else "SPY"

    td = build_ticker_data(ticker, benchmark=bench)
    Path("data").mkdir(exist_ok=True)
    out = Path("data") / f"{ticker}_sale_events.parquet"
    td.events.to_parquet(out)

    summary = hb_summary(td.events.dropna(subset=["strength"]), td.price_features)
    print(f"{ticker}: {len(td.events)} officer Code-S sales -> {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on AAPL**

Run:
```bash
EDGAR_IDENTITY="Joey Scully scully.joey@gmail.com" .venv/bin/python scripts/build_events.py AAPL
```
Expected: prints a sale count, writes `data/AAPL_sale_events.parquet`, and a JSON summary with `strength_sale` ≈ 0.7–0.8 and negative `sale_eq` at the 126/252 horizons (matching reconnaissance: ~0.77 strength, ~−2% at 252d).

- [ ] **Step 3: Run on ORCL**

Run:
```bash
EDGAR_IDENTITY="Joey Scully scully.joey@gmail.com" .venv/bin/python scripts/build_events.py ORCL
```
Expected: writes `data/ORCL_sale_events.parquet`; summary shows `strength_sale` ≈ 0.77 and strongly negative `sale_eq`/`sale_vw` at 252d (reconnaissance: ~−25%). Confirms the pipeline reproduces the hand-run recon numbers.

- [ ] **Step 4: Validation gate — assert reproduction**

Run:
```bash
.venv/bin/python - <<'PY'
import pandas as pd
a = pd.read_parquet("data/AAPL_sale_events.parquet")
o = pd.read_parquet("data/ORCL_sale_events.parquet")
assert a["strength"].dropna().mean() > 0.6, "AAPL strength too low"
assert o["strength"].dropna().mean() > 0.6, "ORCL strength too low"
assert o[o["owner"].str.upper().str.contains("CATZ")]["value"].sum() > 1e9, "Catz $ missing"
print("validation OK:",
      "AAPL n=", len(a), "strength=", round(a['strength'].mean(),2),
      "| ORCL n=", len(o), "strength=", round(o['strength'].mean(),2))
PY
```
Expected: `validation OK: ...` with both strength means ≈ 0.7+ and Catz total > $1B.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_events.py
git commit -m "feat: CLI to build per-ticker sale-event tables; validated on AAPL+ORCL"
```

---

## Self-Review Notes (addressed during authoring)

- **Spec coverage:** Form 4 Code-S + isOfficer (Task 1,7), 10b5-1 flag + adoption date (Task 1), opportunistic/routine split (Task 3), trailing strength + forward excess (Task 4), per-firm descriptive summary (Task 5), validated on AAPL+ORCL (Task 8). Buyback/H-A features intentionally excluded (spec §8 YAGNI for S1).
- **Inference honesty:** `analysis.py` docstring and the spec's §6 caveat are carried into the code so no one mistakes the descriptive summary for a test.
- **Type consistency:** `Form4`/`Txn` fields, `build_ticker_data`→`TickerData.events/price_features`, and `hb_summary(events, exc_all)` signatures are consistent across tasks.
- **No placeholders:** every code/test step contains complete, runnable content.

## Known follow-ups (NOT S1)

- Live tests depend on current EDGAR/yfinance data; the AAPL/ORCL assertions use loose thresholds (strength > 0.55/0.6) so they stay robust as new filings arrive.
- `fetch_form4_xml(limit=200)` caps history; S2 replaces this with async jobs + disk cache for deep/wide pulls.
- S2/S3: panel build + calendar-time-portfolio / block-bootstrap inference.
