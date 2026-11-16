# S2 Panel Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sale-event panel (`data/panel.parquet`, one row per officer Code-S sale) across a curated ~45-firm governance/size universe, by running the S1 per-ticker pipeline behind a resumable disk cache.

**Architecture:** A hand-maintained `data/universe.csv` drives a resumable CLI orchestrator (`scripts/build_panel.py`). Per firm it runs the S1 `build_ticker_data` (now accepting injected fetchers) through a disk cache (`cache.py`), attaches governance columns (`panel.py` pure helpers), and persists a per-firm parquet; completed firms are skipped on rerun. A final concat produces the panel. The S1 10b5-1 adoption-date regex is widened first since it feeds the opportunistic/routine split.

**Tech Stack:** Python 3, pandas, pyarrow, edgartools, yfinance, pytest. Pure logic offline-testable with fixtures; network I/O behind `@pytest.mark.live`.

**Spec:** `docs/superpowers/specs/2026-11-16-insider-hb-s2-panel-design.md`

---

## File Structure

| File | Create/Modify | Responsibility |
|------|---------------|----------------|
| `src/insider_hb/forms.py` | Modify | Widen 10b5-1 adoption-date regex + multi-format date parsing |
| `data/universe.csv` | Create | Curated firm list with hand governance/size tiers |
| `src/insider_hb/universe.py` | Create | Load + validate `universe.csv` → `list[Firm]` (pure) |
| `src/insider_hb/cache.py` | Create | Disk cache: Form 4 XML (per accession) + prices (per ticker), staleness-aware |
| `src/insider_hb/pipeline.py` | Modify | Dependency-inject fetchers into `build_ticker_data` |
| `src/insider_hb/panel.py` | Create | Pure assembly: `attach_governance`, `assemble_panel` |
| `src/insider_hb/fetch.py` | Modify | Add `fetch_insider_own_pct` (best-effort, live) |
| `scripts/build_panel.py` | Create | Resumable CLI orchestrator → `data/panel.parquet` + JSON summary |
| `tests/test_forms.py` | Modify | Fixtures for alternate 10b5-1 phrasings |
| `tests/test_universe.py` | Create | Universe loader + validation tests |
| `tests/test_cache.py` | Create | get-or-fetch miss/hit/staleness tests |
| `tests/test_panel.py` | Create | Governance attach + assemble tests |
| `tests/test_panel_live.py` | Create | `@pytest.mark.live` 2-firm end-to-end |
| `tests/fixtures/form4_10b51_entered_iso.xml` | Create | Fixture: "entered into on YYYY-MM-DD" |
| `tests/fixtures/form4_10b51_slashdate.xml` | Create | Fixture: "adopted on MM/DD/YYYY" |

**Note on event-table columns:** The S1 `build_ticker_data().events` table has columns `ticker, date, owner, role, shares, price, value, aff10b5one, plan_adoption, adoption_gap_days, class, opportunistic, td, strength, exc_21, exc_63, exc_126, exc_252`. S2 **does not rename** these — it drops the `td` helper column and appends `gov_tier, size_tier, dual_class, founder_ceo, insider_own_pct`.

---

## Task 0: Branch

**Files:** none (git only)

- [ ] **Step 1: Create the S2 feature branch from main**

```bash
git checkout main
git checkout -b feat/s2-panel
```

- [ ] **Step 2: Confirm the cache + data dirs are gitignored**

Run: `git check-ignore data/cache/x.parquet data/panel.parquet`
Expected: both paths printed (they match the existing `data/` rule in `.gitignore`). If nothing prints, add `data/` to `.gitignore` and commit.

---

## Task 1: Widen the 10b5-1 adoption-date regex

**Why:** The S1 regex only matches `... adopted ... on <Month DD, YYYY>`. Plans phrased "entered into on" or dated `YYYY-MM-DD` / `MM/DD/YYYY` are missed → opportunistic plans undercounted. This is the publishable-edge signal.

**Files:**
- Modify: `src/insider_hb/forms.py`
- Test: `tests/test_forms.py`
- Create: `tests/fixtures/form4_10b51_entered_iso.xml`, `tests/fixtures/form4_10b51_slashdate.xml`

- [ ] **Step 1: Write the two new fixtures**

`tests/fixtures/form4_10b51_entered_iso.xml`:

```xml
<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerCik>0000000001</issuerCik><issuerTradingSymbol>TST</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Doe John</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <aff10b5One>true</aff10b5One>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2025-06-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes>
    <footnote>This transaction was made pursuant to a Rule 10b5-1 trading plan entered into on 2024-09-15.</footnote>
  </footnotes>
</ownershipDocument>
```

`tests/fixtures/form4_10b51_slashdate.xml` — identical but with this footnote:

```xml
  <footnotes>
    <footnote>Sale under a 10b5-1 plan adopted on 09/15/2024.</footnote>
  </footnotes>
```

- [ ] **Step 2: Write failing tests**

Add to `tests/test_forms.py`:

```python
from pathlib import Path
from insider_hb.forms import parse_form4

FIX = Path(__file__).parent / "fixtures"

def test_adoption_date_entered_into_iso():
    f4 = parse_form4((FIX / "form4_10b51_entered_iso.xml").read_text())
    assert f4.aff10b5one is True
    assert f4.plan_adoption_date == "2024-09-15"

def test_adoption_date_slash_format():
    f4 = parse_form4((FIX / "form4_10b51_slashdate.xml").read_text())
    assert f4.plan_adoption_date == "2024-09-15"
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_forms.py -k "entered_into_iso or slash_format" -v`
Expected: FAIL — `plan_adoption_date` is `None` (current regex/parser miss these).

- [ ] **Step 4: Widen the parser in `forms.py`**

Replace the `_adopt_iso` function and the footnote loop in `parse_form4`.

Replace `_adopt_iso` (lines 49-55) with:

```python
_DATE_FORMATS = [
    (re.compile(r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}"), "%B %d, %Y"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "%Y-%m-%d"),
    (re.compile(r"\d{1,2}/\d{1,2}/\d{4}"), "%m/%d/%Y"),
]

_ADOPT_TRIGGER = re.compile(r"10b5-1.*?(?:adopted|entered into)", re.IGNORECASE | re.DOTALL)


def _find_adoption_date(text: str) -> str | None:
    """Return ISO date of a 10b5-1 plan adoption mentioned in `text`, else None."""
    trig = _ADOPT_TRIGGER.search(text)
    if not trig:
        return None
    tail = text[trig.end():]
    for pat, fmt in _DATE_FORMATS:
        m = pat.search(tail)
        if m:
            try:
                return dt.datetime.strptime(m.group(0), fmt).date().isoformat()
            except ValueError:
                continue
    return None
```

Replace the footnote loop in `parse_form4` (lines 63-69) with:

```python
    adopt_iso = None
    for fn in root.findall("footnotes/footnote"):
        adopt_iso = _find_adoption_date(fn.text or "")
        if adopt_iso:
            break
```

And change the `Form4(...)` field (line 99) from `plan_adoption_date=_adopt_iso(adopt_raw),` to:

```python
        plan_adoption_date=adopt_iso,
```

- [ ] **Step 5: Run the full forms test suite, verify pass**

Run: `.venv/bin/pytest tests/test_forms.py -v`
Expected: PASS — the two new tests plus all pre-existing forms tests (the original `form4_catz_10b51.xml` "adopted on <Month DD, YYYY>" case still resolves via the first date format).

- [ ] **Step 6: Commit**

```bash
git add src/insider_hb/forms.py tests/test_forms.py tests/fixtures/form4_10b51_entered_iso.xml tests/fixtures/form4_10b51_slashdate.xml
git commit -m "feat: widen 10b5-1 adoption-date parsing to more phrasings/date formats"
```

---

## Task 2: Universe loader + curated CSV

**Files:**
- Create: `data/universe.csv`, `src/insider_hb/universe.py`
- Test: `tests/test_universe.py`

- [ ] **Step 1: Write the curated `data/universe.csv`**

This is a hand-maintained starting universe; flags are a best-effort first pass the analyst is expected to vet. Create `data/universe.csv` with exactly this content:

```csv
ticker,gov_tier,size_tier,dual_class,founder_ceo,notes
AAPL,dispersed,mega,false,false,widely held; independent board; recon baseline
MSFT,dispersed,mega,false,false,widely held
JNJ,dispersed,mega,false,false,widely held
PG,dispersed,mega,false,false,widely held
JPM,dispersed,mega,false,false,widely held
XOM,dispersed,mega,false,false,widely held
KO,dispersed,large,false,false,widely held
UNH,dispersed,mega,false,false,widely held
HD,dispersed,large,false,false,widely held
PFE,dispersed,large,false,false,widely held
MRK,dispersed,large,false,false,widely held
CSCO,dispersed,large,false,false,widely held
INTC,dispersed,large,false,false,widely held
VZ,dispersed,large,false,false,widely held
DIS,dispersed,large,false,false,widely held
COST,dispersed,large,false,false,widely held
PEP,dispersed,large,false,false,widely held
ABBV,dispersed,large,false,false,widely held
CAT,dispersed,large,false,false,widely held
IBM,dispersed,large,false,false,widely held
ETSY,dispersed,mid,false,false,widely held mid-cap
WMT,concentrated,mega,false,false,Walton family control ~45pct; not dual-class
ORCL,concentrated,mega,false,false,Ellison ~40pct founder-chair control; recon case
TSLA,concentrated,mega,false,false,Musk large holder outsized influence
META,concentrated,mega,true,true,Zuckerberg super-voting B shares
GOOGL,concentrated,mega,true,false,founder super-voting; Pichai CEO
BRK-B,concentrated,mega,true,false,Buffett control; A/B dual class
NKE,concentrated,large,true,false,Knight family B-share control
PLTR,concentrated,large,true,true,multi-class; Karp CEO Thiel
COIN,concentrated,large,true,true,Armstrong founder-CEO; dual class
SHOP,concentrated,large,true,true,Lutke founder-CEO; founder share
ABNB,concentrated,large,true,true,Chesky founder-CEO; dual class
HOOD,concentrated,mid,true,true,Tenev founder-CEO; dual class
RBLX,concentrated,mid,true,true,Baszucki founder-CEO; dual class
DDOG,concentrated,mid,true,true,dual class; founder-CEO
NET,concentrated,mid,true,true,Cloudflare; Prince founder-CEO; dual class
ROKU,concentrated,mid,true,true,Wood founder-CEO; dual class
DKNG,concentrated,mid,true,true,Robins founder-CEO; multi-class
W,concentrated,mid,true,true,Wayfair; founders dual class
LYFT,concentrated,mid,true,true,co-founders dual class
PINS,concentrated,mid,true,false,dual class; founder stepped down
NVDA,mixed,mega,false,true,Huang founder-CEO; broadly held otherwise
AMZN,mixed,mega,false,false,Bezos founder large holder; Jassy CEO
CRM,mixed,large,false,true,Benioff founder-CEO; widely held
NFLX,mixed,large,false,false,Hastings co-founder exec chair
SNOW,mixed,large,false,false,no founder-CEO; VC heritage
U,mixed,mid,false,false,Unity; no founder-CEO
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_universe.py`:

```python
from pathlib import Path
import pytest
from insider_hb.universe import Firm, load_universe

def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "u.csv"
    p.write_text("ticker,gov_tier,size_tier,dual_class,founder_ceo,notes\n" + body)
    return p

def test_loads_valid_rows(tmp_path):
    p = _write(tmp_path, "AAPL,dispersed,mega,false,false,widely held\nMETA,concentrated,mega,true,true,founder\n")
    firms = load_universe(p)
    assert firms == [
        Firm("AAPL", "dispersed", "mega", False, False, "widely held"),
        Firm("META", "concentrated", "mega", True, True, "founder"),
    ]

def test_bad_gov_tier_raises(tmp_path):
    p = _write(tmp_path, "AAPL,sideways,mega,false,false,x\n")
    with pytest.raises(ValueError, match="gov_tier"):
        load_universe(p)

def test_bad_size_tier_raises(tmp_path):
    p = _write(tmp_path, "AAPL,dispersed,huge,false,false,x\n")
    with pytest.raises(ValueError, match="size_tier"):
        load_universe(p)

def test_bad_bool_raises(tmp_path):
    p = _write(tmp_path, "AAPL,dispersed,mega,yes,false,x\n")
    with pytest.raises(ValueError, match="dual_class"):
        load_universe(p)

def test_curated_universe_file_is_valid():
    firms = load_universe(Path("data/universe.csv"))
    assert len(firms) >= 40
    tiers = {f.gov_tier for f in firms}
    assert {"concentrated", "mixed", "dispersed"} <= tiers
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_universe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insider_hb.universe'`.

- [ ] **Step 4: Implement `src/insider_hb/universe.py`**

```python
"""Load + validate the curated firm universe. Pure, no network."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

GOV_TIERS = {"concentrated", "mixed", "dispersed"}
SIZE_TIERS = {"mega", "large", "mid", "small"}
_REQUIRED = ["ticker", "gov_tier", "size_tier", "dual_class", "founder_ceo", "notes"]


@dataclass
class Firm:
    ticker: str
    gov_tier: str
    size_tier: str
    dual_class: bool
    founder_ceo: bool
    notes: str


def _parse_bool(val: str, field: str, ticker: str) -> bool:
    v = val.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"{field} must be true/false for {ticker}, got {val!r}")


def load_universe(path: str | Path) -> list[Firm]:
    firms: list[Firm] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"universe.csv missing columns: {missing}")
        for row in reader:
            ticker = row["ticker"].strip()
            gov = row["gov_tier"].strip()
            size = row["size_tier"].strip()
            if gov not in GOV_TIERS:
                raise ValueError(f"gov_tier {gov!r} for {ticker} not in {GOV_TIERS}")
            if size not in SIZE_TIERS:
                raise ValueError(f"size_tier {size!r} for {ticker} not in {SIZE_TIERS}")
            firms.append(
                Firm(
                    ticker=ticker,
                    gov_tier=gov,
                    size_tier=size,
                    dual_class=_parse_bool(row["dual_class"], "dual_class", ticker),
                    founder_ceo=_parse_bool(row["founder_ceo"], "founder_ceo", ticker),
                    notes=row["notes"].strip(),
                )
            )
    return firms
```

- [ ] **Step 5: Run tests, verify pass**

Run: `.venv/bin/pytest tests/test_universe.py -v`
Expected: PASS (all 5).

- [ ] **Step 6: Commit**

```bash
git add data/universe.csv src/insider_hb/universe.py tests/test_universe.py
git commit -m "feat: curated firm universe + validating loader"
```

> **Note:** `data/` is gitignored, so `git add data/universe.csv` needs the explicit path (it works — ignore rules don't block explicit adds). Verify with `git status` that `universe.csv` is staged.

---

## Task 3: Disk cache

**Files:**
- Create: `src/insider_hb/cache.py`
- Test: `tests/test_cache.py`

Cache honors the spec: Form 4 XML stored per accession (immutable), prices per ticker with a staleness threshold. A per-ticker Form 4 manifest records which accessions belong to the ticker.

- [ ] **Step 1: Write failing tests**

Create `tests/test_cache.py`:

```python
import json
from datetime import datetime, timedelta
import pandas as pd
from insider_hb import cache

def test_form4_miss_then_hit(tmp_path):
    calls = {"n": 0}
    def fake(ticker, limit):
        calls["n"] += 1
        return [("acc-1", "2025-01-01", "<x>1</x>"), ("acc-2", "2025-01-02", "<x>2</x>")]

    out1 = cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path)
    out2 = cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path)
    assert calls["n"] == 1                       # second call served from disk
    assert out1 == out2
    assert out2[0] == ("acc-1", "2025-01-01", "<x>1</x>")
    assert (tmp_path / "form4" / "acc-1.xml").exists()

def test_form4_refetch_when_stale(tmp_path):
    calls = {"n": 0}
    def fake(ticker, limit):
        calls["n"] += 1
        return [("acc-1", "2025-01-01", "<x>1</x>")]

    cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path)
    man = tmp_path / "form4" / "TST.json"
    data = json.loads(man.read_text())
    data["fetched"] = (datetime.now() - timedelta(days=30)).isoformat()
    man.write_text(json.dumps(data))
    cache.get_or_fetch_form4_xml("TST", 200, fetcher=fake, cache_dir=tmp_path, max_age_days=7)
    assert calls["n"] == 2                        # stale manifest triggered refetch

def test_prices_miss_then_hit(tmp_path):
    calls = {"n": 0}
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-01-01", "2025-01-02"]), name="TST")
    def fake(ticker):
        calls["n"] += 1
        return s

    out1 = cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path)
    out2 = cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path)
    assert calls["n"] == 1
    pd.testing.assert_series_equal(out1, out2)
    assert out2.name == "TST"

def test_prices_refetch_when_stale(tmp_path):
    calls = {"n": 0}
    s = pd.Series([1.0], index=pd.to_datetime(["2025-01-01"]), name="TST")
    def fake(ticker):
        calls["n"] += 1
        return s

    cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path)
    meta = tmp_path / "prices" / "TST.meta.json"
    meta.write_text(json.dumps({"fetched": (datetime.now() - timedelta(days=30)).isoformat()}))
    cache.get_or_fetch_prices("TST", fetcher=fake, cache_dir=tmp_path, max_age_days=7)
    assert calls["n"] == 2
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insider_hb.cache'`.

- [ ] **Step 3: Implement `src/insider_hb/cache.py`**

```python
"""Disk cache over the Form 4 / price fetchers. Keeps the cold panel build
re-runnable and resilient to EDGAR throttling. Pure I/O, no business logic."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .fetch import fetch_form4_xml, fetch_prices

CACHE_DIR = Path("data/cache")


def _fresh(fetched_iso: str, max_age_days: int) -> bool:
    try:
        age = datetime.now() - datetime.fromisoformat(fetched_iso)
    except (ValueError, TypeError):
        return False
    return age.days < max_age_days


def get_or_fetch_form4_xml(
    ticker: str,
    limit: int = 200,
    *,
    fetcher: Callable[[str, int], list[tuple[str, str, str]]] = fetch_form4_xml,
    cache_dir: Path = CACHE_DIR,
    max_age_days: int = 7,
) -> list[tuple[str, str, str]]:
    """Return [(accession, filing_date, xml)], served from disk when fresh."""
    d = Path(cache_dir) / "form4"
    d.mkdir(parents=True, exist_ok=True)
    manifest = d / f"{ticker}.json"

    if manifest.exists():
        man = json.loads(manifest.read_text())
        if _fresh(man.get("fetched", ""), max_age_days):
            out = []
            for acc, fdate in man["accessions"]:
                xml = (d / f"{acc}.xml").read_text()
                out.append((acc, fdate, xml))
            return out

    rows = fetcher(ticker, limit)
    for acc, _fdate, xml in rows:
        (d / f"{acc}.xml").write_text(xml)
    manifest.write_text(
        json.dumps(
            {
                "fetched": datetime.now().isoformat(),
                "accessions": [[acc, fdate] for acc, fdate, _ in rows],
            }
        )
    )
    return rows


def get_or_fetch_prices(
    ticker: str,
    *,
    fetcher: Callable[[str], pd.Series] = fetch_prices,
    cache_dir: Path = CACHE_DIR,
    max_age_days: int = 7,
) -> pd.Series:
    """Return a price Series, served from disk when fresh."""
    d = Path(cache_dir) / "prices"
    d.mkdir(parents=True, exist_ok=True)
    data_path = d / f"{ticker}.parquet"
    meta_path = d / f"{ticker}.meta.json"

    if data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if _fresh(meta.get("fetched", ""), max_age_days):
            s = pd.read_parquet(data_path)["value"]
            s.name = ticker
            return s

    s = fetcher(ticker)
    s.to_frame("value").to_parquet(data_path)
    meta_path.write_text(json.dumps({"fetched": datetime.now().isoformat()}))
    return s
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/pytest tests/test_cache.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/cache.py tests/test_cache.py
git commit -m "feat: staleness-aware disk cache for Form 4 XML and prices"
```

---

## Task 4: Inject fetchers into `build_ticker_data`

**Why:** The panel build must route fetches through the cache. `build_ticker_data` currently hardcodes `fetch_form4_xml` / `fetch_prices`. Add optional injected fetchers (default = the real ones, so S1 behavior is unchanged).

**Files:**
- Modify: `src/insider_hb/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write a failing test**

Add to `tests/test_pipeline.py`:

```python
import pandas as pd
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline.py -k injected -v`
Expected: FAIL — `build_ticker_data() got an unexpected keyword argument 'form4_fetcher'`.

- [ ] **Step 3: Modify `build_ticker_data` signature + calls**

In `src/insider_hb/pipeline.py`, change the signature (line 29-34) to add two keyword-only params, and use them in the body.

New signature:

```python
def build_ticker_data(
    ticker: str,
    benchmark: str = "SPY",
    limit: int = 200,
    horizons=DEFAULT_HORIZONS,
    *,
    form4_fetcher=fetch_form4_xml,
    price_fetcher=fetch_prices,
) -> TickerData:
```

Change the fetch call (line 37) from `for acc, _fdate, xml in fetch_form4_xml(ticker, limit):` to:

```python
    for acc, _fdate, xml in form4_fetcher(ticker, limit):
```

Change the price fetches (lines 65-66) from:

```python
    firm = fetch_prices(ticker)
    bench = fetch_prices(benchmark)
```

to:

```python
    firm = price_fetcher(ticker)
    bench = price_fetcher(benchmark)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: PASS — the new injected test plus all pre-existing offline pipeline tests.

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/pipeline.py tests/test_pipeline.py
git commit -m "feat: dependency-inject fetchers into build_ticker_data"
```

---

## Task 5: Panel assembly helpers

**Files:**
- Create: `src/insider_hb/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_panel.py`:

```python
import numpy as np
import pandas as pd
from insider_hb.universe import Firm
from insider_hb.panel import attach_governance, assemble_panel, PANEL_COLUMNS

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
    assert "gov_tier" in out.columns

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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insider_hb.panel'`.

- [ ] **Step 3: Implement `src/insider_hb/panel.py`**

```python
"""Pure assembly of per-firm sale-event tables into the cross-firm panel.

No network, no disk — orchestration lives in scripts/build_panel.py.
"""
from __future__ import annotations

import pandas as pd

from .universe import Firm

_GOV_COLS = ["gov_tier", "size_tier", "dual_class", "founder_ceo", "insider_own_pct"]

# S1 event columns (minus the `td` helper) followed by governance columns.
PANEL_COLUMNS = [
    "ticker", "date", "owner", "role", "shares", "price", "value",
    "aff10b5one", "plan_adoption", "adoption_gap_days", "class", "opportunistic",
    "strength", "exc_21", "exc_63", "exc_126", "exc_252",
] + _GOV_COLS


def attach_governance(
    events: pd.DataFrame, firm: Firm, insider_own_pct: float | None
) -> pd.DataFrame:
    """Drop the `td` helper and append the firm's governance columns to every row."""
    out = events.drop(columns=["td"], errors="ignore").copy()
    out["gov_tier"] = firm.gov_tier
    out["size_tier"] = firm.size_tier
    out["dual_class"] = firm.dual_class
    out["founder_ceo"] = firm.founder_ceo
    out["insider_own_pct"] = insider_own_pct
    return out


def assemble_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-firm frames into one panel with a stable column order."""
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    panel = pd.concat(non_empty, ignore_index=True)
    return panel.reindex(columns=PANEL_COLUMNS)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/bin/pytest tests/test_panel.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add src/insider_hb/panel.py tests/test_panel.py
git commit -m "feat: pure panel-assembly helpers (governance attach + concat)"
```

---

## Task 6: Best-effort insider-ownership fetch

**Files:**
- Modify: `src/insider_hb/fetch.py`

This is a thin live-only helper; it returns `None` on any failure and is exercised by the live test in Task 8 (no offline unit test — it is pure network passthrough).

- [ ] **Step 1: Add `fetch_insider_own_pct` to `src/insider_hb/fetch.py`**

Append:

```python
def fetch_insider_own_pct(ticker: str) -> float | None:
    """Best-effort current insider-ownership fraction from yfinance. None on failure.

    NOTE: this is a current snapshot, not point-in-time — an audit signal only.
    """
    try:
        pct = yf.Ticker(ticker).info.get("heldPercentInsiders")
        return float(pct) if pct is not None else None
    except Exception:
        return None
```

- [ ] **Step 2: Verify it imports**

Run: `.venv/bin/python -c "from insider_hb.fetch import fetch_insider_own_pct; print('ok')"`
Expected: prints `ok` (set `PYTHONPATH=src` if needed: `PYTHONPATH=src .venv/bin/python -c ...`).

- [ ] **Step 3: Commit**

```bash
git add src/insider_hb/fetch.py
git commit -m "feat: best-effort insider-ownership snapshot fetch"
```

---

## Task 7: Resumable CLI orchestrator

**Files:**
- Create: `scripts/build_panel.py`

- [ ] **Step 1: Implement `scripts/build_panel.py`**

```python
"""Build the cross-firm sale-event panel.

Usage:
    EDGAR_IDENTITY="Name email" PYTHONPATH=src .venv/bin/python scripts/build_panel.py [--force] [--limit N]

Resumable: each firm's events persist to data/cache/panel/{ticker}_events.parquet
and completed firms are skipped unless --force. Writes data/panel.parquet + prints
a JSON coverage summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from insider_hb.universe import load_universe
from insider_hb.cache import get_or_fetch_form4_xml, get_or_fetch_prices
from insider_hb.fetch import fetch_insider_own_pct
from insider_hb.pipeline import build_ticker_data
from insider_hb.panel import attach_governance, assemble_panel, PANEL_COLUMNS

UNIVERSE = Path("data/universe.csv")
PANEL_OUT = Path("data/panel.parquet")
PER_FIRM_DIR = Path("data/cache/panel")
RETRY_DELAY_S = 5.0


def _build_one(firm, limit: int) -> pd.DataFrame:
    td = build_ticker_data(
        firm.ticker,
        limit=limit,
        form4_fetcher=get_or_fetch_form4_xml,
        price_fetcher=get_or_fetch_prices,
    )
    pct = fetch_insider_own_pct(firm.ticker)
    return attach_governance(td.events, firm, pct)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild all firms")
    ap.add_argument("--limit", type=int, default=200, help="Form 4 filings per firm")
    args = ap.parse_args()

    PER_FIRM_DIR.mkdir(parents=True, exist_ok=True)
    firms = load_universe(UNIVERSE)
    failed: list[str] = []

    for firm in firms:
        out_path = PER_FIRM_DIR / f"{firm.ticker}_events.parquet"
        if out_path.exists() and not args.force:
            continue
        try:
            events = _build_one(firm, args.limit)
        except Exception:
            time.sleep(RETRY_DELAY_S)
            try:
                events = _build_one(firm, args.limit)
            except Exception as e:  # noqa: BLE001
                failed.append(firm.ticker)
                sys.stderr.write(f"FAILED {firm.ticker}: {e}\n")
                continue
        events.to_parquet(out_path)

    frames = [pd.read_parquet(p) for p in sorted(PER_FIRM_DIR.glob("*_events.parquet"))]
    panel = assemble_panel(frames) if frames else pd.DataFrame(columns=PANEL_COLUMNS)
    panel.to_parquet(PANEL_OUT)

    horizons = ["exc_21", "exc_63", "exc_126", "exc_252"]
    summary = {
        "n_firms": len(firms),
        "n_firms_failed": len(failed),
        "failed": failed,
        "n_events": int(len(panel)),
        "events_by_gov_tier": (
            panel["gov_tier"].value_counts().to_dict() if not panel.empty else {}
        ),
        "horizon_coverage": {
            h: (float(panel[h].notna().mean()) if not panel.empty else 0.0)
            for h in horizons
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test argument parsing offline**

Run: `PYTHONPATH=src .venv/bin/python scripts/build_panel.py --help`
Expected: argparse usage text printed (no network hit).

- [ ] **Step 3: Commit**

```bash
git add scripts/build_panel.py
git commit -m "feat: resumable CLI panel builder with per-firm cache and retry"
```

---

## Task 8: Live end-to-end test

**Files:**
- Create: `tests/test_panel_live.py`

- [ ] **Step 1: Write the live test**

Create `tests/test_panel_live.py`:

```python
import os
import pytest
import pandas as pd

from insider_hb.universe import Firm
from insider_hb.cache import get_or_fetch_form4_xml, get_or_fetch_prices
from insider_hb.fetch import fetch_insider_own_pct
from insider_hb.pipeline import build_ticker_data
from insider_hb.panel import attach_governance, assemble_panel, PANEL_COLUMNS

pytestmark = pytest.mark.live

FIRMS = [
    Firm("AAPL", "dispersed", "mega", False, False, "recon baseline"),
    Firm("ORCL", "concentrated", "mega", False, False, "recon case"),
]


@pytest.mark.skipif(not os.environ.get("EDGAR_IDENTITY"), reason="EDGAR_IDENTITY unset")
def test_two_firm_panel_end_to_end(tmp_path):
    frames = []
    for firm in FIRMS:
        td = build_ticker_data(
            firm.ticker,
            limit=60,
            form4_fetcher=lambda t, l, cd=tmp_path: get_or_fetch_form4_xml(t, l, cache_dir=cd),
            price_fetcher=lambda t, cd=tmp_path: get_or_fetch_prices(t, cache_dir=cd),
        )
        frames.append(attach_governance(td.events, firm, fetch_insider_own_pct(firm.ticker)))

    panel = assemble_panel(frames)
    assert list(panel.columns) == PANEL_COLUMNS
    assert set(panel["ticker"]) == {"AAPL", "ORCL"}
    assert len(panel) > 0
    assert panel["strength"].dropna().between(0, 1).all()
```

- [ ] **Step 2: Run offline suite — confirm the live test is excluded by default**

Run: `.venv/bin/pytest -q`
Expected: all offline tests pass; the live test is deselected (`addopts="-m 'not live'"`).

- [ ] **Step 3: Run the live test explicitly**

Run: `EDGAR_IDENTITY="Joey Scully scully.joey@gmail.com" .venv/bin/pytest tests/test_panel_live.py -m live -v`
Expected: PASS — panel built for AAPL+ORCL with the full column set and in-range strength. (Transient EDGAR throttling may require one rerun — a known EDGAR behavior, not a code failure.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_panel_live.py
git commit -m "test: live two-firm end-to-end panel build"
```

---

## Task 9: Full cold build + validation

**Files:** none (produces `data/panel.parquet`, gitignored)

- [ ] **Step 1: Run the full build**

Run: `EDGAR_IDENTITY="Joey Scully scully.joey@gmail.com" PYTHONPATH=src .venv/bin/python scripts/build_panel.py`
Expected: JSON summary with `n_firms` ≈ 48, `n_events` > 0, `events_by_gov_tier` spanning all three tiers, `horizon_coverage` near 1.0 for short horizons and lower for `exc_252` (recent sales right-censored). Some `n_firms_failed` is acceptable (log them).

- [ ] **Step 2: Validate determinism (warm cache rerun)**

Run the same command again.
Expected: near-instant (all per-firm parquets exist → skipped); identical summary.

- [ ] **Step 3: Spot-check against S1**

Run: `PYTHONPATH=src .venv/bin/python -c "import pandas as pd; p=pd.read_parquet('data/panel.parquet'); a=p[p.ticker=='ORCL']; print(len(a), round(a['strength'].mean(),2))"`
Expected: ORCL event count and mean strength consistent with the S1 recon (n≈114, strength≈0.85 at limit=200). Minor differences are fine if the fetch `limit` differs.

- [ ] **Step 4: Finish the branch**

Use the **superpowers:finishing-a-development-branch** skill to run the offline suite and merge `feat/s2-panel` to `main`.

---

## Self-Review Notes

- **Spec coverage:** universe (Task 2) ✓; disk cache per-accession XML + per-ticker prices w/ staleness (Task 3) ✓; resumable CLI + retry/delay, no concurrency (Task 7) ✓; panel schema w/ right-censored NaN rows preserved (Task 5, via S1 NaT-snap) ✓; governance hand-label + raw signals + best-effort insider_own_pct (Tasks 2, 5, 6) ✓; 10b5-1 regex widening w/ fixtures (Task 1) ✓; validation incl. determinism + S1 reproduction (Task 9) ✓; survivorship caveat is documented in the spec and not re-litigated here.
- **Column-name reconciliation:** plan uses the real S1 event columns (`owner`, `role`, `value`, `class`, `opportunistic`, `plan_adoption`), not the idealized spec §6 names; `PANEL_COLUMNS` is the single source of truth.
- **Type consistency:** `Firm` fields and `PANEL_COLUMNS` are referenced identically across Tasks 2, 5, 7, 8; fetcher injection names (`form4_fetcher`, `price_fetcher`) match between Task 4 and Tasks 7-8.
