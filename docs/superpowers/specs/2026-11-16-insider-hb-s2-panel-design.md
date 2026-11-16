# S2 — Sale-Event Panel Build — Design Spec

**Date:** 2026-11-16
**Status:** Design — S2 (panel build); follows S1 (single-firm pipeline, merged)
**Author:** Joey Scully
**Parent spec:** `docs/superpowers/specs/2026-11-16-insider-buybacks-design.md` (H-B primary)

---

## 1. Goal

Build a **sale-event panel** — one row per officer open-market sale (Form 4, Code `S`) —
across a curated firm universe spanning governance and size, by running the merged S1 per-ticker
pipeline over the universe behind a disk cache.

**Deliverable:** `data/panel.parquet` + a coverage summary (firms, events, events by governance
tier, horizon coverage).

This stage builds the dataset. It does **not** run the hypothesis test — inference (calendar-time
portfolios / block bootstrap, the opportunistic-vs-routine contrast) is S3.

---

## 2. Scope Decisions (settled in brainstorming)

| Decision | Choice | Consequence / caveat |
|----------|--------|----------------------|
| **Firm universe** | Curated ~40-60 live firms, hand-tiered by governance and size | Fast, controllable, directly tests the recon magnitude gap (AAPL vs ORCL). **Known cost burned in: selection bias + survivorship bias** (all firms live today → no delisted/acquired names). Broad point-in-time universe is deferred to a later stage. |
| **Governance tier** | Hand-label (3 buckets) **plus** objective signals captured alongside | `gov_tier` is a hand label; `dual_class` and `founder_ceo` are stable hand-entered facts; `insider_own_pct` is pulled best-effort. S3 decides which to actually use; the raw signals are the audit trail. |
| **Build architecture** | CLI batch script (`scripts/build_panel.py`), resumable, with disk cache | Matches the repo convention (CLI + pytest, not sidecar magics). Serial with a polite delay + one retry per firm — **no concurrency** (EDGAR rate-limits; S1 already saw throttling, and this is a one-time cold build). |

**Out of scope (YAGNI):**
- Delisted/acquired tickers and point-in-time index membership (deferred; survivorship caveat accepted).
- H-A buyback-authorization / repurchase extraction (H-A stays a secondary, later concern).
- Any inference, factor model, or significance testing (S3).
- Quantitative-only governance scoring (the objective signals are captured but not formula-combined here).

---

## 3. Components

New files; pure logic separated from network I/O so it stays offline-testable (repo convention).

| File | Responsibility | Depends on |
|------|----------------|-----------|
| `data/universe.csv` | The curated firm list. Hand-maintained. | — |
| `src/insider_hb/universe.py` | Load + validate `universe.csv` → `list[Firm]`. Pure. | csv, dataclass |
| `src/insider_hb/cache.py` | Disk cache for Form 4 XML (by accession, immutable) and price series (per ticker, with a fetch-date manifest + staleness check). `get_or_fetch` wrappers around S1 `fetch`. | `fetch.py` |
| `src/insider_hb/panel.py` | `build_panel`: per firm → S1 `build_ticker_data` (through cache) → attach governance columns + best-effort `insider_own_pct` → concat. Assembly logic kept pure; I/O injected. | `pipeline.py`, `universe.py`, `cache.py` |
| `scripts/build_panel.py` | CLI orchestrator: loop universe, resumable, retry+delay, write `data/panel.parquet` + JSON summary. | `panel.py` |

### 3.1 `data/universe.csv` columns

```
ticker, gov_tier, size_tier, dual_class, founder_ceo, notes
```

- `gov_tier` ∈ {`concentrated`, `mixed`, `dispersed`}
- `size_tier` ∈ {`mega`, `large`, `mid`, `small`}
- `dual_class`, `founder_ceo` ∈ {`true`, `false`}
- `notes` — free text (why this tier; e.g. "founder-CEO, 10-vote super-shares")

### 3.2 `Firm` dataclass (`universe.py`)

```
Firm(ticker: str, gov_tier: str, size_tier: str,
     dual_class: bool, founder_ceo: bool, notes: str)
```

Loader validates: required columns present, `gov_tier`/`size_tier` in allowed sets, booleans
parse. A bad row raises `ValueError` (fail loud — the universe is hand-maintained).

---

## 4. Data Flow & Resumability

1. `build_panel.py` loads the universe.
2. For each firm, if `data/cache/panel/{ticker}_events.parquet` exists and `--force` is not set,
   **skip** (already built) — a throttle failure mid-run never loses finished work.
3. Otherwise run the S1 pipeline through the cache:
   - Form 4 XML fetched once per accession, cached forever (`data/cache/form4/{accession}.xml`);
     filings are immutable so the cache is never invalidated.
   - Price series cached per ticker (`data/cache/prices/{ticker}.parquet`) with a fetch date in a
     manifest; refetched only if older than a staleness threshold (default 7 days) or `--force`.
   - On `SECHTMLResponseError` / transient network error: wait a polite delay and retry **once**;
     if it still fails, record the firm as failed and continue (do not abort the whole run).
4. Attach governance columns (`gov_tier`, `size_tier`, `dual_class`, `founder_ceo`) from the CSV
   and `insider_own_pct` (best-effort, see §5) to every event row; persist the per-firm table.
5. After the loop, concatenate all per-firm parquets → `data/panel.parquet`, and print a JSON
   summary: `n_firms`, `n_firms_failed`, `n_events`, `events_by_gov_tier`, `horizon_coverage`
   (share of events with a non-NaN forward return at each horizon).

---

## 5. Governance Signals & Point-in-Time

- `gov_tier`, `size_tier`, `dual_class`, `founder_ceo` — hand-entered, stable firm attributes.
- `insider_own_pct` — pulled best-effort from yfinance `.info['heldPercentInsiders']`.
  **Caveat burned in: this is a current snapshot, not point-in-time.** It is an audit signal for
  S3, not a time-varying regressor. Missing → NaN (never block the build on it).
- Form 4 events are inherently point-in-time (filed ≤2 business days after the trade).
- Prices are split/dividend-adjusted (yfinance `auto_adjust`), consistent with S1.

---

## 6. Panel Schema (one row per officer Code-S sale)

```
ticker, gov_tier, size_tier, dual_class, founder_ceo, insider_own_pct,
event_date, owner_name, officer_title, shares, price, value_usd,
strength, aff10b5one, plan_adoption_date, sale_class, is_opportunistic,
exc_21, exc_63, exc_126, exc_252
```

- Reuses the S1 event table columns; adds the governance columns.
- **Right-censored events** (recent sales lacking a full forward window) keep the row with **NaN**
  at horizons extending past available price data — reusing S1's NaT-snap behavior. S3 decides
  inclusion per horizon; S2 does not silently drop them.

---

## 7. Carried-Forward Fix: 10b5-1 Adoption-Date Regex

The S1 `forms.py` adoption-date regex matches only
`10b5-1 … adopted … on <Month DD, YYYY>`. It misses common alternate phrasings and date formats,
undercounting opportunistic plans — and the opportunistic/routine split is the publishable edge.

**S2 widens it** to also cover, at minimum:
- "entered into on <date>" / "adopted on <date>"
- ISO (`YYYY-MM-DD`) and numeric (`MM/DD/YYYY`) date formats
- "Rule 10b5-1 trading plan adopted <date>"

Each alternate phrasing gets a fixture and a test. Parsing resolves all supported formats to an
ISO date; an unrecognized footnote still yields `None` (conservative → routine), as in S1.

---

## 8. Testing (TDD, offline-first)

Pure logic tested offline with fixtures; network I/O behind `@pytest.mark.live` (repo convention,
`addopts="-m 'not live'"`).

| Target | Test |
|--------|------|
| `universe.py` | CSV fixture → validated `Firm` list; bad tier / missing column / unparseable bool raises `ValueError`. |
| `cache.py` | `get_or_fetch` with a fake fetcher: miss then disk hit; immutable accession never re-fetched; stale price manifest triggers refetch, fresh does not. |
| `panel.py` | Fake per-firm event tables → correct concat, governance columns attached, schema matches §6, right-censored NaN rows preserved. |
| `forms.py` | New fixtures for each alternate 10b5-1 phrasing (§7) → adoption date parsed to ISO; unrecognized → `None`. |
| live | `@pytest.mark.live` 2-firm (AAPL + ORCL) end-to-end build → valid panel with both tickers, correct schema, non-empty events. |

---

## 9. Validation (acceptance)

- Full cold build over the curated universe completes, writing `data/panel.parquet`.
- Panel reproduces the S1 single-firm numbers for AAPL and ORCL (same n_events and strength as the
  S1 CLI, since the same pipeline runs underneath).
- Summary shows events spread across all three `gov_tier` buckets.
- A second run (warm cache) is near-instant and writes an identical panel (determinism check).

---

## 10. Key Risks

- **Selection / survivorship bias** (accepted) — curated live set. Mitigation: documented here and
  carried into S3; report heterogeneity, not a pooled population estimate. A broad point-in-time
  universe is the eventual upgrade path.
- **Thin per-tier N** — 40-60 firms may give few events in the `dispersed` or `small` buckets.
  Mitigation: the summary surfaces `events_by_gov_tier`; top up the universe if a bucket is starved.
- **`insider_own_pct` not point-in-time** — do not use it as a time-varying regressor in S3; it is
  an audit snapshot only.
- **10b5-1 regex still misses phrasings** — widening reduces but will not eliminate undercount.
  Mitigation: `None` stays conservative (routine); S3 can sensitivity-test the split.

---

Each stage gets its own spec → plan → implementation cycle. **This spec covers S2.** S3 (inference)
follows.
