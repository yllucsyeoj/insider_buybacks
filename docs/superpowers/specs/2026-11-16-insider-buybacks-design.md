# Insider Selling & Forward Returns — Feasibility & Design Spec

**Date:** 2026-11-16
**Status:** Design v2 — H-B primary (post two-firm reconnaissance)
**Author:** Joey Scully

---

## 1. Motivation

The project began testing a cynical narrative: **buybacks precede insider selling** — companies
repurchase stock (supporting/signalling the price) and insiders sell into the strength, using the
buyback as exit liquidity.

Reconnaissance on Apple and Oracle (2026-11-16) **refocused the project**. The buyback→selling link
is confound-heavy and, on Oracle, decoupled entirely (buybacks collapsed while the CEO sold $2.5B).
But a cleaner, buyback-independent signal emerged and held on both firms:

> **Officers sell into price strength, and that selling precedes negative forward excess returns.**

That is the "opportunistic insider" effect (Cohen-Malloy-Pomorski; Lakonishok-Lee), with a modern
twist the SEC itself cares about: **10b5-1 plan adoption timing**. This spec makes that signal (H-B)
primary and demotes the buyback linkage (H-A) to a controlled secondary.

---

## 2. Hypotheses

### H-B — Primary

> **H-B:** Officer open-market sales (Form 4, Code `S`) executed at **high trailing-price strength**
> — especially *opportunistic* ones (discretionary, or via recently-adopted 10b5-1 plans) — predict
> **negative forward excess returns** (firm minus benchmark), across a **panel** of firms.

> **H-B0 (null):** Forward excess returns after officer sales are indistinguishable from the firm's
> baseline, after conditioning on strength.

### H-A — Secondary (controlled)

> **H-A:** Discretionary (non-10b5-1) officer selling is elevated in the first open trading window
> following buyback activity, controlling for the earnings/blackout cycle.

Retained because it motivated the project and is testable on the same data, but it is confound-heavy
(see §6) and empirically weak/decoupled in reconnaissance. Reported as a secondary result, not the
headline.

---

## 3. Reconnaissance Evidence (what we already found)

Two-firm, descriptive, **suggestive not inferential** (single-firm, overlapping windows, small N):

| Signal | Apple | Oracle |
|--------|-------|--------|
| Trailing strength at sale (percentile of trailing-1y range) | 0.77 | 0.77 |
| Baseline forward excess @252d (vs SPY) | +1.7% | +9.1% |
| Post-sale forward excess @252d (equal-wt) | **−2.1%** | **−24.8%** |
| Post-sale forward excess @126d (equal-wt) | −1.2% | −22.6% |
| Code-S officer sales (window) | $786M diffuse | $3.2B, 80% CEO Catz |
| Biggest sale | — | Catz $1.83B Jun-2025, **100% 10b5-1**, plan adopted Sep-2024 |

**Read:** "sell into strength" is universal (identical 0.77); "predicts underperformance" is
directionally consistent on both firms but ~10x larger on concentrated-governance Oracle. The
cross-firm *variation* is the thing the panel should explain.

Artifacts: `aapl_buyback_insider_timeline.png`, `orcl_buyback_insider_timeline.png`,
`orcl_hb_forward_returns.png` (project root).

---

## 4. Operational Definitions

| Concept | Definition | Source |
|---------|-----------|--------|
| **Insider** | Form 4 filer with `isOfficer = true`. Exclude pure directors. Keep raw `officerTitle` + flags; normalize role strings (`CEO` == `Chief Executive Officer`, strip double spaces). | Form 4 |
| **Sale** | Transaction **Code `S`** (open-market). Exclude `F` (tax), `M` (exercise), `A` (grant), `G` (gift). | Form 4 |
| **10b5-1 sale** | `<aff10b5One>true</aff10b5One>` in raw Form 4 XML. Capture the **plan adoption date** from the footnote. | Form 4 XML |
| **Opportunistic sale** | Discretionary (non-10b5-1) **or** 10b5-1 with a plan adopted suspiciously recently / after a run-up (adoption-to-trade gap near the 90-day cooling-off floor). The routine/opportunistic split is a key regressor. | Derived |
| **Trailing strength** | Percentile of the sale-day close within the trailing-252-day price range (recon used this; refine vs a drawdown-from-high measure). | Price |
| **Forward excess return** | Firm return minus benchmark return over h ∈ {21,63,126,252} trading days. Benchmark = SPY initially; sector/size-matched later. | Price |
| **Buyback signal** (H-A) | Authorization date+$ (8-K EX-99.1 regex) and/or YTD-differenced quarterly repurchases (`us-gaap:PaymentsForRepurchaseOfCommonStock`). | 8-K / 10-Q |
| **First open window** (H-A) | First non-blackout interval after a buyback event; blackout proxied by an offset around earnings dates. | Derived |

---

## 5. Data Feasibility (proven)

All series pulled end-to-end via sidecar `sec` extension (edgartools) + `%yfinance`. See the
`sec-sidecar-data-access` memory for exact commands and gotchas.

| Series | Status | Caveat |
|--------|--------|--------|
| Officer sales (Form 4, Code S, isOfficer) | ✅ | Per-filing fetch slow → async + disk cache for panel |
| **10b5-1 flag + adoption date** | ✅ **resolved** | In raw Form 4 XML, not the convenience magic |
| Price + benchmark (firm, SPY) | ✅ | `_close` columns; yfinance clobbers same-named vars |
| Repurchases quarterly (H-A) | ✅ | XBRL is YTD → difference successive quarters |
| Authorizations (H-A) | ✅ | 8-K EX-99.1 regex; cross-firm language varies |

---

## 6. Confounds & Controls

| Confound | Threat | Control |
|----------|--------|---------|
| **Overlapping forward windows** | Clustered sales → correlated observations → overstated significance. The recon numbers have *no* honest p-value. | Non-overlapping or decay-weighted windows; block bootstrap / Newey-West; cluster SEs by firm and calendar month. |
| **Single-regime / single-event** | One firm's boom-bust (Oracle 2026) masquerades as many observations. | **Panel across firms and time**; calendar-time portfolio approach so each date counts once. |
| **Benchmark / risk** | "Underperformance" may be beta/size/sector, not insider skill. | Market-, then size/sector-, then factor-adjusted excess returns. |
| **Look-ahead / survivorship** | Using delisted/acquired firms or revised data leaks future info. | Point-in-time filings (Form 4 filed ≤2 business days); include dead tickers; price as-of. |
| **Strength ≠ signal** | Sales cluster at highs mechanically (vesting, liquidity) regardless of skill. | Condition on strength; test whether forward returns are *worse than other high-strength days* (not just vs unconditional baseline). |
| **Blackout / 10b5-1** (H-A) | Selling suppressed at announcement; autopilot severs attribution. | Re-clock to first open window; earnings-cycle fixed effects; discretionary-only primary. |

---

## 7. Methodology

**Primary (H-B):**
1. **Event set** — officer Code-S sales; tag each with strength percentile, $value, 10b5-1 status,
   plan-adoption gap (opportunistic vs routine).
2. **Outcome** — forward excess returns at {21,63,126,252}d, benchmark-adjusted.
3. **Estimation** — calendar-time portfolio (long baseline / short post-sale windows) *and* a panel
   regression of forward excess return on: sale indicator, strength, $value, opportunistic flag,
   with firm + month effects and clustered/Newey-West SEs.
4. **Key contrast** — do *opportunistic* sales (discretionary, or recently-adopted 10b5-1) predict
   more negative forward returns than *routine* sales? That is the publishable edge.
5. **Descriptive layer (per firm)** — the overlay + forward-return charts as sanity checks.

**Secondary (H-A):** event-study of buyback signals → first-open-window discretionary selling, with
earnings-cycle controls. Reported alongside, clearly labelled as confound-limited.

---

## 8. Staging

| Stage | Goal | Deliverable |
|-------|------|-------------|
| **S0 — spike (done)** | Prove access; surface confounds; H-A→H-B pivot; two-firm recon | This spec + 3 PNGs |
| **S1 — single-firm pipeline** | Reusable per-ticker functions: Form 4 (Code S, isOfficer, 10b5-1 + adoption date), price/benchmark, strength + forward-excess features. Validate on AAPL & ORCL. | Parameterized pipeline + validated feature tables |
| **S2 — panel build** | Run across a firm universe (span governance/size); async jobs + disk cache; dead tickers; point-in-time | Panel dataset (sale-event × features × forward returns) |
| **S3 — test** | Estimate H-B (+H-A secondary) with proper inference | Results + writeup |

Each stage gets its own spec → plan → implementation cycle. **This spec covers through S1.**

---

## 9. Out of Scope (YAGNI)

- Monthly 10-Q "Issuer Purchases" buckets (quarterly YTD-differenced suffices).
- Intraday / microstructure.
- Full factor model at S1 (market-adjusted is enough to validate the pipeline; size/sector/factor at S3).
- Options-comp modelling beyond the Code `M` flag.
- Any firm universe before S1 is validated on AAPL + ORCL.

---

## 10. Key Risks

- **Overlapping-window inference done wrong** → false confidence. Mitigation: calendar-time
  portfolio + block bootstrap from the start of S3; treat recon numbers as descriptive only.
- **Signal is regime-driven** (concentrated in a few boom-bust names) → doesn't generalize.
  Mitigation: panel spanning governance/size; report effect heterogeneity, not just a pooled mean.
- **Opportunistic/routine split too coarse** (10b5-1 adoption gap a weak proxy) → the headline
  contrast washes out. Mitigation: validate the split on AAPL/ORCL in S1 before scaling.
- **Cross-firm authorization extraction brittle** (H-A only) → lean on standardized XBRL repurchases;
  H-A is secondary regardless.
