"""Run the S3 H-B inference on the sale-event panel.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_inference.py \
        [--panel data/panel.parquet] [--cache data/cache/prices] \
        [--out-dir results] [--bootstrap-n 2000] [--seed 0]

Loads the panel + price cache, runs three inference engines (firm-cluster block
bootstrap, calendar-time portfolio, panel FE regression) across horizons, and
writes results/s3_hb_inference.json + results/s3_hb_findings.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insider_hb.estimands import (
    pooled_mean, opportunistic_contrast, gov_tier_means, n_valid,
)
from insider_hb.bootstrap import bootstrap_statistic
from insider_hb.calendar_time import daily_portfolio_excess, calendar_time_test
from insider_hb.regression import fe_regression
from insider_hb.features import DEFAULT_HORIZONS


def run_inference(panel, prices, spy, *, horizons=DEFAULT_HORIZONS,
                  bootstrap_n=2000, seed=0) -> dict:
    """Pure core: already-loaded data in, results dict out."""
    out_horizons: dict[str, dict] = {}
    for h in horizons:
        boot = lambda fn: bootstrap_statistic(panel, fn, cluster="ticker",
                                              n=bootstrap_n, seed=seed)
        gov = gov_tier_means(panel, h)
        gov_boot = {}
        for tier in gov:
            gov_boot[tier] = bootstrap_statistic(
                panel[panel["gov_tier"] == tier],
                lambda df, _h=h: pooled_mean(df, _h),
                cluster="ticker", n=bootstrap_n, seed=seed,
            )
        daily = daily_portfolio_excess(panel, prices, spy, h)
        out_horizons[str(h)] = {
            "n_valid": n_valid(panel, h),
            "pooled_mean_equal": boot(lambda df, _h=h: pooled_mean(df, _h)),
            "pooled_mean_value": boot(lambda df, _h=h: pooled_mean(df, _h, weight="value")),
            "opportunistic_contrast": boot(lambda df, _h=h: opportunistic_contrast(df, _h)),
            "gov_tier_means": gov_boot,
            "calendar_time": calendar_time_test(daily, maxlags=h),
            "regression": fe_regression(panel, h),
        }
    return {
        "meta": {
            "n_events": int(len(panel)),
            "bootstrap_n": int(bootstrap_n),
            "seed": int(seed),
        },
        "horizons": out_horizons,
    }


def _fmt(x) -> str:
    return "n/a" if x != x else f"{x:+.4f}"   # x!=x catches NaN


def findings_markdown(res: dict) -> str:
    lines = ["# S3 — H-B Inference Findings", ""]
    lines.append(f"Events: {res['meta']['n_events']} · "
                 f"bootstrap n={res['meta']['bootstrap_n']} · seed={res['meta']['seed']}")
    lines.append("")
    lines.append("## Pooled post-sale forward excess (firm-clustered bootstrap)")
    lines.append("")
    lines.append("| h | n | equal-wt | 95% CI | p | value-wt | opp−routine | CI | p |")
    lines.append("|---|---|----------|--------|---|----------|-------------|----|---|")
    for h, b in res["horizons"].items():
        eq, vw, oc = b["pooled_mean_equal"], b["pooled_mean_value"], b["opportunistic_contrast"]
        lines.append(
            f"| {h} | {b['n_valid']} | {_fmt(eq['estimate'])} | "
            f"[{_fmt(eq['ci_lo'])}, {_fmt(eq['ci_hi'])}] | {eq['p_value']:.3f} | "
            f"{_fmt(vw['estimate'])} | {_fmt(oc['estimate'])} | "
            f"[{_fmt(oc['ci_lo'])}, {_fmt(oc['ci_hi'])}] | {oc['p_value']:.3f} |"
        )
    lines += ["", "## Governance-tier pooled means (bootstrap)", "",
              "| h | concentrated | mixed | dispersed |", "|---|---|---|---|"]
    for h, b in res["horizons"].items():
        g = b["gov_tier_means"]
        lines.append(f"| {h} | {_fmt(g['concentrated']['estimate'])} | "
                     f"{_fmt(g['mixed']['estimate'])} | {_fmt(g['dispersed']['estimate'])} |")
    lines += ["", "## Calendar-time portfolio (Newey-West)", "",
              "| h | mean daily excess | t | p | n_days |", "|---|---|---|---|---|"]
    for h, b in res["horizons"].items():
        c = b["calendar_time"]
        lines.append(f"| {h} | {_fmt(c['mean_daily_excess'])} | {c['t_stat']:.2f} | "
                     f"{c['p_value']:.3f} | {c['n_days']} |")
    lines += ["", "## Panel FE regression (firm-clustered SEs)", ""]
    for h, b in res["horizons"].items():
        lines += [f"### h = {h}", "", "| term | coef | se | t | p | 95% CI |",
                  "|------|------|----|---|---|--------|"]
        for r in b["regression"]:
            lines.append(f"| {r['term']} | {_fmt(r['coef'])} | {r['se']:.4f} | "
                         f"{r['t']:.2f} | {r['p']:.3f} | "
                         f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] |")
        lines.append("")
    lines += ["## Verdict", "", _verdict(res)]
    return "\n".join(lines) + "\n"


def _verdict(res: dict) -> str:
    h = max(res["horizons"], key=int)  # longest horizon, order-independent
    b = res["horizons"][h]
    eq, oc = b["pooled_mean_equal"], b["opportunistic_contrast"]
    neg = eq["ci_hi"] < 0
    opp_worse = oc["estimate"] < 0 and oc["ci_hi"] < 0
    parts = [
        f"At h={h}: pooled post-sale excess {_fmt(eq['estimate'])} "
        f"(95% CI [{_fmt(eq['ci_lo'])}, {_fmt(eq['ci_hi'])}]) — "
        + ("H-B supported: forward excess is significantly negative."
           if neg else "H-B not supported at conventional significance."),
        "Opportunistic sales predict "
        + ("significantly more negative" if opp_worse else "no significantly different")
        + f" forward returns than routine ones (contrast {_fmt(oc['estimate'])}).",
    ]
    return " ".join(parts)


def _load_prices(cache_dir: Path, tickers) -> dict:
    out = {}
    for t in tickers:
        p = cache_dir / f"{t}.parquet"
        if p.exists():
            out[t] = pd.read_parquet(p)["value"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel.parquet")
    ap.add_argument("--cache", default="data/cache/prices")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--bootstrap-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    cache = Path(args.cache)
    tickers = sorted(panel["ticker"].unique())
    prices = _load_prices(cache, tickers)
    missing = [t for t in tickers if t not in prices]
    if missing:
        print(f"warning: no price cache for {len(missing)} tickers: {missing}",
              file=sys.stderr)
    spy_path = cache / "SPY.parquet"
    if not spy_path.exists():
        raise SystemExit(f"SPY benchmark cache missing at {spy_path}")
    spy = pd.read_parquet(spy_path)["value"]

    res = run_inference(panel, prices, spy, bootstrap_n=args.bootstrap_n, seed=args.seed)
    res["meta"]["panel_sha"] = hashlib.sha256(
        Path(args.panel).read_bytes()).hexdigest()[:12]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s3_hb_inference.json").write_text(json.dumps(res, indent=2))
    (out / "s3_hb_findings.md").write_text(findings_markdown(res))
    print(json.dumps({"n_events": res["meta"]["n_events"],
                      "wrote": [str(out / "s3_hb_inference.json"),
                                str(out / "s3_hb_findings.md")]}), file=sys.stderr)


if __name__ == "__main__":
    main()
