"""Run the S4 variance-robust (WLS) strength inference on the sale-event panel.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_s4.py \
        [--panel data/panel.parquet] [--out-dir results] \
        [--time-cutoff 2024-01-01] [--firm-seed 7]

Three estimators per horizon (inverse-variance WLS [primary], equal-weight OLS
[conservative bound], per-firm z-scored DV [corroboration]); a leakage-free
time holdout (weights frozen from the train window) plus a secondary firm-half
split; a verdict; and results/s4_wls_inference.json + results/s4_wls_findings.md.

CAVEAT: the panel is already spent (2024+ was examined during exploration), so
the time holdout is a replication/stability check, not a pristine out-of-sample
test. See the spec section 2.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insider_hb.wls import (
    REGRESSORS, firm_variance, inverse_variance_weights, wls_fe_regression,
    firm_zscore_dv,
)
from insider_hb.holdout import time_split, firm_split

DEFAULT_HORIZONS = (21, 63, 126, 252)


def _derive(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the S4 regressor columns; drop non-positive value (log1p guard)."""
    df = panel[panel["value"] > 0].copy()
    df["discretionary"] = (df["class"] == "discretionary").astype(float)
    df["opp_plan"] = (df["class"] == "opportunistic_plan").astype(float)
    df["log_value"] = np.log1p(df["value"].astype(float))
    df["strength"] = df["strength"].astype(float)
    return df


def _strength(rows: list) -> dict:
    return next(r for r in rows if r["term"] == "strength")


def _wls_strength(df, h):
    var = firm_variance(df, h)
    w = inverse_variance_weights(df, h, var)
    return _strength(wls_fe_regression(df, h, weights=w))


def run_s4(panel, *, horizons=DEFAULT_HORIZONS,
           time_cutoff="2024-01-01", firm_seed=7) -> dict:
    """Pure core: derived panel in, results dict out."""
    df = _derive(panel)
    out: dict = {}
    for h in horizons:
        # full sample (in-sample variances; labelled as such in the writeup)
        full_var = firm_variance(df, h)
        w_full = inverse_variance_weights(df, h, full_var)
        wls_table = wls_fe_regression(df, h, weights=w_full)
        df[f"exc_{h}_z"] = firm_zscore_dv(df, h)
        full_sample = {
            "wls": _strength(wls_table),
            "ols": _strength(wls_fe_regression(df, h, weights=None)),
            "zscore": _strength(wls_fe_regression(df, h, weights=None,
                                                  dv_col=f"exc_{h}_z")),
            "wls_table": wls_table,
        }
        # time holdout: weights FROZEN from train, applied to both fits
        train, test = time_split(df, cutoff=time_cutoff)
        train_var = firm_variance(train, h)
        time_holdout = {
            "train": _strength(wls_fe_regression(
                train, h, weights=inverse_variance_weights(train, h, train_var))),
            "test": _strength(wls_fe_regression(
                test, h, weights=inverse_variance_weights(test, h, train_var))),
        }
        # firm holdout (secondary): each half uses its own in-sample variances
        half_a, half_b = firm_split(df, seed=firm_seed)
        firm_holdout = {"half_a": _wls_strength(half_a, h),
                        "half_b": _wls_strength(half_b, h)}
        out[str(h)] = {"full_sample": full_sample,
                       "time_holdout": time_holdout,
                       "firm_holdout": firm_holdout}
    return {
        "meta": {
            "n_rows": int(len(df)),
            "horizons": list(horizons),
            "time_cutoff": time_cutoff,
            "firm_seed": int(firm_seed),
            "generated": _dt.date.today().isoformat(),
        },
        "horizons": out,
        "verdict": _verdict(out),
    }


def _sig_neg(row: dict) -> bool:
    return row["coef"] < 0 and row["p"] < 0.05


def _verdict(horizons_out: dict) -> dict:
    by_h = {}
    for h, blk in horizons_out.items():
        c1 = _sig_neg(blk["full_sample"]["wls"])
        c2 = _sig_neg(blk["time_holdout"]["test"])
        c3 = (blk["firm_holdout"]["half_a"]["coef"] < 0
              and blk["firm_holdout"]["half_b"]["coef"] < 0)
        by_h[h] = {"robust": bool(c1 and c2 and c3),
                   "full_sample_sig_neg": bool(c1),
                   "test_window_sig_neg": bool(c2),
                   "firm_halves_sign_consistent": bool(c3)}
    return {"by_horizon": by_h,
            "focal_robust": bool(by_h.get("126", {}).get("robust", False))}


def _fmt(x) -> str:
    return "n/a" if x != x else f"{x:+.4f}"   # x!=x catches NaN


def _row(label, r) -> str:
    return (f"| {label} | {_fmt(r['coef'])} | {r['se']:.4f} | {r['t']:.2f} | "
            f"{r['p']:.3f} | [{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | "
            f"{r['n']} | {r['n_firms']} |")


def findings_markdown(res: dict) -> str:
    m = res["meta"]
    lines = ["# S4 — Variance-Robust (WLS) Strength Inference", "",
             f"Rows: {m['n_rows']} · time cutoff: {m['time_cutoff']} · "
             f"firm seed: {m['firm_seed']} · generated: {m['generated']}", "",
             "> **Caveat:** the panel is already spent (2024+ was examined during "
             "exploration). The time holdout below is a replication/stability check, "
             "not a pristine out-of-sample test. WLS shifts the estimand toward "
             "low-variance firms; equal-weight OLS is printed alongside as the "
             "conservative bound.", ""]

    lines += ["## Strength coefficient by estimator (full sample)", "",
              "| h | WLS (primary) | OLS (bound) | z-scored DV |",
              "|---|---|---|---|"]
    for h, blk in res["horizons"].items():
        fs = blk["full_sample"]
        lines.append(
            f"| {h} | {_fmt(fs['wls']['coef'])} (p={fs['wls']['p']:.3f}) | "
            f"{_fmt(fs['ols']['coef'])} (p={fs['ols']['p']:.3f}) | "
            f"{_fmt(fs['zscore']['coef'])} (p={fs['zscore']['p']:.3f}) |")

    lines += ["", "## Time holdout — WLS strength (weights frozen from train)", "",
              "| h | train (pre-cutoff) | test (post-cutoff) |", "|---|---|---|"]
    for h, blk in res["horizons"].items():
        t = blk["time_holdout"]
        lines.append(f"| {h} | {_fmt(t['train']['coef'])} (p={t['train']['p']:.3f}) | "
                     f"{_fmt(t['test']['coef'])} (p={t['test']['p']:.3f}) |")

    lines += ["", "## Firm holdout — WLS strength (secondary, sign check)", "",
              "| h | half A | half B |", "|---|---|---|"]
    for h, blk in res["horizons"].items():
        f = blk["firm_holdout"]
        lines.append(f"| {h} | {_fmt(f['half_a']['coef'])} (p={f['half_a']['p']:.3f}) | "
                     f"{_fmt(f['half_b']['coef'])} (p={f['half_b']['p']:.3f}) |")

    lines += ["", "## Full WLS regressor table", ""]
    for h, blk in res["horizons"].items():
        lines += [f"### h = {h}", "",
                  "| term | coef | se | t | p | 95% CI | n | n_firms |",
                  "|------|------|----|---|---|--------|---|---------|"]
        for r in blk["full_sample"]["wls_table"]:
            lines.append(_row(r["term"], r))
        lines.append("")

    v = res["verdict"]
    lines += ["## Verdict", ""]
    for h, vb in v["by_horizon"].items():
        verdict = ("ROBUST" if vb["robust"] else "not robust")
        lines.append(
            f"- **h={h}: strength {verdict}** — full-sample sig-neg: "
            f"{vb['full_sample_sig_neg']}, test-window sig-neg: "
            f"{vb['test_window_sig_neg']}, firm-halves sign-consistent: "
            f"{vb['firm_halves_sign_consistent']}.")
    lines += ["",
              ("At the focal 126-day horizon, the variance-robust WLS estimate of "
               "sell-into-strength -> forward excess is "
               + ("robust by the pre-registered criteria."
                  if v["focal_robust"] else
                  "not robust by the pre-registered criteria."))]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel.parquet")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--time-cutoff", default="2024-01-01")
    ap.add_argument("--firm-seed", type=int, default=7)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    res = run_s4(panel, time_cutoff=args.time_cutoff, firm_seed=args.firm_seed)
    res["meta"]["panel_sha"] = hashlib.sha256(
        Path(args.panel).read_bytes()).hexdigest()[:12]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s4_wls_inference.json").write_text(json.dumps(res, indent=2))
    (out / "s4_wls_findings.md").write_text(findings_markdown(res))
    print(json.dumps({"n_rows": res["meta"]["n_rows"],
                      "focal_robust": res["verdict"]["focal_robust"],
                      "wrote": [str(out / "s4_wls_inference.json"),
                                str(out / "s4_wls_findings.md")]}), file=sys.stderr)


if __name__ == "__main__":
    main()
