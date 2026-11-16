import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "run_inference", Path(__file__).resolve().parents[1] / "scripts" / "run_inference.py"
)
run_inference_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_inference_mod)
run_inference = run_inference_mod.run_inference
findings_markdown = run_inference_mod.findings_markdown


def _panel():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=400)
    rows = []
    for firm, tier in [("A", "concentrated"), ("B", "dispersed")]:
        for k in range(20):
            sale = idx[10 + 5 * k]
            opp = k % 2 == 0
            rows.append({
                "ticker": firm, "date": sale, "value": 1e6,
                "opportunistic": opp, "gov_tier": tier, "strength": 0.8,
                "exc_21": -0.02 + rng.normal(0, 0.01),
                "exc_63": -0.03 + rng.normal(0, 0.01),
                "exc_126": -0.04 + rng.normal(0, 0.01),
                "exc_252": -0.05 + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def _prices():
    idx = pd.bdate_range("2022-01-03", periods=400)
    a = pd.Series(100.0 * 0.999 ** np.arange(400), index=idx, name="A")
    b = pd.Series(100.0 * 0.998 ** np.arange(400), index=idx, name="B")
    spy = pd.Series(100.0 * 1.0005 ** np.arange(400), index=idx, name="SPY")
    return {"A": a, "B": b}, spy


def test_run_inference_full_schema():
    import json
    prices, spy = _prices()
    res = run_inference(_panel(), prices, spy, horizons=(21, 63, 126, 252),
                        bootstrap_n=200, seed=0)
    json.dumps(res)  # must be plain-Python serializable (no numpy scalars leaking in)
    assert res["meta"]["bootstrap_n"] == 200
    for h in (21, 63, 126, 252):
        block = res["horizons"][str(h)]
        assert set(block) >= {
            "pooled_mean_equal", "pooled_mean_value", "opportunistic_contrast",
            "gov_tier_means", "calendar_time", "regression", "n_valid",
        }
        assert set(block["pooled_mean_equal"]) == {
            "estimate", "ci_lo", "ci_hi", "p_value", "n_boot",
        }
        assert set(block["gov_tier_means"]) == {"concentrated", "mixed", "dispersed"}
        assert block["calendar_time"]["n_days"] > 0
        assert [r["term"] for r in block["regression"]] == [
            "opportunistic", "strength", "log_value",
        ]


def test_findings_markdown_mentions_verdict():
    prices, spy = _prices()
    res = run_inference(_panel(), prices, spy, horizons=(21, 63),
                        bootstrap_n=100, seed=0)
    md = findings_markdown(res)
    assert "# S3" in md
    assert "H-B" in md
    assert "opportunistic" in md.lower()
