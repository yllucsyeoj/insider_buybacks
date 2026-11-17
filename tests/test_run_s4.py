import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "run_s4", Path(__file__).resolve().parents[1] / "scripts" / "run_s4.py")
run_s4_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_s4_mod)
run_s4 = run_s4_mod.run_s4


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2022-01-03", periods=60).append(
        pd.bdate_range("2024-02-01", periods=60))        # straddles the 2024 cutoff
    classes = ["routine_plan", "discretionary", "opportunistic_plan"]
    for i in range(8):
        shift = rng.normal(0, 0.02)
        for d in dates:
            s = rng.uniform(0.2, 0.95)
            rows.append({
                "ticker": f"F{i}", "date": d, "value": float(rng.integers(1e5, 1e7)),
                "class": classes[rng.integers(0, 3)], "strength": s,
                "exc_21": shift - 0.05 * s + rng.normal(0, 0.01),
                "exc_63": shift - 0.08 * s + rng.normal(0, 0.01),
                "exc_126": shift - 0.10 * s + rng.normal(0, 0.01),
                "exc_252": shift - 0.12 * s + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def test_run_s4_full_schema_and_serializable():
    res = run_s4(_panel(), horizons=(63, 126))
    json.dumps(res)                                         # no numpy scalars leak
    assert set(res) >= {"meta", "horizons", "verdict"}
    for h in ("63", "126"):
        blk = res["horizons"][h]
        assert set(blk) == {"full_sample", "time_holdout", "firm_holdout"}
        fs = blk["full_sample"]
        assert set(fs) == {"wls", "ols", "zscore", "wls_table"}
        assert set(fs["wls"]) >= {"coef", "p", "n", "n_firms"}
        assert [r["term"] for r in fs["wls_table"]] == run_s4_mod.REGRESSORS
        assert set(blk["time_holdout"]) == {"train", "test"}
        assert set(blk["firm_holdout"]) == {"half_a", "half_b"}
    assert set(res["verdict"]) >= {"by_horizon", "focal_robust"}


def test_verdict_truth_table():
    _v = run_s4_mod._verdict
    def blk(full_c, full_p, test_c, test_p, ha_c, hb_c):
        return {
            "full_sample": {"wls": {"coef": full_c, "p": full_p}},
            "time_holdout": {"test": {"coef": test_c, "p": test_p}},
            "firm_holdout": {"half_a": {"coef": ha_c}, "half_b": {"coef": hb_c}},
        }
    allpass = _v({"126": blk(-0.2, 0.01, -0.2, 0.02, -0.1, -0.1)})
    assert allpass["by_horizon"]["126"]["robust"] is True
    assert allpass["focal_robust"] is True
    testfail = _v({"126": blk(-0.2, 0.01, -0.2, 0.40, -0.1, -0.1)})
    assert testfail["by_horizon"]["126"]["robust"] is False
    signflip = _v({"126": blk(-0.2, 0.01, -0.2, 0.02, -0.1, +0.1)})
    assert signflip["by_horizon"]["126"]["robust"] is False


def test_findings_markdown_has_estimators_verdict_and_caveat():
    findings_markdown = run_s4_mod.findings_markdown
    res = run_s4(_panel(), horizons=(63, 126))
    md = findings_markdown(res)
    assert "# S4" in md
    assert "WLS" in md and "OLS" in md
    assert "strength" in md.lower()
    assert "spent" in md.lower() or "replication" in md.lower()   # honest caveat present
    assert "Verdict" in md
