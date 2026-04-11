from __future__ import annotations

import json

import pandas as pd

from src.institutional import run_institutional_pipeline


MANDATORY = [
    "report.json",
    "optimization_results.csv",
    "optimization_summary.json",
    "best_model_report.json",
    "improvement_log.md",
    "stress_test_summary.json",
    "oos_comparison.json",
    "regime_analysis.json",
]


def test_institutional_pipeline_outputs(tmp_path):
    idx = pd.date_range("2023-01-01", periods=24 * 500, freq="h", tz="UTC")
    base = pd.Series(range(len(idx)), index=idx, dtype=float) + 20_000
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": base,
            "high": base + 80,
            "low": base - 80,
            "close": base + 10,
            "volume": 10000,
        }
    )
    csv_path = tmp_path / "btc.csv"
    df.to_csv(csv_path, index=False)

    run_institutional_pipeline(str(csv_path), str(tmp_path / "outputs"), starting_equity=20_000)

    for name in MANDATORY:
        assert (tmp_path / "outputs" / name).exists(), f"Missing {name}"

    report = json.loads((tmp_path / "outputs" / "report.json").read_text())
    assert "best_oos" in report
