import pandas as pd

from src.backtest import run_backtest


def test_backtest_smoke(tmp_path):
    idx = pd.date_range("2024-01-01", periods=24 * 300, freq="h", tz="UTC")
    base = pd.Series(range(len(idx)), index=idx, dtype=float) + 30_000
    df = pd.DataFrame(index=idx)
    df["open"] = base
    df["high"] = base + 50
    df["low"] = base - 50
    df["close"] = base + 10
    df["volume"] = 10000

    report = run_backtest(df, output_dir=tmp_path, data_source="synthetic_demo")

    assert "ending_equity" in report
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "report.json").exists()
