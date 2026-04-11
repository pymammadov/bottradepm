import pandas as pd

from src.backtest import run_backtest
from src.strategy import StrategyConfig


def make_fixture() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=24 * 140, freq="H", tz="UTC")
    base = pd.Series([10000 + i for i in range(len(idx))], index=idx, dtype=float)
    df = pd.DataFrame(index=idx)
    df["open"] = base
    df["high"] = base + 20
    df["low"] = base - 20
    df["close"] = base + 5
    df["volume"] = 10_000

    # engineer a breakout and TP sequence near the end
    pivot = idx[-10]
    df.loc[pivot, ["open", "close", "high", "low", "volume"]] = [13000, 13300, 13350, 12980, 50_000]
    df.loc[idx[-9], ["open", "close", "high", "low", "volume"]] = [13300, 13600, 15000, 13200, 60_000]
    df.loc[idx[-8], ["open", "close", "high", "low", "volume"]] = [13600, 13900, 17000, 13550, 60_000]
    return df


def test_partial_tps_and_force_close_and_reproducibility(tmp_path):
    cfg = StrategyConfig(fee_rate=0.0, slippage_rate=0.0)
    df = make_fixture()

    out1 = tmp_path / "o1"
    out2 = tmp_path / "o2"
    r1 = run_backtest(df, config=cfg, output_dir=out1, data_source="synthetic_demo")
    r2 = run_backtest(df, config=cfg, output_dir=out2, data_source="synthetic_demo")

    t1 = pd.read_csv(out1 / "trades.csv")
    assert (t1["event_type"] == "tp1").any()
    assert (t1["event_type"] == "tp2").any()
    # tp3 or forced close acceptable depending on levels reached
    assert (t1["event_type"].isin(["tp3", "force_close"])).any()

    entries = t1[t1["event_type"] == "entry"]
    assert len(entries) >= 1

    # deterministic report
    assert r1["ending_equity"] == r2["ending_equity"]


def test_monthly_kill_switch_disables_entries(tmp_path):
    idx = pd.date_range("2025-01-01", periods=24 * 220, freq="H", tz="UTC")
    df = pd.DataFrame(index=idx)
    trend = pd.Series(10000.0, index=idx)
    df["open"] = trend
    df["high"] = trend + 100
    df["low"] = trend - 100
    df["close"] = trend + 50
    df["volume"] = 50_000

    # create very large down move after likely entry month to trigger -4%
    df.iloc[-50:, df.columns.get_loc("low")] = 9000
    df.iloc[-50:, df.columns.get_loc("close")] = 9050

    cfg = StrategyConfig(fee_rate=0.0, slippage_rate=0.0)
    run_backtest(df, config=cfg, output_dir=tmp_path, data_source="synthetic_demo")
    trades = pd.read_csv(tmp_path / "trades.csv")

    # entries should be limited due to month kill-switch; sanity bound
    assert (trades["event_type"] == "entry").sum() < 8
