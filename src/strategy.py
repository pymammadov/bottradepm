from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.indicators import adx, atr, ema, rsi


@dataclass
class StrategyConfig:
    risk_pct: float = 0.0075
    max_leverage: float = 2.0
    stop_atr_mult: float = 1.8
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    tp1_r: float = 1.2
    tp2_r: float = 2.2
    tp3_r: float = 4.0
    adx_threshold: float = 18.0
    breakout_lookback: int = 20
    pullback_rsi_min: float = 48.0
    pullback_rsi_max: float = 62.0
    trailing_atr_mult: float = 0.5
    monthly_min_return_pct: float = 0.10  # Minimum 10% monthly return target
    scale_up_threshold: float = 0.08  # Scale up position size at 8%
    scale_down_threshold: float = 0.12  # Scale down at 12%
    loss_stop_threshold: float = -0.04  # Stop entries at -4% monthly loss


@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    remaining_qty: float
    stop_price: float
    initial_stop: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    tp1_done: bool = False
    tp2_done: bool = False
    tp3_done: bool = False


@dataclass
class MonthlyState:
    month: pd.Period | None = None
    start_equity: float | None = None
    size_multiplier: float = 1.0
    entries_disabled: bool = False
    monthly_return_pct: float = 0.0
    target_met: bool = False


def add_features(df_4h: pd.DataFrame, df_daily: pd.DataFrame, config: StrategyConfig | None = None) -> pd.DataFrame:
    cfg = config or StrategyConfig()
    out = df_4h.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["atr14"] = atr(out, 14)
    out["rsi14"] = rsi(out["close"], 14)
    out["prior_lookback_high"] = out["high"].rolling(cfg.breakout_lookback).max().shift(1)
    out["vol_ma20"] = out["volume"].rolling(20).mean()

    d = df_daily.copy()
    d["d_ema50"] = ema(d["close"], 50)
    d["d_ema200"] = ema(d["close"], 200)
    d["d_adx14"] = adx(d, 14)
    d["daily_regime"] = (
        (d["close"] > d["d_ema200"])
        & (d["d_ema50"] > d["d_ema200"])
        & (d["d_adx14"] > cfg.adx_threshold)
    ).shift(1)

    cols = d[["daily_regime", "d_ema50", "d_ema200", "d_adx14"]]
    out = out.merge(cols, left_index=True, right_index=True, how="left")
    out[["daily_regime", "d_ema50", "d_ema200", "d_adx14"]] = out[
        ["daily_regime", "d_ema50", "d_ema200", "d_adx14"]
    ].ffill()
    return out


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_distance: float,
    risk_pct: float,
    max_leverage: float,
    size_multiplier: float = 1.0,
) -> float:
    risk_amount = equity * risk_pct * size_multiplier
    raw_qty = risk_amount / stop_distance
    cap_qty = (equity * max_leverage) / entry_price
    return max(0.0, min(raw_qty, cap_qty))


def is_breakout_entry(row: pd.Series, config: StrategyConfig | None = None) -> bool:
    _ = config or StrategyConfig()
    return bool(
        row["daily_regime"]
        and row["close"] > row["prior_lookback_high"]
        and row["volume"] > 1.5 * row["vol_ma20"]
        and row["close"] > row["ema20"]
        and row["close"] > row["ema50"]
    )


def is_pullback_entry(row: pd.Series, config: StrategyConfig | None = None) -> bool:
    cfg = config or StrategyConfig()
    touch_ema = (row["low"] <= row["ema20"] <= row["close"]) or (row["low"] <= row["ema50"] <= row["close"])
    bullish = row["close"] > row["open"]
    rsi_ok = cfg.pullback_rsi_min <= row["rsi14"] <= cfg.pullback_rsi_max
    return bool(row["daily_regime"] and touch_ema and bullish and rsi_ok and row["close"] > row["ema50"])
