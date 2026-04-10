from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import resample_ohlcv
from src.performance import summarize_report
from src.strategy import (
    MonthlyState,
    Position,
    StrategyConfig,
    add_features,
    compute_position_size,
    is_breakout_entry,
    is_pullback_entry,
)


def _entry_fill(price: float, slippage: float) -> float:
    return price * (1 + slippage)


def _exit_fill(price: float, slippage: float) -> float:
    return price * (1 - slippage)


def run_backtest(
    df_1h: pd.DataFrame,
    config: StrategyConfig | None = None,
    starting_equity: float = 10_000.0,
    output_dir: str | Path = "outputs",
    data_source: str = "public_csv",
) -> dict:
    cfg = config or StrategyConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_4h = resample_ohlcv(df_1h, "4H")
    df_daily = resample_ohlcv(df_1h, "1D")
    df = add_features(df_4h, df_daily).dropna().copy()

    equity = starting_equity
    position: Position | None = None
    trades: list[dict] = []
    curve: list[dict] = []
    monthly = MonthlyState()
    forced_close_count = 0

    for ts, row in df.iterrows():
        current_month = ts.to_period("M")
        if monthly.month != current_month:
            monthly.month = current_month
            monthly.start_equity = equity
            monthly.size_multiplier = 1.0
            monthly.entries_disabled = False

        if monthly.start_equity and monthly.start_equity > 0:
            month_ret = (equity / monthly.start_equity) - 1
            if month_ret >= 0.10:
                monthly.size_multiplier = 0.5
            if month_ret <= -0.04:
                monthly.entries_disabled = True

        if position is not None:
            stop_hit = row["low"] <= position.stop_price
            tp1_hit = (not position.tp1_done) and row["high"] >= position.tp1_price
            tp2_hit = (not position.tp2_done) and row["high"] >= position.tp2_price
            tp3_hit = (not position.tp3_done) and row["high"] >= position.tp3_price

            if stop_hit:
                qty = position.remaining_qty
                fill = _exit_fill(position.stop_price, cfg.slippage_rate)
                pnl = (fill - position.entry_price) * qty
                fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                equity += pnl - fee
                r_mult = (fill - position.entry_price) / (position.entry_price - position.initial_stop)
                trades.append({
                    "timestamp": ts,
                    "event_type": "stop",
                    "price": fill,
                    "qty": qty,
                    "realized_pnl": pnl - fee,
                    "equity": equity,
                    "r_multiple": r_mult,
                })
                position = None
            else:
                if tp1_hit:
                    qty = position.qty * 0.4
                    fill = _exit_fill(position.tp1_price, cfg.slippage_rate)
                    pnl = (fill - position.entry_price) * qty
                    fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                    equity += pnl - fee
                    position.remaining_qty -= qty
                    position.tp1_done = True
                    position.stop_price = position.entry_price
                    r_mult = (fill - position.entry_price) / (position.entry_price - position.initial_stop)
                    trades.append({"timestamp": ts, "event_type": "tp1", "price": fill, "qty": qty, "realized_pnl": pnl - fee, "equity": equity, "r_multiple": r_mult})

                if tp2_hit and position is not None:
                    qty = position.qty * 0.3
                    fill = _exit_fill(position.tp2_price, cfg.slippage_rate)
                    pnl = (fill - position.entry_price) * qty
                    fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                    equity += pnl - fee
                    position.remaining_qty -= qty
                    position.tp2_done = True
                    trail = row["ema20"] - 0.5 * row["atr14"]
                    position.stop_price = max(position.stop_price, trail)
                    r_mult = (fill - position.entry_price) / (position.entry_price - position.initial_stop)
                    trades.append({"timestamp": ts, "event_type": "tp2", "price": fill, "qty": qty, "realized_pnl": pnl - fee, "equity": equity, "r_multiple": r_mult})

                if tp3_hit and position is not None:
                    qty = position.remaining_qty
                    fill = _exit_fill(position.tp3_price, cfg.slippage_rate)
                    pnl = (fill - position.entry_price) * qty
                    fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                    equity += pnl - fee
                    r_mult = (fill - position.entry_price) / (position.entry_price - position.initial_stop)
                    trades.append({"timestamp": ts, "event_type": "tp3", "price": fill, "qty": qty, "realized_pnl": pnl - fee, "equity": equity, "r_multiple": r_mult})
                    position = None

        if position is None and not monthly.entries_disabled:
            if is_breakout_entry(row) or is_pullback_entry(row):
                entry_price = _entry_fill(row["close"], cfg.slippage_rate)
                stop_dist = cfg.stop_atr_mult * row["atr14"]
                qty = compute_position_size(
                    equity=equity,
                    entry_price=entry_price,
                    stop_distance=stop_dist,
                    risk_pct=cfg.risk_pct,
                    max_leverage=cfg.max_leverage,
                    size_multiplier=monthly.size_multiplier,
                )
                if qty > 0:
                    stop_price = entry_price - stop_dist
                    fee = entry_price * qty * cfg.fee_rate
                    equity -= fee
                    position = Position(
                        entry_time=ts,
                        entry_price=entry_price,
                        qty=qty,
                        remaining_qty=qty,
                        stop_price=stop_price,
                        initial_stop=stop_price,
                        tp1_price=entry_price + cfg.tp1_r * stop_dist,
                        tp2_price=entry_price + cfg.tp2_r * stop_dist,
                        tp3_price=entry_price + cfg.tp3_r * stop_dist,
                    )
                    trades.append({
                        "timestamp": ts,
                        "event_type": "entry",
                        "price": entry_price,
                        "qty": qty,
                        "realized_pnl": -fee,
                        "equity": equity,
                        "r_multiple": None,
                        "size_multiplier": monthly.size_multiplier,
                    })

        mark_to_market = equity
        if position is not None:
            mark_to_market += position.remaining_qty * (row["close"] - position.entry_price)
        curve.append({"timestamp": ts, "equity": mark_to_market})

    if position is not None:
        final_ts = df.index[-1]
        final_price = _exit_fill(df.iloc[-1]["close"], cfg.slippage_rate)
        qty = position.remaining_qty
        pnl = (final_price - position.entry_price) * qty
        fee = (position.entry_price * qty + final_price * qty) * cfg.fee_rate
        equity += pnl - fee
        r_mult = (final_price - position.entry_price) / (position.entry_price - position.initial_stop)
        trades.append({
            "timestamp": final_ts,
            "event_type": "force_close",
            "price": final_price,
            "qty": qty,
            "realized_pnl": pnl - fee,
            "equity": equity,
            "r_multiple": r_mult,
        })
        forced_close_count += 1

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(curve).set_index("timestamp")

    report = summarize_report(
        starting_equity=starting_equity,
        ending_equity=equity,
        trades=trades_df,
        equity_curve=equity_df,
        forced_close_count=forced_close_count,
        data_source=data_source,
    )

    trades_df.to_csv(out_dir / "trades.csv", index=False)
    equity_df.to_csv(out_dir / "equity_curve.csv")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    plt.figure(figsize=(10, 4))
    plt.plot(equity_df.index, equity_df["equity"])
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png")
    plt.close()

    return report
