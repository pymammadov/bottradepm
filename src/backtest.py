from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import resample_ohlcv
from .performance import summarize_report
from .strategy import (
    MonthlyState,
    Position,
    StrategyConfig,
    add_features,
    compute_position_size,
    is_breakdown_entry,
    is_breakout_entry,
    is_pullback_entry,
    is_short_pullback_entry,
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
    save_plot: bool = True,
) -> dict:
    cfg = config or StrategyConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "timestamp" in df_1h.columns:
        df_1h = df_1h.set_index("timestamp")

    df_4h = resample_ohlcv(df_1h, "4h")
    df_daily = resample_ohlcv(df_1h, "1d")
    df = add_features(df_4h, df_daily, cfg).dropna().copy()
    if df.empty:
        report = summarize_report(
            starting_equity=starting_equity,
            ending_equity=starting_equity,
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame({"equity": []}),
            forced_close_count=0,
            data_source=data_source,
            monthly_min_target_pct=cfg.monthly_min_return_pct * 100,
        )
        pd.DataFrame().to_csv(out_dir / "trades.csv", index=False)
        pd.DataFrame({"equity": []}).to_csv(out_dir / "equity_curve.csv")
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    equity = starting_equity
    position: Position | None = None
    trades: list[dict] = []
    curve: list[dict] = []
    monthly = MonthlyState()
    forced_close_count = 0
    cooldown_bars = 0

    for ts, row in df.iterrows():
        current_month = ts.strftime("%Y-%m")
        if monthly.month != current_month:
            monthly.month = current_month
            monthly.start_equity = equity
            monthly.size_multiplier = 1.0
            monthly.entries_disabled = False
            monthly.monthly_return_pct = 0.0
            monthly.target_met = False

        if monthly.start_equity and monthly.start_equity > 0:
            month_ret = (equity / monthly.start_equity) - 1
            monthly.monthly_return_pct = month_ret

            if cfg.use_monthly_controls:
                if month_ret >= cfg.monthly_min_return_pct:
                    monthly.target_met = True
                    monthly.size_multiplier = max(0.3, monthly.size_multiplier - 0.1)
                elif month_ret >= cfg.scale_up_threshold:
                    monthly.size_multiplier = min(1.5, monthly.size_multiplier + 0.15)
                elif month_ret <= cfg.loss_stop_threshold:
                    monthly.entries_disabled = True

        if position is not None:
            is_long = position.side == 1
            stop_hit = row["low"] <= position.stop_price if is_long else row["high"] >= position.stop_price
            tp1_hit = (not position.tp1_done) and (row["high"] >= position.tp1_price if is_long else row["low"] <= position.tp1_price)
            tp2_hit = (not position.tp2_done) and (row["high"] >= position.tp2_price if is_long else row["low"] <= position.tp2_price)
            tp3_hit = (not position.tp3_done) and (row["high"] >= position.tp3_price if is_long else row["low"] <= position.tp3_price)

            if stop_hit:
                qty = position.remaining_qty
                fill = _exit_fill(position.stop_price, cfg.slippage_rate)
                pnl = (fill - position.entry_price) * qty * position.side
                fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                equity += pnl - fee
                risk_per_unit = abs(position.entry_price - position.initial_stop)
                r_mult = ((fill - position.entry_price) * position.side) / risk_per_unit
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
                cooldown_bars = cfg.reentry_cooldown_bars
            else:
                if tp1_hit:
                    qty = min(position.remaining_qty, position.qty * cfg.tp1_size)
                    fill = _exit_fill(position.tp1_price, cfg.slippage_rate)
                    pnl = (fill - position.entry_price) * qty * position.side
                    fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                    equity += pnl - fee
                    position.remaining_qty -= qty
                    position.tp1_done = True
                    if cfg.move_stop_to_breakeven_after_tp1:
                        position.stop_price = position.entry_price
                    risk_per_unit = abs(position.entry_price - position.initial_stop)
                    r_mult = ((fill - position.entry_price) * position.side) / risk_per_unit
                    trades.append({"timestamp": ts, "event_type": "tp1", "price": fill, "qty": qty, "realized_pnl": pnl - fee, "equity": equity, "r_multiple": r_mult})

                if tp2_hit and position is not None:
                    qty = min(position.remaining_qty, position.qty * cfg.tp2_size)
                    fill = _exit_fill(position.tp2_price, cfg.slippage_rate)
                    pnl = (fill - position.entry_price) * qty * position.side
                    fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                    equity += pnl - fee
                    position.remaining_qty -= qty
                    position.tp2_done = True
                    trail = row["ema20"] - cfg.trailing_atr_mult * row["atr14"] if is_long else row["ema20"] + cfg.trailing_atr_mult * row["atr14"]
                    position.stop_price = max(position.stop_price, trail) if is_long else min(position.stop_price, trail)
                    risk_per_unit = abs(position.entry_price - position.initial_stop)
                    r_mult = ((fill - position.entry_price) * position.side) / risk_per_unit
                    trades.append({"timestamp": ts, "event_type": "tp2", "price": fill, "qty": qty, "realized_pnl": pnl - fee, "equity": equity, "r_multiple": r_mult})

                if tp3_hit and position is not None:
                    qty = position.remaining_qty
                    fill = _exit_fill(position.tp3_price, cfg.slippage_rate)
                    pnl = (fill - position.entry_price) * qty * position.side
                    fee = (position.entry_price * qty + fill * qty) * cfg.fee_rate
                    equity += pnl - fee
                    risk_per_unit = abs(position.entry_price - position.initial_stop)
                    r_mult = ((fill - position.entry_price) * position.side) / risk_per_unit
                    trades.append({"timestamp": ts, "event_type": "tp3", "price": fill, "qty": qty, "realized_pnl": pnl - fee, "equity": equity, "r_multiple": r_mult})
                    position = None
                    cooldown_bars = cfg.reentry_cooldown_bars

        if cooldown_bars > 0:
            cooldown_bars -= 1

        if position is None and not monthly.entries_disabled and cooldown_bars == 0:
            breakout_signal = is_breakout_entry(row, cfg)
            pullback_signal = is_pullback_entry(row, cfg)
            should_enter = (
                (cfg.entry_mode == "combined" and (breakout_signal or pullback_signal))
                or (cfg.entry_mode == "breakout" and breakout_signal)
                or (cfg.entry_mode == "pullback" and pullback_signal)
            )
            side = 1
            if cfg.enable_short:
                short_breakdown = is_breakdown_entry(row, cfg)
                short_pullback = is_short_pullback_entry(row, cfg)
                short_ok = (
                    (cfg.short_entry_mode == "combined" and (short_breakdown or short_pullback))
                    or (cfg.short_entry_mode == "breakdown" and short_breakdown)
                    or (cfg.short_entry_mode == "pullback" and short_pullback)
                )
                if short_ok and not should_enter:
                    should_enter = True
                    side = -1
                elif short_ok and should_enter and bool(row.get("high_vol_regime", False)):
                    side = -1
            if should_enter:
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
                    stop_price = entry_price - stop_dist if side == 1 else entry_price + stop_dist
                    fee = entry_price * qty * cfg.fee_rate
                    equity -= fee
                    position = Position(
                        entry_time=ts,
                        entry_price=entry_price,
                        qty=qty,
                        remaining_qty=qty,
                        stop_price=stop_price,
                        initial_stop=stop_price,
                        tp1_price=entry_price + side * cfg.tp1_r * stop_dist,
                        tp2_price=entry_price + side * cfg.tp2_r * stop_dist,
                        tp3_price=entry_price + side * cfg.tp3_r * stop_dist,
                        side=side,
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
                        "side": side,
                    })

        mark_to_market = equity
        if position is not None:
            mark_to_market += position.remaining_qty * (row["close"] - position.entry_price) * position.side
        curve.append({"timestamp": ts, "equity": mark_to_market})

    if position is not None:
        final_ts = df.index[-1]
        final_price = _exit_fill(df.iloc[-1]["close"], cfg.slippage_rate)
        qty = position.remaining_qty
        pnl = (final_price - position.entry_price) * qty * position.side
        fee = (position.entry_price * qty + final_price * qty) * cfg.fee_rate
        equity += pnl - fee
        risk_per_unit = abs(position.entry_price - position.initial_stop)
        r_mult = ((final_price - position.entry_price) * position.side) / risk_per_unit
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

    trade_columns = ["timestamp", "event_type", "price", "qty", "realized_pnl", "equity", "r_multiple", "size_multiplier"]
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=trade_columns)
    else:
        for col in trade_columns:
            if col not in trades_df.columns:
                trades_df[col] = None

    equity_curve_raw = pd.DataFrame(curve)
    if equity_curve_raw.empty:
        equity_df = pd.DataFrame({"equity": [starting_equity]}, index=[df.index[0]])
    else:
        equity_df = equity_curve_raw.set_index("timestamp")

    report = summarize_report(
        starting_equity=starting_equity,
        ending_equity=equity,
        trades=trades_df,
        equity_curve=equity_df,
        forced_close_count=forced_close_count,
        data_source=data_source,
        monthly_min_target_pct=cfg.monthly_min_return_pct * 100,
    )

    trades_df.to_csv(out_dir / "trades.csv", index=False)
    equity_df.to_csv(out_dir / "equity_curve.csv")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    if save_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(equity_df.index, equity_df["equity"])
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png")
        plt.close()

    return report
