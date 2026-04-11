from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def max_drawdown_pct(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min() * 100)


def average_monthly_return_pct(equity_df: pd.DataFrame) -> float:
    monthly = equity_df.resample("ME").last()["equity"]
    if len(monthly) < 2:
        return float("nan")
    returns = monthly.pct_change().dropna()
    if returns.empty:
        return float("nan")
    return float(returns.mean() * 100)


def summarize_report(
    starting_equity: float,
    ending_equity: float,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    forced_close_count: int,
    data_source: str,
) -> dict[str, Any]:
    total_return_pct = ((ending_equity / starting_equity) - 1) * 100
    if trades.empty or "event_type" not in trades.columns:
        closed = pd.DataFrame()
        total_trades = 0
    else:
        closed = trades[trades["event_type"].isin(["stop", "tp1", "tp2", "tp3", "force_close"])]
        total_trades = int((trades["event_type"] == "entry").sum())

    trade_pnls = closed["realized_pnl"].dropna() if not closed.empty else pd.Series(dtype=float)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    win_rate = (len(wins) / len(trade_pnls) * 100) if len(trade_pnls) else math.nan
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else math.nan
    expectancy = float(trade_pnls.mean()) if len(trade_pnls) else 0.0
    avg_r = float(closed["r_multiple"].dropna().mean()) if not closed.empty and "r_multiple" in closed else 0.0

    return {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return_pct": total_return_pct,
        "average_monthly_return_pct": average_monthly_return_pct(equity_curve),
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct(equity_curve["equity"]),
        "expectancy_per_trade": expectancy,
        "average_r_multiple": avg_r,
        "forced_closes": forced_close_count,
        "data_source": data_source,
    }
