from __future__ import annotations

import random
from itertools import product
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest
from src.strategy import StrategyConfig


OPTIMIZATION_PARAM_KEYS = [
    "strategy_family",
    "risk_pct",
    "stop_atr_mult",
    "adx_threshold",
    "breakout_lookback",
    "pullback_rsi_min",
    "pullback_rsi_max",
    "trailing_atr_mult",
    "tp1_r",
    "tp2_r",
    "tp3_r",
    "entry_mode",
    "breakout_volume_mult",
    "breakout_close_buffer_atr",
    "reentry_cooldown_bars",
    "tp1_size",
    "tp2_size",
    "move_stop_to_breakeven_after_tp1",
    "use_monthly_controls",
    "use_regime_filter",
    "enable_short",
    "short_entry_mode",
    "v2_regime_mode",
    "v2_vol_stop_floor",
    "v2_vol_stop_ceiling",
    "v2_tp1_size",
    "v2_tp2_size",
    "v2_tp1_r",
    "v2_tp2_r",
    "v2_tp3_r",
    "v2_trail_activation_r",
    "v2_trail_swing_lookback",
    "v2_reentry_bars",
]


def _split_train_validation(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("train_ratio produced an empty train or validation segment")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _train_score(report: dict) -> float:
    avg_monthly = float(report.get("average_monthly_return_pct", 0.0) or 0.0)
    drawdown_abs = abs(float(report.get("max_drawdown_pct", 0.0) or 0.0))
    profit_factor = float(report.get("profit_factor", 0.0) or 0.0)
    trades = int(report.get("total_trades", 0) or 0)

    score = avg_monthly * 3.8
    score += min(max(profit_factor - 1.0, -1.0), 2.0) * 4.0
    score -= drawdown_abs * 1.2
    score += min(trades, 120) * 0.04

    if trades < 20:
        score -= (20 - trades) * 0.9
    if profit_factor < 1.2:
        score -= (1.2 - profit_factor) * 12.0
    if drawdown_abs > 25:
        score -= (drawdown_abs - 25) * 1.0

    return score


def _oos_score(row: pd.Series) -> float:
    val_monthly = float(row.get("validation_average_monthly_return_pct", 0.0) or 0.0)
    val_pf = float(row.get("validation_profit_factor", 0.0) or 0.0)
    val_dd = abs(float(row.get("validation_max_drawdown_pct", 0.0) or 0.0))
    val_trades = int(row.get("validation_total_trades", 0) or 0)

    train_monthly = float(row.get("train_average_monthly_return_pct", 0.0) or 0.0)
    overfit_gap = max(0.0, train_monthly - val_monthly)

    score = val_monthly * 4.0 + (val_pf - 1.0) * 5.0 - val_dd * 1.1 + min(val_trades, 80) * 0.05
    score -= overfit_gap * 2.2
    if val_trades < 20:
        score -= (20 - val_trades) * 2.0
    if val_pf < 1.1:
        score -= (1.1 - val_pf) * 10.0
    return score


def _config_grid() -> list[dict]:
    tp_structures = [
        (1.0, 2.2, 4.2),
        (1.2, 2.6, 4.8),
        (1.4, 3.0, 6.0),
    ]
    rows: list[dict] = []
    report_keys: list[str] | None = None
    rsi_ranges = [(44.0, 64.0), (46.0, 62.0), (48.0, 60.0)]
    baseline_rows = []
    for (
        risk_pct,
        stop_atr_mult,
        adx_threshold,
        breakout_lookback,
        rsi_range,
        trailing_atr_mult,
        tp,
        entry_mode,
        volume_mult,
        breakout_buffer,
        cooldown,
        tp_split,
        move_be,
        use_monthly_controls,
        use_regime_filter,
    ) in product(
        [0.006, 0.008, 0.010, 0.012],
        [1.5, 1.8, 2.1],
        [16.0, 20.0, 24.0],
        [15, 20, 30],
        rsi_ranges,
        [0.35, 0.5, 0.8],
        tp_structures,
        ["combined", "breakout", "pullback"],
        [1.2, 1.5],
        [0.0, 0.2],
        [0, 1],
        [(0.25, 0.25), (0.35, 0.30), (0.45, 0.25)],
        [True, False],
        [True, False],
        [True, False],
    ):
        pullback_rsi_min, pullback_rsi_max = rsi_range
        tp1_size, tp2_size = tp_split
        baseline_rows.append(
            {
                "strategy_family": "baseline",
                "risk_pct": risk_pct,
                "stop_atr_mult": stop_atr_mult,
                "adx_threshold": adx_threshold,
                "breakout_lookback": breakout_lookback,
                "pullback_rsi_min": pullback_rsi_min,
                "pullback_rsi_max": pullback_rsi_max,
                "trailing_atr_mult": trailing_atr_mult,
                "tp1_r": tp[0],
                "tp2_r": tp[1],
                "tp3_r": tp[2],
                "entry_mode": entry_mode,
                "breakout_volume_mult": volume_mult,
                "breakout_close_buffer_atr": breakout_buffer,
                "reentry_cooldown_bars": cooldown,
                "tp1_size": tp1_size,
                "tp2_size": tp2_size,
                "move_stop_to_breakeven_after_tp1": move_be,
                "use_monthly_controls": use_monthly_controls,
                "use_regime_filter": use_regime_filter,
                "enable_short": False,
                "short_entry_mode": "breakdown",
                "v2_regime_mode": "hybrid",
                "v2_vol_stop_floor": 1.1,
                "v2_vol_stop_ceiling": 2.6,
                "v2_tp1_size": 0.2,
                "v2_tp2_size": 0.2,
                "v2_tp1_r": 1.6,
                "v2_tp2_r": 3.2,
                "v2_tp3_r": 7.0,
                "v2_trail_activation_r": 2.4,
                "v2_trail_swing_lookback": 6,
                "v2_reentry_bars": 6,
            }
        )
    rows.extend(baseline_rows[:18])
    for (
        risk_pct,
        stop_atr_mult,
        adx_threshold,
        breakout_lookback,
        entry_mode,
        short_mode,
        v2_mode,
        stop_floor,
        stop_ceiling,
        trail_activation,
        trail_lb,
        reentry_bars,
    ) in product(
        [0.006, 0.008, 0.010],
        [1.5, 1.8, 2.1],
        [18.0, 22.0],
        [15, 24],
        ["combined", "breakout", "pullback"],
        ["combined", "breakdown", "pullback"],
        ["hybrid", "breakout", "pullback"],
        [1.0, 1.2],
        [2.4, 2.8],
        [2.0, 2.8, 3.6],
        [5, 8],
        [4, 8],
    ):
        rows.append(
            {
                "strategy_family": "v2",
                "risk_pct": risk_pct,
                "stop_atr_mult": stop_atr_mult,
                "adx_threshold": adx_threshold,
                "breakout_lookback": breakout_lookback,
                "pullback_rsi_min": 46.0,
                "pullback_rsi_max": 62.0,
                "trailing_atr_mult": 0.5,
                "tp1_r": 1.2,
                "tp2_r": 2.2,
                "tp3_r": 4.0,
                "entry_mode": entry_mode,
                "breakout_volume_mult": 1.2,
                "breakout_close_buffer_atr": 0.1,
                "reentry_cooldown_bars": 0,
                "tp1_size": 0.4,
                "tp2_size": 0.3,
                "move_stop_to_breakeven_after_tp1": False,
                "use_monthly_controls": False,
                "use_regime_filter": True,
                "enable_short": True,
                "short_entry_mode": short_mode,
                "v2_regime_mode": v2_mode,
                "v2_vol_stop_floor": stop_floor,
                "v2_vol_stop_ceiling": stop_ceiling,
                "v2_tp1_size": 0.15,
                "v2_tp2_size": 0.2,
                "v2_tp1_r": 1.8,
                "v2_tp2_r": 3.8,
                "v2_tp3_r": 8.0,
                "v2_trail_activation_r": trail_activation,
                "v2_trail_swing_lookback": trail_lb,
                "v2_reentry_bars": reentry_bars,
            }
        )
    max_grid_size = 60
    if len(rows) <= max_grid_size:
        return rows
    rng = random.Random(42)
    return rng.sample(rows, max_grid_size)


def _extract_config(row: pd.Series) -> dict:
    cfg = {}
    for key in OPTIMIZATION_PARAM_KEYS:
        value = row[key]
        if key in {"breakout_lookback", "reentry_cooldown_bars", "v2_trail_swing_lookback", "v2_reentry_bars"}:
            cfg[key] = int(value)
        elif key in {"move_stop_to_breakeven_after_tp1", "use_monthly_controls", "use_regime_filter", "enable_short"}:
            cfg[key] = bool(value)
        else:
            cfg[key] = value
    return cfg


def run_parameter_sweep(
    df_1h: pd.DataFrame,
    starting_equity: float = 10_000.0,
    output_dir: str | Path = "outputs",
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, validation_df = _split_train_validation(df_1h, train_ratio)
    grid = _config_grid()

    rows: list[dict] = []
    report_keys: list[str] | None = None
    for idx, params in enumerate(grid, start=1):
        cfg = StrategyConfig(**params)
        train_report = run_backtest(
            df_1h=train_df,
            config=cfg,
            starting_equity=starting_equity,
            output_dir=out_dir / "train_runs" / f"cfg_{idx:05d}",
            data_source="optimization_train",
            save_plot=False,
        )
        row = {**params}
        row.update({f"train_{k}": v for k, v in train_report.items()})
        row["train_score"] = _train_score(train_report)
        rows.append(row)
        if report_keys is None:
            report_keys = list(train_report.keys())

    results_df = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)

    top_candidates = results_df.head(10)
    oos_rows: list[dict] = []
    for rank, row in top_candidates.iterrows():
        cfg_params = _extract_config(row)
        cfg = StrategyConfig(**cfg_params)
        val_report = run_backtest(
            df_1h=validation_df,
            config=cfg,
            starting_equity=starting_equity,
            output_dir=out_dir / f"validation_rank_{rank + 1}",
            data_source="optimization_validation",
            save_plot=False,
        )
        out_row = {
            "validation_rank": rank + 1,
            **cfg_params,
            **{f"train_{k}": row[f"train_{k}"] for k in (report_keys or [])},
            **{f"validation_{k}": v for k, v in val_report.items()},
        }
        out_row["oos_score"] = _oos_score(pd.Series(out_row))
        oos_rows.append(out_row)

    oos_df = pd.DataFrame(oos_rows).sort_values("oos_score", ascending=False).reset_index(drop=True)

    meta = {
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "validation_start": str(validation_df.index.min()),
        "validation_end": str(validation_df.index.max()),
        "train_ratio": train_ratio,
        "tested_parameter_sets": len(grid),
        "train_score_formula": "3.8*avg_monthly + 4*(pf-1) - 1.2*|dd| + 0.04*trades with penalties for low trades/pf and extreme dd",
        "oos_score_formula": "4*val_monthly + 5*(val_pf-1) - 1.1*|val_dd| + 0.05*val_trades - 2.2*max(0, train_monthly-val_monthly) with low trades/pf penalties",
    }

    results_df.to_csv(out_dir / "optimization_results.csv", index=False)
    results_df.head(10).to_csv(out_dir / "optimization_top10.csv", index=False)
    oos_df.to_csv(out_dir / "optimization_top10_oos.csv", index=False)

    return results_df, oos_df, meta
