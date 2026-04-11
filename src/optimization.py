from __future__ import annotations

import random
from itertools import product
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest
from src.strategy import StrategyConfig


OPTIMIZATION_PARAM_KEYS = [
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
        rows.append(
            {
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
            }
        )
    max_grid_size = 320
    if len(rows) <= max_grid_size:
        return rows
    rng = random.Random(42)
    return rng.sample(rows, max_grid_size)


def _extract_config(row: pd.Series) -> dict:
    cfg = {}
    for key in OPTIMIZATION_PARAM_KEYS:
        value = row[key]
        if key in {"breakout_lookback", "reentry_cooldown_bars"}:
            cfg[key] = int(value)
        elif key in {"move_stop_to_breakeven_after_tp1", "use_monthly_controls", "use_regime_filter"}:
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
