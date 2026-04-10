from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest
from src.strategy import StrategyConfig


def _split_train_validation(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("train_ratio produced an empty train or validation segment")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _balanced_score(report: dict) -> float:
    avg_monthly = float(report.get("average_monthly_return_pct", 0.0) or 0.0)
    drawdown_abs = abs(float(report.get("max_drawdown_pct", 0.0) or 0.0))
    profit_factor = float(report.get("profit_factor", 0.0) or 0.0)
    trades = int(report.get("total_trades", 0) or 0)

    score = avg_monthly * 3.0
    score -= drawdown_abs * 0.9

    if profit_factor < 1.4:
        score -= (1.4 - profit_factor) * 8.0
    else:
        score += min((profit_factor - 1.4) * 2.0, 2.5)

    if trades < 25:
        score -= (25 - trades) * 0.3
    else:
        score += min((trades - 25) * 0.03, 1.5)

    return score


def _config_grid() -> list[dict]:
    tp_structures = [
        (1.0, 2.0, 3.2),
        (1.2, 2.2, 4.0),
        (1.4, 2.6, 4.5),
    ]
    rows: list[dict] = []
    rsi_ranges = [(45.0, 60.0), (48.0, 62.0), (50.0, 65.0)]
    for (
        risk_pct,
        stop_atr_mult,
        adx_threshold,
        breakout_lookback,
        rsi_range,
        trailing_atr_mult,
        tp,
    ) in product(
        [0.006, 0.008, 0.01],
        [1.6, 1.8],
        [18.0, 22.0],
        [20, 30],
        rsi_ranges,
        [0.4, 0.7],
        tp_structures,
    ):
        pullback_rsi_min, pullback_rsi_max = rsi_range
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
            }
        )
    return rows


def run_parameter_sweep(
    df_1h: pd.DataFrame,
    starting_equity: float = 10_000.0,
    output_dir: str | Path = "outputs",
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, validation_df = _split_train_validation(df_1h, train_ratio)
    grid = _config_grid()

    rows: list[dict] = []
    for params in grid:
        cfg = StrategyConfig(**params)
        train_report = run_backtest(
            df_1h=train_df,
            config=cfg,
            starting_equity=starting_equity,
            output_dir=out_dir / "train_runs",
            data_source="optimization_train",
        )
        row = {**params}
        row.update({f"train_{k}": v for k, v in train_report.items()})
        row["balanced_score"] = _balanced_score(train_report)
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("balanced_score", ascending=False).reset_index(drop=True)

    top3_rows = results_df.head(3)
    oos_rows: list[dict] = []
    for rank, row in top3_rows.iterrows():
        cfg_params = {
            "risk_pct": row["risk_pct"],
            "stop_atr_mult": row["stop_atr_mult"],
            "adx_threshold": row["adx_threshold"],
            "breakout_lookback": int(row["breakout_lookback"]),
            "pullback_rsi_min": row["pullback_rsi_min"],
            "pullback_rsi_max": row["pullback_rsi_max"],
            "trailing_atr_mult": row["trailing_atr_mult"],
            "tp1_r": row["tp1_r"],
            "tp2_r": row["tp2_r"],
            "tp3_r": row["tp3_r"],
        }
        cfg = StrategyConfig(**cfg_params)
        val_report = run_backtest(
            df_1h=validation_df,
            config=cfg,
            starting_equity=starting_equity,
            output_dir=out_dir / f"validation_rank_{rank + 1}",
            data_source="optimization_validation",
        )
        out_row = {
            "validation_rank": rank + 1,
            **cfg_params,
            **{f"validation_{k}": v for k, v in val_report.items()},
        }
        oos_rows.append(out_row)

    oos_df = pd.DataFrame(oos_rows)

    results_df.to_csv(out_dir / "optimization_results.csv", index=False)
    results_df.head(10).to_csv(out_dir / "optimization_top10.csv", index=False)
    oos_df.to_csv(out_dir / "optimization_top3_oos.csv", index=False)

    return results_df, oos_df
