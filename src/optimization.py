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


def build_optimization_summary(
    baseline_metrics: dict[str, float],
    results_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    output_dir: str | Path = "outputs",
    train_ratio: float = 0.7,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "optimization_summary.md"

    top10 = results_df.head(10)

    best_monthly_row = top10.sort_values("train_average_monthly_return_pct", ascending=False).iloc[0].to_dict()

    if oos_df.empty:
        best_tradeoff_row = {}
    else:
        ranked = oos_df.copy()
        ranked["tradeoff_score"] = (
            ranked["validation_average_monthly_return_pct"] * 3.0
            - ranked["validation_max_drawdown_pct"].abs() * 1.2
        )
        best_tradeoff_row = ranked.sort_values("tradeoff_score", ascending=False).iloc[0].to_dict()

    lines: list[str] = []
    lines.append("# Optimization Summary")
    lines.append("")
    lines.append("## Baseline metrics (current reproducible run)")
    for k, v in baseline_metrics.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Method and anti-overfitting assumptions")
    lines.append(
        f"- Data is split chronologically into train/validation using train_ratio={train_ratio:.2f}; "
        "no shuffling is used."
    )
    lines.append("- Parameter sweep rankings are based on train metrics only, then validated out-of-sample on the top 3.")
    lines.append("- Balanced score rewards average monthly return, penalizes drawdown, and enforces PF/trade-count quality gates.")
    lines.append("- Validation is used for selection sanity-checks rather than repeated retuning.")
    lines.append("")
    lines.append("## Top 10 train-ranked parameter sets")
    lines.append(f"- Saved to `{(out_dir / 'optimization_top10.csv').as_posix()}`.")
    lines.append("- Full grid results saved to `outputs/optimization_results.csv`.")
    lines.append("")
    lines.append("## Out-of-sample validation (top 3)")
    lines.append(f"- Saved to `{(out_dir / 'optimization_top3_oos.csv').as_posix()}`.")
    lines.append("")
    lines.append("## Best return/drawdown tradeoff set (from OOS top 3)")
    if best_tradeoff_row:
        lines.append(
            "- Validation rank "
            f"{int(best_tradeoff_row['validation_rank'])}: "
            f"avg_monthly={best_tradeoff_row['validation_average_monthly_return_pct']:.2f}%, "
            f"max_drawdown={best_tradeoff_row['validation_max_drawdown_pct']:.2f}%, "
            f"profit_factor={best_tradeoff_row['validation_profit_factor']:.2f}, "
            f"trades={int(best_tradeoff_row['validation_total_trades'])}"
        )
        lines.append(
            "- Parameters: "
            f"risk_pct={best_tradeoff_row['risk_pct']}, stop_atr_mult={best_tradeoff_row['stop_atr_mult']}, "
            f"tp=({best_tradeoff_row['tp1_r']}, {best_tradeoff_row['tp2_r']}, {best_tradeoff_row['tp3_r']}), "
            f"adx_threshold={best_tradeoff_row['adx_threshold']}, "
            f"breakout_lookback={int(best_tradeoff_row['breakout_lookback'])}, "
            f"pullback_rsi=[{best_tradeoff_row['pullback_rsi_min']}, {best_tradeoff_row['pullback_rsi_max']}], "
            f"trailing_atr_mult={best_tradeoff_row['trailing_atr_mult']}"
        )
    else:
        lines.append("- No validation rows available.")
    lines.append("")
    lines.append("## Set with strongest average monthly return improvement (train top 10)")
    lines.append(
        f"- avg_monthly={best_monthly_row['train_average_monthly_return_pct']:.2f}% "
        f"(baseline {baseline_metrics.get('average_monthly_return_pct', 0.0):.2f}%)."
    )
    lines.append(
        "- Parameters: "
        f"risk_pct={best_monthly_row['risk_pct']}, stop_atr_mult={best_monthly_row['stop_atr_mult']}, "
        f"tp=({best_monthly_row['tp1_r']}, {best_monthly_row['tp2_r']}, {best_monthly_row['tp3_r']}), "
        f"adx_threshold={best_monthly_row['adx_threshold']}, "
        f"breakout_lookback={int(best_monthly_row['breakout_lookback'])}, "
        f"pullback_rsi=[{best_monthly_row['pullback_rsi_min']}, {best_monthly_row['pullback_rsi_max']}], "
        f"trailing_atr_mult={best_monthly_row['trailing_atr_mult']}"
    )
    lines.append("")
    lines.append("## Caution")
    lines.append(
        "- Before deployment, run walk-forward testing across multiple market regimes to reduce single-split bias."
    )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path
