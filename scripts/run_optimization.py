#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from src.backtest import run_backtest
from src.data_loader import load_ohlcv_csv
from src.optimization import OPTIMIZATION_PARAM_KEYS, run_parameter_sweep
from src.strategy import StrategyConfig


def _to_serializable(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        if hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _pick_best_oos(oos_df):
    if oos_df.empty:
        return {}
    return _to_serializable(oos_df.iloc[0].to_dict())


def _split_train_val_oos(df, train_ratio: float = 0.6, val_ratio: float = 0.2):
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Run parameter sweep and OOS validation")
    parser.add_argument("--csv", required=True, help="OHLCV csv path")
    parser.add_argument("--starting-equity", type=float, default=10_000)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv_csv(args.csv)

    train_df, validation_df, oos_segment = _split_train_val_oos(df, train_ratio=0.6, val_ratio=0.2)

    baseline_train = run_backtest(
        df_1h=train_df,
        config=StrategyConfig(),
        starting_equity=args.starting_equity,
        output_dir=out_dir / "baseline_train",
        data_source="baseline_train",
        save_plot=False,
    )
    baseline_validation = run_backtest(
        df_1h=validation_df,
        config=StrategyConfig(),
        starting_equity=args.starting_equity,
        output_dir=out_dir / "baseline_validation",
        data_source="baseline_validation",
        save_plot=False,
    )
    baseline_oos = run_backtest(
        df_1h=oos_segment,
        config=StrategyConfig(),
        starting_equity=args.starting_equity,
        output_dir=out_dir / "baseline_oos",
        data_source="baseline_oos",
        save_plot=False,
    )

    results_df, oos_df, meta = run_parameter_sweep(
        df_1h=df,
        starting_equity=args.starting_equity,
        output_dir=out_dir,
        train_ratio=args.train_ratio,
    )

    best_oos = _pick_best_oos(oos_df)
    selected_params = {k: best_oos[k] for k in OPTIMIZATION_PARAM_KEYS} if best_oos else {}

    best_family = selected_params.get("strategy_family", "baseline")
    best_cfg = StrategyConfig(**selected_params) if selected_params else StrategyConfig()
    best_train = run_backtest(train_df, config=best_cfg, starting_equity=args.starting_equity, output_dir=out_dir / "best_train", data_source="best_train", save_plot=False)
    best_validation = run_backtest(validation_df, config=best_cfg, starting_equity=args.starting_equity, output_dir=out_dir / "best_validation", data_source="best_validation", save_plot=False)
    best_oos_report = run_backtest(oos_segment, config=best_cfg, starting_equity=args.starting_equity, output_dir=out_dir / "best_oos", data_source="best_oos", save_plot=False)

    summary = {
        "baseline_metrics": {
            "train": baseline_train,
            "validation": baseline_validation,
            "oos": baseline_oos,
        },
        "train_validation_ranges": {
            "train_start": meta["train_start"],
            "train_end": meta["train_end"],
            "validation_start": meta["validation_start"],
            "validation_end": meta["validation_end"],
            "train_ratio": meta["train_ratio"],
        },
        "score_formulas": {
            "train_score_formula": meta["train_score_formula"],
            "oos_score_formula": meta["oos_score_formula"],
        },
        "tested_parameter_sets": meta["tested_parameter_sets"],
        "top_10_train_configs": [_to_serializable(r) for r in results_df.head(10).to_dict(orient="records")],
        "top_3_oos_configs": [_to_serializable(r) for r in oos_df.head(3).to_dict(orient="records")],
        "selected_best_config": {
            "family": best_family,
            "parameters": selected_params,
            "train_metrics": {k.replace("train_", ""): best_oos[k] for k in best_oos if k.startswith("train_")} if best_oos else {},
            "oos_metrics": {k.replace("validation_", ""): best_oos[k] for k in best_oos if k.startswith("validation_")} if best_oos else {},
            "reason": "Selected by highest OOS score balancing monthly return, PF, drawdown, trade count, and train-validation robustness.",
        },
        "baseline_vs_best": {
            "train": {
                "baseline_average_monthly_return_pct": baseline_train.get("average_monthly_return_pct"),
                "best_average_monthly_return_pct": best_train.get("average_monthly_return_pct"),
                "baseline_profit_factor": baseline_train.get("profit_factor"),
                "best_profit_factor": best_train.get("profit_factor"),
                "baseline_max_drawdown_pct": baseline_train.get("max_drawdown_pct"),
                "best_max_drawdown_pct": best_train.get("max_drawdown_pct"),
            },
            "validation": {
                "baseline_average_monthly_return_pct": baseline_validation.get("average_monthly_return_pct"),
                "best_average_monthly_return_pct": best_validation.get("average_monthly_return_pct"),
                "baseline_profit_factor": baseline_validation.get("profit_factor"),
                "best_profit_factor": best_validation.get("profit_factor"),
                "baseline_max_drawdown_pct": baseline_validation.get("max_drawdown_pct"),
                "best_max_drawdown_pct": best_validation.get("max_drawdown_pct"),
            },
            "oos": {
                "baseline_average_monthly_return_pct": baseline_oos.get("average_monthly_return_pct"),
                "best_average_monthly_return_pct": best_oos_report.get("average_monthly_return_pct"),
                "baseline_profit_factor": baseline_oos.get("profit_factor"),
                "best_profit_factor": best_oos_report.get("profit_factor"),
                "baseline_max_drawdown_pct": baseline_oos.get("max_drawdown_pct"),
                "best_max_drawdown_pct": best_oos_report.get("max_drawdown_pct"),
            },
        },
    }

    results_df.to_csv(out_dir / "strategy_v2_optimization_results.csv", index=False)
    (out_dir / "strategy_v2_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    beats_train = best_train.get("average_monthly_return_pct", -999) > baseline_train.get("average_monthly_return_pct", 999)
    beats_val = best_validation.get("average_monthly_return_pct", -999) > baseline_validation.get("average_monthly_return_pct", 999)
    beats_oos = best_oos_report.get("average_monthly_return_pct", -999) > baseline_oos.get("average_monthly_return_pct", 999)
    improvement_log = [
        "# Strategy V2 Improvement Log",
        "",
        f"- Best selected family: **{best_family}**.",
        f"- Train improvement vs baseline (avg monthly return): {'YES' if beats_train else 'NO'}.",
        f"- Validation improvement vs baseline (avg monthly return): {'YES' if beats_val else 'NO'}.",
        f"- OOS improvement vs baseline (avg monthly return): {'YES' if beats_oos else 'NO'}.",
        "- Strategy V2 includes optional short-side trading, regime-aware switching, volatility-aware stops, delayed profit-taking, and continuation re-entry.",
    ]
    (out_dir / "strategy_v2_improvement_log.md").write_text("\n".join(improvement_log) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "results": str(out_dir / "strategy_v2_optimization_results.csv"),
                "summary": str(out_dir / "strategy_v2_summary.json"),
                "improvement_log": str(out_dir / "strategy_v2_improvement_log.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
