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

    baseline_report = run_backtest(
        df_1h=df,
        config=StrategyConfig(),
        starting_equity=args.starting_equity,
        output_dir=out_dir,
        data_source="baseline_full",
    )

    results_df, oos_df, meta = run_parameter_sweep(
        df_1h=df,
        starting_equity=args.starting_equity,
        output_dir=out_dir,
        train_ratio=args.train_ratio,
    )

    best_oos = _pick_best_oos(oos_df)
    selected_params = {k: best_oos[k] for k in OPTIMIZATION_PARAM_KEYS} if best_oos else {}

    summary = {
        "baseline_metrics": baseline_report,
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
            "parameters": selected_params,
            "train_metrics": {k.replace("train_", ""): best_oos[k] for k in best_oos if k.startswith("train_")} if best_oos else {},
            "oos_metrics": {k.replace("validation_", ""): best_oos[k] for k in best_oos if k.startswith("validation_")} if best_oos else {},
            "reason": "Selected by highest OOS score balancing monthly return, PF, drawdown, trade count, and train-validation robustness.",
        },
    }

    (out_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    best_report = {
        "best_model": summary["selected_best_config"],
        "comparison_vs_baseline": {
            "baseline_average_monthly_return_pct": baseline_report.get("average_monthly_return_pct"),
            "best_train_average_monthly_return_pct": summary["selected_best_config"]["train_metrics"].get("average_monthly_return_pct"),
            "best_oos_average_monthly_return_pct": summary["selected_best_config"]["oos_metrics"].get("average_monthly_return_pct"),
            "baseline_profit_factor": baseline_report.get("profit_factor"),
            "best_oos_profit_factor": summary["selected_best_config"]["oos_metrics"].get("profit_factor"),
            "baseline_max_drawdown_pct": baseline_report.get("max_drawdown_pct"),
            "best_oos_max_drawdown_pct": summary["selected_best_config"]["oos_metrics"].get("max_drawdown_pct"),
        },
    }
    (out_dir / "best_model_report.json").write_text(json.dumps(best_report, indent=2, default=str), encoding="utf-8")

    print(json.dumps({"summary": str(out_dir / "optimization_summary.json"), "best": str(out_dir / "best_model_report.json")}, indent=2))


if __name__ == "__main__":
    main()
