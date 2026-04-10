#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.data_loader import load_ohlcv_csv
from src.optimization import build_optimization_summary, run_parameter_sweep


def _pick_best_tradeoff(oos_df):
    if oos_df.empty:
        return {}
    ranked = oos_df.copy()
    ranked["tradeoff_score"] = (
        ranked["validation_average_monthly_return_pct"] * 3.0
        - ranked["validation_max_drawdown_pct"].abs() * 1.2
    )
    ranked = ranked.sort_values("tradeoff_score", ascending=False)
    return ranked.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parameter sweep and OOS validation")
    parser.add_argument("--csv", required=True, help="OHLCV csv path")
    parser.add_argument("--starting-equity", type=float, default=10_000)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--baseline-total-return-pct", type=float, default=10.23)
    parser.add_argument("--baseline-average-monthly-return-pct", type=float, default=0.38)
    parser.add_argument("--baseline-total-trades", type=int, default=44)
    parser.add_argument("--baseline-win-rate-pct", type=float, default=70.45)
    parser.add_argument("--baseline-profit-factor", type=float, default=1.73)
    parser.add_argument("--baseline-max-drawdown-pct", type=float, default=-4.44)
    parser.add_argument("--baseline-average-r-multiple", type=float, default=0.96)
    args = parser.parse_args()

    df = load_ohlcv_csv(args.csv)
    results_df, oos_df = run_parameter_sweep(
        df_1h=df,
        starting_equity=args.starting_equity,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )

    top10 = results_df.head(10)
    best_tradeoff = _pick_best_tradeoff(oos_df)
    baseline_metrics = {
        "total_return_pct": float(args.baseline_total_return_pct),
        "average_monthly_return_pct": float(args.baseline_average_monthly_return_pct),
        "total_trades": int(args.baseline_total_trades),
        "win_rate_pct": float(args.baseline_win_rate_pct),
        "profit_factor": float(args.baseline_profit_factor),
        "max_drawdown_pct": float(args.baseline_max_drawdown_pct),
        "average_r_multiple": float(args.baseline_average_r_multiple),
    }
    summary_path = build_optimization_summary(
        baseline_metrics=baseline_metrics,
        results_df=results_df,
        oos_df=oos_df,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )
    payload = {
        "tested_parameter_sets": int(len(results_df)),
        "top_10": top10.to_dict(orient="records"),
        "top_3_oos": oos_df.to_dict(orient="records"),
        "best_tradeoff_oos": best_tradeoff,
        "summary_report": str(summary_path),
        "notes": (
            "Train/validation split reduces overfitting risk, but repeated tuning on one validation slice "
            "can still bias results. Consider walk-forward revalidation before deployment."
        ),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
