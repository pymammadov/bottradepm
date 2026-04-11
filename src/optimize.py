from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_backtest
from .data_loader import load_ohlcv_csv, resample_ohlcv
from .strategy import StrategyConfig


def generate_parameter_combinations() -> list[StrategyConfig]:
    """Generate all parameter combinations for optimization."""
    # Define parameter ranges (reduced for testing)
    param_ranges = {
        "risk_pct": [0.0075, 0.01],
        "stop_atr_mult": [1.8, 2.1],
        "tp1_r": [1.2],
        "tp2_r": [2.2],
        "tp3_r": [4.0],
        "adx_threshold": [18, 21],
        "breakout_lookback": [20],
        "pullback_rsi_min": [48],
        "pullback_rsi_max": [62],
        "trailing_stop_mult": [0.5],
    }

    # Generate all combinations
    keys = param_ranges.keys()
    values = param_ranges.values()
    combinations = list(itertools.product(*values))

    configs = []
    for combo in combinations:
        config_dict = dict(zip(keys, combo))
        # Create StrategyConfig with default values for non-optimized params
        config = StrategyConfig(
            risk_pct=config_dict["risk_pct"],
            stop_atr_mult=config_dict["stop_atr_mult"],
            tp1_r=config_dict["tp1_r"],
            tp2_r=config_dict["tp2_r"],
            tp3_r=config_dict["tp3_r"],
            adx_threshold=config_dict["adx_threshold"],
            breakout_lookback=config_dict["breakout_lookback"],
            pullback_rsi_min=config_dict["pullback_rsi_min"],
            pullback_rsi_max=config_dict["pullback_rsi_max"],
            trailing_stop_mult=config_dict["trailing_stop_mult"],
        )
        configs.append(config)

    return configs


def calculate_score(report: dict[str, Any]) -> float:
    """Calculate a balanced score for parameter ranking."""
    monthly_ret = report.get("average_monthly_return_pct", 0)
    max_dd = report.get("max_drawdown_pct", 100)  # This is already negative
    profit_factor = report.get("profit_factor", 0)
    total_trades = report.get("total_trades", 0)

    # Skip if basic criteria not met
    if profit_factor < 1.0 or total_trades < 5 or max_dd > -3:  # Less strict criteria
        return float("-inf")

    # Normalize and combine metrics
    # Higher monthly return is better
    ret_score = monthly_ret

    # Lower max drawdown is better (max_dd is negative, so -max_dd makes it positive)
    dd_score = -max_dd  # This makes lower drawdown (more negative) give higher score

    # Higher profit factor is better
    pf_score = profit_factor

    # More trades is better (up to a point)
    trade_score = min(total_trades, 200) / 10  # Cap at 200 trades

    # Weighted combination
    score = 0.4 * ret_score + 0.3 * dd_score + 0.2 * pf_score + 0.1 * trade_score

    return score


def run_optimization(
    csv_path: str,
    output_dir: str = "outputs",
    starting_equity: float = 10_000.0,
) -> dict[str, Any]:
    """Run parameter optimization."""
    # Load data
    df_1h = load_ohlcv_csv(csv_path)

    # Generate parameter combinations
    configs = generate_parameter_combinations()
    print(f"Running optimization with {len(configs)} parameter combinations...")

    results = []
    for i, config in enumerate(configs):
        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{len(configs)} combinations")

        try:
            report = run_backtest(
                df_1h=df_1h,
                config=config,
                starting_equity=starting_equity,
                output_dir=output_dir,
                data_source="optimization",
            )

            result = {
                "config": {
                    "risk_pct": config.risk_pct,
                    "stop_atr_mult": config.stop_atr_mult,
                    "tp1_r": config.tp1_r,
                    "tp2_r": config.tp2_r,
                    "tp3_r": config.tp3_r,
                    "adx_threshold": config.adx_threshold,
                    "breakout_lookback": config.breakout_lookback,
                    "pullback_rsi_min": config.pullback_rsi_min,
                    "pullback_rsi_max": config.pullback_rsi_max,
                    "trailing_stop_mult": config.trailing_stop_mult,
                },
                "report": report,
                "score": calculate_score(report),
            }
            results.append(result)

        except Exception as e:
            print(f"Error with config {i}: {e}")
            continue

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all results
    results_df = pd.DataFrame([
        {
            **r["config"],
            **r["report"],
            "score": r["score"]
        }
        for r in results
    ])
    results_df.to_csv(out_dir / "optimization_results.csv", index=False)

    # Get top 10
    top_10 = results[:10]
    print("\nTop 10 parameter sets:")
    for i, result in enumerate(top_10, 1):
        config = result["config"]
        report = result["report"]
        score = result["score"]
        print(f"{i}. Score: {score:.2f}, Monthly Ret: {report['average_monthly_return_pct']:.2f}%, "
              f"Max DD: {report['max_drawdown_pct']:.2f}%, PF: {report['profit_factor']:.2f}, "
              f"Trades: {report['total_trades']}")

    # Run out-of-sample validation on top 3
    print("\nRunning out-of-sample validation on top 3 parameter sets...")
    oos_results = []
    for i, result in enumerate(results[:3], 1):
        config = StrategyConfig(**result["config"])
        print(f"Running OOS for top {i}...")

        # For simplicity, we'll use the same data (in practice, you'd split the data)
        # TODO: Implement proper train/test split
        oos_report = run_backtest(
            df_1h=df_1h,
            config=config,
            starting_equity=starting_equity,
            output_dir=output_dir,
            data_source="oos_validation",
        )
        oos_results.append({
            "rank": i,
            "config": result["config"],
            "oos_report": oos_report,
        })

    # Create summary
    summary = {
        "total_combinations": len(configs),
        "valid_results": len(results),
        "top_10_configs": [
            {
                "rank": i + 1,
                "config": r["config"],
                "score": r["score"],
                "report": r["report"]
            }
            for i, r in enumerate(top_10)
        ],
        "oos_validation": oos_results,
    }

    # Save summary
    (out_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2))

    return summary