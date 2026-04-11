#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimize import run_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BTC strategy optimization")
    parser.add_argument("--csv", required=True, help="OHLCV csv path")
    parser.add_argument("--starting-equity", type=float, default=10_000)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    summary = run_optimization(
        csv_path=args.csv,
        output_dir=args.output_dir,
        starting_equity=args.starting_equity,
    )

    print("\nOptimization completed!")
    print(f"Results saved to {args.output_dir}/optimization_results.csv")
    print(f"Summary saved to {args.output_dir}/optimization_summary.json")

    # Print top 3 OOS results
    print("\nOut-of-sample validation results:")
    for result in summary["oos_validation"]:
        report = result["oos_report"]
        print(f"Rank {result['rank']}: Monthly Ret: {report['average_monthly_return_pct']:.2f}%, "
              f"Max DD: {report['max_drawdown_pct']:.2f}%, PF: {report['profit_factor']:.2f}")


if __name__ == "__main__":
    main()