from __future__ import annotations

import argparse
import json

from .backtest import run_backtest
from .data_loader import load_ohlcv_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BTC strategy backtest")
    parser.add_argument("--csv", required=True, help="OHLCV csv path")
    parser.add_argument("--starting-equity", type=float, default=10_000)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-source", default="public_csv")
    args = parser.parse_args()

    df = load_ohlcv_csv(args.csv)
    report = run_backtest(
        df_1h=df,
        starting_equity=args.starting_equity,
        output_dir=args.output_dir,
        data_source=args.data_source,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
