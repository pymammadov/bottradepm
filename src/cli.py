from __future__ import annotations

import argparse
import json

from .backtest import run_backtest
from .data_loader import load_ohlcv_csv
from .institutional import run_institutional_pipeline
from .optimization import run_parameter_sweep
from .strategy import StrategyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSD systematic research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    baseline = sub.add_parser("backtest", help="Run baseline backtest")
    baseline.add_argument("--csv", required=True, help="OHLCV csv path")
    baseline.add_argument("--starting-equity", type=float, default=10_000)
    baseline.add_argument("--output-dir", default="outputs")
    baseline.add_argument("--data-source", default="public_csv")

    opt = sub.add_parser("optimize", help="Run train/validation optimization sweep")
    opt.add_argument("--csv", required=True)
    opt.add_argument("--starting-equity", type=float, default=10_000)
    opt.add_argument("--output-dir", default="outputs")
    opt.add_argument("--train-ratio", type=float, default=0.7)

    inst = sub.add_parser("institutional", help="Run full institutional workflow")
    inst.add_argument("--csv", required=True)
    inst.add_argument("--starting-equity", type=float, default=10_000)
    inst.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if args.command == "backtest":
        df = load_ohlcv_csv(args.csv)
        report = run_backtest(
            df_1h=df,
            config=StrategyConfig(),
            starting_equity=args.starting_equity,
            output_dir=args.output_dir,
            data_source=args.data_source,
        )
    elif args.command == "optimize":
        df = load_ohlcv_csv(args.csv)
        results_df, oos_df, meta = run_parameter_sweep(df, args.starting_equity, args.output_dir, args.train_ratio)
        report = {"meta": meta, "top_train": results_df.head(5).to_dict(orient="records"), "top_oos": oos_df.head(5).to_dict(orient="records")}
    else:
        report = run_institutional_pipeline(args.csv, args.output_dir, args.starting_equity)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
