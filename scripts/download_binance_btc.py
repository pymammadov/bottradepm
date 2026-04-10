#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"


def to_ms(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def download_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    cur = start_ms

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        cur = int(rows[-1][0]) + 1

    if not all_rows:
        raise RuntimeError("No klines returned from Binance")

    df = pd.DataFrame(all_rows)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[0], unit="ms", utc=True),
            "open": pd.to_numeric(df[1]),
            "high": pd.to_numeric(df[2]),
            "low": pd.to_numeric(df[3]),
            "close": pd.to_numeric(df[4]),
            "volume": pd.to_numeric(df[5]),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance klines to CSV")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="data/btcusdt_1h.csv")
    args = parser.parse_args()

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)
    df = download_klines(args.symbol, args.interval, start_ms, end_ms)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
