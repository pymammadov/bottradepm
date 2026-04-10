# BTCUSDT Strategy Research Repo (10% Monthly Target Profile)

This repository provides a **reproducible strategy research framework** for BTCUSDT.

> Important: the 10% monthly figure is a **target profile**, not a guaranteed outcome.

## Strategy Summary

- **Regime filter (Daily):** prior close > EMA200, EMA50 > EMA200, ADX(14) > 18.
- **Execution timeframe:** 4H candles (resampled from input data).
- **Entries:**
  - Breakout: close > prior 20-bar high, volume > 1.5x 20-bar average, close above EMA20/EMA50.
  - Pullback: touch EMA20/EMA50 and reclaim, bullish close, RSI(14) 48-62, close > EMA50.
- **Stop loss:** 1.8 * ATR(14).
- **Position sizing:** risk-based, leverage only as notional cap.
- **Profit-taking:** TP1 1.2R (40%), TP2 2.2R (30%), TP3 4.0R (remaining 30%).
- **After TP1:** move stop to breakeven.
- **After TP2:** trail stop via `max(current_stop, EMA20 - 0.5*ATR14)`.
- **Monthly controls:**
  - +10% monthly return => reduce new trade size by 50%.
  - -4% monthly return => disable new entries for month remainder.
- **Execution assumptions:** candle high/low for trigger checks, SL priority if TP and SL both hit in same candle.

## Repository Layout

```
.
├── data/
├── outputs/
├── src/
│   ├── indicators.py
│   ├── data_loader.py
│   ├── strategy.py
│   ├── backtest.py
│   ├── performance.py
│   └── cli.py
├── tests/
├── scripts/
│   ├── download_binance_btc.py
│   └── run_backtest.py
├── BACKTEST_AUDIT.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download public Binance data (no API key)

```bash
python scripts/download_binance_btc.py \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2026-04-11 \
  --out data/btcusdt_1h.csv
```

CSV format expected:

`timestamp,open,high,low,close,volume`

## Run backtest

```bash
python scripts/run_backtest.py --csv data/btcusdt_1h.csv
```

Outputs:
- `outputs/trades.csv`
- `outputs/equity_curve.csv`
- `outputs/report.json`
- `outputs/equity_curve.png`

## Report fields

- starting equity
- ending equity
- total return %
- actual average monthly return %
- total trades
- win rate
- average win
- average loss
- profit factor
- max drawdown %
- expectancy per trade
- average R multiple
- number of forced closes
- data source tag (public CSV vs synthetic)

## Tests

```bash
pytest -q
```

Test suite covers:
1. EMA/ATR/RSI/ADX sanity
2. risk sizing correctness
3. leverage cap and stop-defined risk
4. partial TP behavior and quantity reductions
5. stop-loss accounting
6. force-close behavior
7. monthly kill-switch behavior
8. deterministic reproducibility
9. full backtest smoke test
