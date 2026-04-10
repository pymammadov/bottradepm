# Backtest Audit

## What was already present
- `README.md` with only project name.
- `btcusd_bot_simple.py` containing a random-price toy simulation and console prints.

## Major flaws found
- No real OHLCV ingestion, no reproducible data source, and no timeframe handling.
- Randomness drives entries/prices, so results are non-deterministic and not auditable.
- No fee/slippage modeling, no realistic risk sizing, no leverage controls.
- No proper stop/target engine, no partial exits, no force-close rule.
- No tests, no structured outputs, and no reporting artifacts.

## Quantitative logic errors
- PnL logic ignored quantity/risk sizing and treated price difference as full trade PnL.
- Entry/exit logic mixed random triggers with no market-structure confirmation.
- No prevention of lookahead bias and no multi-timeframe regime logic.

## Methodology risks
- Synthetic random walk can overstate strategy behavior and hide tail risk.
- Missing drawdown/profit-factor/expectancy metrics prevents robust evaluation.
- Missing deterministic tests allows silent regressions in accounting logic.

## What I changed
- Rebuilt repo into modular research structure (`src`, `tests`, `scripts`, `data`, `outputs`).
- Implemented indicators, data loading/resampling, strategy filters, risk sizing, partial TP, trailing stop, monthly controls, and force-close behavior.
- Added Binance public downloader (no API key) and runnable CLI backtest script.
- Added pytest coverage for indicators, sizing/leverage, trade management, kill-switch behavior, reproducibility, and smoke execution.
- Added reporting outputs and README with exact commands.

## What remains uncertain
- Bar-internal execution ordering is conservative (SL priority on same bar); live microstructure can differ.
- Binance API availability/rate limits can impact large historical pulls.
- Results depend on dataset quality and interval alignment (1H source resampled to 4H/1D).
