# BTCUSDT Strategy Improvement Log

## Iteration 1
- **What changed:** Expanded optimizer search space from a tiny fixed grid to a broader mixed-mode search (risk, ATR stop, TP structure, entry mode, breakout filter, cooldown, TP split, monthly controls) with deterministic sampling and explicit OOS scoring.
- **Why:** The original sweep was too narrow and favored train-only winners with weak robustness controls.
- **Previous best metrics:** N/A (no optimization artifacts present in repo).
- **New best metrics:**
  - Train average_monthly_return_pct: **1.02%**
  - OOS average_monthly_return_pct: **1.18%**
  - OOS profit_factor: **8.22**
  - OOS max_drawdown_pct: **-1.61%**
  - OOS total_trades: **6**
- **Decision:** **Kept** (framework improvements are valid, but objective target still missed).

## Iteration 2
- **What changed:** Strengthened anti-overfitting ranking by adding harder OOS low-trade penalties and searching `use_regime_filter` on/off.
- **Why:** Iteration-1 winner had too few validation trades and inflated PF from tiny sample size.
- **Previous best metrics:** same as iteration 1 above.
- **New best metrics:**
  - Train average_monthly_return_pct: **1.02%**
  - OOS average_monthly_return_pct: **1.18%**
  - OOS profit_factor: **8.22**
  - OOS max_drawdown_pct: **-1.61%**
  - OOS total_trades: **6**
  - Best train monthly found in full sweep: **1.64%**
- **Decision:** **Kept** for stricter validity logic; no material performance lift.

## Conclusion after iterative runs
- On the available dataset and this strategy family, no tested configuration approaches the **10% average monthly** target.
- Current bottleneck: long-only breakout/pullback logic with partial-profit clipping and strict daily trend dependency yields low upside capture per month.
- Next architecture step needed: directional symmetry (short-side), volatility-regime model, and walk-forward objective with multi-split selection.
