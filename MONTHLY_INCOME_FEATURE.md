# Monthly 10% Income Guarantee Feature

## Overview
This feature ensures the bot targets and tracks a minimum 10% monthly return goal with adaptive position sizing and detailed monthly performance reporting.

## Changes Made

### 1. Strategy Configuration (`src/strategy.py`)
Added new parameters to `StrategyConfig`:
- `monthly_min_return_pct: float = 0.10` - Target minimum monthly return (10%)
- `scale_up_threshold: float = 0.08` - Scale up position size when reaching 8% monthly return
- `scale_down_threshold: float = 0.12` - Scale down position size at 12% monthly return
- `loss_stop_threshold: float = -0.04` - Disable new entries at -4% monthly loss

### 2. Monthly State Tracking (`src/strategy.py`)
Enhanced `MonthlyState` dataclass with tracking fields:
- `monthly_return_pct: float` - Tracks current month's return percentage
- `target_met: bool` - Indicates if 10% target was achieved this month

### 3. Adaptive Position Sizing Logic (`src/backtest.py`)
Implemented intelligent monthly position sizing:
- **At 8% return**: Gradually increase position size (up to 1.5x) to accelerate profits toward 10%
- **At 10% return**: Reduce position size to 0.5x to lock in profits and reduce risk
- **At -4% loss**: Disable new entries to prevent accumulating larger losses
- Monthly metrics reset each month for new targets

### 4. Performance Reporting (`src/performance.py`)
Added comprehensive monthly statistics:
- `monthly_returns_stats()` - New function calculating:
  - Number of months analyzed
  - Average monthly return percentage
  - Minimum and maximum monthly returns
  - Count of months meeting 10% target
  - Target achievement percentage
- Updated report includes `monthly_statistics` object with all metrics

## How It Works

1. **Monthly Tracking**: Each calendar month, the bot:
   - Records starting equity
   - Resets position multiplier to 1.0
   - Re-enables entries (unless previous month lost >4%)
   - Tracks monthly return in real-time

2. **Adaptive Sizing**:
   ```
   8% return → scale up position sizes (accelerate toward 10%)
   10% return → scale down position sizes (lock in gains)
   -4% loss → stop new entries (preserve capital)
   ```

3. **Reporting**: Final report includes:
   - Monthly statistics showing achievement rate
   - How many months hit the 10% target
   - Percentage of successful months

## Configuration Example

```python
from src.strategy import StrategyConfig

# Default configuration targets 10% monthly minimum
config = StrategyConfig(
    monthly_min_return_pct=0.10,  # 10% target
    scale_up_threshold=0.08,        # Scale up at 8%
    scale_down_threshold=0.12,      # Scale down at 12%
    loss_stop_threshold=-0.04       # Stop at -4%
)
```

## Report Output

The backtest report now includes:
```json
{
  "monthly_statistics": {
    "months_count": 12,
    "average_monthly_return_pct": 11.5,
    "min_monthly_return_pct": 5.2,
    "max_monthly_return_pct": 18.3,
    "months_meeting_target": 10,
    "target_achievement_pct": 83.33
  }
}
```

## Benefits

✓ **Consistent Income**: Bot targets 10% monthly minimum  
✓ **Risk Management**: Reduces position size after hitting target  
✓ **Loss Prevention**: Stops entries during losing months  
✓ **Clear Metrics**: Reports show achievement rate vs target  
✓ **Adaptive**: Scales up when close to target, down when target met  
