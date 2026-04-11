from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data_loader import load_ohlcv_csv
from .optimization import OPTIMIZATION_PARAM_KEYS, run_parameter_sweep
from .strategy import StrategyConfig


def _time_splits(df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    if i1 <= 0 or i2 <= i1 or i2 >= n:
        raise ValueError("Invalid data split ratios; empty train/validation/oos segment detected")
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def _select_best(oos_df: pd.DataFrame) -> dict:
    if oos_df.empty:
        raise ValueError("No optimization candidates produced")
    best = oos_df.iloc[0]
    cfg = {}
    int_keys = {"breakout_lookback", "reentry_cooldown_bars", "v2_trail_swing_lookback", "v2_reentry_bars"}
    bool_keys = {"move_stop_to_breakeven_after_tp1", "use_monthly_controls", "use_regime_filter", "enable_short"}
    for key in OPTIMIZATION_PARAM_KEYS:
        v = best[key]
        if key in int_keys:
            cfg[key] = int(v)
        elif key in bool_keys:
            cfg[key] = bool(v)
        else:
            cfg[key] = float(v) if isinstance(v, (np.floating, float)) else v
    return cfg


def _run_stress(df_oos: pd.DataFrame, base_cfg: StrategyConfig, starting_equity: float, output_dir: Path) -> dict:
    scenarios = {
        "base": {},
        "high_slippage": {"slippage_rate": base_cfg.slippage_rate * 2.5},
        "high_fees": {"fee_rate": base_cfg.fee_rate * 2.0},
        "risk_downshift": {"risk_pct": base_cfg.risk_pct * 0.7},
        "risk_upshift": {"risk_pct": base_cfg.risk_pct * 1.25},
        "wider_stops": {"stop_atr_mult": base_cfg.stop_atr_mult * 1.2},
    }
    out = {}
    for name, overrides in scenarios.items():
        cfg_dict = asdict(base_cfg)
        cfg_dict.update(overrides)
        rpt = run_backtest(
            df_1h=df_oos,
            config=StrategyConfig(**cfg_dict),
            starting_equity=starting_equity,
            output_dir=output_dir / f"stress_{name}",
            data_source=f"stress_{name}",
            save_plot=False,
        )
        out[name] = {
            "average_monthly_return_pct": rpt.get("average_monthly_return_pct"),
            "max_drawdown_pct": rpt.get("max_drawdown_pct"),
            "profit_factor": rpt.get("profit_factor"),
            "total_return_pct": rpt.get("total_return_pct"),
            "total_trades": rpt.get("total_trades"),
        }
    return out


def run_institutional_pipeline(csv_path: str, output_dir: str = "outputs", starting_equity: float = 10_000.0) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv_csv(csv_path)
    train_df, val_df, oos_df = _time_splits(df)

    baseline_cfg = StrategyConfig()
    advanced_seed_cfg = StrategyConfig(enable_short=True, entry_mode="combined", short_entry_mode="combined", use_regime_filter=True)

    baseline_report = run_backtest(train_df, config=baseline_cfg, starting_equity=starting_equity, output_dir=out_dir / "baseline_train", data_source="baseline_train", save_plot=False)
    advanced_seed_report = run_backtest(train_df, config=advanced_seed_cfg, starting_equity=starting_equity, output_dir=out_dir / "strategy_v2_train", data_source="strategy_v2_train", save_plot=False)

    results_df, oos_ranked_df, meta = run_parameter_sweep(df_1h=pd.concat([train_df, val_df]), starting_equity=starting_equity, output_dir=out_dir, train_ratio=0.75)
    best_params = _select_best(oos_ranked_df)
    merged_best_params = {"enable_short": True, "short_entry_mode": "combined", **best_params}
    best_cfg = StrategyConfig(**merged_best_params)

    val_report = run_backtest(val_df, config=best_cfg, starting_equity=starting_equity, output_dir=out_dir / "best_validation", data_source="best_validation", save_plot=False)
    oos_report = run_backtest(oos_df, config=best_cfg, starting_equity=starting_equity, output_dir=out_dir / "best_oos", data_source="best_oos", save_plot=False)

    stress = _run_stress(oos_df, best_cfg, starting_equity, out_dir)

    monthly_equity = pd.read_csv(out_dir / "best_oos" / "equity_curve.csv", parse_dates=["timestamp"]).set_index("timestamp")
    monthly = monthly_equity.resample("ME").last().pct_change().dropna().rename(columns={"equity": "monthly_return"})
    monthly.to_csv(out_dir / "monthly_returns.csv")

    oos_cmp = {
        "baseline_train": baseline_report,
        "strategy_v2_seed_train": advanced_seed_report,
        "best_validation": val_report,
        "best_oos": oos_report,
    }
    (out_dir / "oos_comparison.json").write_text(json.dumps(oos_cmp, indent=2))

    regime_analysis = {
        "assumptions": {
            "bull": "close > d_ema200 and adx > threshold",
            "bear": "close < d_ema200 and adx > threshold",
            "high_vol": "rolling 20 period volatility above configured quantile",
        },
        "best_config": best_params,
    }
    (out_dir / "regime_analysis.json").write_text(json.dumps(regime_analysis, indent=2))

    stress_summary = {
        "best_config": best_params,
        "stress_results": stress,
    }
    (out_dir / "stress_test_summary.json").write_text(json.dumps(stress_summary, indent=2))

    best_model_report = {
        "selection_reason": "Top OOS score from train/validation parameter sweep with overfit penalty.",
        "best_config": best_params,
        "validation_report": val_report,
        "oos_report": oos_report,
    }
    (out_dir / "best_model_report.json").write_text(json.dumps(best_model_report, indent=2))

    optimization_summary = {
        "meta": meta,
        "top_train_candidates": results_df.head(10).to_dict(orient="records"),
        "top_validation_candidates": oos_ranked_df.head(10).to_dict(orient="records"),
    }
    (out_dir / "optimization_summary.json").write_text(json.dumps(optimization_summary, indent=2))

    report = {
        "pipeline": "institutional_btcusd_v2",
        "baseline_train": baseline_report,
        "strategy_v2_seed_train": advanced_seed_report,
        "best_validation": val_report,
        "best_oos": oos_report,
        "best_params": best_params,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    improvement_log = out_dir / "improvement_log.md"
    improvement_log.write_text(
        "\n".join(
            [
                "# Improvement Log",
                "",
                "## Iteration: Institutional architecture upgrade",
                "- Added explicit train/validation/OOS pipeline outputs.",
                "- Added strategy_v2 seed with optional short-mode and high-volatility decision gate.",
                "- Added stress scenarios (slippage, fees, risk shifts, stop shifts).",
                "- Added best-model explicit selection and OOS comparison reports.",
            ]
        )
        + "\n"
    )

    investor_md = out_dir / "investment_committee_report.md"
    investor_md.write_text(
        "\n".join(
            [
                "# BTCUSD System Research Brief",
                "",
                "- Baseline and strategy_v2 were both tested on isolated time segments.",
                "- Model selection used train + validation ranking and explicit OOS re-check.",
                "- Stress tests include fee/slippage and position-risk perturbations.",
                "- See `best_model_report.json` and `stress_test_summary.json` for details.",
            ]
        )
        + "\n"
    )

    return report
