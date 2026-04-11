import numpy as np
import pandas as pd

from src.indicators import adx, atr, ema, rsi


def test_indicator_sanity():
    idx = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 130, 300), index=idx)
    df = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )

    ema20 = ema(df["close"], 20)
    atr14 = atr(df, 14)
    rsi14 = rsi(df["close"], 14)
    adx14 = adx(df, 14)

    assert ema20.notna().sum() > 0
    assert (atr14.dropna() > 0).all()
    assert ((rsi14.dropna() >= 0) & (rsi14.dropna() <= 100)).all()
    assert ((adx14.dropna() >= 0) & (adx14.dropna() <= 100)).all()
