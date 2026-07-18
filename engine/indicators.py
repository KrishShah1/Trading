"""Technical indicators. Pure numpy/pandas — no I/O, no UI.

These operate on the wide (date x ticker) price frames produced by data.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """True Range = max(high-low, |high-prev_close|, |low-prev_close|), per cell."""
    prev_close = close.shift(1)
    return _rowwise_max(high - low, high - prev_close, low - prev_close)


def _rowwise_max(hl: pd.DataFrame, hc: pd.DataFrame, lc: pd.DataFrame) -> pd.DataFrame:
    """Element-wise max of |hl|, |hc|, |lc| across three aligned wide frames."""
    stacked = pd.concat([hl.abs(), hc.abs(), lc.abs()])
    return stacked.groupby(level=0).max()


def atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14
) -> pd.DataFrame:
    """Wilder's Average True Range (wide date x ticker frame).

    Uses Wilder smoothing (an EMA with alpha = 1/period), the standard ATR.
    """
    tr = _rowwise_max(high - low, high - close.shift(1), low - close.shift(1))
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_pct(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14
) -> pd.DataFrame:
    """ATR as a fraction of price — comparable across tickers of different price."""
    return atr(high, low, close, period) / close


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a wide price frame (use adjusted close)."""
    return prices.pct_change()


def realized_vol(prices: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.DataFrame:
    """Rolling realized volatility of daily returns, optionally annualized (252d)."""
    vol = daily_returns(prices).rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol
