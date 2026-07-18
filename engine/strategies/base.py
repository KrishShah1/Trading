"""Strategy interface — the one contract every strategy implements.

A strategy is pure decision logic. Each trading day the backtest hands it a
StrategyContext (market state up to *today*, plus the current portfolio) and the
strategy returns the target weights it wants, or None to mean "no change today".

Keeping strategies to this single method is what makes the engine a platform:
data, indicators, execution, and scoring all live elsewhere. Adding a strategy is
one subclass + one line in the registry (strategies/__init__.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyContext:
    """Everything a strategy is allowed to see on a given day.

    Only data up to and including `date` is exposed — no look-ahead.
    """

    date: pd.Timestamp
    tickers: list[str]
    close: pd.DataFrame        # adjusted close, dates <= today, columns = tickers
    high: pd.DataFrame
    low: pd.DataFrame
    atr: pd.DataFrame          # Wilder ATR, same shape
    current_weights: dict[str, float]

    @property
    def prices_today(self) -> dict[str, float]:
        row = self.close.iloc[-1]
        return {t: float(row[t]) for t in self.tickers if pd.notna(row.get(t))}


class Strategy(ABC):
    """Base class. Subclasses set `name` and implement target_weights()."""

    name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abstractmethod
    def target_weights(self, ctx: StrategyContext) -> dict[str, float] | None:
        """Return desired {ticker: weight} for today, or None for no change."""

    # --- shared helpers available to all strategies -------------------------

    def _tradeable(self, ctx: StrategyContext) -> list[str]:
        """Tickers that have a valid price today (can actually be traded)."""
        return list(ctx.prices_today.keys())

    def _equal_weights(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
