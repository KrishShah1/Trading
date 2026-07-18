"""The backtest loop — the heart of the engine.

Walks forward one trading day at a time. Each day it builds a StrategyContext from
data *up to that day only* (no look-ahead), asks the strategy for target weights,
and — if the strategy wants a change — moves the portfolio toward those weights.
Records an equity curve, weights over time, and a full trade log.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind
from .portfolio import Portfolio
from .strategies.base import Strategy, StrategyContext


@dataclass
class BacktestResult:
    strategy: str
    equity_curve: pd.Series          # date -> total equity
    weights: pd.DataFrame            # date x ticker weights
    trades: pd.DataFrame             # one row per executed trade
    params: dict


def run_backtest(
    strategy: Strategy,
    prices: dict[str, pd.DataFrame],
    start_cash: float = 100_000.0,
    cost_bps: float = 0.0,
    atr_period: int = 14,
    warmup: int | None = None,
) -> BacktestResult:
    """Run one strategy over the fetched price panel.

    Args:
        prices: dict from data.fetch_prices — needs 'adjclose', 'high', 'low', 'close'.
        warmup: trading days to skip before trading (defaults to atr_period so ATR
            is defined on day one of trading).
    """
    close = prices["adjclose"]
    high = prices["high"]
    low = prices["low"]
    raw_close = prices["close"]
    tickers = list(close.columns)

    atr = ind.atr(high, low, raw_close, period=atr_period)
    dates = close.index
    warmup = atr_period if warmup is None else warmup

    pf = Portfolio(cash=start_cash, cost_bps=cost_bps)
    equity: dict[pd.Timestamp, float] = {}
    weight_rows: dict[pd.Timestamp, dict[str, float]] = {}

    for i, date in enumerate(dates):
        prices_today = {t: float(close.iloc[i][t]) for t in tickers if pd.notna(close.iloc[i][t])}
        if not prices_today:
            continue

        if i >= warmup:
            ctx = StrategyContext(
                date=date,
                tickers=tickers,
                close=close.iloc[: i + 1],
                high=high.iloc[: i + 1],
                low=low.iloc[: i + 1],
                atr=atr.iloc[: i + 1],
                current_weights=pf.weights(prices_today),
            )
            target = strategy.target_weights(ctx)
            if target is not None:
                pf.apply_target_weights(date, target, prices_today)

        equity[date] = pf.value(prices_today)
        weight_rows[date] = pf.weights(prices_today)

    equity_curve = pd.Series(equity, name=strategy.name).sort_index()
    weights = pd.DataFrame.from_dict(weight_rows, orient="index").sort_index().fillna(0.0)
    trades = pd.DataFrame(
        [
            {"date": t.date, "ticker": t.ticker, "shares": t.shares, "price": t.price, "cost": t.cost}
            for t in pf.trades
        ]
    )
    return BacktestResult(
        strategy=strategy.name,
        equity_curve=equity_curve,
        weights=weights,
        trades=trades,
        params=dict(strategy.params),
    )
