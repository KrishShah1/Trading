"""Performance metrics and benchmark comparison.

Turns an equity curve into the numbers that answer "did this strategy work?":
total/annualized return, risk-adjusted return, drawdown, and — the headline —
alpha vs the Buy & Hold benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = len(equity) / TRADING_DAYS
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annualized_vol(equity: pd.Series) -> float:
    return _returns(equity).std() * np.sqrt(TRADING_DAYS)


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity)
    excess = r - rf / TRADING_DAYS
    if excess.std() == 0:
        return float("nan")
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity) - rf / TRADING_DAYS
    downside = r[r < 0].std()
    if not downside or np.isnan(downside):
        return float("nan")
    return (r.mean() / downside) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline (a negative number)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    return equity.iloc[-1] / equity.iloc[0] - 1


def summarize(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Full metric bundle for one strategy, plus alpha vs benchmark if provided."""
    out = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "ann_vol": annualized_vol(equity),
        "sharpe": sharpe(equity),
        "sortino": sortino(equity),
        "max_drawdown": max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
    }
    if benchmark is not None and len(benchmark) > 1:
        out["alpha_vs_benchmark"] = total_return(equity) - total_return(benchmark)
    return out


def summary_table(results: dict[str, pd.Series], benchmark_name: str | None = None) -> pd.DataFrame:
    """Build a comparison table across strategies.

    Args:
        results: {strategy_name: equity_curve}.
        benchmark_name: key in `results` to compute alpha against.
    """
    bench = results.get(benchmark_name) if benchmark_name else None
    rows = {name: summarize(curve, bench) for name, curve in results.items()}
    return pd.DataFrame(rows).T
