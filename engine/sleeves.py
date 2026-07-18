"""Sleeves — turn "pinned" into a comparison dimension.

A *sleeve* is a labeled subset of tickers to trade. We compare three:
  pinned      your owned holdings alone         ("what if I just held my stuff?")
  systematic  the screened universe, pins NOT force-kept  (pure algorithmic picks)
  combo       pinned ∪ systematic               (holdings + engine picks together)

Each sleeve is run through the *same* backtest engine, optionally with a different
strategy, and the results are compared like any other set of equity curves — so
nothing in backtest/portfolio/metrics needs to change.
"""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .strategies import make_strategy

SLEEVES = ("pinned", "systematic", "combo")


def pinned_tickers(universe: pd.DataFrame, sources: tuple[str, ...] = ("owned",)) -> list[str]:
    """Tickers that make up the 'pinned' sleeve. Defaults to owned holdings only
    (ETFs are treated as tradeable universe, not personal holdings)."""
    return universe.loc[universe["source"].isin(sources), "ticker"].tolist()


def build_sleeves(
    universe: pd.DataFrame,
    systematic: list[str],
    available: set[str] | None = None,
    pin_sources: tuple[str, ...] = ("owned",),
) -> dict[str, list[str]]:
    """Assemble the three sleeves as {label: [tickers]}.

    Args:
        universe: full tagged universe.
        systematic: the watchlist from refine run with keep_pinned=False.
        available: if given, restrict every sleeve to tickers we actually have
            prices for (drops names that failed to download).
    """
    pins = pinned_tickers(universe, pin_sources)
    sleeves = {
        "pinned": list(pins),
        "systematic": list(systematic),
        "combo": sorted(set(pins) | set(systematic)),
    }
    if available is not None:
        sleeves = {k: [t for t in v if t in available] for k, v in sleeves.items()}
    return {k: v for k, v in sleeves.items() if v}  # drop empty sleeves


def label(sleeve: str, strategy: str) -> str:
    return f"{sleeve} · {strategy}"


def holdings_weights(shares: dict[str, float], prices: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Convert {ticker: shares} into starting weights using the first available price.

    Uses the earliest price in the panel so the pinned sleeve reproduces "buy & hold
    my current allocation from the start of the window". Returns normalized weights.
    """
    if not shares or "adjclose" not in prices:
        return {}
    close = prices["adjclose"]
    first = close.bfill().iloc[0]  # earliest non-NaN price per ticker
    value = {t: shares[t] * float(first[t]) for t in shares if t in close.columns and pd.notna(first.get(t))}
    total = sum(value.values())
    return {t: v / total for t, v in value.items()} if total > 0 else {}


def run_matrix(
    sleeves: dict[str, list[str]],
    strategy_specs: dict[str, dict],
    prices: dict[str, pd.DataFrame],
    start_cash: float = 100_000.0,
    cost_bps: float = 0.0,
    atr_period: int = 14,
    pinned_strategy: str = "buy_and_hold",
    pinned_weights: dict[str, float] | None = None,
) -> dict[str, BacktestResult]:
    """Run every (sleeve × strategy) combination through the engine.

    The 'pinned' sleeve is run with `pinned_strategy` only (default Buy & Hold —
    the "just hold my holdings" baseline). When `pinned_weights` is given, that
    sleeve holds your real allocation instead of equal weight. Other sleeves run
    each strategy in `strategy_specs`. Returns {label: BacktestResult}.
    """
    results: dict[str, BacktestResult] = {}
    for sleeve, tickers in sleeves.items():
        if not tickers:
            continue
        priced = {field: df[[t for t in tickers if t in df.columns]] for field, df in prices.items()}
        if sleeve == "pinned":
            params = dict(strategy_specs.get(pinned_strategy, {}))
            if pinned_weights and pinned_strategy == "buy_and_hold":
                params["weights"] = pinned_weights
            specs = {pinned_strategy: params}
        else:
            specs = strategy_specs
        for strat_name, params in specs.items():
            strat = make_strategy(strat_name, params)
            res = run_backtest(strat, priced, start_cash=start_cash, cost_bps=cost_bps, atr_period=atr_period)
            results[label(sleeve, strat_name)] = res
    return results


def default_benchmark(results: dict, pinned_strategy: str = "buy_and_hold") -> str | None:
    """Pick the benchmark label: 'pinned · buy_and_hold' if present (does the engine
    beat just holding my holdings?), else the first buy_and_hold sleeve, else None."""
    preferred = label("pinned", pinned_strategy)
    if preferred in results:
        return preferred
    for key in results:
        if key.endswith(f"· {pinned_strategy}"):
            return key
    return next(iter(results), None)
