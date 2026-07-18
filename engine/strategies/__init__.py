"""Strategy registry.

Adding a strategy = import it here and add one line to STRATEGIES. The CLI and the
Streamlit cockpit both discover strategies through this dict, so registration is
the only wiring needed.
"""

from __future__ import annotations

from .atr_harvest import AtrHarvest
from .base import Strategy, StrategyContext
from .buy_and_hold import BuyAndHold
from .rebalance import Rebalance

STRATEGIES: dict[str, type[Strategy]] = {
    BuyAndHold.name: BuyAndHold,
    AtrHarvest.name: AtrHarvest,
    Rebalance.name: Rebalance,
}

# The benchmark every other strategy is scored against.
BENCHMARK = BuyAndHold.name


def make_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}. Known: {sorted(STRATEGIES)}")
    return STRATEGIES[name](params)


__all__ = ["STRATEGIES", "BENCHMARK", "make_strategy", "Strategy", "StrategyContext"]
