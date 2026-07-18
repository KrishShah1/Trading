"""Buy & Hold — the benchmark / control group.

Sets target weights once on the first day it runs, then returns None forever after
(no trading). Everything else is scored against this.

By default weights are equal. Pass params["weights"] = {ticker: weight} to hold a
specific allocation — this is how the `pinned` sleeve reproduces your real portfolio
(holdings × price), rather than an equal-weight approximation.
"""

from __future__ import annotations

from .base import Strategy, StrategyContext


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._initialized = False

    def target_weights(self, ctx: StrategyContext):
        if self._initialized:
            return None  # never trade again
        tradeable = self._tradeable(ctx)
        if not tradeable:
            return None
        self._initialized = True

        preset = self.params.get("weights")
        if preset:
            # Keep only tradeable names, then renormalize so they sum to 1.
            kept = {t: float(preset[t]) for t in tradeable if t in preset and preset[t] > 0}
            total = sum(kept.values())
            if total > 0:
                return {t: w / total for t, w in kept.items()}
        return self._equal_weights(tradeable)

