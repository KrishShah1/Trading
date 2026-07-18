"""Threshold + calendar rebalancing — harvests the "rebalancing premium".

Holds fixed target weights (equal by default). Rebalances back to target when:
  * any position drifts more than `drift_band` from its target, OR
  * a calendar trigger fires (`freq`: none | weekly | monthly | quarterly).

Mechanically sells what's risen and buys what's fallen — a simple, transparent
baseline to test whether ATR bands actually beat plain rebalancing.
"""

from __future__ import annotations

from .base import Strategy, StrategyContext

_FREQ_ATTR = {"weekly": "week", "monthly": "month", "quarterly": "quarter"}


class Rebalance(Strategy):
    name = "rebalance"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.drift_band = float(self.params.get("drift_band", 0.05))
        self.freq = self.params.get("freq", "monthly")
        self._targets: dict[str, float] | None = None
        self._last_period = None

    def _target(self, ctx: StrategyContext) -> dict[str, float]:
        if self._targets is None:
            self._targets = self._equal_weights(self._tradeable(ctx))
        return self._targets

    def _calendar_trigger(self, ctx: StrategyContext) -> bool:
        if self.freq in (None, "none"):
            return False
        ts = ctx.date
        if self.freq == "weekly":
            period = (ts.year, ts.isocalendar().week)
        elif self.freq == "quarterly":
            period = (ts.year, ts.quarter)
        else:  # monthly (default)
            period = (ts.year, ts.month)
        if period != self._last_period:
            self._last_period = period
            return True
        return False

    def target_weights(self, ctx: StrategyContext):
        target = self._target(ctx)
        if not target:
            return None

        first_time = not ctx.current_weights
        calendar = self._calendar_trigger(ctx)
        drifted = any(
            abs(ctx.current_weights.get(t, 0.0) - w) > self.drift_band for t, w in target.items()
        )

        if first_time or calendar or drifted:
            return target
        return None
