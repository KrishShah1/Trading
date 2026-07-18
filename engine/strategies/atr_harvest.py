"""ATR Harvest — volatility harvesting (strategy #1).

Each position carries a running *reference price*. Measured in ATR units from that
reference:
  * price >= ref + trim_mult * ATR   -> TRIM (take profit toward cash)
  * price <= ref - buy_mult  * ATR   -> ADD  (deploy cash into the dip)

On any trigger we adjust that name's target weight by `trade_fraction`, reset its
reference to today's price, and return the new target weights (which need not sum
to 1 — the slack is cash). Days with no trigger return None.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategyContext


class AtrHarvest(Strategy):
    name = "atr_harvest"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.buy_mult = float(self.params.get("buy_mult", 1.5))
        self.trim_mult = float(self.params.get("trim_mult", 3.0))
        self.trade_fraction = float(self.params.get("trade_fraction", 0.25))
        self._target: dict[str, float] | None = None
        self._ref: dict[str, float] = {}  # reference price per ticker

    def _init_targets(self, ctx: StrategyContext) -> dict[str, float] | None:
        tradeable = self._tradeable(ctx)
        if not tradeable:
            return None
        self._target = self._equal_weights(tradeable)
        self._ref = dict(ctx.prices_today)
        return dict(self._target)

    def target_weights(self, ctx: StrategyContext):
        if self._target is None:
            return self._init_targets(ctx)  # establish the initial book

        prices = ctx.prices_today
        atr_row = ctx.atr.iloc[-1]
        triggered = False

        for t in list(self._target):
            px = prices.get(t)
            a = atr_row.get(t)
            ref = self._ref.get(t)
            if px is None or ref is None or a is None or pd.isna(a) or a <= 0:
                continue

            if px >= ref + self.trim_mult * a:
                # Winner ran — trim toward cash, reset the band at today's price.
                self._target[t] = max(0.0, self._target[t] * (1 - self.trade_fraction))
                self._ref[t] = px
                triggered = True
            elif px <= ref - self.buy_mult * a:
                # Dip — add from cash, but keep the book from exceeding fully invested.
                headroom = max(0.0, 1.0 - sum(self._target.values()))
                add = min(self._target[t] * self.trade_fraction, headroom)
                self._target[t] += add
                self._ref[t] = px
                triggered = True

        return dict(self._target) if triggered else None
