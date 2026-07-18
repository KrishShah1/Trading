"""Portfolio accounting: cash, share positions, valuation, and trade execution.

Strategies never touch this directly — they only emit target weights. The backtest
loop calls apply_target_weights() to move the book toward those targets, which is
where trades, transaction costs, and cash bookkeeping actually happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trade:
    date: object
    ticker: str
    shares: float      # +buy / -sell
    price: float
    cost: float        # transaction cost paid on this trade


@dataclass
class Portfolio:
    """A long-only book. Weights are expressed as fractions of total equity."""

    cash: float
    cost_bps: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)  # ticker -> shares
    trades: list[Trade] = field(default_factory=list)

    def value(self, prices: dict[str, float]) -> float:
        """Total equity = cash + market value of holdings."""
        holdings = sum(sh * prices.get(t, 0.0) for t, sh in self.positions.items())
        return self.cash + holdings

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Current position weights as fractions of total equity."""
        total = self.value(prices)
        if total <= 0:
            return {}
        return {t: (sh * prices.get(t, 0.0)) / total for t, sh in self.positions.items()}

    def apply_target_weights(self, date, target: dict[str, float], prices: dict[str, float]) -> None:
        """Trade the book toward `target` weights at today's `prices`.

        Sells execute before buys so freed cash funds purchases. Transaction cost
        (cost_bps of traded notional) is charged to cash. Tickers with no price
        today are left untouched.
        """
        total = self.value(prices)
        if total <= 0:
            return

        # Desired share count per ticker (0 for anything not in target).
        desired: dict[str, float] = {}
        tickers = set(target) | set(self.positions)
        for t in tickers:
            px = prices.get(t)
            if px is None or px <= 0:
                desired[t] = self.positions.get(t, 0.0)  # can't trade -> hold
                continue
            w = target.get(t, 0.0)
            desired[t] = (w * total) / px

        deltas = {t: desired[t] - self.positions.get(t, 0.0) for t in tickers}

        # Sells first (delta < 0), then buys (delta > 0).
        for t in sorted(deltas, key=lambda k: deltas[k]):
            d = deltas[t]
            px = prices.get(t)
            if px is None or px <= 0 or abs(d * px) < 1e-6:
                continue
            notional = d * px
            cost = abs(notional) * self.cost_bps / 1e4
            self.cash -= notional + cost
            self.positions[t] = self.positions.get(t, 0.0) + d
            if abs(self.positions[t]) < 1e-9:
                self.positions.pop(t, None)
            self.trades.append(Trade(date=date, ticker=t, shares=d, price=px, cost=cost))
