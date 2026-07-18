"""Headless backtest runner — reproducible, no UI.

Builds the universe, screens it, pulls 2yr of prices once, derives the pinned /
systematic / combo sleeves, runs every (sleeve × strategy) combination, and writes
CSV "ground truth" outputs. Mirrors what will become a Next.js Route Handler later.

Usage:
    python3 scripts/run_backtest.py
    python3 scripts/run_backtest.py --years 2 --no-sp500   # ETFs + owned only (fast)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import data, indicators, refine, sleeves, universe  # noqa: E402
from engine.metrics import summary_table  # noqa: E402
from engine.strategies import STRATEGIES  # noqa: E402

CONFIG = ROOT / "config"
OUTPUTS = ROOT / "outputs"


def prepare_data(use_sp500: bool, start: str, end: str, params: refine.RefineParams):
    """Run the funnel and return (universe, prices, systematic_watchlist).

    Prices are pulled once for the keep_pinned=True fundamentals shortlist (a superset
    that always contains the owned/ETF names the 'pinned' sleeve needs). The systematic
    sleeve is then derived with keep_pinned=False from the same cached data.
    """
    sp500 = None if use_sp500 else pd.DataFrame(columns=universe.UNIVERSE_COLUMNS[:3])
    uni = universe.build_universe(CONFIG / "universe.yaml", sp500=sp500)
    print(f"Universe: {len(uni)} tickers ({uni['source'].value_counts().to_dict()})")

    fundamentals = data.fetch_fundamentals(uni["ticker"].tolist())

    # Superset shortlist (pins kept) drives the single price pull.
    superset = refine.screen_fundamentals(uni, fundamentals, replace(params, keep_pinned=True))
    prices = data.fetch_prices(superset["ticker"].tolist(), start, end)
    if not prices or "adjclose" not in prices:
        raise SystemExit("No price data returned — check network / tickers.")
    have = set(prices["adjclose"].columns)

    atr_pct = indicators.atr_pct(prices["high"], prices["low"], prices["close"], 14)
    latest_atr_pct = atr_pct.ffill().iloc[-1]

    # Systematic sleeve: screen the full universe with pins NOT force-kept.
    sys_params = replace(params, keep_pinned=False)
    sys_short = refine.screen_fundamentals(uni, fundamentals, sys_params)
    sys_watch = refine.screen_volatility(sys_short, latest_atr_pct, sys_params)
    systematic = [t for t in sys_watch["ticker"] if t in have]
    print(f"Systematic watchlist ({len(systematic)}): {systematic}")
    return uni, prices, systematic


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the algorithmic trading backtest engine.")
    ap.add_argument("--years", type=float, default=2.0, help="years of history (default 2)")
    ap.add_argument("--no-sp500", action="store_true", help="skip S&P 500 pull (ETFs + owned only)")
    ap.add_argument("--start-cash", type=float, default=None)
    ap.add_argument("--min-market-cap", type=float, default=0.0)
    ap.add_argument("--max-per-sector", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((CONFIG / "strategies.yaml").read_text())
    execu = cfg.get("execution", {})
    start_cash = args.start_cash or execu.get("start_cash", 100_000)
    cost_bps = execu.get("cost_bps", 0)

    end = date.today()
    start = end - timedelta(days=int(args.years * 365) + 5)
    start_s, end_s = start.isoformat(), end.isoformat()

    params = refine.RefineParams(
        min_market_cap=args.min_market_cap,
        max_per_sector=args.max_per_sector,
    )

    uni, prices, systematic = prepare_data(not args.no_sp500, start_s, end_s, params)

    available = set(prices["adjclose"].columns)
    books = sleeves.build_sleeves(uni, systematic, available=available)
    if not books:
        raise SystemExit("All sleeves are empty — loosen the filters or check config.")
    print(f"Sleeves: {{ {', '.join(f'{k}:{len(v)}' for k, v in books.items())} }}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.Series({k: ",".join(v) for k, v in books.items()}).to_csv(OUTPUTS / "sleeves.csv", header=["tickers"])

    strategy_specs = {name: cfg.get(name, {}) for name in STRATEGIES}
    pinned_weights = sleeves.holdings_weights(universe.load_holdings(CONFIG / "universe.yaml"), prices)
    results = sleeves.run_matrix(
        books, strategy_specs, prices, start_cash=start_cash, cost_bps=cost_bps,
        pinned_weights=pinned_weights,
    )

    equity_curves: dict[str, pd.Series] = {}
    for label, res in results.items():
        equity_curves[label] = res.equity_curve
        safe = label.replace(" · ", "__").replace(" ", "_")
        res.equity_curve.to_csv(OUTPUTS / f"equity_{safe}.csv", header=["equity"])
        if not res.trades.empty:
            res.trades.to_csv(OUTPUTS / f"trades_{safe}.csv", index=False)
        print(f"  {label:28} {len(res.trades):5d} trades, final ${res.equity_curve.iloc[-1]:,.0f}")

    benchmark = sleeves.default_benchmark(results)
    table = summary_table(equity_curves, benchmark_name=benchmark)
    table.to_csv(OUTPUTS / "metrics.csv")
    print(f"\n=== Performance (alpha vs benchmark: {benchmark}) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:,.4f}"):
        print(table)
    print(f"\nOutputs written to {OUTPUTS}/")


if __name__ == "__main__":
    main()
