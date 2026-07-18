# Algorithmic Trading Backtest Engine — Local Lab

A local Python laboratory for **backtesting algorithmic trading strategies** over historical
data. Volatility harvesting is strategy #1, not the whole product — the engine treats every
strategy identically and is built to grow a zoo of them (momentum, mean-reversion, ...).

## The core idea

Every strategy answers exactly one question each trading day:

> *"Given everything I know up to today, what target weights do I want to hold?"*

The **engine** owns everything else — data, indicators, portfolio accounting, execution,
scoring. That separation is what makes this a trading platform rather than a single script:
a new strategy is pure decision logic (one file + one registry line), no plumbing.

## Flow

```
1. UNIVERSE      S&P 500 + ETFs + your owned holdings  (deduped, tagged by source)
2. SCREEN        cheap fundamentals filter (market cap, sector, margin, P/E)  → shortlist
3. DATA          pull 2yr OHLC for the shortlist only, cached to parquet
4. REFINE        ATR% / volatility band + sector caps                         → watchlist
5. SLEEVES       split into books to compare: pinned · systematic · combo
6. BACKTEST      for each day: strategy -> target weights -> execute trades -> equity curve
7. METRICS       CAGR, Sharpe, max drawdown, and ALPHA vs benchmark
8. OUTPUT        CSV logs (ground truth) + Streamlit charts
```

The 2-year price pull is **scoped to the shortlist and cached** — the backtest reuses that
cache, and moving refinement sliders re-filters in memory (no re-download).

## Sleeves — "pinned" is a comparison, not a rule

Rather than force your owned holdings into every watchlist, the engine runs three **sleeves**
side-by-side through the same backtest so you can see what actually earns its keep:

| Sleeve       | What it is                                    | Default strategy     |
| ------------ | --------------------------------------------- | -------------------- |
| `pinned`     | your owned holdings, at your real allocation  | Buy & Hold (baseline) |
| `systematic` | the screened universe, pins **not** force-kept | your active strategy |
| `combo`      | pinned ∪ systematic                           | your active strategy |

The `pinned` sleeve is **holdings-weighted** — it reads the share quantities in
`config/universe.yaml` (`holdings:`) and reproduces your true current allocation, so the
comparison is against *your actual portfolio*, not an equal-weight stand-in.

You pick which sleeves and which strategies to compare; the engine runs the cross-product
(`systematic · atr_harvest`, `pinned · buy_and_hold`, …). The benchmark for alpha is
`pinned · buy_and_hold` — so alpha answers *"does this beat just holding my forever-list?"*
The `keep_pinned` flag on `RefineParams` is what makes the `systematic` sleeve honest (pins must
qualify on the metrics like anything else).

## Layout

```
engine/            # PURE python, zero UI imports — this is what ports to TypeScript later
  universe.py      # build the tagged universe
  data.py          # yfinance fetch (fundamentals + OHLC) with parquet cache
  indicators.py    # Wilder ATR, ATR%, returns, realized vol
  refine.py        # two-tier screen (fundamentals then volatility) -> watchlist
  sleeves.py       # pinned / systematic / combo books + sleeve×strategy runner
  portfolio.py     # cash, positions, value/weights, trade execution w/ costs
  backtest.py      # the daily loop
  metrics.py       # performance stats + alpha vs benchmark
  strategies/      # the strategy zoo (base + registry)
config/            # universe.yaml (ETFs/owned/exclude), strategies.yaml (params)
scripts/           # run_backtest.py — headless, reproducible
app.py             # Streamlit cockpit (disposable local UI)
data/ outputs/     # cache + generated CSVs (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
```

## Run

```bash
# headless backtest (uses config defaults) → CSVs in outputs/
python3 scripts/run_backtest.py

# interactive cockpit — tune refinement + strategy params with sliders
streamlit run app.py
```

## Adding a strategy

1. Create `engine/strategies/my_strategy.py` subclassing `Strategy`.
2. Implement `target_weights(...)` — return a `{ticker: weight}` dict, or `None` for "no change".
3. Register it in `engine/strategies/__init__.py` and (optionally) add defaults to
   `config/strategies.yaml`. It's now available to the CLI and cockpit automatically.

## Path to the wider suite (Phase 2+)

The `engine/` package is deliberately UI-agnostic and dependency-light so it ports cleanly:

| Local lab (now)            | Next.js / Supabase suite (later)             |
| -------------------------- | -------------------------------------------- |
| `engine/*.py` pure funcs   | TypeScript utility functions                 |
| `yfinance`                 | `yahoo-finance2` in a Route Handler          |
| CSVs in `outputs/`         | Supabase Postgres tables                     |
| Streamlit `app.py`         | shadcn/ui + Recharts (rebuilt, not ported)   |
