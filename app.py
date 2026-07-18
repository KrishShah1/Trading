"""Streamlit cockpit for the trading backtest engine (LOCAL LAB — disposable UI).

This file is the only place Streamlit is imported. It orchestrates and renders; all
logic lives in engine/. When the project moves to the Next.js/Supabase suite this UI
is rebuilt in shadcn/ui + Recharts and thrown away, while engine/ ports to TypeScript.

Run:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from engine import data, indicators, refine, sleeves, universe
from engine.metrics import summary_table
from engine.strategies import STRATEGIES

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"

st.set_page_config(page_title="Trading Backtest Lab", layout="wide")


# --- cached data access (pull once, filter in memory) -----------------------

@st.cache_data(show_spinner="Building universe…")
def _universe(use_sp500: bool) -> pd.DataFrame:
    sp = None if use_sp500 else pd.DataFrame(columns=universe.UNIVERSE_COLUMNS[:3])
    return universe.build_universe(CONFIG / "universe.yaml", sp500=sp)


@st.cache_data(show_spinner="Fetching fundamentals…")
def _fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    return data.fetch_fundamentals(list(tickers))


@st.cache_data(show_spinner="Pulling price history…")
def _prices(tickers: tuple[str, ...], start: str, end: str) -> dict[str, pd.DataFrame]:
    return data.fetch_prices(list(tickers), start, end)


@st.cache_data
def _defaults() -> dict:
    return yaml.safe_load((CONFIG / "strategies.yaml").read_text())


defaults = _defaults()

# --- sidebar: the control panel ---------------------------------------------

st.sidebar.title("⚙️ Controls")

with st.sidebar.expander("Universe & window", expanded=True):
    use_sp500 = st.checkbox("Include S&P 500 (slow first pull)", value=False,
                            help="Off = ETFs + owned only, for fast iteration.")
    years = st.slider("Years of history", 1.0, 5.0, 2.0, 0.5)
    start_cash = st.number_input("Starting capital ($)", 1000, 10_000_000,
                                 int(defaults["execution"]["start_cash"]), step=1000)
    cost_bps = st.slider("Transaction cost (bps)", 0, 50,
                         int(defaults["execution"]["cost_bps"]))

st.sidebar.subheader("Refine — Tier 1 (fundamentals)")
min_mktcap_b = st.sidebar.slider("Min market cap ($B)", 0.0, 500.0, 0.0, 5.0)
max_per_sector = st.sidebar.slider("Max names per sector (0 = no cap)", 0, 20, 0)
require_profit = st.sidebar.checkbox("Require positive profit margin", value=False)
max_pe = st.sidebar.slider("Max trailing P/E (0 = no cap)", 0, 100, 0)

st.sidebar.subheader("Refine — Tier 2 (volatility)")
atr_lo, atr_hi = st.sidebar.slider("ATR% band (daily)", 0.0, 0.15, (0.0, 0.15), 0.005)

params = refine.RefineParams(
    min_market_cap=min_mktcap_b * 1e9,
    max_per_sector=max_per_sector or None,
    min_profit_margin=0.0 if require_profit else None,
    max_pe=float(max_pe) if max_pe else None,
    atr_pct_min=atr_lo,
    atr_pct_max=atr_hi,
)

st.sidebar.subheader("Strategies")
selected = st.sidebar.multiselect("Run", list(STRATEGIES), default=list(STRATEGIES))

# Per-strategy param sliders (only for the ones selected & that have knobs).
strat_params: dict[str, dict] = {name: dict(defaults.get(name, {})) for name in selected}
if "atr_harvest" in selected:
    with st.sidebar.expander("atr_harvest params"):
        strat_params["atr_harvest"]["buy_mult"] = st.slider("Buy mult (×ATR below entry)", 0.5, 5.0, 1.5, 0.25)
        strat_params["atr_harvest"]["trim_mult"] = st.slider("Trim mult (×ATR above entry)", 0.5, 6.0, 3.0, 0.25)
        strat_params["atr_harvest"]["trade_fraction"] = st.slider("Trade fraction", 0.05, 1.0, 0.25, 0.05)
if "rebalance" in selected:
    with st.sidebar.expander("rebalance params"):
        strat_params["rebalance"]["drift_band"] = st.slider("Drift band", 0.01, 0.25, 0.05, 0.01)
        strat_params["rebalance"]["freq"] = st.selectbox("Rebalance freq", ["none", "weekly", "monthly", "quarterly"], index=2)

st.sidebar.subheader("Sleeves to compare")
sleeve_sel = st.sidebar.multiselect(
    "Books",
    list(sleeves.SLEEVES),
    default=list(sleeves.SLEEVES),
    help="pinned = just your owned holdings (Buy & Hold) · systematic = pure screened "
         "picks (pins NOT force-kept) · combo = both together.",
)

# --- run the funnel ---------------------------------------------------------

end = date.today()
start = end - timedelta(days=int(years * 365) + 5)
start_s, end_s = start.isoformat(), end.isoformat()

try:
    uni = _universe(use_sp500)
    fund = _fundamentals(tuple(uni["ticker"]))
    # Superset shortlist (pins kept) drives the single price pull; the systematic
    # sleeve is derived from the same cached data with keep_pinned=False.
    superset = refine.screen_fundamentals(uni, fund, replace(params, keep_pinned=True))
    prices = _prices(tuple(superset["ticker"]), start_s, end_s)
except Exception as exc:  # noqa: BLE001 — surface data/network issues in the UI
    st.error(f"Data fetch failed: {exc}\n\nInstall deps (`pip3 install -r requirements.txt`) "
             f"and ensure network access to Yahoo Finance / Wikipedia.")
    st.stop()

if not prices or "adjclose" not in prices or prices["adjclose"].empty:
    st.warning("No price data for the current shortlist. Loosen the Tier-1 filters.")
    st.stop()

have = set(prices["adjclose"].columns)
atr_pct_panel = indicators.atr_pct(prices["high"], prices["low"], prices["close"], 14)
latest_atr_pct = atr_pct_panel.ffill().iloc[-1]

# Systematic sleeve: full screen with pins NOT force-kept.
sys_params = replace(params, keep_pinned=False)
sys_short = refine.screen_fundamentals(uni, fund, sys_params)
sys_watch = refine.screen_volatility(sys_short, latest_atr_pct, sys_params)
systematic = [t for t in sys_watch["ticker"] if t in have]

books = sleeves.build_sleeves(uni, systematic, available=have)
books = {k: v for k, v in books.items() if k in sleeve_sel}

st.title("📈 Algorithmic Trading Backtest Lab")
st.caption("Volatility harvesting is strategy #1 — the engine runs any strategy the same way. "
           "Compare your holdings (pinned) vs the algorithm (systematic) vs both (combo).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe", len(uni))
c2.metric("Systematic picks", len(systematic))
c3.metric("Sleeves", len(books))
c4.metric("History", f"{years:g} yr")

tab_universe, tab_backtest = st.tabs(["🔎 Universe & Sleeves", "🚀 Backtest"])

# --- Tab 1: universe & sleeves (live) ---------------------------------------

with tab_universe:
    st.subheader("Sleeve composition")
    if not books:
        st.warning("No sleeves selected / all empty — check the sidebar and filters.")
    else:
        for name, tickers in books.items():
            st.markdown(f"**{name}** — {len(tickers)} names")
            st.caption(", ".join(tickers) if tickers else "_(empty)_")

    if systematic:
        view = sys_watch[sys_watch["ticker"].isin(systematic)].copy()
        view["atr_pct"] = view["ticker"].map(latest_atr_pct.round(4))
        view["market_cap_$B"] = view["ticker"].map((fund["market_cap"] / 1e9).round(1))
        left, right = st.columns([2, 1])
        left.subheader("Systematic watchlist")
        left.dataframe(view, use_container_width=True, hide_index=True)
        right.subheader("Sector breakdown")
        right.bar_chart(view["sector"].value_counts())

    with st.expander("Full universe (pre-refinement)"):
        st.dataframe(uni, use_container_width=True, hide_index=True)

# --- Tab 2: backtest ---------------------------------------------------------

with tab_backtest:
    if not books or not selected:
        st.info("Pick at least one sleeve and one strategy to run a backtest.")
    else:
        results = sleeves.run_matrix(
            books, {n: strat_params.get(n, {}) for n in selected},
            prices, start_cash=start_cash, cost_bps=cost_bps,
            pinned_weights=sleeves.holdings_weights(
                universe.load_holdings(CONFIG / "universe.yaml"), prices
            ),
        )
        curves = {lbl: res.equity_curve for lbl, res in results.items()}
        benchmark = sleeves.default_benchmark(results)

        # Equity curves.
        fig = go.Figure()
        for lbl, curve in curves.items():
            is_bench = lbl == benchmark
            fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=lbl,
                                     line=dict(width=4 if is_bench else 2,
                                               dash="dash" if is_bench else "solid")))
        fig.update_layout(title=f"Equity curves (benchmark: {benchmark})", height=450,
                          yaxis_title="Portfolio value ($)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table.
        table = summary_table(curves, benchmark_name=benchmark)
        st.subheader(f"Performance — alpha vs `{benchmark}`")
        st.dataframe(
            table.style.format({
                "total_return": "{:.2%}", "cagr": "{:.2%}", "ann_vol": "{:.2%}",
                "sharpe": "{:.2f}", "sortino": "{:.2f}", "max_drawdown": "{:.2%}",
                "final_equity": "${:,.0f}", "alpha_vs_benchmark": "{:+.2%}",
            }),
            use_container_width=True,
        )

        # Drawdown.
        dd = go.Figure()
        for lbl, curve in curves.items():
            drawdown = curve / curve.cummax() - 1
            dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name=lbl, fill="tozeroy"))
        dd.update_layout(title="Drawdown", height=300, yaxis_tickformat=".0%", hovermode="x unified")
        st.plotly_chart(dd, use_container_width=True)

        # Downloads + trade logs.
        st.subheader("Export (ground truth)")
        st.download_button("metrics.csv", table.to_csv().encode(), "metrics.csv", "text/csv")
        with st.expander("Trade logs"):
            for lbl, res in results.items():
                st.write(f"**{lbl}** — {len(res.trades)} trades")
                if not res.trades.empty:
                    st.dataframe(res.trades.tail(200), use_container_width=True, hide_index=True)
