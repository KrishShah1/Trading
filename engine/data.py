"""Market data access: fundamentals (cheap screen) + OHLC prices (heavy pull).

Two-tier by design, matching the funnel:
  Tier 1  fetch_fundamentals()  -> cheap per-ticker fields, screens the full universe
  Tier 2  fetch_prices()        -> 2yr OHLC, pulled only for the surviving shortlist

Both tiers cache to parquet under data/ so repeated runs and slider-driven UI
re-filtering never re-hit the network unless the request changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Fundamental fields we screen on, mapped from yfinance's .info keys.
_FUNDAMENTAL_FIELDS = {
    "marketCap": "market_cap",
    "profitMargins": "profit_margin",
    "trailingPE": "trailing_pe",
}


def _cache_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def fetch_fundamentals(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Fetch cheap per-ticker fundamentals for the Stage 1 screen.

    Returns a DataFrame indexed by ticker with columns
    [market_cap, profit_margin, trailing_pe]. Missing values are NaN (common for
    ETFs and some names) and are treated as "unknown" — never auto-rejected — by
    the refine step. Cached to data/fundamentals.parquet.
    """
    cache = _cache_path("fundamentals.parquet")
    cached = pd.read_parquet(cache) if (cache.exists() and not refresh) else pd.DataFrame()

    have = set(cached.index) if not cached.empty else set()
    missing = [t for t in tickers if t not in have]

    rows = []
    for t in missing:
        try:
            info = yf.Ticker(t).info
        except Exception:  # noqa: BLE001 — network/parse failures shouldn't be fatal
            info = {}
        rows.append(
            {"ticker": t, **{dst: info.get(src) for src, dst in _FUNDAMENTAL_FIELDS.items()}}
        )

    fresh = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
    combined = pd.concat([cached, fresh]) if not cached.empty else fresh
    if not combined.empty:
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(cache)

    # Return only requested tickers, preserving order, with all expected columns.
    cols = list(_FUNDAMENTAL_FIELDS.values())
    out = combined.reindex(tickers)
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


def fetch_prices(
    tickers: list[str], start: str, end: str, refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLC for the shortlist (Tier 2 heavy pull).

    Returns a dict {field: wide DataFrame} where field is one of
    {"open","high","low","close","adjclose"} and each DataFrame is indexed by date
    with one column per ticker. High/Low/Close feed ATR; Adj Close feeds returns.

    Cached to data/prices.parquet. The cache is reused when it already covers the
    requested [start, end] range for all requested tickers; otherwise re-downloaded.
    """
    cache = _cache_path("prices.parquet")

    if cache.exists() and not refresh:
        cached = pd.read_parquet(cache)
        if _cache_covers(cached, tickers, start, end):
            return _reshape_prices(_slice(cached, tickers, start, end))

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    long_df = _to_long(raw, tickers)
    long_df.to_parquet(cache)
    return _reshape_prices(long_df)


# --- internal helpers -------------------------------------------------------

_PRICE_FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjclose",
}


def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output (which varies by ticker count) into a long frame
    with columns [date, ticker, open, high, low, close, adjclose]."""
    records = []
    single = len(tickers) == 1
    for field, dst in _PRICE_FIELDS.items():
        if single:
            if field not in raw.columns:
                continue
            series = raw[field]
            sub = pd.DataFrame({"date": raw.index, "ticker": tickers[0], dst: series.values})
        else:
            if field not in raw.columns.get_level_values(0):
                continue
            block = raw[field]
            sub = block.reset_index().melt(id_vars="Date", var_name="ticker", value_name=dst)
            sub = sub.rename(columns={"Date": "date"})
        records.append(sub)

    if not records:
        return pd.DataFrame(columns=["date", "ticker", *_PRICE_FIELDS.values()])

    out = records[0]
    for sub in records[1:]:
        out = out.merge(sub, on=["date", "ticker"], how="outer")
    out = out.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def _reshape_prices(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long frame -> {field: wide DataFrame indexed by date, columns=tickers}."""
    fields = {}
    for dst in _PRICE_FIELDS.values():
        if dst in long_df.columns:
            fields[dst] = long_df.pivot(index="date", columns="ticker", values=dst).sort_index()
    return fields


def _cache_covers(cached: pd.DataFrame, tickers: list[str], start: str, end: str) -> bool:
    if cached.empty:
        return False
    have = set(cached["ticker"].unique())
    if not set(tickers).issubset(have):
        return False
    dmin, dmax = cached["date"].min(), cached["date"].max()
    return dmin <= pd.Timestamp(start) and dmax >= pd.Timestamp(end) - pd.Timedelta(days=5)


def _slice(cached: pd.DataFrame, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    mask = (
        cached["ticker"].isin(tickers)
        & (cached["date"] >= pd.Timestamp(start))
        & (cached["date"] <= pd.Timestamp(end))
    )
    return cached[mask].copy()
