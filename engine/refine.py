"""Refinement: turn the full universe into a tradeable watchlist.

Two-tier, matching the funnel:
  Tier 1 (fundamentals)  screen_fundamentals()  — market cap, sector caps, margin, P/E.
                         Runs on the full universe *before* the heavy price pull.
  Tier 2 (volatility)    screen_volatility()     — ATR% band + final sector caps.
                         Runs on the shortlist *after* prices are fetched.

Pinned names (owned holdings + ETFs) always survive both tiers. Missing fundamentals
are treated as "unknown" and never auto-rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RefineParams:
    """Slider-driven refinement knobs. Defaults are permissive (keep everything)."""

    # Tier 1 — fundamentals
    min_market_cap: float = 0.0          # dollars; 0 = no floor
    min_profit_margin: float | None = None  # e.g. 0.0 to require profitability
    max_pe: float | None = None          # e.g. 40 to drop expensive names

    # Tier 2 — volatility
    atr_pct_min: float = 0.0
    atr_pct_max: float = 1.0             # fraction of price; 1.0 = no ceiling

    # Applied per tier
    max_per_sector: int | None = None    # cap names per GICS sector; None = no cap

    # Whether owned/ETF ("pinned") names bypass the screens. True = current behavior
    # (always kept). Set False for the "systematic" sleeve so pins must earn their
    # place on the same footing as everything else.
    keep_pinned: bool = True


def screen_fundamentals(universe: pd.DataFrame, fundamentals: pd.DataFrame, params: RefineParams) -> pd.DataFrame:
    """Tier 1: cheap fundamental screen over the full universe -> shortlist.

    Args:
        universe: DataFrame with columns [ticker, name, sector, source, pinned].
        fundamentals: DataFrame indexed by ticker with market_cap/profit_margin/trailing_pe.
    Returns:
        The surviving rows of `universe` (pinned names always kept).
    """
    df = universe.set_index("ticker")
    f = fundamentals.reindex(df.index)
    pinned = df["pinned"].fillna(False) & params.keep_pinned

    keep = pinned.copy()  # pinned survive only when keep_pinned is on
    passes = pd.Series(True, index=df.index)

    if params.min_market_cap > 0:
        # NaN market cap -> unknown -> not auto-rejected only if we choose to keep it.
        passes &= f["market_cap"].fillna(0) >= params.min_market_cap
    if params.min_profit_margin is not None:
        passes &= f["profit_margin"].fillna(-1e9) >= params.min_profit_margin
    if params.max_pe is not None:
        # A name with no P/E (unprofitable / ETF) fails a max-PE gate unless pinned.
        passes &= f["trailing_pe"].fillna(1e9) <= params.max_pe

    keep = keep | passes
    shortlist = df[keep].reset_index()
    return _apply_sector_cap(
        shortlist, params.max_per_sector, rank_by=f["market_cap"], respect_pinned=params.keep_pinned
    )


def screen_volatility(shortlist: pd.DataFrame, atr_pct_latest: pd.Series, params: RefineParams) -> pd.DataFrame:
    """Tier 2: ATR% band + final sector cap on the price-fetched shortlist -> watchlist.

    Args:
        shortlist: output of screen_fundamentals (or any universe subset).
        atr_pct_latest: Series indexed by ticker of the most recent ATR% value.
    """
    df = shortlist.set_index("ticker")
    pinned = df["pinned"].fillna(False) & params.keep_pinned
    a = atr_pct_latest.reindex(df.index)

    in_band = (a >= params.atr_pct_min) & (a <= params.atr_pct_max)
    # NaN ATR% (insufficient history) is kept only if pinned.
    keep = pinned | in_band.fillna(False)

    watchlist = df[keep].reset_index()
    return _apply_sector_cap(
        watchlist, params.max_per_sector, rank_by=a, ascending=False, respect_pinned=params.keep_pinned
    )


def _apply_sector_cap(
    df: pd.DataFrame,
    max_per_sector: int | None,
    rank_by: pd.Series,
    ascending: bool = False,
    respect_pinned: bool = True,
) -> pd.DataFrame:
    """Keep at most `max_per_sector` names per sector, ranked by `rank_by`.

    When respect_pinned is True, pinned names never count against the cap and are
    always kept; when False they compete for slots like everyone else.
    """
    if not max_per_sector or df.empty:
        return df.reset_index(drop=True)

    rank = rank_by.reindex(df["ticker"].values).fillna(-1e18 if not ascending else 1e18).values
    work = df.assign(_rank=rank)

    is_pinned = work["pinned"].fillna(False) & respect_pinned
    pinned = work[is_pinned]
    contested = work[~is_pinned]

    capped = (
        contested.sort_values("_rank", ascending=ascending)
        .groupby("sector", group_keys=False)
        .head(max_per_sector)
    )
    out = pd.concat([pinned, capped]).drop(columns="_rank")
    return out.drop_duplicates(subset="ticker").sort_values("ticker").reset_index(drop=True)
