"""Build the tradeable universe: S&P 500 + configured ETFs + owned holdings.

The universe is the full set of things we *could* trade, tagged by source so the
refinement step knows what to screen (S&P 500 names) vs. what to always keep
(owned holdings and ETFs are "pinned").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

# Columns every universe DataFrame carries.
UNIVERSE_COLUMNS = ["ticker", "name", "sector", "source", "pinned"]

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_universe_config(path: str | Path) -> dict:
    """Load config/universe.yaml, merging private quantities from holdings.local.yaml.

    Tickers live in `universe.yaml` (`owned:` list, safe to commit). Real share
    quantities live in a sibling `holdings.local.yaml` (gitignored). Either file may
    also carry an inline `holdings:` map for backward compatibility. All owned tickers
    plus any holdings keys are merged into the owned/pinned set.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    holdings = {str(t).upper(): float(q) for t, q in (cfg.get("holdings") or {}).items()}

    local = path.parent / "holdings.local.yaml"
    if local.exists():
        with open(local) as f:
            lcfg = yaml.safe_load(f) or {}
        holdings.update({str(t).upper(): float(q) for t, q in (lcfg.get("holdings") or {}).items()})

    owned_list = [t.upper() for t in (cfg.get("owned") or [])]
    owned = sorted(set(holdings) | set(owned_list))
    return {
        "etfs": [t.upper() for t in (cfg.get("etfs") or [])],
        "owned": owned,
        "holdings": holdings,
        "exclude": {t.upper() for t in (cfg.get("exclude") or [])},
    }


def load_holdings(path: str | Path) -> dict[str, float]:
    """Return {ticker: shares} for the owned portfolio (empty if none configured)."""
    return load_universe_config(path)["holdings"]


def sp500_constituents() -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia.

    Returns a DataFrame [ticker, name, sector]. Tickers are normalized to the
    format yfinance expects (e.g. BRK.B -> BRK-B).
    """
    tables = pd.read_html(_SP500_WIKI_URL)
    df = tables[0]
    out = pd.DataFrame(
        {
            "ticker": df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False),
            "name": df["Security"].astype(str),
            "sector": df["GICS Sector"].astype(str),
        }
    )
    return out.drop_duplicates(subset="ticker").reset_index(drop=True)


def build_universe(config_path: str | Path, sp500: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the full tagged universe.

    Args:
        config_path: path to universe.yaml.
        sp500: optional pre-fetched constituents (lets callers cache / mock the
            network call). If None, fetches from Wikipedia.

    Returns:
        DataFrame with columns UNIVERSE_COLUMNS, deduped by ticker. When a ticker
        appears in more than one source, priority is owned > etf > sp500 so pinned
        status is never lost.
    """
    cfg = load_universe_config(config_path)
    if sp500 is None:
        sp500 = sp500_constituents()

    frames = []

    sp = sp500.copy()
    sp["source"] = "sp500"
    sp["pinned"] = False
    frames.append(sp[UNIVERSE_COLUMNS])

    if cfg["etfs"]:
        frames.append(
            pd.DataFrame(
                {
                    "ticker": cfg["etfs"],
                    "name": cfg["etfs"],
                    "sector": "ETF",
                    "source": "etf",
                    "pinned": True,
                }
            )
        )

    if cfg["owned"]:
        # For owned names we may already know the sector from the S&P 500 table.
        sector_lookup = sp500.set_index("ticker")["sector"].to_dict()
        name_lookup = sp500.set_index("ticker")["name"].to_dict()
        frames.append(
            pd.DataFrame(
                {
                    "ticker": cfg["owned"],
                    "name": [name_lookup.get(t, t) for t in cfg["owned"]],
                    "sector": [sector_lookup.get(t, "Unknown") for t in cfg["owned"]],
                    "source": "owned",
                    "pinned": True,
                }
            )
        )

    universe = pd.concat(frames, ignore_index=True)

    # Drop excluded tickers.
    universe = universe[~universe["ticker"].isin(cfg["exclude"])]

    # Dedupe with priority owned > etf > sp500 (higher priority = kept).
    priority = {"owned": 3, "etf": 2, "sp500": 1}
    universe = universe.assign(_prio=universe["source"].map(priority))
    universe = (
        universe.sort_values("_prio", ascending=False)
        .drop_duplicates(subset="ticker", keep="first")
        .drop(columns="_prio")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return universe[UNIVERSE_COLUMNS]
