"""Index constituents (NIFTY 50/100/200/500) with sector labels.

Lists come from niftyindices.com and are cached for a week. A small
built-in NIFTY 50 list is used if the download fails.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from .config import CACHE_DIR

INDEX_FILES = {
    "nifty50": "ind_nifty50list.csv",
    "nifty100": "ind_nifty100list.csv",
    "nifty200": "ind_nifty200list.csv",
    "nifty500": "ind_nifty500list.csv",
}
SOURCES = [
    "https://niftyindices.com/IndexConstituent/{f}",
    "https://archives.nseindia.com/content/indices/{f}",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}

# Benchmarks and context series on Yahoo Finance.
NIFTY = "^NSEI"
BANKNIFTY = "^NSEBANK"
INDIA_VIX = "^INDIAVIX"
# Yahoo's history for most NSE sector indices is too gappy to use, so sector
# strength is computed from constituents instead. These few are reliable.
SECTOR_INDICES = {
    "Bank Nifty": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Midcap 50": "^NSEMDCP50",
    "Nifty 500": "^CRSLDX",
}

_FALLBACK_NIFTY50 = """ADANIENT,Metals & Mining
ADANIPORTS,Services
APOLLOHOSP,Healthcare
ASIANPAINT,Consumer Durables
AXISBANK,Financial Services
BAJAJ-AUTO,Automobile and Auto Components
BAJFINANCE,Financial Services
BAJAJFINSV,Financial Services
BEL,Capital Goods
BHARTIARTL,Telecommunication
CIPLA,Healthcare
COALINDIA,Oil Gas & Consumable Fuels
DRREDDY,Healthcare
EICHERMOT,Automobile and Auto Components
ETERNAL,Consumer Services
GRASIM,Construction Materials
HCLTECH,Information Technology
HDFCBANK,Financial Services
HDFCLIFE,Financial Services
HEROMOTOCO,Automobile and Auto Components
HINDALCO,Metals & Mining
HINDUNILVR,Fast Moving Consumer Goods
ICICIBANK,Financial Services
INDUSINDBK,Financial Services
INFY,Information Technology
ITC,Fast Moving Consumer Goods
JIOFIN,Financial Services
JSWSTEEL,Metals & Mining
KOTAKBANK,Financial Services
LT,Construction
M&M,Automobile and Auto Components
MARUTI,Automobile and Auto Components
NESTLEIND,Fast Moving Consumer Goods
NTPC,Power
ONGC,Oil Gas & Consumable Fuels
POWERGRID,Power
RELIANCE,Oil Gas & Consumable Fuels
SBILIFE,Financial Services
SHRIRAMFIN,Financial Services
SBIN,Financial Services
SUNPHARMA,Healthcare
TCS,Information Technology
TATACONSUM,Fast Moving Consumer Goods
TATAMOTORS,Automobile and Auto Components
TATASTEEL,Metals & Mining
TECHM,Information Technology
TITAN,Consumer Durables
TRENT,Consumer Services
ULTRACEMCO,Construction Materials
WIPRO,Information Technology"""


def to_yahoo(symbol: str) -> str:
    return symbol if symbol.startswith("^") or symbol.endswith((".NS", ".BO")) else f"{symbol}.NS"


def from_yahoo(ticker: str) -> str:
    return ticker.removesuffix(".NS").removesuffix(".BO")


def _download(fname: str) -> pd.DataFrame | None:
    for url in SOURCES:
        try:
            r = requests.get(url.format(f=fname), headers=HEADERS, timeout=20)
            if r.ok and "Symbol" in r.text[:500]:
                return pd.read_csv(io.StringIO(r.text))
        except requests.RequestException:
            continue
    return None


def get_universe(index: str = "nifty200", max_age_days: int = 7) -> pd.DataFrame:
    """Return DataFrame[symbol, name, sector, ticker] for an index."""
    index = index.lower()
    if index not in INDEX_FILES:
        raise ValueError(f"Unknown index {index!r}; choose from {list(INDEX_FILES)}")
    cache = CACHE_DIR / f"universe_{index}.csv"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_days * 86400
    df = pd.read_csv(cache) if fresh else None
    if df is None:
        raw = _download(INDEX_FILES[index])
        if raw is not None:
            raw = raw[raw.get("Series", "EQ") == "EQ"] if "Series" in raw else raw
            df = pd.DataFrame({
                "symbol": raw["Symbol"].str.strip(),
                "name": raw["Company Name"].str.strip(),
                "sector": raw["Industry"].str.strip(),
            })
            df.to_csv(cache, index=False)
        elif cache.exists():
            df = pd.read_csv(cache)
        else:
            rows = [line.split(",", 1) for line in _FALLBACK_NIFTY50.splitlines()]
            df = pd.DataFrame(rows, columns=["symbol", "sector"])
            df["name"] = df["symbol"]
    df["ticker"] = df["symbol"].map(to_yahoo)
    return df.reset_index(drop=True)
