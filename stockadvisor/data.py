"""Price data from Yahoo Finance with an on-disk parquet cache.

Daily bars are split- and dividend-adjusted. The cache is updated
incrementally, so after the first download each refresh only fetches the
last few days.
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import CACHE_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

DAILY_DIR = CACHE_DIR / "daily"
INTRA_DIR = CACHE_DIR / "intraday"
DAILY_DIR.mkdir(parents=True, exist_ok=True)
INTRA_DIR.mkdir(parents=True, exist_ok=True)

COLS = ["open", "high", "low", "close", "volume"]


def _path(ticker: str, folder: Path) -> Path:
    return folder / (ticker.replace("^", "IDX_").replace("&", "and") + ".parquet")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    df = df[[c for c in COLS if c in df.columns]].dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _split_batch(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in level0:
                part = _normalise(raw[t].copy())
                if not part.empty:
                    out[t] = part
    elif len(tickers) == 1:
        part = _normalise(raw.copy())
        if not part.empty:
            out[tickers[0]] = part
    return out


def _yf_download(tickers: list[str], chunk: int = 60, **kw) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                raw = yf.download(batch, group_by="ticker", auto_adjust=True,
                                  progress=False, threads=True, **kw)
                out.update(_split_batch(raw, batch))
                break
            except Exception:  # yfinance raises a variety of network errors
                time.sleep(2 * (attempt + 1))
    return out


def get_daily(tickers: list[str] | str, years: int = 6, max_age_hours: float = 6,
              refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for each ticker, keyed by ticker. Missing tickers are skipped."""
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(dict.fromkeys(tickers))
    now = time.time()
    result: dict[str, pd.DataFrame] = {}
    missing, stale = [], []
    for t in tickers:
        p = _path(t, DAILY_DIR)
        if p.exists():
            df = pd.read_parquet(p)
            result[t] = df
            if refresh or now - p.stat().st_mtime > max_age_hours * 3600:
                stale.append(t)
        else:
            missing.append(t)

    if missing:
        start = (datetime.now() - timedelta(days=int(365.25 * years) + 5)).strftime("%Y-%m-%d")
        for t, df in _yf_download(missing, start=start).items():
            df.to_parquet(_path(t, DAILY_DIR))
            result[t] = df

    if stale:
        oldest = min(result[t].index.max() for t in stale)
        start = (oldest - timedelta(days=7)).strftime("%Y-%m-%d")
        fresh = _yf_download(stale, start=start)
        for t in stale:
            if t in fresh:
                merged = pd.concat([result[t], fresh[t]])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                result[t] = merged
            result[t].to_parquet(_path(t, DAILY_DIR))  # also bumps mtime

    _fill_missing_last_bars(result)
    return result


def _fill_missing_last_bars(result: dict[str, pd.DataFrame]) -> None:
    """Yahoo's daily feed sometimes lags a day for some NSE stocks. Rebuild the
    missing session(s) from hourly bars so every stock ends on the same date."""
    stocks = {t: df for t, df in result.items() if not t.startswith("^") and len(df)}
    if len(stocks) < 5:
        return
    latest = max(df.index.max() for df in stocks.values())
    behind = [t for t, df in stocks.items() if df.index.max() < latest]
    if not behind:
        return
    hourly = _yf_download(behind, period="5d", interval="60m", chunk=60)
    for t in behind:
        h = hourly.get(t)
        if h is None or h.empty:
            continue
        idx = h.index.tz_convert("Asia/Kolkata") if h.index.tz is not None else h.index
        h = h.set_axis(idx)
        days = h.groupby(h.index.date).agg(open=("open", "first"), high=("high", "max"),
                                           low=("low", "min"), close=("close", "last"),
                                           volume=("volume", "sum"))
        days.index = pd.to_datetime(days.index)
        new = days[(days.index > result[t].index.max()) & (days.index <= latest)]
        if not new.empty:
            result[t] = pd.concat([result[t], new.astype(result[t].dtypes.to_dict())]).sort_index()


def get_one(ticker: str, years: int = 6, **kw) -> pd.DataFrame:
    return get_daily([ticker], years=years, **kw).get(ticker, pd.DataFrame(columns=COLS))


def get_intraday(tickers: list[str] | str, interval: str = "15m", period: str = "60d",
                 max_age_minutes: float = 5) -> dict[str, pd.DataFrame]:
    """Intraday bars in IST. Yahoo keeps 60 days of 5m/15m and ~2 years of 1h."""
    if isinstance(tickers, str):
        tickers = [tickers]
    now = time.time()
    result, todo = {}, []
    for t in tickers:
        p = INTRA_DIR / f"{_path(t, INTRA_DIR).stem}_{interval}.parquet"
        if p.exists() and now - p.stat().st_mtime < max_age_minutes * 60:
            result[t] = pd.read_parquet(p)
        else:
            todo.append(t)
    for t, df in _yf_download(todo, interval=interval, period=period, chunk=40).items():
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("Asia/Kolkata")
        df.to_parquet(INTRA_DIR / f"{_path(t, INTRA_DIR).stem}_{interval}.parquet")
        result[t] = df
    return result


def panel(frames: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Wide DataFrame (dates x tickers) of one field."""
    return pd.DataFrame({t: df[field] for t, df in frames.items() if field in df}).sort_index()


def fix_corporate_actions(df: pd.DataFrame, threshold: float = 0.30) -> tuple[pd.DataFrame, list[str]]:
    """Back-adjust overnight gaps of 30%+ that Yahoo left unadjusted.

    Bonus issues, splits and demergers sometimes arrive unadjusted, which
    looks like a crash and corrupts indicators and backtests. A gap this
    size where the day opened at the new level is treated as a corporate
    action and earlier prices are scaled to match. Real 30%+ one-day falls
    in index stocks are very rare (Adani in Feb 2023 was -28%).
    """
    if len(df) < 2:
        return df, []
    df = df.astype(float)
    notes = []
    prev_close = df["close"].shift()
    gap = df["open"] / prev_close - 1
    for d in gap[gap.abs() >= threshold].index:
        i = df.index.get_loc(d)
        ratio = df["open"].iloc[i] / df["close"].iloc[i - 1]
        cols = ["open", "high", "low", "close"]
        df.iloc[:i, [df.columns.get_loc(c) for c in cols]] *= ratio
        if "volume" in df:
            df.iloc[:i, df.columns.get_loc("volume")] /= ratio
        notes.append(f"{d.date()}: adjusted earlier prices by x{ratio:.3f} (likely bonus/split/demerger)")
    return df, notes
