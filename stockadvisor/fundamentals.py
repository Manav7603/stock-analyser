"""Company fundamentals and upcoming results dates from Yahoo Finance.

Snapshot data only (today's values), cached for a week. They are used for
long-term ranking and for the "results due soon" warning, never in
backtests - there is no point-in-time history, so using them in the past
would leak future information.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import yfinance as yf

from .config import CACHE_DIR

FILE = CACHE_DIR / "fundamentals.json"
FIELDS = ["returnOnEquity", "debtToEquity", "trailingPE", "forwardPE", "priceToBook",
          "earningsGrowth", "revenueGrowth", "profitMargins", "operatingMargins",
          "marketCap", "dividendYield", "earningsQuarterlyGrowth", "pegRatio",
          "heldPercentInstitutions", "heldPercentInsiders", "recommendationKey",
          "targetMeanPrice", "numberOfAnalystOpinions", "currentRatio"]
MAX_AGE = 7 * 86400


def _load() -> dict:
    try:
        return json.loads(FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _fetch_one(ticker: str) -> dict:
    out = {"_ts": time.time()}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        out.update({k: info.get(k) for k in FIELDS})
        try:
            cal = tk.calendar or {}
            ed = cal.get("Earnings Date")
            if ed:
                out["next_results"] = str(ed[0] if isinstance(ed, list) else ed)
            exd = cal.get("Ex-Dividend Date")
            if exd:
                out["ex_dividend"] = str(exd)
        except Exception:
            pass
    except Exception as e:  # network / parsing - keep going for other tickers
        out["_error"] = str(e)[:120]
    return out


def get_fundamentals(tickers: list[str], refresh: bool = False, workers: int = 8) -> dict[str, dict]:
    cache = _load()
    now = time.time()
    todo = [t for t in tickers if refresh or t not in cache or now - cache[t].get("_ts", 0) > MAX_AGE]
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for t, res in zip(todo, ex.map(_fetch_one, todo)):
                cache[t] = res
        FILE.write_text(json.dumps(cache, default=str))
    return {t: cache.get(t, {}) for t in tickers}


def days_to_results(f: dict, today: date | None = None) -> int | None:
    d = f.get("next_results")
    if not d:
        return None
    try:
        dt = datetime.fromisoformat(str(d)[:10]).date()
    except ValueError:
        return None
    delta = (dt - (today or date.today())).days
    return delta if delta >= -1 else None
