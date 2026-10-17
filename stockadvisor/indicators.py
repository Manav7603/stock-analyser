"""Technical indicators, written with pandas so there is no TA-Lib dependency.

Every function only looks backwards: the value on day t uses bars up to
and including t, never later ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, down = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(up.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def bollinger(close: pd.Series, n=20, k=2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n).std()
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_mid": mid, "bb_up": up, "bb_lo": lo,
                         "bb_width": (up - lo) / mid, "bb_pct": (close - lo) / (up - lo)})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    upper, lower = (hl2 + mult * a).to_numpy(), (hl2 - mult * a).to_numpy()
    close = df["close"].to_numpy()
    fu, fl = upper.copy(), lower.copy()
    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if np.isnan(upper[i]) or np.isnan(fu[i - 1]):
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if close[i] < fl[i] else 1
        else:
            trend[i] = 1 if close[i] > fu[i] else -1
    line = np.where(trend == 1, fl, fu)
    return pd.DataFrame({"st_line": line, "st_dir": trend}, index=df.index)


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    """Session VWAP that resets every trading day (index must be tz-aware IST)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vol


def pivots(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Classic floor pivots + Central Pivot Range, for intraday levels."""
    p = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = 2 * p - bc
    return {
        "P": p, "R1": 2 * p - prev_low, "S1": 2 * p - prev_high,
        "R2": p + (prev_high - prev_low), "S2": p - (prev_high - prev_low),
        "CPR_top": max(tc, bc), "CPR_bottom": min(tc, bc),
        "PDH": prev_high, "PDL": prev_low, "PDC": prev_close,
    }


def add_all(df: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with the full indicator set added."""
    o = df.copy()
    c = o["close"]
    for n in (10, 20, 50, 100, 200):
        o[f"ema{n}"] = ema(c, n)
    o["sma200"] = sma(c, 200)
    o["rsi14"] = rsi(c, 14)
    o["rsi2"] = rsi(c, 2)
    o["atr14"] = atr(o, 14)
    o["atr_pct"] = o["atr14"] / c * 100
    o = o.join(macd(c)).join(adx(o)).join(bollinger(c)).join(supertrend(o))
    o["obv"] = obv(o)
    o["obv_slope"] = o["obv"].diff(20) / o["volume"].rolling(20).mean() / 20
    o["vol20"] = o["volume"].rolling(20).mean()
    o["vol_ratio"] = o["volume"] / o["vol20"]
    o["turnover_cr"] = (c * o["volume"]).rolling(20).mean() / 1e7
    o["hh20"] = o["high"].rolling(20).max()
    o["ll20"] = o["low"].rolling(20).min()
    o["hh55"] = o["high"].rolling(55).max()
    o["hh252"] = o["high"].rolling(252, min_periods=120).max()
    o["ll252"] = o["low"].rolling(252, min_periods=120).min()
    o["dist_52wh"] = c / o["hh252"] - 1
    o["dist_52wl"] = c / o["ll252"] - 1
    for n in (5, 21, 63, 126, 252):
        o[f"ret{n}"] = c.pct_change(n)
    o["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    o["volatility63"] = c.pct_change().rolling(63).std() * np.sqrt(252)
    o["ema20_slope"] = o["ema20"].pct_change(5)
    o["ema50_slope"] = o["ema50"].pct_change(10)
    o["above200_share"] = (c > o["sma200"]).rolling(126, min_periods=60).mean()
    o["range_pct"] = (o["high"] - o["low"]) / c
    o["nr7"] = o["range_pct"] <= o["range_pct"].rolling(7).min()
    o["inside_day"] = (o["high"] < o["high"].shift()) & (o["low"] > o["low"].shift())
    o["drawdown"] = c / c.cummax() - 1
    if bench is not None:
        b = bench.reindex(o.index).ffill()
        o["rs_line"] = c / b
        o["rs63"] = c.pct_change(63) - b.pct_change(63)
        o["rs126"] = c.pct_change(126) - b.pct_change(126)
        o["rs_line_high"] = o["rs_line"] >= o["rs_line"].rolling(126, min_periods=60).max()
    return o
