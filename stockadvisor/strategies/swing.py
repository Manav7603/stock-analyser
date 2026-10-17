"""Swing setups (holding period: days to a few weeks).

Each setup is a vectorised rule on an indicator frame, so the same code
drives both today's scan and the historical backtest. A rule on day t only
uses data up to the close of day t; trades are entered on day t+1.

Setups
------
Breakout         New 20-day closing high on 1.5x volume, in a stock already
                 above its 200-day average and outperforming NIFTY.
Trend pullback   Strong uptrend that dipped to the 20-day EMA and held it.
                 Entry is a buy-stop above the pullback day's high.
Oversold bounce  Short-term washout (RSI-2 < 10) inside a long-term uptrend.
                 Exits on the first close back above the 10-day EMA.
Supertrend flip  Supertrend turns up with MACD positive and price above
                 the 50-day EMA - early stage of a new leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import grade

SETUPS = ["Breakout", "Trend pullback", "Oversold bounce", "Supertrend flip"]

# How each setup is entered and exited in backtests and advice.
SETUP_RULES = {
    "Breakout": {"entry": "open", "exit": "targets"},
    "Trend pullback": {"entry": "stop_above_high", "exit": "targets"},
    "Oversold bounce": {"entry": "open", "exit": "close_above_ema10"},
    "Supertrend flip": {"entry": "open", "exit": "targets"},
}


def _base_filters(f: pd.DataFrame, min_turnover_cr: float) -> pd.Series:
    return (f["turnover_cr"] >= min_turnover_cr) & f["sma200"].notna() & f["atr14"].notna()


def setup_masks(f: pd.DataFrame, min_turnover_cr: float = 20) -> dict[str, pd.Series]:
    c = f["close"]
    base = _base_filters(f, min_turnover_cr)
    uptrend = (c > f["sma200"]) & (f["ema50"] > f["sma200"])
    rs_ok = f.get("rs63", pd.Series(0.0, index=f.index)).fillna(0) > 0
    bar_strength = (c - f["low"]) / (f["high"] - f["low"]).replace(0, np.nan)

    breakout = (base & (c > f["sma200"]) & rs_ok
                & (c > f["hh20"].shift(1)) & (f["vol_ratio"] >= 1.5)
                & (bar_strength >= 0.6) & (f["rsi14"] < 80)
                & (f["dist_52wh"] > -0.25))

    pullback = (base & uptrend & (f["ema20"] > f["ema50"]) & rs_ok
                & (f["adx"] >= 18)
                & (f["low"] <= f["ema20"] * 1.01) & (c > f["ema20"])
                & (f["rsi14"].between(40, 62))
                & (f["ret5"] < 0.02))

    bounce = (base & (c > f["sma200"]) & (f["ema50"] > f["sma200"])
              & (f["rsi2"] < 10) & (f["ret5"] < -0.03)
              & (c > f["sma200"] * 1.02))

    st_flip = (base & (f["st_dir"] == 1) & (f["st_dir"].shift(1) == -1)
               & (f["hist"] > 0) & (c > f["ema50"]) & (c > f["sma200"])
               & (f["vol_ratio"] >= 1.0))

    return {"Breakout": breakout.fillna(False), "Trend pullback": pullback.fillna(False),
            "Oversold bounce": bounce.fillna(False), "Supertrend flip": st_flip.fillna(False)}


def levels(setup: str, row: pd.Series, recent: pd.DataFrame, stop_atr: float = 2.0,
           t1_r: float = 1.5, t2_r: float = 3.0) -> dict:
    """Entry, stop and targets for a setup on the signal day's bar."""
    a = row["atr14"]
    if setup == "Trend pullback":
        entry = row["high"] * 1.001
        structural = recent["low"].tail(5).min() - 0.25 * a
    elif setup == "Oversold bounce":
        entry = row["close"]
        structural = row["close"] - 2.5 * a
    elif setup == "Breakout":
        entry = row["close"]
        structural = min(row["low"], recent["low"].tail(3).min()) - 0.25 * a
    else:
        entry = row["close"]
        structural = row["st_line"] - 0.1 * a
    # keep the stop between 1 and stop_atr+1 ATRs away
    stop = min(max(structural, entry - (stop_atr + 1) * a), entry - 1.0 * a)
    r = entry - stop
    t1, t2 = entry + t1_r * r, entry + t2_r * r
    if setup == "Oversold bounce":
        t1, t2 = max(row["ema10"], entry + r), max(row["ema20"], entry + 1.5 * r)
    return {"entry": float(entry), "stop": float(stop), "target1": float(t1), "target2": float(t2)}


def score_row(setup: str, row: pd.Series, regime_score: float) -> tuple[float, list[str], list[str]]:
    """0-100 quality score with human-readable reasons and risks."""
    s, reasons, risks = 40.0, [], []
    c = row["close"]

    if c > row["sma200"]:
        pct = (c / row["sma200"] - 1) * 100
        reasons.append(f"Long-term uptrend: price is {pct:.0f}% above its 200-day average")
        s += 5
    if row["ema20"] > row["ema50"] > row["sma200"]:
        reasons.append("Moving averages stacked bullishly (20 > 50 > 200 EMA)")
        s += 7
    adx_v = row.get("adx", np.nan)
    if adx_v >= 25:
        reasons.append(f"Strong trend: ADX {adx_v:.0f} (above 25 means a real trend)")
        s += 7
    elif adx_v < 15:
        risks.append(f"Weak trend strength (ADX {adx_v:.0f}) - moves may fizzle")
        s -= 4
    rs = row.get("rs63", np.nan)
    if pd.notna(rs):
        if rs > 0.10:
            reasons.append(f"Outperforming NIFTY by {rs*100:.0f}% over 3 months (market leader)")
            s += 10
        elif rs > 0:
            reasons.append(f"Beating NIFTY by {rs*100:.1f}% over 3 months")
            s += 4
        else:
            risks.append(f"Lagging NIFTY by {abs(rs)*100:.0f}% over 3 months")
            s -= 5
    if row.get("rs_line_high", False):
        reasons.append("Relative-strength line at a 6-month high - institutions likely accumulating")
        s += 5
    vr = row.get("vol_ratio", np.nan)
    if setup in ("Breakout", "Supertrend flip") and vr >= 2:
        reasons.append(f"Volume {vr:.1f}x the 20-day average confirms buying interest")
        s += 8
    elif setup in ("Breakout", "Supertrend flip") and vr >= 1.5:
        reasons.append(f"Above-average volume ({vr:.1f}x)")
        s += 4
    if setup == "Trend pullback" and vr < 0.9:
        reasons.append("Pullback on light volume - sellers are not aggressive")
        s += 4
    d52 = row.get("dist_52wh", np.nan)
    if pd.notna(d52):
        if d52 > -0.05:
            reasons.append(f"Within {abs(d52)*100:.1f}% of its 52-week high (little overhead supply)")
            s += 6
        elif d52 < -0.30:
            risks.append(f"{abs(d52)*100:.0f}% below its 52-week high - lots of trapped sellers above")
            s -= 4
    if row.get("obv_slope", 0) > 0.1:
        reasons.append("On-balance volume rising - accumulation")
        s += 3
    r14 = row.get("rsi14", 50)
    if r14 > 75:
        risks.append(f"RSI {r14:.0f} is overbought - risk of a quick pullback; don't chase")
        s -= 6
    atrp = row.get("atr_pct", 2)
    if atrp > 4.5:
        risks.append(f"Very volatile ({atrp:.1f}% daily range) - keep size small")
        s -= 4
    if setup == "Oversold bounce":
        reasons.append(f"Short-term oversold (RSI-2 at {row['rsi2']:.0f}) inside a long-term uptrend - "
                       "historically these snap back within days")
    if setup == "Breakout":
        reasons.append(f"Closed at a new 20-day high ({row['close']:.2f} > {row['hh20']:.2f} prior)")
    if setup == "Trend pullback":
        reasons.append(f"Pulled back to and held the 20-day EMA ({row['ema20']:.2f})")
    if setup == "Supertrend flip":
        reasons.append("Supertrend just turned bullish with MACD above signal")

    # market regime matters a lot for long trades
    if regime_score >= 70:
        s += 6
    elif regime_score < 30:
        risks.append("Overall market (NIFTY) is in a downtrend - most setups fail in this phase")
        s -= 12
    elif regime_score < 50:
        risks.append("Market regime is weak - expect lower follow-through")
        s -= 6
    return float(np.clip(s, 0, 100)), reasons, risks


def latest_signals(frames: dict[str, pd.DataFrame], regime_score: float, cfg: dict,
                   lookback_days: int = 1) -> list[dict]:
    """Setups that triggered on the most recent bar(s)."""
    sw = cfg["swing"]
    out = []
    for t, f in frames.items():
        if len(f) < 210:
            continue
        masks = setup_masks(f, sw["min_avg_traded_value_cr"])
        for setup, m in masks.items():
            hits = m.tail(lookback_days)
            if not hits.any():
                continue
            day = hits[hits].index[-1]
            i = f.index.get_loc(day)
            row = f.iloc[i]
            lv = levels(setup, row, f.iloc[max(0, i - 10):i + 1], sw["stop_atr"],
                        sw["target1_r"], sw["target2_r"])
            sc, reasons, risks = score_row(setup, row, regime_score)
            out.append({"ticker": t, "setup": setup, "date": day, "row": row, "score": sc,
                        "grade": grade(sc), "reasons": reasons, "risks": risks, **lv})
    return sorted(out, key=lambda d: d["score"], reverse=True)
