"""Intraday tools: a pre-market watchlist with key levels, and an
Opening Range Breakout (ORB) scanner + backtest on 15-minute bars.

Intraday trading has the worst odds of any style for retail traders -
costs and slippage are paid on every trade. Use small size, always use a
stop, and square off by 3:15 PM.
"""

from __future__ import annotations

import pandas as pd

from ..costs import trade_cost
from ..indicators import atr, pivots, vwap_intraday

SQUARE_OFF = pd.Timestamp("15:15").time()


def premarket_watchlist(ctx, top: int = 20, min_turnover_cr: float = 50) -> pd.DataFrame:
    rows = []
    for t, f in ctx.frames.items():
        if len(f) < 60:
            continue
        r = f.iloc[-1]
        if r["turnover_cr"] < min_turnover_cr or r["atr_pct"] < 1.2:
            continue
        lv = pivots(r["high"], r["low"], r["close"])
        cpr_w = abs(lv["CPR_top"] - lv["CPR_bottom"]) / lv["P"] * 100
        long_bias = r["close"] > r["ema20"] > r["ema50"]
        short_bias = r["close"] < r["ema20"] < r["ema50"]
        score, why = 0.0, []
        if r["nr7"]:
            score += 25
            why.append("NR7: narrowest range in 7 days - range expansion likely")
        if r["inside_day"]:
            score += 15
            why.append("Inside day - coiled; trade the break of yesterday's high/low")
        if cpr_w < 0.25:
            score += 25
            why.append(f"Very narrow CPR ({cpr_w:.2f}%) - trending day more likely")
        elif cpr_w < 0.5:
            score += 10
        if long_bias or short_bias:
            score += 15
            why.append("Clean daily trend: " + ("up - favour longs" if long_bias else "down - favour shorts"))
        if r["vol_ratio"] > 1.5:
            score += 10
            why.append(f"Yesterday's volume {r['vol_ratio']:.1f}x average - stock is in play")
        score += min(r["atr_pct"], 4) * 5
        rows.append({"ticker": t, "symbol": t.removesuffix(".NS"), "sector": ctx.sector_of(t),
                     "bias": "Long" if long_bias else "Short" if short_bias else "Neutral",
                     "score": score, "atr_pct": r["atr_pct"], "cpr_width_pct": cpr_w,
                     "turnover_cr": r["turnover_cr"], "why": why, **lv})
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).head(top).reset_index(drop=True) if len(df) else df


def _orb_day(day: pd.DataFrame, or_bars: int = 1, rr: float = 2.0, bias: str = "Both",
             max_or_pct: float = 2.5) -> dict | None:
    """Simulate one ORB trade on one day's bars. Returns trade dict or None."""
    if len(day) < or_bars + 3:
        return None
    o = day.iloc[:or_bars]
    hi, lo = o["high"].max(), o["low"].min()
    if (hi - lo) / lo * 100 > max_or_pct:      # opening range too wide - poor R:R
        return None
    vw = vwap_intraday(day)
    avg_vol = day["volume"].iloc[:or_bars].mean()
    for i in range(or_bars, len(day)):
        bar = day.iloc[i]
        if bar.name.time() >= pd.Timestamp("14:30").time():
            return None
        long_ok = bias in ("Both", "Long") and bar["close"] > hi and bar["close"] > vw.iloc[i]
        short_ok = bias in ("Both", "Short") and bar["close"] < lo and bar["close"] < vw.iloc[i]
        if not (long_ok or short_ok) or bar["volume"] < 0.6 * avg_vol:
            continue
        side = 1 if long_ok else -1
        entry = bar["close"]
        mid = (hi + lo) / 2
        stop = (lo if (entry - lo) / entry < 0.012 else mid) if side == 1 else \
               (hi if (hi - entry) / entry < 0.012 else mid)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        target = entry + side * rr * risk
        exit_px, reason, exit_time = None, "", None
        for j in range(i + 1, len(day)):
            b = day.iloc[j]
            if (side == 1 and b["low"] <= stop) or (side == -1 and b["high"] >= stop):
                exit_px, reason = stop, "Stop"
            elif (side == 1 and b["high"] >= target) or (side == -1 and b["low"] <= target):
                exit_px, reason = target, "Target"
            elif b.name.time() >= SQUARE_OFF:
                exit_px, reason = b["open"], "Square-off"
            if exit_px is not None:
                exit_time = b.name
                break
        if exit_px is None:
            exit_px, reason, exit_time = day["close"].iloc[-1], "Square-off", day.index[-1]
        return {"side": "LONG" if side == 1 else "SHORT", "entry_time": bar.name, "entry": entry,
                "stop": stop, "target": target, "exit_time": exit_time, "exit": exit_px,
                "exit_reason": reason, "or_high": hi, "or_low": lo}
    return None


def backtest_orb(bars: dict[str, pd.DataFrame], capital: float = 500000, risk_pct: float = 0.5,
                 rr: float = 2.0, broker: str = "zerodha", daily_bias: dict | None = None,
                 slippage_pct: float = 0.05) -> pd.DataFrame:
    trades = []
    for t, df in bars.items():
        for d, day in df.groupby(df.index.date):
            day = day.between_time("09:15", "15:30")
            bias = (daily_bias or {}).get(t, "Both")
            tr = _orb_day(day, rr=rr, bias=bias)
            if not tr:
                continue
            side = 1 if tr["side"] == "LONG" else -1
            entry = tr["entry"] * (1 + side * slippage_pct / 100)
            exit_ = tr["exit"] * (1 - side * slippage_pct / 100)
            qty = int(capital * risk_pct / 100 / abs(entry - tr["stop"])) if entry != tr["stop"] else 0
            qty = min(qty, int(capital * 2 / entry))          # MIS leverage cap ~5x; stay at 2x
            if qty <= 0:
                continue
            buy_v, sell_v = (entry * qty, exit_ * qty) if side == 1 else (exit_ * qty, entry * qty)
            cost = trade_cost(buy_v, sell_v, True, broker)["total"]
            gross = (exit_ - entry) * side * qty
            trades.append({"ticker": t, "date": pd.Timestamp(d), **tr, "qty": qty,
                           "gross_pnl": gross, "costs": cost, "net_pnl": gross - cost,
                           "r_multiple": (gross - cost) / (abs(entry - tr["stop"]) * qty)})
    return pd.DataFrame(trades)


def live_orb_scan(bars: dict[str, pd.DataFrame], watch: pd.DataFrame | None = None, rr: float = 2.0) -> pd.DataFrame:
    """ORB triggers on today's bars so far (run during market hours)."""
    rows = []
    bias_map = dict(zip(watch["ticker"], watch["bias"])) if watch is not None and len(watch) else {}
    for t, df in bars.items():
        if df.empty:
            continue
        today = df.index[-1].date()
        day = df[df.index.date == today].between_time("09:15", "15:30")
        b = bias_map.get(t, "Neutral")
        tr = _orb_day(day, rr=rr, bias="Both" if b == "Neutral" else b)
        if tr:
            last = day["close"].iloc[-1]
            running = tr["exit_reason"] == "Square-off" and day.index[-1].time() < SQUARE_OFF
            status = "Running" if running else tr["exit_reason"]
            rows.append({"ticker": t, "symbol": t.removesuffix(".NS"), **tr, "last": last, "status": status,
                         "with_daily_trend": b == ("Long" if tr["side"] == "LONG" else "Short")})
    return pd.DataFrame(rows)


def intraday_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return atr(df, n)


def orb_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"Trades": 0}
    daily = trades.groupby("date")["net_pnl"].sum()
    return {
        "Trades": len(trades), "Win rate %": (trades["net_pnl"] > 0).mean() * 100,
        "Expectancy (R)": trades["r_multiple"].mean(),
        "Net P&L Rs": trades["net_pnl"].sum(), "Costs Rs": trades["costs"].sum(),
        "Costs as % of gross profit": trades["costs"].sum() / max(trades.loc[trades.gross_pnl > 0, "gross_pnl"].sum(), 1) * 100,
        "Profitable days %": (daily > 0).mean() * 100,
        "Worst day Rs": daily.min(), "Best day Rs": daily.max(),
        "Targets hit %": (trades["exit_reason"] == "Target").mean() * 100,
    }

