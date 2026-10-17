"""Your own positions (entered manually after you place the order) and
daily hold / trail / exit advice on each one.
"""

from __future__ import annotations

import json
import uuid
from datetime import date

import pandas as pd

from .config import DATA_DIR
from .costs import trade_cost
from .fundamentals import days_to_results

FILE = DATA_DIR / "portfolio.json"


def _load() -> dict:
    if FILE.exists():
        return json.loads(FILE.read_text())
    return {"open": [], "closed": []}


def _save(p: dict) -> None:
    FILE.write_text(json.dumps(p, indent=2, default=str))


def positions() -> list[dict]:
    return _load()["open"]


def closed() -> list[dict]:
    return _load()["closed"]


def add(symbol: str, qty: int, entry: float, stop: float, target: float | None = None,
        horizon: str = "swing", setup: str = "", entry_date: str | None = None, notes: str = "") -> dict:
    p = _load()
    pos = {"id": uuid.uuid4().hex[:8], "symbol": symbol.upper().removesuffix(".NS"),
           "ticker": symbol.upper() if symbol.upper().endswith(".NS") else symbol.upper() + ".NS",
           "qty": int(qty), "entry": float(entry), "stop": float(stop), "initial_stop": float(stop),
           "target": float(target) if target else None, "horizon": horizon, "setup": setup,
           "entry_date": entry_date or str(date.today()), "notes": notes, "half_booked": False}
    p["open"].append(pos)
    _save(p)
    return pos


def update(pos_id: str, **fields) -> None:
    p = _load()
    for pos in p["open"]:
        if pos["id"] == pos_id:
            pos.update(fields)
    _save(p)


def close(pos_id: str, exit_price: float, qty: int | None = None, broker: str = "zerodha",
          exit_date: str | None = None) -> dict | None:
    p = _load()
    for pos in p["open"]:
        if pos["id"] != pos_id:
            continue
        q = int(qty or pos["qty"])
        intraday = pos["horizon"] == "intraday"
        cost = trade_cost(pos["entry"] * q, exit_price * q, intraday, broker)["total"]
        rec = {**pos, "qty": q, "exit": float(exit_price), "exit_date": exit_date or str(date.today()),
               "gross_pnl": (exit_price - pos["entry"]) * q, "costs": cost}
        rec["net_pnl"] = rec["gross_pnl"] - cost
        risk = (pos["entry"] - pos["initial_stop"]) * q
        rec["r_multiple"] = rec["net_pnl"] / risk if risk > 0 else None
        p["closed"].append(rec)
        if q >= pos["qty"]:
            p["open"] = [x for x in p["open"] if x["id"] != pos_id]
        else:
            pos["qty"] -= q
        _save(p)
        return rec
    return None


def remove(pos_id: str) -> None:
    p = _load()
    p["open"] = [x for x in p["open"] if x["id"] != pos_id]
    _save(p)


def evaluate(ctx, fund: dict | None = None) -> pd.DataFrame:
    """Mark positions to market and say what to do with each one."""
    rows = []
    for pos in positions():
        f = ctx.frames.get(pos["ticker"])
        if f is None:
            from . import data, indicators
            raw = data.get_one(pos["ticker"], years=2)
            f = indicators.add_all(raw) if len(raw) > 60 else None
        if f is None or f.empty:
            rows.append({**pos, "last": None, "action": "No price data", "advice": []})
            continue
        r = f.iloc[-1]
        last = float(r["close"])
        risk_ps = pos["entry"] - pos["initial_stop"]
        r_now = (last - pos["entry"]) / risk_ps if risk_ps > 0 else 0.0
        pnl = (last - pos["entry"]) * pos["qty"]
        target = pos.get("target") or pos["entry"] + 3 * risk_ps
        t1 = pos["entry"] + 1.5 * risk_ps
        trail = max(pos["stop"], last - 2.5 * float(r["atr14"]))
        advice, action = [], "Hold"
        if last <= pos["stop"]:
            action = "EXIT - stop hit"
            advice.append(f"Closed at Rs {last:,.2f}, at or below your stop Rs {pos['stop']:,.2f}. Exit at the open; don't hope.")
        elif last >= target:
            action = "Book profit"
            advice.append(f"Target Rs {target:,.2f} reached. Book the rest, or trail tightly at Rs {last - 1.5 * r['atr14']:,.2f}.")
        elif not pos.get("half_booked") and last >= t1 and pos["horizon"] != "longterm":
            action = "Book half + move stop to cost"
            advice.append(f"Up {r_now:.1f}R. Sell half ({pos['qty'] // 2} shares) and move the stop to your entry "
                          f"Rs {pos['entry']:,.2f} - the trade can no longer lose money.")
        elif r_now >= 1 and trail > pos["stop"]:
            action = "Trail stop"
            advice.append(f"Raise the stop from Rs {pos['stop']:,.2f} to Rs {trail:,.2f} (2.5 ATR below the close).")
        if pos["horizon"] == "longterm":
            if last < r["sma200"]:
                action = "Review - below 200 DMA"
                advice.append("Long-term holding has fallen below its 200-day average - the trend has broken. "
                              "Reconsider the thesis; most long-term exits start here.")
        else:
            held = (pd.Timestamp(ctx.asof) - pd.Timestamp(pos["entry_date"])).days
            if held > 28 and r_now < 0.5:
                advice.append(f"Held {held} days with little progress - dead money. Consider freeing the capital.")
                if action == "Hold":
                    action = "Consider exit (time)"
            if last < r["ema20"] and r_now > 0:
                advice.append("Closed below the 20-day EMA - momentum is fading; tighten the stop.")
        dtr = days_to_results((fund or {}).get(pos["ticker"], {}))
        if dtr is not None and 0 <= dtr <= 7:
            advice.append(f"Results in {dtr} day(s). Decide now: book part, tighten the stop, or accept gap risk.")
        if not advice:
            advice.append("On track. Keep the stop where it is and let the trade work.")
        rows.append({**pos, "last": last, "pnl": pnl, "pnl_pct": (last / pos["entry"] - 1) * 100,
                     "r_now": r_now, "value": last * pos["qty"], "suggested_stop": trail,
                     "action": action, "advice": advice, "rsi14": float(r["rsi14"])})
    return pd.DataFrame(rows)
