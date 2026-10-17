"""Portfolio-level swing backtester with realistic Indian execution.

Rules that keep it honest:
* Signals use data up to day t's close; orders fill on day t+1 or later.
* Gaps are respected - a stop below the open fills at the open, not the stop.
* If a stop and a target are both inside one day's range, the stop is
  assumed to have hit first (conservative).
* Every exit pays the full broker + statutory cost stack, plus slippage.
* Position size = fixed % risk of current equity, scaled down by the market
  regime, capped by max position size and available cash.

Known bias: the universe is today's index members, so stocks that were
dropped from the index (often the losers) are missing. Real results are
usually somewhat worse than a backtest - treat numbers as an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, market
from .costs import trade_cost
from .strategies import swing


@dataclass
class SwingParams:
    setups: list[str] = field(default_factory=lambda: list(swing.SETUPS))
    capital: float = 500000
    risk_pct: float = 1.0
    max_positions: int = 8
    max_position_pct: float = 20.0
    stop_atr: float = 2.0
    t1_r: float = 1.5
    t2_r: float = 3.0
    max_hold: int = 20
    trail_atr: float = 2.5
    min_score: float = 0.0
    use_regime: bool = True
    skip_bearish: bool = True
    min_turnover_cr: float = 20
    slippage_pct: float = 0.10
    broker: str = "zerodha"
    start: str | None = None
    end: str | None = None
    ml_probs: pd.DataFrame | None = None      # optional out-of-sample P(win), dates x tickers
    min_ml_prob: float = 0.0


def _candidate_table(ctx, p: SwingParams, regime: pd.Series) -> pd.DataFrame:
    """All setup triggers across the universe with levels and scores."""
    rows = []
    for t, f in ctx.frames.items():
        if len(f) < 210:
            continue
        masks = swing.setup_masks(f, p.min_turnover_cr)
        for setup in p.setups:
            m = masks[setup]
            if p.start:
                m = m[m.index >= pd.Timestamp(p.start) - pd.Timedelta(days=5)]
            for d in m[m].index:
                i = f.index.get_loc(d)
                row = f.iloc[i]
                lv = swing.levels(setup, row, f.iloc[max(0, i - 10):i + 1], p.stop_atr, p.t1_r, p.t2_r)
                rs = float(regime.get(d, 50))
                sc, _, _ = swing.score_row(setup, row, rs)
                rows.append({"date": d, "ticker": t, "setup": setup, "score": sc,
                             "atr": row["atr14"], **lv})
    return pd.DataFrame(rows)


def run_swing(ctx, p: SwingParams) -> dict:
    br = market.breadth(ctx)
    regime = market.regime_series(ctx.nifty, ctx.vix, br)
    cands = _candidate_table(ctx, p, regime)
    if p.ml_probs is not None and not cands.empty:
        cands["ml_prob"] = [p.ml_probs.at[d, t] if (d in p.ml_probs.index and t in p.ml_probs.columns)
                            else np.nan for d, t in zip(cands["date"], cands["ticker"])]
        cands = cands[cands["ml_prob"].fillna(0) >= p.min_ml_prob]
    cands = cands[cands["score"] >= p.min_score] if not cands.empty else cands
    by_date = {d: g.sort_values("score", ascending=False) for d, g in cands.groupby("date")} if not cands.empty else {}

    dates = ctx.nifty.index
    if p.start:
        dates = dates[dates >= pd.Timestamp(p.start)]
    if p.end:
        dates = dates[dates <= pd.Timestamp(p.end)]
    frames = ctx.frames
    last_close = data.panel(frames, "close").ffill()

    cash = p.capital
    positions: dict[str, dict] = {}
    pending: list[dict] = []
    trades, equity, exposure = [], [], []

    def close_pos(t, pos, d, px, qty, reason):
        nonlocal cash
        px_eff = px * (1 - p.slippage_pct / 100)
        buy_v, sell_v = pos["entry"] * qty, px_eff * qty
        cost = trade_cost(buy_v, sell_v, False, p.broker)
        gross = sell_v - buy_v
        cash += sell_v - cost["total"]
        trades.append({"ticker": t, "setup": pos["setup"], "entry_date": pos["date"], "exit_date": d,
                       "entry": pos["entry"], "exit": px_eff, "qty": qty, "stop0": pos["stop0"],
                       "gross_pnl": gross, "costs": cost["total"], "net_pnl": gross - cost["total"],
                       "r_multiple": (gross - cost["total"]) / (pos["risk_ps"] * pos["qty0"]),
                       "hold_days": pos["bars"], "exit_reason": reason, "score": pos["score"]})

    for d in dates:
        # 1) manage open positions on today's bar
        for t in list(positions):
            f = frames[t]
            if d not in f.index:
                continue
            bar = f.loc[d]
            pos = positions[t]
            pos["bars"] += 1
            exit_rule = swing.SETUP_RULES[pos["setup"]]["exit"]
            if bar["open"] <= pos["stop"]:
                close_pos(t, pos, d, bar["open"], pos["qty"], "Stop (gap)")
                del positions[t]
                continue
            if bar["low"] <= pos["stop"]:
                close_pos(t, pos, d, pos["stop"], pos["qty"],
                          "Trailing stop" if pos["half_done"] else "Stop")
                del positions[t]
                continue
            if exit_rule == "targets":
                if not pos["half_done"] and bar["high"] >= pos["t1"]:
                    half = pos["qty"] // 2
                    if half > 0:
                        close_pos(t, pos, d, max(pos["t1"], bar["open"]), half, "Target 1 (half)")
                        pos["qty"] -= half
                    pos["half_done"] = True
                    pos["stop"] = max(pos["stop"], pos["entry"])          # risk-free from here
                if bar["high"] >= pos["t2"]:
                    close_pos(t, pos, d, max(pos["t2"], bar["open"]), pos["qty"], "Target 2")
                    del positions[t]
                    continue
                if pos["half_done"]:
                    pos["stop"] = max(pos["stop"], bar["close"] - p.trail_atr * bar["atr14"])
            elif exit_rule == "close_above_ema10" and bar["close"] > bar["ema10"]:
                close_pos(t, pos, d, bar["close"], pos["qty"], "Mean-reversion exit")
                del positions[t]
                continue
            if pos["bars"] >= p.max_hold:
                close_pos(t, pos, d, bar["close"], pos["qty"], "Time stop")
                del positions[t]

        # 2) fill yesterday's pending orders
        mv = sum(pos["qty"] * last_close.at[d, t] for t, pos in positions.items())
        eq_now = cash + mv
        rs = float(regime.get(d, 50))
        _, mult, _ = market.regime_label(rs)
        if not p.use_regime:
            mult = 1.0
        for o in pending:
            if len(positions) >= p.max_positions:
                break
            t = o["ticker"]
            if t in positions or d not in frames[t].index:
                continue
            bar = frames[t].loc[d]
            rule = swing.SETUP_RULES[o["setup"]]["entry"]
            if rule == "stop_above_high":
                if bar["high"] < o["entry"]:
                    continue
                px = max(bar["open"], o["entry"])
            else:
                px = bar["open"]
                if px > o["entry"] + 1.0 * o["atr"]:       # gapped too far - don't chase
                    continue
            if px <= o["stop"]:
                continue
            px *= 1 + p.slippage_pct / 100
            risk_ps = px - o["stop"]
            qty = int(min(eq_now * p.risk_pct / 100 * mult / risk_ps,
                          eq_now * p.max_position_pct / 100 / px, cash / px))
            if qty <= 0:
                continue
            cash -= px * qty
            shift = px - o["entry"]
            positions[t] = {"setup": o["setup"], "date": d, "entry": px, "qty": qty, "qty0": qty,
                            "stop": o["stop"], "stop0": o["stop"], "risk_ps": risk_ps,
                            "t1": o["target1"] + shift, "t2": o["target2"] + shift,
                            "half_done": False, "bars": 0, "score": o["score"]}
        pending = []

        # 3) mark to market
        mv = sum(pos["qty"] * last_close.at[d, t] for t, pos in positions.items())
        equity.append((d, cash + mv))
        exposure.append((d, mv / (cash + mv) if cash + mv else 0))

        # 4) queue tonight's signals for tomorrow
        if d in by_date and not (p.skip_bearish and p.use_regime and rs < 30):
            pending = [r for _, r in by_date[d].iterrows() if r["ticker"] not in positions]

    # close anything still open at the last price
    if dates.size:
        last = dates[-1]
        for t, pos in list(positions.items()):
            px = frames[t]["close"].loc[:last].iloc[-1]
            close_pos(t, pos, last, px, pos["qty"], "End of test")
    eq = pd.Series(dict(equity)).sort_index()
    bench = ctx.nifty["close"].reindex(eq.index).ffill()
    return {"equity": eq, "benchmark": bench / bench.iloc[0] * p.capital if len(bench) else bench,
            "trades": pd.DataFrame(trades), "exposure": pd.Series(dict(exposure)).sort_index(),
            "candidates": cands, "regime": regime}


def setup_edge(trades: pd.DataFrame, recent_days: int = 365) -> pd.DataFrame:
    """Per-setup track record: is this setup actually making money after costs?"""
    if trades.empty:
        return pd.DataFrame()
    # a partial exit and its remainder belong to one trade - group them
    t = trades.copy()
    t["trade_id"] = t["ticker"] + "|" + t["entry_date"].astype(str)
    g = t.groupby("trade_id").agg(setup=("setup", "first"), entry_date=("entry_date", "first"),
                                  net_pnl=("net_pnl", "sum"), r=("r_multiple", "sum"),
                                  hold=("hold_days", "max"))
    cutoff = g["entry_date"].max() - pd.Timedelta(days=recent_days)
    rows = []
    for s, x in g.groupby("setup"):
        rec = x[x["entry_date"] >= cutoff]
        wins, losses = x[x.net_pnl > 0].net_pnl.sum(), -x[x.net_pnl <= 0].net_pnl.sum()
        rows.append({"setup": s, "trades": len(x), "win_rate_pct": (x.net_pnl > 0).mean() * 100,
                     "expectancy_r": x.r.mean(), "profit_factor": wins / losses if losses else np.inf,
                     "avg_hold_days": x.hold.mean(), "net_pnl": x.net_pnl.sum(),
                     "recent_trades": len(rec),
                     "recent_expectancy_r": rec.r.mean() if len(rec) else np.nan,
                     "recent_win_rate_pct": (rec.net_pnl > 0).mean() * 100 if len(rec) else np.nan})
    df = pd.DataFrame(rows)
    df["verdict"] = np.select(
        [(df.expectancy_r > 0.15) & (df.profit_factor > 1.3) & (df.trades >= 30),
         (df.expectancy_r > 0) & (df.profit_factor > 1.0)],
        ["Proven edge", "Marginal"], "No edge - avoid")
    return df.sort_values("expectancy_r", ascending=False).reset_index(drop=True)
