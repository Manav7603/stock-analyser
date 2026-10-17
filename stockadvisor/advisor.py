"""The advisor: combines every layer into one daily report.

For each swing idea the final score blends four independent views:

  45%  technical quality of today's setup          (rules, strategies/swing.py)
  35%  ML rank - will this stock beat the others?   (walk-forward tested, ml.py)
  20%  track record of this setup after costs       (backtest.py: setup_edge)
  +/-  news tone, results-date risk and market regime adjustments

A setup that has not made money historically is never shown as a buy,
however good today's chart looks.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import ai, ml, news
from .backtest import SwingParams, run_swing, setup_edge
from .config import CACHE_DIR, DATA_DIR, load_config
from .costs import breakeven_move_pct
from .fundamentals import days_to_results, get_fundamentals
from .market import MarketContext, market_snapshot, sector_strength
from .strategies import longterm, swing
from .strategies.base import Signal, grade, position_size
from .strategies.intraday import premarket_watchlist

REPORT_DIR = DATA_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Out-of-sample tests (2022-2026, NIFTY 200) showed that skipping setups the ML
# model gives < 30% odds lifted every swing configuration: all setups went from
# 2.2% to 6.1% CAGR and drawdowns fell. The same filter is applied to live ideas.
ML_MIN_PROB = 0.30


def setup_track_record(ctx: MarketContext, years: int = 4, refresh: bool = False) -> pd.DataFrame:
    """Backtest every swing setup over recent years; cached per trading day."""
    model_ts = (ml.load()[1] or {}).get("trained_at", "none").replace(":", "")
    key = f"edge_{len(ctx.frames)}_{ctx.asof.date()}_{model_ts}.parquet"
    p = CACHE_DIR / key
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    cfg = load_config()
    start = (ctx.asof - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    probs = ml.oos_prob_panel()
    if probs is not None:
        start = max(start, str(probs.index.min().date()))
    res = run_swing(ctx, SwingParams(start=start, capital=cfg["account"]["capital"],
                                     broker=cfg["account"]["broker"], use_regime=True,
                                     ml_probs=probs, min_ml_prob=ML_MIN_PROB if probs is not None else 0.0))
    edge = setup_edge(res["trades"])
    if not edge.empty:
        edge.to_parquet(p)
    return edge


def _edge_score(row: dict | None) -> float:
    if not row:
        return 50.0
    e = row.get("expectancy_r", 0) or 0
    rec = row.get("recent_expectancy_r")
    if rec is not None and not (isinstance(rec, float) and np.isnan(rec)):
        e = 0.6 * e + 0.4 * rec
    return float(np.clip(50 + e * 120, 0, 100))


def swing_ideas(ctx: MarketContext, snap: dict, edge: pd.DataFrame, mlp: pd.DataFrame,
                fund: dict, cfg: dict, use_ai: bool = True, ai_top: int = 5,
                news_top: int = 12, lookback_days: int = 1) -> list[Signal]:
    acct = cfg["account"]
    raw = swing.latest_signals(ctx.frames, snap["score"], cfg, lookback_days=lookback_days)
    edge_map = {r["setup"]: r for r in edge.to_dict("records")} if not edge.empty else {}
    sigs: list[Signal] = []
    for c in raw:
        t = c["ticker"]
        er = edge_map.get(c["setup"])
        ml_rank = float(mlp.at[t, "ml_rank"]) if t in mlp.index else None
        ml_p = float(mlp.at[t, "prob"]) if t in mlp.index else None
        tech = c["score"]
        combined = 0.45 * tech + 0.35 * (ml_rank if ml_rank is not None else tech) + 0.20 * _edge_score(er)
        reasons, risks = list(c["reasons"]), list(c["risks"])
        if ml_rank is not None:
            if ml_rank >= 80:
                reasons.insert(0, f"ML model ranks it in the top {100 - ml_rank:.0f}% of stocks for the next 10 days")
            elif ml_rank < 40:
                risks.append(f"ML model ranks it in the bottom {ml_rank:.0f}% - odds of outperforming are low")
        if ml_p is not None and ml_p < ML_MIN_PROB:
            combined -= 12
            risks.insert(0, f"Fails the ML filter ({ml_p:.0%} odds < {ML_MIN_PROB:.0%}) - in backtests, "
                            "skipping these trades improved results")
        if er:
            if er["verdict"] == "Proven edge":
                reasons.append(f"This setup has a proven edge: {er['win_rate_pct']:.0f}% wins, "
                               f"{er['expectancy_r']:+.2f}R per trade after costs over {er['trades']} trades")
            elif er["verdict"] == "No edge - avoid":
                risks.insert(0, f"This setup has NOT made money recently after costs "
                                f"({er['expectancy_r']:+.2f}R per trade over {er['trades']} trades)")
        f = fund.get(t, {})
        dtr = days_to_results(f)
        if dtr is not None and dtr <= 7:
            combined -= 10
            risks.insert(0, f"Quarterly results in {dtr} day(s) - large gap risk")
        sym = t.removesuffix(".NS")
        qty = position_size(acct["capital"], acct["risk_per_trade_pct"], c["entry"], c["stop"],
                            acct["max_position_pct"], snap["exposure"])
        s = Signal(ticker=t, symbol=sym, name=ctx.name_of(t), sector=ctx.sector_of(t),
                   horizon="swing", setup=c["setup"], side="BUY", date=str(c["date"].date()),
                   close=float(c["row"]["close"]), entry=round(c["entry"], 2),
                   entry_type="buy above" if c["setup"] == "Trend pullback" else "at open",
                   stop=round(c["stop"], 2), target1=round(c["target1"], 2), target2=round(c["target2"], 2),
                   score=float(combined), reasons=reasons, risks=risks,
                   ml_prob=ml_rank, qty=qty)
        s.capital_needed = round(qty * s.entry, 0)
        s.risk_amount = round(qty * s.risk_per_share, 0)
        s.cost_breakeven_pct = round(breakeven_move_pct(s.entry, max(qty, 1), False, acct["broker"]), 3)
        s.extra = {"technical_score": tech, "edge": er, "days_to_results": dtr, "ml_prob": ml_p,
                   "passes_ml_filter": ml_p is None or ml_p >= ML_MIN_PROB,
                   "atr": float(c["row"]["atr14"]), "rsi14": float(c["row"]["rsi14"]),
                   "adx": float(c["row"]["adx"]), "rs63": float(c["row"].get("rs63", np.nan)),
                   "fundamentals": {k: f.get(k) for k in ("trailingPE", "returnOnEquity", "debtToEquity",
                                                          "earningsGrowth", "recommendationKey", "targetMeanPrice")}}
        sigs.append(s)
    sigs.sort(key=lambda x: x.score, reverse=True)

    # one idea per stock (best setup), then news + AI on the top ones
    seen, uniq = set(), []
    for s in sigs:
        if s.ticker not in seen:
            uniq.append(s)
            seen.add(s.ticker)
    days = cfg["news"]["lookback_days"]
    for i, s in enumerate(uniq[:news_top]):
        items = news.stock_news(s.symbol, s.name, days, cfg["news"]["max_items"])
        summ = news.summarise(items)
        s.news, s.news_sentiment = items, summ["sentiment"]
        s.score = float(np.clip(s.score + 10 * summ["sentiment"], 0, 100))
        if summ["events"].get("Regulatory"):
            s.risks.insert(0, "Regulatory headlines in recent news - read them before entering")
        view_ctx = {
            "symbol": s.symbol, "name": s.name, "sector": s.sector, "setup": s.setup,
            "close": s.close, "entry": s.entry, "entry_type": s.entry_type, "stop": s.stop,
            "target1": s.target1, "target2": s.target2, "technical_score": round(s.extra["technical_score"], 1),
            "ml_rank": None if s.ml_prob is None else round(s.ml_prob, 1),
            "setup_track_record": s.extra["edge"], "reasons": s.reasons, "risks": s.risks,
            "days_to_results": s.extra["days_to_results"], "fundamentals": s.extra["fundamentals"],
            "indicators": {"rsi14": round(s.extra["rsi14"], 1), "adx": round(s.extra["adx"], 1),
                           "atr": round(s.extra["atr"], 2), "rs_vs_nifty_3m_pct": round(s.extra["rs63"] * 100, 1)},
            "market": {"label": snap["label"], "score": snap["score"], "note": snap["note"]},
            "news_summary": summ, "news_days": days,
            "headlines": [{"title": n["title"], "source": n["source"], "date": n["published"][:10]} for n in items],
        }
        s.ai = ai.stock_view(view_ctx, use_ai=use_ai and i < ai_top)
    for s in uniq:
        s.grade = grade(s.score)
    uniq.sort(key=lambda x: x.score, reverse=True)
    return uniq


def longterm_picks(ctx: MarketContext, fund: dict, cfg: dict) -> dict:
    lt = cfg["longterm"]
    ranked = longterm.rank_today(ctx, fund)
    port = longterm.pick_portfolio(ranked, lt["portfolio_size"], lt["max_per_sector"])
    capital = cfg["account"]["capital"]
    port["amount"] = (port["weight_pct"] / 100 * capital).round(0)
    port["qty"] = (port["amount"] / port["close"]).astype(int)
    port["why"] = [longterm.explain(r) for _, r in port.iterrows()]
    port["results_in_days"] = [days_to_results(fund.get(t, {})) for t in port.index]
    return {"ranked": ranked, "portfolio": port}


def build_report(ctx: MarketContext, use_ai: bool = True, ai_top: int = 5,
                 with_fundamentals: bool = True, lookback_days: int = 1) -> dict:
    cfg = load_config()
    snap = market_snapshot(ctx)
    sectors = sector_strength(ctx)
    edge = setup_track_record(ctx)
    mlp = ml.predict_latest(ctx)
    fund = get_fundamentals(list(ctx.frames)) if with_fundamentals else {}
    ideas = swing_ideas(ctx, snap, edge, mlp, fund, cfg, use_ai, ai_top, lookback_days=lookback_days)
    lt = longterm_picks(ctx, fund, cfg)
    watch = premarket_watchlist(ctx)
    mnews = news.market_news()
    mview = ai.market_view({
        "date": str(ctx.asof.date()), "label": snap["label"], "score": snap["score"],
        "note": snap["note"], "reasons": snap["reasons"], "vix": snap["vix"],
        "nifty_close": snap["nifty_close"], "nifty_1w_pct": snap["nifty_chg_pct"],
        "breadth_above50": float(snap["breadth"]["above50"].iloc[-1]),
        "sectors": sectors[["sector", "1M %", "3M %", "RS 3M", "% above 50D"]].round(1).to_dict("records"),
        "market_news": [{"title": n["title"], "source": n["source"]} for n in mnews[:12]],
        "setup_edge": edge[["setup", "expectancy_r", "verdict"]].round(2).to_dict("records") if not edge.empty else [],
    }, use_ai=use_ai)
    ml_top = pd.DataFrame()
    if not mlp.empty:
        ml_top = mlp.sort_values("prob", ascending=False).head(15)[["prob", "ml_rank"]].copy()
        ml_top["name"] = [ctx.name_of(t) for t in ml_top.index]
        ml_top["sector"] = [ctx.sector_of(t) for t in ml_top.index]
        ml_top["close"] = [ctx.frames[t]["close"].iloc[-1] for t in ml_top.index]
        ml_top["above_200dma"] = [bool(ctx.frames[t]["close"].iloc[-1] > ctx.frames[t]["sma200"].iloc[-1])
                                  for t in ml_top.index]
    report = {"asof": str(ctx.asof.date()), "generated": datetime.now().isoformat(timespec="seconds"),
              "market": snap, "market_view": mview, "market_news": mnews, "sectors": sectors,
              "edge": edge, "swing": ideas, "longterm": lt, "intraday_watch": watch,
              "ml_top": ml_top, "ml_meta": ml.load()[1], "adjustments": ctx.adjustments}
    save_report(report)
    return report


def save_report(r: dict) -> None:
    out = {
        "asof": r["asof"], "generated": r["generated"],
        "market": {k: v for k, v in r["market"].items() if k not in ("breadth", "regime")},
        "market_view": r["market_view"],
        "swing": [s.to_dict() for s in r["swing"]],
        "longterm": r["longterm"]["portfolio"].reset_index().rename(columns={"index": "ticker"})
        .to_dict("records"),
        "edge": r["edge"].to_dict("records") if not r["edge"].empty else [],
    }
    (REPORT_DIR / f"report_{r['asof']}.json").write_text(json.dumps(out, indent=2, default=str))

