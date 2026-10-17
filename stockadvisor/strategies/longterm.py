"""Long-term (months to years) stock selection.

Ranks stocks on a blend of factors that have been persistent in Indian
equities: momentum, trend quality, low volatility and, when available,
business quality, growth and valuation. Only stocks in a long-term
uptrend are eligible, and a sector cap keeps the portfolio diversified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data
from ..costs import trade_cost


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std()
    if not sd or np.isnan(sd):
        return s * 0
    return ((s - s.mean()) / sd).clip(-3, 3)


PRICE_FACTORS = {          # name: (panel column, sign, weight)
    "Momentum 12-1M": ("mom_12_1", 1, 0.30),
    "Momentum 6M": ("ret126", 1, 0.15),
    "Low volatility": ("volatility63", -1, 0.20),
    "Trend quality": ("above200_share", 1, 0.20),
    "Near 52W high": ("dist_52wh", 1, 0.15),
}


def price_factor_scores(cross: pd.DataFrame) -> pd.DataFrame:
    """cross: rows=tickers, columns=indicator values on one date."""
    out = pd.DataFrame(index=cross.index)
    for name, (col, sign, _) in PRICE_FACTORS.items():
        out[name] = _z(sign * cross[col])
    out["price_score"] = sum(out[n] * w for n, (_, _, w) in PRICE_FACTORS.items())
    return out


def fundamental_scores(fund: dict[str, dict], sectors: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(fund).T
    num = lambda c: pd.to_numeric(f.get(c), errors="coerce")  # noqa: E731
    df = pd.DataFrame(index=f.index)
    roe, opm, de = num("returnOnEquity"), num("operatingMargins"), num("debtToEquity")
    eg, rg = num("earningsGrowth"), num("revenueGrowth")
    pe = num("trailingPE").where(lambda x: x > 0)
    df["Quality"] = (_z(roe.clip(-0.5, 0.8)).fillna(0) + _z(opm.clip(-0.5, 0.8)).fillna(0)
                     - _z(np.log1p(de.clip(0, 500))).fillna(0)) / 3
    df["Growth"] = (_z(eg.clip(-1, 2)).fillna(0) + _z(rg.clip(-1, 2)).fillna(0)) / 2
    # value relative to the stock's own sector (banks vs IT have very different PEs)
    sec = sectors.reindex(df.index)
    ey = (1 / pe)
    rel = ey / ey.groupby(sec).transform("median")
    df["Value (vs sector)"] = _z(np.log(rel.clip(0.2, 5))).fillna(0)
    df["fund_score"] = 0.45 * df["Quality"] + 0.35 * df["Growth"] + 0.20 * df["Value (vs sector)"]
    df["has_fund"] = roe.notna() | eg.notna()
    return df


def rank_today(ctx, fund: dict[str, dict] | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Ranked long-term candidates as of the latest bar."""
    rows = {t: f.iloc[-1] for t, f in ctx.frames.items() if len(f) > 260}
    cross = pd.DataFrame(rows).T
    cols = [c for _, (c, _, _) in PRICE_FACTORS.items()] + ["close", "sma200", "ema50", "atr14", "ret21", "ret252", "turnover_cr"]
    cross = cross[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["mom_12_1", "volatility63"])
    ps = price_factor_scores(cross)
    sectors = ctx.universe.set_index("ticker")["sector"]
    out = cross.join(ps)
    if fund:
        fs = fundamental_scores({t: fund.get(t, {}) for t in out.index}, sectors)
        out = out.join(fs)
        out["score"] = np.where(out["has_fund"], 0.6 * out["price_score"] + 0.4 * out["fund_score"],
                                out["price_score"])
    else:
        out["score"] = out["price_score"]
    out["sector"] = sectors.reindex(out.index)
    out["name"] = ctx.universe.set_index("ticker")["name"].reindex(out.index)
    out["eligible"] = (out["close"] > out["sma200"]) & (out["turnover_cr"] >= 5)
    out["rank"] = out["score"].rank(ascending=False).astype(int)
    out["percentile"] = out["score"].rank(pct=True) * 100
    return out.sort_values("score", ascending=False)


def pick_portfolio(ranked: pd.DataFrame, n: int = 15, max_per_sector: int = 3) -> pd.DataFrame:
    picks, per_sector = [], {}
    for t, r in ranked[ranked["eligible"]].iterrows():
        if per_sector.get(r["sector"], 0) >= max_per_sector:
            continue
        picks.append(t)
        per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        if len(picks) == n:
            break
    p = ranked.loc[picks].copy()
    inv_vol = 1 / p["volatility63"].clip(lower=0.10)
    p["weight_pct"] = (inv_vol / inv_vol.sum() * 100).round(1)   # calmer stocks get more weight
    return p


def explain(r: pd.Series) -> list[str]:
    out = []
    if r.get("Momentum 12-1M", 0) > 0.8:
        out.append(f"Strong 12-month momentum ({r['mom_12_1']*100:.0f}% excluding last month) - winners tend to keep winning")
    if r.get("Low volatility", 0) > 0.5:
        out.append(f"Calmer than peers ({r['volatility63']*100:.0f}% annualised volatility) - smoother compounding")
    if r.get("Trend quality", 0) > 0.5:
        out.append(f"Above its 200-day average {r['above200_share']*100:.0f}% of the last 6 months - persistent trend")
    if r.get("Near 52W high", 0) > 0.5:
        out.append("Trading close to its 52-week high")
    if r.get("Quality", 0) > 0.5:
        out.append("High business quality (strong ROE / margins, manageable debt)")
    if r.get("Growth", 0) > 0.5:
        out.append("Earnings and revenue growing faster than peers")
    if r.get("Value (vs sector)", 0) > 0.5:
        out.append("Cheaper than its sector on P/E")
    if r.get("Value (vs sector)", 0) < -1:
        out.append("Expensive vs its sector - momentum is carrying the valuation")
    if not out:
        out.append("Balanced scores across factors; no single standout")
    return out


def backtest_monthly(ctx, n: int = 15, max_per_sector: int = 3, start: str | None = None,
                     capital: float = 1_000_000, broker: str = "zerodha",
                     trend_filter: bool = True) -> dict:
    """Monthly-rebalanced factor portfolio vs NIFTY (price factors only, no lookahead).

    Ranks at each month-end close, trades at the next day's open.
    """
    close_raw = data.panel(ctx.frames, "close")
    close = close_raw.ffill()            # a missing bar must not zero out a holding
    opn = data.panel(ctx.frames, "open").reindex(close.index)
    panels = {col: data.panel(ctx.frames, col).reindex(close.index)
              for col in {c for c, _, _ in PRICE_FACTORS.values()} | {"sma200", "turnover_cr"}}
    sectors = ctx.universe.set_index("ticker")["sector"]
    idx = close.index[close.index >= pd.Timestamp(start)] if start else close.index[260:]
    month_ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last().values[:-1]

    cash, holdings = capital, {}
    equity, turnover_log, picks_log = [], [], []
    rebalance_next = None
    costs_total = 0.0
    for d in idx:
        if rebalance_next is not None:
            target = rebalance_next
            rebalance_next = None
            px = opn.loc[d]
            value = cash + sum(q * px.get(t, np.nan) for t, q in holdings.items() if pd.notna(px.get(t)))
            new_hold = {}
            per = value / max(len(target), 1)
            for t in target:
                if pd.notna(px.get(t)) and px[t] > 0:
                    new_hold[t] = int(per / px[t])
            traded = 0.0
            for t in set(holdings) | set(new_hold):
                p = px.get(t)
                if pd.isna(p):
                    new_hold[t] = holdings.get(t, 0)
                    continue
                dq = new_hold.get(t, 0) - holdings.get(t, 0)
                traded += abs(dq) * p
            c = trade_cost(traded / 2, traded / 2, False, broker)["total"] if traded else 0.0
            costs_total += c
            cash = value - sum(q * px.get(t, 0) for t, q in new_hold.items() if pd.notna(px.get(t))) - c
            holdings = {t: q for t, q in new_hold.items() if q > 0}
            turnover_log.append(traded / value if value else 0)
        mv = sum(q * close.at[d, t] for t, q in holdings.items() if pd.notna(close.at[d, t]))
        equity.append((d, cash + mv))
        if d in month_ends:
            cross = pd.DataFrame({k: v.loc[d] for k, v in panels.items()}).dropna(
                subset=["mom_12_1", "volatility63"])
            cross = cross[cross["turnover_cr"] >= 5]
            if trend_filter:
                cross = cross[close_raw.loc[d].reindex(cross.index) > cross["sma200"]]
            if len(cross) >= 5:
                sc = price_factor_scores(cross)["price_score"].sort_values(ascending=False)
                chosen, per_sec = [], {}
                for t in sc.index:
                    s = sectors.get(t, "?")
                    if per_sec.get(s, 0) < max_per_sector:
                        chosen.append(t)
                        per_sec[s] = per_sec.get(s, 0) + 1
                    if len(chosen) == n:
                        break
                rebalance_next = chosen
                picks_log.append((d, chosen))
            else:
                rebalance_next = []   # go to cash when nothing qualifies
    eq = pd.Series(dict(equity)).sort_index()
    bench = ctx.nifty["close"].reindex(eq.index).ffill()
    bench = bench / bench.iloc[0] * capital
    return {"equity": eq, "benchmark": bench, "costs": costs_total,
            "avg_turnover_pct": float(np.mean(turnover_log) * 100) if turnover_log else 0.0,
            "picks": picks_log}
