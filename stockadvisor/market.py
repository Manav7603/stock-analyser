"""Loads the universe once and computes market regime, breadth and sector strength.

The regime decides how aggressive the advice is: in a weak market the
advisor cuts position sizes and raises the bar for new swing trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, indicators
from .config import load_config
from .universe import INDIA_VIX, NIFTY, SECTOR_INDICES, get_universe


@dataclass
class MarketContext:
    universe: pd.DataFrame                      # symbol, name, sector, ticker
    frames: dict[str, pd.DataFrame]             # ticker -> OHLCV + indicators
    nifty: pd.DataFrame
    vix: pd.DataFrame
    sectors: dict[str, pd.DataFrame] = field(default_factory=dict)
    adjustments: dict[str, list[str]] = field(default_factory=dict)

    @property
    def asof(self) -> pd.Timestamp:
        return self.nifty.index[-1]

    def sector_of(self, ticker: str) -> str:
        row = self.universe.loc[self.universe["ticker"] == ticker, "sector"]
        return row.iloc[0] if len(row) else "Unknown"

    def name_of(self, ticker: str) -> str:
        row = self.universe.loc[self.universe["ticker"] == ticker, "name"]
        return row.iloc[0] if len(row) else ticker


def load_context(index: str | None = None, years: int | None = None,
                 refresh: bool = False) -> MarketContext:
    cfg = load_config()
    index = index or cfg["universe"]["index"]
    years = years or cfg["universe"]["history_years"]
    uni = get_universe(index)
    extra = [NIFTY, INDIA_VIX, *SECTOR_INDICES.values()]
    raw = data.get_daily(list(uni["ticker"]) + extra, years=years, refresh=refresh)
    nifty = indicators.add_all(raw[NIFTY]) if NIFTY in raw else pd.DataFrame()
    bench = nifty["close"] if not nifty.empty else None
    frames, adjustments = {}, {}
    for t in uni["ticker"]:
        df = raw.get(t)
        if df is not None and len(df) > 60:
            df, notes = data.fix_corporate_actions(df)
            if notes:
                adjustments[t] = notes
            frames[t] = indicators.add_all(df, bench)
    uni = uni[uni["ticker"].isin(frames)].reset_index(drop=True)
    sectors = {k: raw[v] for k, v in SECTOR_INDICES.items() if v in raw and len(raw[v]) > 60}
    return MarketContext(uni, frames, nifty, raw.get(INDIA_VIX, pd.DataFrame()), sectors, adjustments)


def breadth(ctx: MarketContext) -> pd.DataFrame:
    """Share of universe stocks above their 50/200-day averages, per day."""
    close = data.panel(ctx.frames, "close")
    e50 = data.panel(ctx.frames, "ema50")
    s200 = data.panel(ctx.frames, "sma200")
    high = data.panel(ctx.frames, "high")
    low = data.panel(ctx.frames, "low")
    hh = data.panel(ctx.frames, "hh252")
    ll = data.panel(ctx.frames, "ll252")
    valid = close.notna().sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({
        "above50": (close > e50).sum(axis=1) / valid * 100,
        "above200": (close > s200).sum(axis=1) / valid * 100,
        "new_highs": (high >= hh).sum(axis=1),
        "new_lows": (low <= ll).sum(axis=1),
        "adv_dec": (close.pct_change() > 0).sum(axis=1) / valid * 100,
    }).dropna(how="all")


def regime_series(nifty: pd.DataFrame, vix: pd.DataFrame, br: pd.DataFrame | None = None) -> pd.Series:
    """Daily regime score 0..100 (higher = friendlier for long trades). No lookahead."""
    c = nifty["close"]
    score = pd.Series(0.0, index=nifty.index)
    score += np.where(c > nifty["sma200"], 25, 0)
    score += np.where(c > nifty["ema50"], 15, 0)
    score += np.where(nifty["ema50"] > nifty["sma200"], 10, 0)
    score += np.where(nifty["ema20_slope"] > 0, 10, 0)
    if not vix.empty:
        v = vix["close"].reindex(nifty.index).ffill()
        vpct = v.rolling(252, min_periods=60).rank(pct=True)
        score += np.where(vpct < 0.5, 15, np.where(vpct < 0.8, 7, 0))
    else:
        score += 7
    if br is not None and not br.empty:
        a50 = br["above50"].reindex(nifty.index).ffill()
        score += np.where(a50 > 60, 25, np.where(a50 > 40, 12, 0))
    else:
        score += 12
    # 3-day smoothing stops one noisy day from flipping position sizes
    return score.clip(0, 100).rolling(3, min_periods=1).mean()


def regime_label(score: float) -> tuple[str, float, str]:
    """(label, exposure multiplier, plain-English guidance)."""
    if score >= 70:
        return ("Bullish", 1.0, "Trend and breadth support longs. Normal position sizes; "
                "breakouts tend to follow through.")
    if score >= 50:
        return ("Neutral-positive", 0.75, "Market is mixed. Prefer the strongest stocks "
                "(high relative strength), trim size by about a quarter.")
    if score >= 30:
        return ("Cautious", 0.5, "Weak trend or poor breadth. Take only A-grade setups at half "
                "size and book profits faster.")
    return ("Bearish", 0.25, "NIFTY is in a downtrend. Most long setups fail here - "
            "mostly stay in cash, protect capital.")


def market_snapshot(ctx: MarketContext) -> dict:
    br = breadth(ctx)
    reg = regime_series(ctx.nifty, ctx.vix, br)
    score = float(reg.iloc[-1])
    label, mult, note = regime_label(score)
    n = ctx.nifty.iloc[-1]
    vix_last = float(ctx.vix["close"].iloc[-1]) if not ctx.vix.empty else float("nan")
    reasons = [
        f"NIFTY {'above' if n.close > n.sma200 else 'below'} its 200-day average "
        f"({n.close:,.0f} vs {n.sma200:,.0f})",
        f"NIFTY {'above' if n.close > n.ema50 else 'below'} its 50-day average",
        f"{br['above50'].iloc[-1]:.0f}% of {len(ctx.frames)} stocks are above their 50-day average",
        f"India VIX at {vix_last:.1f}" + (" (calm)" if vix_last < 15 else " (elevated)" if vix_last > 20 else ""),
        f"{int(br['new_highs'].iloc[-1])} stocks at 52-week highs vs {int(br['new_lows'].iloc[-1])} at lows",
    ]
    return {"score": score, "label": label, "exposure": mult, "note": note, "reasons": reasons,
            "nifty_close": float(n.close), "nifty_chg_pct": float(n.ret5 * 100) if pd.notna(n.ret5) else 0.0,
            "vix": vix_last, "breadth": br, "regime": reg}


def sector_strength(ctx: MarketContext) -> pd.DataFrame:
    """Sector returns over 1W/1M/3M/6M plus relative strength vs NIFTY (from constituents)."""
    rows = []
    for sector, grp in ctx.universe.groupby("sector"):
        fr = [ctx.frames[t] for t in grp["ticker"] if t in ctx.frames]
        if not fr:
            continue
        last = pd.DataFrame([f.iloc[-1] for f in fr])
        rows.append({
            "sector": sector, "stocks": len(fr),
            "1W %": last["ret5"].median() * 100, "1M %": last["ret21"].median() * 100,
            "3M %": last["ret63"].median() * 100, "6M %": last["ret126"].median() * 100,
            "RS 3M": last.get("rs63", pd.Series(dtype=float)).median() * 100,
            "% above 50D": (last["close"] > last["ema50"]).mean() * 100,
        })
    return pd.DataFrame(rows).sort_values("RS 3M", ascending=False).reset_index(drop=True)
