"""Performance statistics for equity curves and trade lists."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def equity_stats(equity: pd.Series, rf: float = 0.065) -> dict:
    """rf defaults to ~Indian T-bill yield."""
    r = equity.pct_change().dropna()
    if r.empty:
        return {}
    ex = r - rf / TRADING_DAYS
    downside = r[r < 0].std()
    mdd = max_drawdown(equity)
    g = cagr(equity)
    return {
        "Total return %": (equity.iloc[-1] / equity.iloc[0] - 1) * 100,
        "CAGR %": g * 100,
        "Volatility %": r.std() * np.sqrt(TRADING_DAYS) * 100,
        "Sharpe": ex.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() else 0.0,
        "Sortino": ex.mean() / downside * np.sqrt(TRADING_DAYS) if downside else 0.0,
        "Max drawdown %": mdd * 100,
        "Calmar": g / abs(mdd) if mdd else 0.0,
        "Best day %": r.max() * 100,
        "Worst day %": r.min() * 100,
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"Trades": 0}
    pnl = trades["net_pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() else float("inf")
    return {
        "Trades": int(len(trades)),
        "Win rate %": len(wins) / len(trades) * 100,
        "Avg win Rs": wins.mean() if len(wins) else 0.0,
        "Avg loss Rs": losses.mean() if len(losses) else 0.0,
        "Profit factor": pf,
        "Expectancy (R)": trades["r_multiple"].mean() if "r_multiple" in trades else np.nan,
        "Avg hold days": trades["hold_days"].mean() if "hold_days" in trades else np.nan,
        "Total costs Rs": trades["costs"].sum() if "costs" in trades else 0.0,
        "Net P&L Rs": pnl.sum(),
    }


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    m = equity.resample("ME").last().pct_change().dropna() * 100
    t = pd.DataFrame({"year": m.index.year, "month": m.index.strftime("%b"), "ret": m.values})
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pv = t.pivot(index="year", columns="month", values="ret").reindex(columns=order)
    pv["Year"] = (equity.resample("YE").last().pct_change() * 100).reindex(
        pd.to_datetime([f"{y}-12-31" for y in pv.index])).values
    first_year = equity.index[0].year
    if first_year in pv.index:
        e = equity[equity.index.year == first_year]
        pv.loc[first_year, "Year"] = (e.iloc[-1] / e.iloc[0] - 1) * 100
    return pv


def monte_carlo(r_multiples: pd.Series, risk_pct: float, n_trades: int | None = None,
                runs: int = 2000, seed: int = 7) -> dict:
    """Reshuffle trade outcomes to see the range of drawdowns luck alone could produce."""
    r = r_multiples.dropna().to_numpy()
    if len(r) < 10:
        return {}
    rng = np.random.default_rng(seed)
    n = n_trades or len(r)
    finals, mdds = [], []
    for _ in range(runs):
        path = np.cumprod(1 + rng.choice(r, n, replace=True) * risk_pct / 100)
        finals.append(path[-1] - 1)
        mdds.append((path / np.maximum.accumulate(path) - 1).min())
    finals, mdds = np.array(finals) * 100, np.array(mdds) * 100
    return {
        "median_return_pct": float(np.median(finals)),
        "p5_return_pct": float(np.percentile(finals, 5)),
        "p95_return_pct": float(np.percentile(finals, 95)),
        "median_max_dd_pct": float(np.median(mdds)),
        "p95_max_dd_pct": float(np.percentile(mdds, 5)),
        "prob_loss_pct": float((finals < 0).mean() * 100),
        "finals": finals, "mdds": mdds,
    }
