"""Machine-learning layer: which stocks will beat the rest of the universe
over the next 10 trading days?

* Target = cross-sectional. Label is 1 when a stock's 10-day return (from
  the next day's open) lands in the top 30% of the universe that day. The
  model ranks stocks against each other; it does not try to time the
  market - an earlier version that did scored AUC 0.50 out of sample.
  Market timing is left to the regime filter.
* Features = the stock's own trend/momentum/volatility/volume state and its
  rank versus the rest of the universe that day.
* Validation = walk-forward by calendar year with a purge gap. The model
  is always scored on a year it has never seen, which is how it would
  have been used in real time. In-sample accuracy is never reported.
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .config import MODEL_DIR

HORIZON = 10
TOP_SHARE = 0.30

STOCK_FEATURES = [
    "rsi14", "rsi2", "adx", "plus_di", "minus_di", "bb_pct", "bb_width", "atr_pct",
    "vol_ratio", "ret5", "ret21", "ret63", "ret126", "mom_12_1", "dist_52wh", "dist_52wl",
    "ema20_slope", "ema50_slope", "volatility63", "above200_share", "obv_slope", "st_dir",
    "rs63", "rs126", "range_pct",
]
DERIVED = ["px_ema20", "px_ema50", "px_sma200", "macd_norm", "rank_ret63", "rank_rs63",
           "rank_vol_ratio", "nr7_f", "inside_f"]
FEATURES = STOCK_FEATURES + DERIVED

PRETTY = {
    "rsi14": "RSI (14)", "rsi2": "RSI (2)", "adx": "ADX trend strength", "plus_di": "+DI", "minus_di": "-DI",
    "bb_pct": "Position in Bollinger band", "bb_width": "Bollinger width (squeeze)", "atr_pct": "Volatility (ATR %)",
    "vol_ratio": "Volume vs average", "ret5": "1-week return", "ret21": "1-month return", "ret63": "3-month return",
    "ret126": "6-month return", "mom_12_1": "12-1 month momentum", "dist_52wh": "Distance from 52W high",
    "dist_52wl": "Distance from 52W low", "ema20_slope": "20 EMA slope", "ema50_slope": "50 EMA slope",
    "volatility63": "3M volatility", "above200_share": "Time above 200 DMA", "obv_slope": "OBV slope",
    "st_dir": "Supertrend direction", "rs63": "3M relative strength", "rs126": "6M relative strength",
    "range_pct": "Day range %", "px_ema20": "Price vs 20 EMA", "px_ema50": "Price vs 50 EMA",
    "px_sma200": "Price vs 200 DMA", "macd_norm": "MACD histogram", "rank_ret63": "3M return rank",
    "rank_rs63": "Relative strength rank", "rank_vol_ratio": "Volume surge rank", "nr7_f": "NR7",
    "inside_f": "Inside day", "mkt_ret21": "NIFTY 1-month return", "mkt_above200": "NIFTY vs 200 DMA",
    "mkt_vix": "India VIX", "mkt_breadth50": "Market breadth", "mkt_regime": "Market regime score",
}


def build_dataset(ctx, with_labels: bool = True) -> pd.DataFrame:
    parts = []
    for t, f in ctx.frames.items():
        if len(f) < 260:
            continue
        x = f[[c for c in STOCK_FEATURES if c in f]].copy()
        x["px_ema20"] = f["close"] / f["ema20"] - 1
        x["px_ema50"] = f["close"] / f["ema50"] - 1
        x["px_sma200"] = f["close"] / f["sma200"] - 1
        x["macd_norm"] = f["hist"] / f["close"]
        x["nr7_f"] = f["nr7"].astype(float)
        x["inside_f"] = f["inside_day"].astype(float)
        x["turnover_cr"] = f["turnover_cr"]
        if with_labels:
            x["fwd_ret"] = f["open"].shift(-HORIZON - 1) / f["open"].shift(-1) - 1
        x["ticker"] = t
        parts.append(x)
    df = pd.concat(parts)
    df.index.name = "date"
    df = df.reset_index()
    df = df[df["turnover_cr"] >= 5]
    for col, src in (("rank_ret63", "ret63"), ("rank_rs63", "rs63"), ("rank_vol_ratio", "vol_ratio")):
        df[col] = df.groupby("date")[src].rank(pct=True)
    if with_labels:
        has = df["fwd_ret"].notna()
        xs = df[has].groupby("date")["fwd_ret"].rank(pct=True)
        df["label"] = np.nan
        df.loc[has, "label"] = (xs >= 1 - TOP_SHARE).astype(float)
        df["excess_ret"] = df["fwd_ret"] - df.groupby("date")["fwd_ret"].transform("median")
    return df


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.04, max_iter=350,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=42)


def walk_forward(ds: pd.DataFrame, first_test_year: int | None = None, purge_days: int = 30) -> dict:
    lab = ds.dropna(subset=["label"])
    years = sorted(lab["date"].dt.year.unique())
    first_test_year = first_test_year or years[min(2, len(years) - 1)]
    oos, folds = [], []
    for y in [yr for yr in years if yr >= first_test_year]:
        test = lab[lab["date"].dt.year == y]
        train = lab[lab["date"] < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=purge_days)]
        if len(train) < 5000 or test.empty:
            continue
        m = _model().fit(train[FEATURES], train["label"])
        t = test[["date", "ticker", "label", "fwd_ret", "excess_ret"]].copy()
        t["prob"] = m.predict_proba(test[FEATURES])[:, 1]
        oos.append(t)
        top = t[t.groupby("date")["prob"].rank(pct=True) >= 0.9]
        bottom = t[t.groupby("date")["prob"].rank(pct=True) <= 0.1]
        folds.append({
            "year": int(y), "rows": len(t),
            "auc": roc_auc_score(t["label"], t["prob"]) if t["label"].nunique() > 1 else np.nan,
            "top10_beat_median_pct": (top["excess_ret"] > 0).mean() * 100,
            "top10_excess_10d_pct": top["excess_ret"].mean() * 100,
            "bottom10_excess_10d_pct": bottom["excess_ret"].mean() * 100,
        })
    return {"folds": pd.DataFrame(folds), "oos": pd.concat(oos) if oos else pd.DataFrame()}


def decile_table(oos: pd.DataFrame) -> pd.DataFrame:
    if oos.empty:
        return oos
    o = oos.copy()
    o["decile"] = np.ceil(o.groupby("date")["prob"].rank(pct=True) * 10).clip(1, 10).astype(int)
    return o.groupby("decile").agg(avg_prob=("prob", "mean"),
                                   beat_median_pct=("excess_ret", lambda s: (s > 0).mean() * 100),
                                   excess_10d_pct=("excess_ret", lambda s: s.mean() * 100),
                                   rows=("label", "size")).reset_index()


def train(ctx, name: str = "swing_model") -> dict:
    """Walk-forward evaluate, then fit the live model on all labelled history."""
    ds = build_dataset(ctx)
    wf = walk_forward(ds)
    lab = ds.dropna(subset=["label"])
    model = _model().fit(lab[FEATURES], lab["label"])
    sample = lab.sample(min(20000, len(lab)), random_state=1)
    imp = permutation_importance(model, sample[FEATURES], sample["label"], n_repeats=3,
                                 random_state=1, scoring="roc_auc")
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .assign(label=lambda d: d.feature.map(PRETTY))
                  .sort_values("importance", ascending=False))
    meta = {"trained_at": datetime.now().isoformat(timespec="seconds"),
            "universe_size": len(ctx.frames), "rows": int(len(lab)),
            "horizon": HORIZON, "top_share": TOP_SHARE,
            "mean_auc": float(wf["folds"]["auc"].mean()) if len(wf["folds"]) else None,
            "mean_top10_excess_pct": float(wf["folds"]["top10_excess_10d_pct"].mean()) if len(wf["folds"]) else None,
            "years_positive": int((wf["folds"]["top10_excess_10d_pct"] > 0).sum()) if len(wf["folds"]) else 0,
            "years_tested": int(len(wf["folds"]))}
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))
    wf["folds"].to_csv(MODEL_DIR / f"{name}_folds.csv", index=False)
    importance.to_csv(MODEL_DIR / f"{name}_importance.csv", index=False)
    if not wf["oos"].empty:
        wf["oos"].to_parquet(MODEL_DIR / f"{name}_oos.parquet")
    return {"meta": meta, "folds": wf["folds"], "importance": importance,
            "deciles": decile_table(wf["oos"]), "oos": wf["oos"]}


def load(name: str = "swing_model"):
    p = MODEL_DIR / f"{name}.joblib"
    if not p.exists():
        return None, None
    meta = json.loads((MODEL_DIR / f"{name}.json").read_text())
    return joblib.load(p), meta


def predict_latest(ctx, name: str = "swing_model") -> pd.DataFrame:
    """Model probability and cross-sectional rank (0-100) for every stock on the latest bar."""
    model, _ = load(name)
    if model is None:
        return pd.DataFrame(columns=["prob", "ml_rank"])
    ds = build_dataset(ctx, with_labels=False)
    last = ds[ds["date"] == ds["date"].max()].set_index("ticker")
    out = pd.DataFrame({"prob": model.predict_proba(last[FEATURES])[:, 1]}, index=last.index)
    out["ml_rank"] = out["prob"].rank(pct=True) * 100
    return out.join(last[FEATURES])


def oos_prob_panel(name: str = "swing_model") -> pd.DataFrame | None:
    """Out-of-sample probabilities (dates x tickers) for honest ML-filtered backtests."""
    p = MODEL_DIR / f"{name}_oos.parquet"
    if not p.exists():
        return None
    o = pd.read_parquet(p)
    return o.pivot_table(index="date", columns="ticker", values="prob")


def explain_prediction(model, row: pd.DataFrame, top: int = 5) -> list[tuple[str, float]]:
    """Which features push this stock's probability up or down (one-at-a-time ablation)."""
    base = model.predict_proba(row[FEATURES])[:, 1][0]
    out = []
    ref = {"rsi14": 50, "rsi2": 50, "adx": 20, "plus_di": 20, "minus_di": 20, "bb_pct": 0.5,
           "vol_ratio": 1.0, "st_dir": 0, "atr_pct": 2.0, "bb_width": 0.1, "volatility63": 0.3,
           "above200_share": 0.5, "range_pct": 0.02, "dist_52wh": -0.15, "dist_52wl": 0.3}
    for f in FEATURES:
        r2 = row[FEATURES].copy()
        r2[f] = ref.get(f, 0.5 if f.startswith("rank_") else 0.0)
        out.append((PRETTY.get(f, f), base - model.predict_proba(r2)[:, 1][0]))
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out[:top]
