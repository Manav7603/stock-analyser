"""Fast offline tests for the maths that money depends on."""

import numpy as np
import pandas as pd
import pytest

from stockadvisor import indicators
from stockadvisor.costs import trade_cost
from stockadvisor.data import fix_corporate_actions
from stockadvisor.metrics import cagr, max_drawdown
from stockadvisor.news import _score_text
from stockadvisor.strategies.base import position_size


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    idx = pd.bdate_range("2022-01-03", periods=n)
    o = close * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, close) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(o, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": close,
                         "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


def test_zerodha_delivery_costs():
    c = trade_cost(100000, 100000, intraday=False, broker="zerodha")
    assert c["brokerage"] == 0
    assert c["stt"] == pytest.approx(200)          # 0.1% on each side
    assert c["stamp"] == pytest.approx(15)         # 0.015% on buy
    assert 230 < c["total"] < 250


def test_intraday_brokerage_capped_at_20():
    c = trade_cost(1_000_000, 1_000_000, intraday=True, broker="zerodha")
    assert c["brokerage"] == pytest.approx(40)     # Rs 20 per order, two orders
    assert c["stt"] == pytest.approx(250)          # 0.025% sell side only


def test_indicators_have_no_lookahead():
    f = _frame()
    full = indicators.add_all(f)
    cut = indicators.add_all(f.iloc[:200])
    cols = ["ema20", "sma200", "rsi14", "atr14", "adx", "macd", "st_dir", "hh20", "bb_up"]
    pd.testing.assert_frame_equal(full[cols].iloc[:200], cut[cols], check_exact=False, rtol=1e-9)


def test_rsi_bounds():
    r = indicators.rsi(_frame()["close"]).dropna()
    assert r.between(0, 100).all()


def test_position_size_respects_risk_and_cap():
    q = position_size(500000, 1.0, entry=100, stop=95, max_position_pct=20)
    assert q == 1000                               # 5000 risk / 5 per share
    q = position_size(500000, 1.0, entry=100, stop=99.9, max_position_pct=20)
    assert q == 1000                               # capped at 20% of capital
    assert position_size(500000, 1.0, entry=100, stop=100, max_position_pct=20) == 0


def test_corporate_action_back_adjusts_bonus():
    f = _frame(50)
    f.iloc[25:, :4] /= 2                           # unadjusted 1:1 bonus on day 25
    fixed, notes = fix_corporate_actions(f)
    assert notes
    assert abs(fixed["close"].pct_change().iloc[25]) < 0.1


def test_metrics():
    eq = pd.Series([100, 120, 90, 135], index=pd.date_range("2020-01-01", periods=4, freq="365D"))
    assert max_drawdown(eq) == pytest.approx(-0.25)
    assert cagr(eq) == pytest.approx(1.35 ** (1 / 3) - 1, rel=1e-2)


def test_news_sentiment_word_boundaries():
    assert _score_text("Mukesh Ambani taps Reliance workforce") == 0      # 'ban' inside 'Ambani'
    assert _score_text("Infosys Q2 net profit jumps, beats estimates") > 0.5
    assert _score_text("SEBI probe into promoter; shares plunge") < -0.5
