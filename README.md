# Indian Stock Advisor

Research, signals and backtests for NSE stocks: swing, long-term and intraday.
It tells you **what** to consider buying, **why**, **how much**, **where to exit** and
**whether the idea has actually made money in the past**. You place every order
yourself in Zerodha / Groww / Dhan.

> Decision support only. Not investment advice and not a SEBI-registered adviser.
> No tool can guarantee profit; the edge here is small and shows up across many trades.

## Start

```bash
cd ~/indian-stock-advisor
.venv/bin/streamlit run app.py
```

Opens at http://localhost:8501. The first load downloads ~6 years of prices (1-2 minutes);
after that it's cached.

Command line:

```bash
.venv/bin/python -m stockadvisor report            # today's regime, ideas, long-term picks
.venv/bin/python -m stockadvisor train             # retrain the ML model (monthly)
.venv/bin/python -m stockadvisor backtest          # swing + long-term backtests
.venv/bin/python -m pytest -q tests                # tests
```

Settings (capital, risk %, broker, universe) are in `config.toml`.
The Claude API key lives in `.env` (git-ignored, readable only by you).

## How it decides

| Layer | What it does | Where |
|---|---|---|
| Market regime | NIFTY vs 50/200 DMA, India VIX, breadth → Bullish/Neutral/Cautious/Bearish; cuts position size in weak markets | `market.py` |
| Swing setups | Breakout, Trend pullback, Oversold bounce, Supertrend flip with entry/stop/targets | `strategies/swing.py` |
| Track record | Every setup is backtested after full Indian costs; setups that don't make money are flagged | `backtest.py` |
| ML ranking | Gradient boosting ranks stocks by odds of beating the universe over 10 days; validated walk-forward | `ml.py` |
| News | Google News per stock, finance sentiment, event tags (results, rating, regulatory, deals) | `news.py` |
| Fundamentals | P/E, ROE, debt, growth, analyst view, **next results date** | `fundamentals.py` |
| AI analyst | Claude reads all of the above + headlines and gives a verdict, catalysts, risks, what to watch | `ai.py` |
| Long-term | Momentum + low-vol + trend quality + quality/growth/value, sector-capped, monthly rebalance | `strategies/longterm.py` |
| Intraday | Pre-market watchlist (CPR, NR7, pivots) + Opening Range Breakout scanner and backtest | `strategies/intraday.py` |
| Portfolio | Log your trades; daily hold / book half / trail / exit advice; trade journal | `portfolio.py` |

## What the tests showed (NIFTY 200, Oct 2026)

Honest numbers, after brokerage, STT, GST, stamp duty and slippage:

- **Long-term factor portfolio:** ~22.7% CAGR vs NIFTY ~9.1% over 5 years, but deeper drawdowns (-29% vs -17%).
- **ML model:** out-of-sample AUC 0.54. The top-ranked 10% beat the median stock in **5 of 5** test years.
- **Swing setups alone:** roughly break-even, underperforming NIFTY. **With the ML filter**, every configuration
  improved (all setups 2.2% → 6.1% CAGR; drawdown -32% → -26%). The filter is on by default.
- **Intraday ORB:** lost money over the last 60 days after costs. Treat intraday as the hardest game.

All backtests use today's index members (survivorship bias), so real results will be somewhat lower.

## Daily routine

1. Evening or pre-market: open **Market Today**. The regime decides how much to trade.
2. **Swing Ideas**: shortlist A/B grades that pass the ML filter, have a proven or marginal setup, and an AI verdict that isn't "Avoid".
3. Read the risks, especially results dates and bonus/split ex-dates.
4. Place the order with the shown quantity, then immediately a **GTT OCO** (stop + target).
5. Log it in **My Portfolio** and follow its daily action.
6. First trading day of each month: rebalance the long-term portfolio and retrain the ML model.

## Data notes

- Prices come from Yahoo Finance (free). Verify in your broker app before ordering.
- If Yahoo's daily bar lags, the last session is rebuilt from hourly bars.
- Unadjusted bonus/split/demerger gaps of 30%+ are back-adjusted automatically.
- Claude responses are cached for 12 hours to keep API costs low.
