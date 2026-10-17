"""Intraday: pre-market watchlist with levels, live ORB scan and ORB backtest."""

from datetime import datetime
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import streamlit as st

from dashboard import ui
from stockadvisor import data
from stockadvisor.costs import trade_cost
from stockadvisor.indicators import vwap_intraday
from stockadvisor.strategies.intraday import backtest_orb, live_orb_scan, orb_summary


@st.cache_resource(ttl=1800, show_spinner="Backtesting ORB on 60 days of 15-minute bars…")
def _orb_bt(tickers: tuple, bias: tuple, capital: float, risk: float, broker: str):
    bars = data.get_intraday(list(tickers), "15m", "60d", max_age_minutes=30)
    return backtest_orb(bars, capital, risk, broker=broker, daily_bias=dict(bias))


def render():
    ctx, rep = ui.current()
    cfg = ui.settings()
    acct = cfg["account"]
    watch = rep["intraday_watch"]
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    open_now = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)

    st.title("Intraday")
    st.warning("SEBI's study found about 7 in 10 individual intraday traders lose money, and costs are paid on "
               "every trade. Trade small, keep a hard stop, square off by 3:15 PM, and stop for the day after "
               "2 losing trades.")
    c = trade_cost(100000, 100000, True, acct["broker"])
    st.caption(f"Cost check: a ₹1 lakh intraday round trip costs about {ui.inr(c['total'], 0)} "
               f"({c['pct_of_turnover'] * 2:.3f}% of the position) with {acct['broker'].title()}. "
               "Your target must be well beyond that.")

    t1, t2, t3 = st.tabs(["Pre-market watchlist", "Live ORB scanner", "ORB backtest"])
    with t1:
        st.caption(f"Built from the {rep['asof']} close for the next session. Narrow CPR + NR7/inside day + clean "
                   "trend = higher chance of a trending day.")
        if watch.empty:
            st.info("No liquid, volatile-enough stocks found.")
        else:
            for _, r in watch.head(12).iterrows():
                with st.expander(f"{r['symbol']} · bias {r['bias']} · score {r['score']:.0f} · ATR {r['atr_pct']:.1f}%"):
                    c1, c2 = st.columns([1.3, 1])
                    with c1:
                        st.markdown(ui.bullet_list(r["why"] or ["Liquid and volatile enough to trade"], "good"),
                                    unsafe_allow_html=True)
                        st.markdown(
                            "**Plan:** " + ("Look for longs above PDH / opening-range high while price is above VWAP."
                                            if r["bias"] == "Long" else
                                            "Look for shorts below PDL / opening-range low while price is below VWAP."
                                            if r["bias"] == "Short" else
                                            "No daily bias - trade only a clean break of the opening range with volume."))
                    with c2:
                        lv = [("R2", r["R2"]), ("R1", r["R1"]), ("PDH", r["PDH"]), ("CPR top", r["CPR_top"]),
                              ("Pivot", r["P"]), ("CPR bottom", r["CPR_bottom"]), ("PDL", r["PDL"]),
                              ("S1", r["S1"]), ("S2", r["S2"])]
                        st.markdown("".join(f'<div class="sa-level"><span>{k}</span><b>{v:,.2f}</b></div>' for k, v in lv),
                                    unsafe_allow_html=True)

    with t2:
        st.caption("Opening Range Breakout: the first 15-minute candle sets the range; a 15-minute close beyond "
                   "it, on the right side of VWAP, is the trigger. Stop at the other side (or the middle if the "
                   "range is wide), target 2R.")
        if not open_now:
            st.info("Market is closed now - the scan shows the last session's triggers.")
        if st.button("Scan watchlist now", type="primary") and not watch.empty:
            bars = data.get_intraday(list(watch["ticker"]), "15m", "5d", max_age_minutes=2)
            st.session_state["orb_live"] = (live_orb_scan(bars, watch), bars)
        if "orb_live" in st.session_state:
            res, bars = st.session_state["orb_live"]
            if res.empty:
                st.info("No ORB triggers yet.")
            else:
                st.dataframe(res[["symbol", "side", "entry_time", "entry", "stop", "target", "last", "status",
                                  "with_daily_trend"]], use_container_width=True, hide_index=True)
                sym = st.selectbox("Chart", res["ticker"])
                d = bars[sym]
                d = d[d.index.date == d.index[-1].date()]
                fig = go.Figure(go.Candlestick(x=d.index.strftime("%H:%M"), open=d["open"], high=d["high"],
                                               low=d["low"], close=d["close"], increasing_line_color=ui.UP,
                                               decreasing_line_color=ui.DOWN, name="15m"))
                fig.add_trace(go.Scatter(x=d.index.strftime("%H:%M"), y=vwap_intraday(d), name="VWAP",
                                         line=dict(color=ui.AMBER)))
                row = res[res["ticker"] == sym].iloc[0]
                for k, v, col in (("OR high", row["or_high"], ui.MUTED), ("OR low", row["or_low"], ui.MUTED),
                                  ("Stop", row["stop"], ui.DOWN), ("Target", row["target"], ui.UP)):
                    fig.add_hline(y=v, line=dict(color=col, dash="dot"), annotation_text=k)
                fig.update_xaxes(type="category")
                st.plotly_chart(ui._layout(fig, 420), use_container_width=True)

    with t3:
        st.caption("Tests the ORB rules on the watchlist over the last 60 trading days (Yahoo's 15-minute limit), "
                   "with intraday costs and slippage. 60 days is a small sample - treat it as a sanity check.")
        trend_only = st.toggle("Only trade in the direction of the daily trend", value=True)
        if st.button("Run ORB backtest") and not watch.empty:
            bias = tuple((t, b if (trend_only and b != "Neutral") else "Both")
                         for t, b in zip(watch["ticker"], watch["bias"]))
            tr = _orb_bt(tuple(watch["ticker"]), bias, acct["capital"],
                         cfg["intraday"]["risk_per_trade_pct"], acct["broker"])
            st.session_state["orb_bt"] = tr
        tr = st.session_state.get("orb_bt")
        if tr is not None:
            if tr.empty:
                st.info("No trades.")
            else:
                s = orb_summary(tr)
                k = st.columns(5)
                k[0].metric("Trades", s["Trades"])
                k[1].metric("Win rate", f"{s['Win rate %']:.0f}%")
                k[2].metric("Expectancy", f"{s['Expectancy (R)']:+.2f}R")
                k[3].metric("Net P&L", ui.inr(s["Net P&L Rs"]))
                k[4].metric("Costs / gross profit", f"{s['Costs as % of gross profit']:.0f}%")
                daily = tr.groupby("date")["net_pnl"].sum().cumsum()
                fig = go.Figure(go.Scatter(x=daily.index, y=daily, fill="tozeroy", line=dict(color=ui.ACCENT)))
                fig.update_layout(title="Cumulative net P&L (₹)")
                st.plotly_chart(ui._layout(fig, 300), use_container_width=True)
                st.dataframe(tr[["date", "ticker", "side", "entry", "exit", "exit_reason", "qty", "net_pnl",
                                 "r_multiple"]].sort_values("date", ascending=False),
                             use_container_width=True, hide_index=True)
                if s["Expectancy (R)"] <= 0:
                    st.error("Negative expectancy after costs - this would have lost money. Don't trade it live.")
