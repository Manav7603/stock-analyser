"""Backtest Lab: test swing and long-term strategies with your own settings."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import ui
from stockadvisor import ml
from stockadvisor.backtest import SwingParams, run_swing, setup_edge
from stockadvisor.metrics import equity_stats, monte_carlo, monthly_returns, trade_stats
from stockadvisor.strategies import longterm, swing


def _stats_block(eq, bench):
    s, b = equity_stats(eq), equity_stats(bench)
    k = st.columns(6)
    k[0].metric("CAGR", f"{s['CAGR %']:.1f}%", f"{s['CAGR %'] - b['CAGR %']:+.1f}% vs NIFTY")
    k[1].metric("Total return", f"{s['Total return %']:.0f}%", f"NIFTY {b['Total return %']:.0f}%", delta_color="off")
    k[2].metric("Max drawdown", f"{s['Max drawdown %']:.1f}%", f"NIFTY {b['Max drawdown %']:.1f}%", delta_color="off")
    k[3].metric("Sharpe", f"{s['Sharpe']:.2f}", f"NIFTY {b['Sharpe']:.2f}", delta_color="off")
    k[4].metric("Sortino", f"{s['Sortino']:.2f}")
    k[5].metric("Calmar", f"{s['Calmar']:.2f}")


def render():
    ctx, rep = ui.current()
    cfg = ui.settings()
    acct = cfg["account"]
    st.title("Backtest Lab")
    st.caption("Every test trades on the next day's prices, respects gaps, and pays full Indian costs + slippage. "
               "Universe = today's index members, so results are an optimistic upper bound.")
    t1, t2 = st.tabs(["Swing strategy", "Long-term portfolio"])

    with t1:
        with st.form("swing_bt"):
            c = st.columns(4)
            setups = c[0].multiselect("Setups", swing.SETUPS, default=swing.SETUPS)
            start = c[1].date_input("Start", value=pd.Timestamp("2022-02-01"),
                                    help="ML out-of-sample predictions start in 2022")
            risk = c[2].slider("Risk per trade %", 0.25, 3.0, float(acct["risk_per_trade_pct"]), 0.25)
            maxpos = c[3].slider("Max positions", 2, 20, int(acct["max_open_positions"]))
            c = st.columns(4)
            use_reg = c[0].toggle("Regime sizing", value=True, help="Cut size in weak markets")
            skip_bear = c[1].toggle("No new trades when bearish", value=True)
            min_score = c[2].slider("Min technical score", 0, 90, 0, 5)
            ml_filter = c[3].slider("ML filter: min OOS probability", 0.0, 0.6, 0.30, 0.02,
                                    help="Uses walk-forward (out-of-sample) predictions only, so no lookahead.")
            c = st.columns(4)
            stop_atr = c[0].slider("Stop (ATR)", 1.0, 4.0, float(cfg["swing"]["stop_atr"]), 0.5)
            t2r = c[1].slider("Final target (R)", 1.5, 6.0, float(cfg["swing"]["target2_r"]), 0.5)
            hold = c[2].slider("Max hold days", 5, 60, int(cfg["swing"]["max_hold_days"]), 5)
            broker = c[3].selectbox("Broker costs", ["zerodha", "dhan", "groww"],
                                    index=["zerodha", "dhan", "groww"].index(acct["broker"]))
            go_ = st.form_submit_button("Run backtest", type="primary")
        if go_:
            probs = ml.oos_prob_panel() if ml_filter > 0 else None
            if ml_filter > 0 and probs is None:
                st.warning("Train the ML model first (ML Model page) to use the ML filter.")
            p = SwingParams(setups=setups, capital=acct["capital"], risk_pct=risk, max_positions=maxpos,
                            max_position_pct=acct["max_position_pct"], stop_atr=stop_atr, t2_r=t2r, max_hold=hold,
                            use_regime=use_reg, skip_bearish=skip_bear, min_score=min_score, broker=broker,
                            start=str(start), ml_probs=probs, min_ml_prob=ml_filter)
            with st.spinner("Simulating every trading day…"):
                st.session_state["bt_swing"] = (run_swing(ctx, p), risk)
        if "bt_swing" in st.session_state:
            res, risk_used = st.session_state["bt_swing"]
            eq, tr = res["equity"], res["trades"]
            if tr.empty:
                st.info("No trades with these settings.")
            else:
                _stats_block(eq, res["benchmark"])
                ts = trade_stats(tr)
                k = st.columns(6)
                k[0].metric("Trades (incl. partials)", ts["Trades"])
                k[1].metric("Win rate", f"{ts['Win rate %']:.0f}%")
                k[2].metric("Profit factor", f"{ts['Profit factor']:.2f}")
                k[3].metric("Expectancy", f"{ts['Expectancy (R)']:+.2f}R")
                k[4].metric("Avg hold", f"{ts['Avg hold days']:.0f} days")
                k[5].metric("Costs paid", ui.inr(ts["Total costs Rs"]))
                st.plotly_chart(ui.equity_chart(eq, res["benchmark"]), use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Setup track record**")
                    st.dataframe(setup_edge(tr).round(2), use_container_width=True, hide_index=True)
                    ex = tr["exit_reason"].value_counts()
                    fig = go.Figure(go.Bar(x=ex.values, y=ex.index, orientation="h", marker_color=ui.ACCENT))
                    fig.update_layout(title="How trades ended")
                    st.plotly_chart(ui._layout(fig, 280), use_container_width=True)
                with c2:
                    mc = monte_carlo(tr["r_multiple"], risk_used)
                    if mc:
                        st.markdown("**Monte Carlo: 2,000 reshuffles of these trades**")
                        fig = go.Figure(go.Histogram(x=mc["mdds"], nbinsx=40, marker_color=ui.DOWN, opacity=0.8))
                        fig.update_layout(title="Range of max drawdowns luck alone could give you")
                        st.plotly_chart(ui._layout(fig, 260), use_container_width=True)
                        st.markdown(ui.bullet_list([
                            f"Median outcome {mc['median_return_pct']:+.0f}% (5th-95th pct: "
                            f"{mc['p5_return_pct']:+.0f}% to {mc['p95_return_pct']:+.0f}%)",
                            f"Typical worst drawdown {mc['median_max_dd_pct']:.0f}%, bad-luck case {mc['p95_max_dd_pct']:.0f}%",
                            f"Chance of ending with a loss: {mc['prob_loss_pct']:.0f}%"], "tip"), unsafe_allow_html=True)
                st.markdown("**Monthly returns %**")
                st.plotly_chart(ui.heatmap_monthly(monthly_returns(eq)), use_container_width=True)
                with st.expander("All trades"):
                    st.dataframe(tr.sort_values("entry_date", ascending=False).round(2),
                                 use_container_width=True, hide_index=True)

    with t2:
        with st.form("lt_bt"):
            c = st.columns(4)
            n = c[0].slider("Stocks held", 5, 30, int(cfg["longterm"]["portfolio_size"]))
            cap = c[1].slider("Max per sector", 1, 8, int(cfg["longterm"]["max_per_sector"]))
            start = c[2].date_input("Start", value=ctx.asof - pd.DateOffset(years=5), key="lts")
            tf = c[3].toggle("Only stocks above 200 DMA", value=True)
            go2 = st.form_submit_button("Run backtest", type="primary")
        if go2:
            with st.spinner("Rebalancing month by month…"):
                st.session_state["bt_lt"] = longterm.backtest_monthly(
                    ctx, n=n, max_per_sector=cap, start=str(start), capital=acct["capital"],
                    broker=acct["broker"], trend_filter=tf)
        if "bt_lt" in st.session_state:
            r = st.session_state["bt_lt"]
            _stats_block(r["equity"], r["benchmark"])
            st.plotly_chart(ui.equity_chart(r["equity"], r["benchmark"], "Factor portfolio vs NIFTY 50"),
                            use_container_width=True)
            st.markdown("**Monthly returns %**")
            st.plotly_chart(ui.heatmap_monthly(monthly_returns(r["equity"])), use_container_width=True)
            st.caption(f"Average monthly turnover {r['avg_turnover_pct']:.0f}% · total costs {ui.inr(r['costs'])}")
            if r["picks"]:
                d, picks = r["picks"][-1]
                st.markdown(f"Latest rebalance ({d.date()}): " + ", ".join(p.removesuffix(".NS") for p in picks))
