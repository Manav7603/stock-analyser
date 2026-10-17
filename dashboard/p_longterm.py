"""Long-Term Portfolio: factor-ranked picks, weights, and the strategy's track record."""

import plotly.express as px
import streamlit as st

from dashboard import ui
from stockadvisor.metrics import equity_stats
from stockadvisor.strategies import longterm


@st.cache_resource(ttl=3600, show_spinner="Backtesting the long-term strategy…")
def _bt(index: str, asof: str, n: int, cap: int, _ctx):
    return longterm.backtest_monthly(_ctx, n=n, max_per_sector=cap,
                                     start=str(_ctx.asof.year - 5) + "-01-01")


def render():
    ctx, rep = ui.current()
    cfg = ui.settings()
    lt = rep["longterm"]
    port, ranked = lt["portfolio"], lt["ranked"]
    st.title("Long-Term Portfolio")
    st.caption("Months to years. A diversified basket ranked on momentum, trend quality, low volatility and "
               "(where data exists) business quality, growth and valuation. Rebalance once a month.")

    bt = _bt(st.session_state.get("universe", "nifty200"), rep["asof"], cfg["longterm"]["portfolio_size"],
             cfg["longterm"]["max_per_sector"], ctx)
    s, b = equity_stats(bt["equity"]), equity_stats(bt["benchmark"])
    k = st.columns(5)
    k[0].metric("Strategy CAGR (5Y test)", f"{s['CAGR %']:.1f}%", f"{s['CAGR %'] - b['CAGR %']:+.1f}% vs NIFTY")
    k[1].metric("NIFTY CAGR", f"{b['CAGR %']:.1f}%")
    k[2].metric("Max drawdown", f"{s['Max drawdown %']:.1f}%", f"NIFTY {b['Max drawdown %']:.1f}%", delta_color="off")
    k[3].metric("Sharpe", f"{s['Sharpe']:.2f}", f"NIFTY {b['Sharpe']:.2f}", delta_color="off")
    k[4].metric("Avg monthly turnover", f"{bt['avg_turnover_pct']:.0f}%", f"costs {ui.inr(bt['costs'])}", delta_color="off")
    with st.expander("Equity curve vs NIFTY", expanded=False):
        st.plotly_chart(ui.equity_chart(bt["equity"], bt["benchmark"], "Monthly-rebalanced factor portfolio vs NIFTY 50"),
                        use_container_width=True)
        st.caption("Price factors only (no historical fundamentals), today's index members, costs included. "
                   "Expect real results to be somewhat lower; drawdowns are deeper than NIFTY - size accordingly.")

    if rep["market"]["label"] in ("Bearish", "Cautious"):
        st.warning("Market regime is weak. For long-term money, consider deploying in 3-4 tranches over the "
                   "next few months instead of all at once.")

    st.subheader(f"Suggested portfolio · {len(port)} stocks · {ui.inr(cfg['account']['capital'])}")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        show = port.reset_index().rename(columns={"index": "ticker"})
        show["ticker"] = show["ticker"].str.removesuffix(".NS")
        show["Pct rank"] = show["percentile"].round(0)
        st.dataframe(show[["ticker", "name", "sector", "Pct rank", "weight_pct", "amount", "qty", "close",
                           "ret252", "volatility63"]],
                     use_container_width=True, hide_index=True,
                     column_config={"Pct rank": st.column_config.ProgressColumn("Score pct", min_value=0, max_value=100, format="%d"),
                                    "weight_pct": st.column_config.NumberColumn("Weight", format="%.1f%%"),
                                    "amount": st.column_config.NumberColumn("Amount", format="₹%d"),
                                    "close": st.column_config.NumberColumn("Price", format="₹%.2f"),
                                    "ret252": st.column_config.NumberColumn("1Y return", format="%.0f%%"),
                                    "volatility63": st.column_config.NumberColumn("Volatility", format="%.2f")})
    with c2:
        sec = port.groupby("sector")["weight_pct"].sum().reset_index()
        fig = px.pie(sec, names="sector", values="weight_pct", hole=0.55)
        fig.update_traces(textinfo="percent+label", textposition="outside")
        fig.update_layout(showlegend=False, title="Sector mix")
        st.plotly_chart(ui._layout(fig, 360), use_container_width=True)

    st.subheader("Why each stock is in")
    factor_cols = [c for c in ["Momentum 12-1M", "Momentum 6M", "Low volatility", "Trend quality",
                               "Near 52W high", "Quality", "Growth", "Value (vs sector)"] if c in port]
    for t, r in port.iterrows():
        with st.expander(f"{t.removesuffix('.NS')} · {r['name']} · weight {r['weight_pct']}%"):
            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.markdown(ui.bullet_list(r["why"], "good"), unsafe_allow_html=True)
                if r.get("results_in_days") is not None and r["results_in_days"] <= 10:
                    st.markdown(ui.bullet_list([f"Results in {r['results_in_days']} days - "
                                                "consider buying after the numbers"], "bad"), unsafe_allow_html=True)
            with c2:
                vals = r[factor_cols].astype(float)
                fig = px.bar(x=vals.values, y=factor_cols, orientation="h",
                             color=vals.values > 0, color_discrete_map={True: ui.UP, False: ui.DOWN})
                fig.update_layout(showlegend=False, title="Factor scores (z, 0 = universe average)")
                st.plotly_chart(ui._layout(fig, 260), use_container_width=True, key=f"lt_{t}")

    st.subheader("Full ranking")
    full = ranked.reset_index().rename(columns={"index": "ticker"})
    full["ticker"] = full["ticker"].str.removesuffix(".NS")
    q = st.text_input("Search", placeholder="Symbol or company name")
    if q:
        full = full[full["ticker"].str.contains(q, case=False) | full["name"].str.contains(q, case=False, na=False)]
    st.dataframe(full[["rank", "ticker", "name", "sector", "percentile", "eligible"] + factor_cols],
                 use_container_width=True, hide_index=True,
                 column_config={"percentile": st.column_config.ProgressColumn("Score pct", min_value=0, max_value=100, format="%.0f"),
                                "eligible": st.column_config.CheckboxColumn("Above 200 DMA")})
    st.markdown("**Rules for long-term money**")
    st.markdown(ui.bullet_list([
        "Rebalance once a month (first trading day). Sell names that drop out of the top ranks or below the 200-day average.",
        "Never put more than ~10% in one stock or 3 stocks in one sector.",
        "Long-term gains (held > 1 year) are taxed at 12.5% above ₹1.25 lakh; short-term at 20%. "
        "Holding winners past 12 months saves real money.",
        "Expect 25-30% drawdowns at some point. The edge comes from sticking with the process through them.",
    ], "tip"), unsafe_allow_html=True)
