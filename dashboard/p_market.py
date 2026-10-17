"""Market Today: regime, breadth, sectors, AI brief and market news."""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import ui


def render():
    ctx, rep = ui.current()
    snap, mv = rep["market"], rep["market_view"]

    st.title("Market Today")
    st.caption(f"Data as of close on {rep['asof']} · {len(ctx.frames)} stocks in "
               f"{st.session_state.get('universe', 'universe').upper()} · generated {rep['generated'][11:16]}")

    # ---- hero: AI brief + gauge
    left, right = st.columns([2.2, 1])
    with left:
        stance_tone = {"Aggressive": "green", "Normal": "blue", "Defensive": "amber", "Stay in cash": "red"}
        st.markdown(
            f'<div class="sa-hero"><div>{ui.badge(snap["label"], ui.tone_for_regime(snap["label"]))}'
            f'{ui.badge("Stance: " + mv.get("stance", ""), stance_tone.get(mv.get("stance"), "grey"))}'
            f'{ui.badge("Size multiplier x" + format(snap["exposure"], ".2f"), "grey")}</div>'
            f'<h3 style="margin-top:10px">{ui.esc(mv.get("headline", ""))}</h3>'
            f'<div style="font-size:0.95rem;line-height:1.5">{ui.esc(mv.get("summary", ""))}</div>'
            f'<div class="sa-muted" style="margin-top:8px">{ui.esc(mv.get("_source", ""))}</div></div>',
            unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Why the regime is what it is**")
            st.markdown(ui.bullet_list(snap["reasons"], "tip"), unsafe_allow_html=True)
        with c2:
            st.markdown("**Tips for today**")
            st.markdown(ui.bullet_list(mv.get("tips", []), "good"), unsafe_allow_html=True)
    with right:
        st.plotly_chart(ui.gauge(snap["score"], "Market regime"), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(f'<div class="sa-muted" style="text-align:center">{ui.esc(snap["note"])}</div>',
                    unsafe_allow_html=True)

    # ---- KPIs
    br = snap["breadth"]
    n = ctx.nifty
    k = st.columns(5)
    k[0].metric("NIFTY 50", f"{n['close'].iloc[-1]:,.0f}", f"{n['close'].pct_change().iloc[-1] * 100:+.2f}% today")
    k[1].metric("NIFTY 1 month", f"{n['ret21'].iloc[-1] * 100:+.1f}%")
    k[2].metric("India VIX", f"{snap['vix']:.1f}",
                f"{(ctx.vix['close'].iloc[-1] / ctx.vix['close'].iloc[-6] - 1) * 100:+.0f}% 1W" if len(ctx.vix) > 6 else None,
                delta_color="inverse")
    k[3].metric("Stocks above 50 DMA", f"{br['above50'].iloc[-1]:.0f}%",
                f"{br['above50'].iloc[-1] - br['above50'].iloc[-6]:+.0f} pts 1W")
    k[4].metric("52W highs / lows", f"{int(br['new_highs'].iloc[-1])} / {int(br['new_lows'].iloc[-1])}")

    # ---- NIFTY chart with regime, breadth
    tab1, tab2, tab3 = st.tabs(["NIFTY & regime", "Market breadth", "Sector strength"])
    with tab1:
        st.plotly_chart(ui.price_chart(n, "NIFTY 50 with 20/50/200-day averages", months=12),
                        use_container_width=True)
        reg = snap["regime"].iloc[-500:]
        fig = go.Figure(go.Scatter(x=reg.index, y=reg, fill="tozeroy", line=dict(color=ui.ACCENT, width=1.5),
                                   fillcolor="rgba(96,165,250,.15)", name="Regime score"))
        for y, c in ((70, ui.UP), (50, ui.ACCENT), (30, ui.DOWN)):
            fig.add_hline(y=y, line=dict(color=c, dash="dot", width=1))
        fig.update_layout(title="Regime score history (above 70 = bullish, below 30 = bearish)")
        st.plotly_chart(ui._layout(fig, 260), use_container_width=True)
    with tab2:
        b = br.iloc[-500:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=b.index, y=b["above50"], name="% above 50 DMA", line=dict(color=ui.ACCENT)), 1, 1)
        fig.add_trace(go.Scatter(x=b.index, y=b["above200"], name="% above 200 DMA", line=dict(color=ui.PURPLE)), 1, 1)
        fig.add_hline(y=50, line=dict(color=ui.MUTED, dash="dot"), row=1, col=1)
        fig.add_trace(go.Bar(x=b.index, y=b["new_highs"], name="52W highs", marker_color=ui.UP), 2, 1)
        fig.add_trace(go.Bar(x=b.index, y=-b["new_lows"], name="52W lows", marker_color=ui.DOWN), 2, 1)
        fig.update_layout(title="Breadth: how many stocks are participating", barmode="relative")
        st.plotly_chart(ui._layout(fig, 480), use_container_width=True)
        st.caption("Healthy rallies have most stocks rising together. When NIFTY rises but breadth falls, "
                   "the rally is narrow and fragile.")
    with tab3:
        sec = rep["sectors"]
        fig = go.Figure(go.Bar(y=sec["sector"], x=sec["RS 3M"], orientation="h",
                               marker_color=[ui.UP if v > 0 else ui.DOWN for v in sec["RS 3M"]],
                               text=[f"{v:+.1f}%" for v in sec["RS 3M"]], textposition="outside"))
        fig.update_layout(title="Sector relative strength vs NIFTY (3 months, median stock)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(ui._layout(fig, 40 + 26 * len(sec)), use_container_width=True)
        st.dataframe(sec.style.format({c: "{:+.1f}" for c in ["1W %", "1M %", "3M %", "6M %", "RS 3M"]}
                                      | {"% above 50D": "{:.0f}"})
                     .background_gradient(subset=["RS 3M"], cmap="RdYlGn"),
                     use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.markdown("**Focus on**  " + " ".join(ui.badge(s, "green") for s in mv.get("sectors_to_focus", [])),
                    unsafe_allow_html=True)
        c2.markdown("**Avoid**  " + " ".join(ui.badge(s, "red") for s in mv.get("sectors_to_avoid", [])),
                    unsafe_allow_html=True)

    # ---- news
    st.subheader("Market news")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(ui.news_html(rep["market_news"], 12), unsafe_allow_html=True)
    with c2:
        st.markdown("**What the news means**")
        st.markdown(ui.bullet_list(mv.get("news_takeaways", []), "tip"), unsafe_allow_html=True)
        st.info("Why news matters: price setups can't see results dates, rating changes, regulatory action "
                "or global shocks. The advisor checks headlines for every idea and downgrades setups when the "
                "news contradicts the chart.")
