"""Swing Ideas: today's setups ranked by technicals + ML + track record + news + AI."""

import pandas as pd
import streamlit as st

from dashboard import ui


def _edge_table(edge: pd.DataFrame):
    if edge.empty:
        st.info("Not enough history to judge setups yet.")
        return
    cols = st.columns(len(edge))
    for c, (_, r) in zip(cols, edge.iterrows()):
        c.markdown(
            f'<div class="sa-card"><div style="font-weight:600">{ui.esc(r["setup"])}</div>'
            f'{ui.badge(r["verdict"], ui.tone_for_verdict(r["verdict"]))}'
            f'<div style="margin-top:8px;font-size:1.3rem;font-weight:700">{r["expectancy_r"]:+.2f}R</div>'
            f'<div class="sa-muted">per trade after costs · {r["win_rate_pct"]:.0f}% wins · '
            f'{int(r["trades"])} trades · PF {r["profit_factor"]:.2f}</div>'
            f'<div class="sa-muted">last 12 months: {r["recent_expectancy_r"]:+.2f}R</div></div>',
            unsafe_allow_html=True)


def _idea(s, ctx, broker: str):
    ai_view = s.ai or {}
    head = (f"{s.symbol} · {s.setup} · score {s.score:.0f}"
            + (f" · AI: {ai_view.get('verdict')}" if ai_view else ""))
    with st.expander(head, expanded=False):
        st.markdown(
            f'{ui.badge("Grade " + s.grade, ui.tone_for_grade(s.grade))}'
            + (ui.badge(ai_view.get("verdict", ""), ui.tone_for_verdict(ai_view.get("verdict", ""))) if ai_view else "")
            + ui.badge(s.sector, "grey")
            + (ui.badge(f"ML rank {s.ml_prob:.0f}/100", "blue") if s.ml_prob is not None else "")
            + ui.sentiment_badge(s.news_sentiment)
            + f'<span class="sa-muted">{ui.esc(s.name)} · signal on {s.date}</span>',
            unsafe_allow_html=True)
        m = st.columns(6)
        m[0].metric("Entry", ui.inr(s.entry, 2), s.entry_type, delta_color="off")
        m[1].metric("Stop-loss", ui.inr(s.stop, 2), f"-{s.risk_per_share / s.entry * 100:.1f}%", delta_color="off")
        m[2].metric("Target 1", ui.inr(s.target1, 2), f"+{(s.target1 / s.entry - 1) * 100:.1f}%", delta_color="off")
        m[3].metric("Target 2", ui.inr(s.target2, 2), f"+{(s.target2 / s.entry - 1) * 100:.1f}%", delta_color="off")
        if s.qty > 0:
            m[4].metric("Quantity", f"{s.qty:,}", ui.inr(s.capital_needed), delta_color="off")
            m[5].metric("Money at risk", ui.inr(s.risk_amount), f"R:R 1:{s.reward_risk:.1f}", delta_color="off")
        else:
            m[4].metric("Quantity", "0", "risk budget too small", delta_color="off")
            m[5].metric("R:R", f"1:{s.reward_risk:.1f}")
        f = ctx.frames[s.ticker]
        st.plotly_chart(ui.price_chart(f, f"{s.symbol} - last 6 months", months=6,
                                       levels={"Entry": s.entry, "Stop": s.stop, "Target 1": s.target1,
                                               "Target 2": s.target2}),
                        use_container_width=True, key=f"chart_{s.ticker}_{s.setup}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Why this trade**")
            st.markdown(ui.bullet_list(s.reasons, "good"), unsafe_allow_html=True)
        with c2:
            st.markdown("**Risks**")
            st.markdown(ui.bullet_list(s.risks or ["No specific red flags"], "bad"), unsafe_allow_html=True)
        if ai_view:
            st.markdown(
                f'<div class="sa-card"><div style="font-weight:600;margin-bottom:6px">🤖 AI analyst · '
                f'confidence {ai_view.get("confidence", 0)}/100 · {ui.esc(ai_view.get("news_view", ""))}</div>'
                f'<div style="line-height:1.5">{ui.esc(ai_view.get("plain_english", ""))}</div>'
                f'<div class="sa-muted" style="margin-top:6px">News: {ui.esc(ai_view.get("news_summary", ""))}</div>'
                f'<div class="sa-muted" style="margin-top:4px">{ui.esc(ai_view.get("_source", ""))}</div></div>',
                unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            a1.markdown("**Catalysts**" + ui.bullet_list(ai_view.get("catalysts", []), "good"), unsafe_allow_html=True)
            a2.markdown("**Watch for**" + ui.bullet_list(ai_view.get("what_to_watch", []), "tip"), unsafe_allow_html=True)
            a3.markdown("**Execution tips**" + ui.bullet_list(ai_view.get("tips", []), "tip"), unsafe_allow_html=True)
        if s.news:
            st.markdown("**Recent headlines**")
            st.markdown(ui.news_html(s.news, 6), unsafe_allow_html=True)
        e = s.extra.get("fundamentals") or {}
        dtr = s.extra.get("days_to_results")
        st.markdown(
            '<div class="sa-muted">'
            + f"P/E {e.get('trailingPE') or '–'} · ROE "
            + (f"{e['returnOnEquity'] * 100:.0f}%" if e.get("returnOnEquity") else "–")
            + " · Debt/Equity " + (f"{e['debtToEquity']:.0f}" if e.get("debtToEquity") else "–")
            + f" · Analyst view {e.get('recommendationKey') or '–'}"
            + (f" · Results in {dtr} days" if dtr is not None else "")
            + f" · Costs eat {s.cost_breakeven_pct:.2f}% of the move ({broker.title()})</div>",
            unsafe_allow_html=True)
        st.markdown("**How to place it**")
        place = (f"Buy **{s.qty}** {s.symbol} "
                 + (f"with a stop-limit **buy above ₹{s.entry:,.2f}** (trigger) tomorrow"
                    if s.entry_type == "buy above" else
                    f"at tomorrow's open (skip if it opens above ₹{s.entry + s.extra.get('atr', 0):,.2f})")
                 + f". Immediately set a GTT **OCO**: stop ₹{s.stop:,.2f} / target ₹{s.target2:,.2f}. "
                 f"At ₹{s.target1:,.2f} sell half and move the stop to your entry.")
        st.markdown(place)


def render():
    ctx, rep = ui.current()
    cfg = ui.settings()
    ideas = rep["swing"]
    st.title("Swing Ideas")
    st.caption("Holding period days to a few weeks. Signals use the last close; act at the next session.")

    snap = rep["market"]
    if snap["label"] == "Bearish":
        st.error(f"Market regime is **Bearish** ({snap['score']:.0f}/100). In this phase most long setups fail. "
                 "Position sizes below are already cut to 25%. Best action is usually to wait.")
    elif snap["label"] == "Cautious":
        st.warning("Market regime is **Cautious** - sizes are halved; take only A-grade ideas.")

    st.subheader("What is actually working (ML-filtered backtest since 2022, after costs)")
    _edge_table(rep["edge"])
    st.caption("R = the amount you risk per trade. +0.20R means on average you make 20% of your risk per trade "
               "after brokerage, STT, GST, stamp duty and slippage. Setups marked 'No edge' are ranked lower. "
               "Only trades that passed the ML filter (≥30% odds, predicted out-of-sample) are counted - "
               "the same filter applied to today's ideas.")

    st.divider()
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    grades = c1.multiselect("Grade", ["A", "B", "C"], default=["A", "B"])
    setups = sorted({s.setup for s in ideas})
    pick = c2.multiselect("Setup", setups, default=setups)
    hide_noedge = c3.toggle("Only proven setups + ML filter", value=False,
                            help="Hide setups with no edge and ideas that fail the ML filter")
    edge_map = {r["setup"]: r["verdict"] for r in rep["edge"].to_dict("records")} if not rep["edge"].empty else {}
    shown = [s for s in ideas if s.grade in grades and s.setup in pick
             and not (hide_noedge and (edge_map.get(s.setup) == "No edge - avoid"
                                       or not s.extra.get("passes_ml_filter", True)))]
    if not ideas:
        st.info("No setups triggered on the last close. That's normal - good trades are not available every day.")
        return
    tbl = pd.DataFrame([{
        "Stock": s.symbol, "Setup": s.setup, "Grade": s.grade, "Score": round(s.score),
        "ML rank": None if s.ml_prob is None else round(s.ml_prob),
        "ML filter": "✓" if s.extra.get("passes_ml_filter", True) else "✗",
        "AI": (s.ai or {}).get("verdict", "–"), "News": None if s.news_sentiment is None else round(s.news_sentiment, 2),
        "Close": s.close, "Entry": s.entry, "Stop": s.stop, "Target 2": s.target2,
        "Qty": s.qty, "Capital ₹": s.capital_needed, "Risk ₹": s.risk_amount, "Sector": s.sector,
    } for s in shown])
    if not tbl.empty:
        st.dataframe(tbl, use_container_width=True, hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                                    "ML rank": st.column_config.ProgressColumn("ML rank", min_value=0, max_value=100, format="%d"),
                                    "Close": st.column_config.NumberColumn(format="₹%.2f"),
                                    "Entry": st.column_config.NumberColumn(format="₹%.2f"),
                                    "Stop": st.column_config.NumberColumn(format="₹%.2f"),
                                    "Target 2": st.column_config.NumberColumn(format="₹%.2f"),
                                    "Capital ₹": st.column_config.NumberColumn(format="₹%d"),
                                    "Risk ₹": st.column_config.NumberColumn(format="₹%d")})
    st.subheader(f"Idea details ({len(shown)})")
    for s in shown[:25]:
        _idea(s, ctx, cfg["account"]["broker"])

    st.divider()
    st.subheader("ML model's top-ranked stocks (no setup needed)")
    st.caption("Stocks the model expects to beat the rest of the universe over the next 10 trading days. "
               "Use as a shortlist to look for entries, not as buy signals on their own.")
    mt = rep["ml_top"]
    if mt is not None and len(mt):
        show = mt.reset_index().rename(columns={"index": "ticker"})
        show["ticker"] = show["ticker"].str.removesuffix(".NS")
        st.dataframe(show[["ticker", "name", "sector", "ml_rank", "close", "above_200dma"]],
                     use_container_width=True, hide_index=True,
                     column_config={"ml_rank": st.column_config.ProgressColumn("ML rank", min_value=0, max_value=100, format="%.0f"),
                                    "close": st.column_config.NumberColumn("Close", format="₹%.2f"),
                                    "above_200dma": st.column_config.CheckboxColumn("Above 200 DMA")})
    else:
        st.info("Train the ML model on the ML Model page to see rankings.")
