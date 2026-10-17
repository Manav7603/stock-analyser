"""My Portfolio: log trades you placed, get hold/trail/exit advice, review your journal."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import ui
from stockadvisor import portfolio
from stockadvisor.fundamentals import get_fundamentals

TONE = {"EXIT - stop hit": "red", "Book profit": "green", "Book half + move stop to cost": "green",
        "Trail stop": "blue", "Review - below 200 DMA": "amber", "Consider exit (time)": "amber", "Hold": "grey"}


def render():
    ctx, rep = ui.current()
    cfg = ui.settings()
    broker = cfg["account"]["broker"]
    st.title("My Portfolio")
    st.caption("Log each order after you place it in your broker app. The advisor checks every position "
               "against today's close and tells you what to do.")

    with st.expander("➕ Add a position", expanded=not portfolio.positions()):
        ideas = {f"{s.symbol} · {s.setup}": s for s in rep["swing"]}
        pre = st.selectbox("Prefill from a swing idea (optional)", ["—"] + list(ideas))
        s = ideas.get(pre)
        with st.form("add"):
            c = st.columns(4)
            sym = c[0].text_input("Symbol", value=s.symbol if s else "")
            qty = c[1].number_input("Quantity", min_value=1, value=max(s.qty, 1) if s else 1)
            entry = c[2].number_input("Buy price", min_value=0.0, value=float(s.entry) if s else 0.0, step=0.05)
            stop = c[3].number_input("Stop-loss", min_value=0.0, value=float(s.stop) if s else 0.0, step=0.05)
            c = st.columns(4)
            target = c[0].number_input("Target (optional)", min_value=0.0, value=float(s.target2) if s else 0.0, step=0.05)
            horizon = c[1].selectbox("Type", ["swing", "longterm", "intraday"])
            d = c[2].date_input("Entry date")
            notes = c[3].text_input("Notes", value=s.setup if s else "")
            if st.form_submit_button("Save position", type="primary"):
                if not sym or entry <= 0 or stop <= 0 or stop >= entry:
                    st.error("Enter a symbol, a buy price, and a stop-loss below the buy price.")
                else:
                    portfolio.add(sym, qty, entry, stop, target or None, horizon, s.setup if s else "",
                                  str(d), notes)
                    st.success(f"Saved {sym.upper()}")
                    st.rerun()

    pos = portfolio.positions()
    if pos:
        fund = get_fundamentals([p["ticker"] for p in pos])
        ev = portfolio.evaluate(ctx, fund)
        invested = (ev["entry"] * ev["qty"]).sum()
        value = ev["value"].sum()
        k = st.columns(4)
        k[0].metric("Open positions", len(ev))
        k[1].metric("Invested", ui.inr(invested))
        k[2].metric("Current value", ui.inr(value), f"{(value / invested - 1) * 100:+.2f}%" if invested else None)
        k[3].metric("Open P&L", ui.inr(ev["pnl"].sum()))
        for _, r in ev.iterrows():
            st.markdown(
                f'<div class="sa-card"><div style="display:flex;justify-content:space-between;flex-wrap:wrap">'
                f'<div><b style="font-size:1.05rem">{ui.esc(r["symbol"])}</b> '
                f'{ui.badge(r["action"], TONE.get(r["action"], "grey"))}{ui.badge(r["horizon"], "grey")}'
                f'<span class="sa-muted">{r["qty"]} @ {ui.inr(r["entry"], 2)} · stop {ui.inr(r["stop"], 2)}'
                f'{" · target " + ui.inr(r["target"], 2) if r.get("target") else ""} · since {r["entry_date"]}</span></div>'
                f'<div><b style="color:{ui.UP if r["pnl"] >= 0 else ui.DOWN}">{ui.inr(r["pnl"])} '
                f'({r["pnl_pct"]:+.1f}%, {r["r_now"]:+.1f}R)</b> <span class="sa-muted">last {ui.inr(r["last"], 2)}</span></div></div>'
                + ui.bullet_list(r["advice"], "tip") + "</div>", unsafe_allow_html=True)
            c = st.columns([1, 1, 1, 1, 2])
            new_stop = c[0].number_input("New stop", value=float(round(r["suggested_stop"], 2)), step=0.05,
                                         key=f"ns_{r['id']}", label_visibility="collapsed")
            if c[1].button("Update stop", key=f"us_{r['id']}"):
                portfolio.update(r["id"], stop=new_stop)
                st.rerun()
            if c[2].button("Booked half", key=f"bh_{r['id']}"):
                portfolio.close(r["id"], r["last"], r["qty"] // 2, broker)
                portfolio.update(r["id"], half_booked=True, stop=max(r["stop"], r["entry"]))
                st.rerun()
            exit_px = c[3].number_input("Exit price", value=float(round(r["last"], 2)), step=0.05,
                                        key=f"ex_{r['id']}", label_visibility="collapsed")
            if c[4].button("Close position at this price", key=f"cl_{r['id']}"):
                portfolio.close(r["id"], exit_px, None, broker)
                st.rerun()
    else:
        st.info("No open positions yet.")

    st.subheader("Trade journal")
    cl = pd.DataFrame(portfolio.closed())
    if cl.empty:
        st.caption("Closed trades will appear here with costs and R-multiples so you can see what is working for you.")
        return
    wins = cl[cl["net_pnl"] > 0]
    k = st.columns(5)
    k[0].metric("Closed trades", len(cl))
    k[1].metric("Win rate", f"{len(wins) / len(cl) * 100:.0f}%")
    k[2].metric("Net P&L", ui.inr(cl["net_pnl"].sum()))
    k[3].metric("Avg R", f"{cl['r_multiple'].dropna().mean():+.2f}R" if cl["r_multiple"].notna().any() else "–")
    k[4].metric("Costs paid", ui.inr(cl["costs"].sum()))
    cl = cl.sort_values("exit_date")
    fig = go.Figure(go.Scatter(x=cl["exit_date"], y=cl["net_pnl"].cumsum(), mode="lines+markers",
                               line=dict(color=ui.ACCENT)))
    fig.update_layout(title="Cumulative realised P&L (₹)")
    st.plotly_chart(ui._layout(fig, 280), use_container_width=True)
    st.dataframe(cl[["symbol", "setup", "entry_date", "exit_date", "qty", "entry", "exit", "net_pnl", "costs",
                     "r_multiple"]].iloc[::-1], use_container_width=True, hide_index=True)
    if "setup" in cl and cl["setup"].astype(bool).any():
        st.markdown("**Your results by setup** - keep doing what works for *you*")
        st.dataframe(cl.groupby("setup").agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
                                             avg_r=("r_multiple", "mean")).reset_index(),
                     use_container_width=True, hide_index=True)
