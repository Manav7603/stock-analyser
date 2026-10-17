"""Playbook & Tips: how to use the tool and the rules that keep traders solvent."""

import streamlit as st

from dashboard import ui
from stockadvisor.costs import trade_cost


def render():
    cfg = ui.settings()
    st.title("Playbook & Tips")

    st.subheader("Daily routine (15 minutes)")
    st.markdown(ui.bullet_list([
        "After 4 PM or before 9 AM: click Refresh, read Market Today. The regime decides how much to trade.",
        "Swing Ideas: shortlist A/B grades whose setup shows a proven edge and whose AI/news view is not 'Avoid'.",
        "Open each shortlisted idea and read the risks. Skip anything with results due within a week unless you want that gamble.",
        "Place the order with the exact quantity shown, then immediately place a GTT OCO (stop + target) in Kite / Dhan / Groww.",
        "Log the trade in My Portfolio. Each evening, follow the action shown (hold / trail / book half / exit).",
        "Month start: rebalance the long-term portfolio.",
    ], "tip"), unsafe_allow_html=True)

    st.subheader("The rules that matter more than any signal")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk**" + ui.bullet_list([
            f"Risk at most {cfg['account']['risk_per_trade_pct']}% of capital per trade. 10 losses in a row then cost ~10%, not 50%.",
            "The stop-loss is placed before you think about the target. Never move it down.",
            "Daily loss limit: after 2 stopped-out trades in a day, stop trading for the day.",
            "Max ~8 swing positions and ≤3 in one sector - correlated positions are one big position.",
            "In a Bearish regime, cash is a position. Most money is lost fighting the market trend.",
        ], "bad"), unsafe_allow_html=True)
    with c2:
        st.markdown("**Execution**" + ui.bullet_list([
            "Don't chase: if a stock opens more than 1 ATR above the entry, skip it.",
            "Book half at Target 1 and move the stop to cost - you can no longer lose on that trade.",
            "Let the rest run with a trailing stop; a few big winners pay for many small losers.",
            "Avoid entries 1-2 days before results, budget, RBI policy or big global events.",
            "Avoid stocks with turnover below ₹20 crore/day - slippage eats the edge.",
        ], "good"), unsafe_allow_html=True)

    st.subheader("Why news is part of the decision")
    st.markdown(ui.bullet_list([
        "Results: a stock can gap 5-15% on the numbers - no stop-loss protects you from a gap.",
        "Rating changes and target revisions move large-caps for days; upgrades support breakouts, downgrades kill them.",
        "Regulatory action (SEBI, RBI, tax raids), promoter pledging or selling are reasons to skip a good chart.",
        "Order wins, capacity expansion and strong guidance are catalysts that help breakouts follow through.",
        "The AI analyst reads the headlines for each idea and tells you whether the news supports or contradicts the chart.",
    ], "tip"), unsafe_allow_html=True)

    st.subheader("What it really costs to trade")
    b = cfg["account"]["broker"]
    rows = []
    for v in (25000, 100000, 500000):
        d = trade_cost(v, v * 1.02, False, b)
        i = trade_cost(v, v * 1.005, True, b)
        rows.append({"Position": ui.inr(v), "Delivery round trip": ui.inr(d["total"], 0),
                     "Delivery %": f"{d['total'] / v * 100:.2f}%", "Intraday round trip": ui.inr(i["total"], 0),
                     "Intraday %": f"{i['total'] / v * 100:.3f}%"})
    st.table(rows)
    st.caption("STT (0.1% each side on delivery) is the biggest cost. Frequent small trades pay it again and again.")

    st.subheader("Tax basics (FY 2025-26 rules)")
    st.markdown(ui.bullet_list([
        "Short-term capital gains (held ≤ 12 months, delivery): 20%.",
        "Long-term capital gains (> 12 months): 12.5% on gains above ₹1.25 lakh per year.",
        "Intraday profits are speculative business income, taxed at your slab rate; F&O is non-speculative business income.",
        "Keep your broker's tax P&L statement; consult a CA if you trade actively.",
    ], "tip"), unsafe_allow_html=True)

    st.subheader("Limits of this tool")
    st.markdown(ui.bullet_list([
        "Data comes from Yahoo Finance (free, end-of-day for daily; ~15-minute bars for intraday). Verify prices in your broker app before ordering.",
        "Backtests use today's index members, so failed companies that left the index are missing - real returns are lower.",
        "Fundamentals are a current snapshot and are not used in backtests.",
        "No model predicts the market. The edge is small and shows up over many trades, not on any one trade.",
    ], "bad"), unsafe_allow_html=True)
    ui.disclaimer()
