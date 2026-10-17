"""Indian Stock Advisor - dashboard entry point.

Run:  .venv/bin/streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Indian Stock Advisor", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

from dashboard import (p_analyzer, p_backtest, p_guide, p_intraday, p_longterm,  # noqa: E402
                       p_market, p_ml, p_portfolio, p_swing, ui)
from stockadvisor import ai  # noqa: E402

ui.setup_page()
cfg = ui.settings()

with st.sidebar:
    st.markdown("### 📈 Indian Stock Advisor")
    st.caption("NSE research, signals and backtests")
    st.selectbox("Universe", ["nifty50", "nifty100", "nifty200", "nifty500"],
                 index=["nifty50", "nifty100", "nifty200", "nifty500"].index(cfg["universe"]["index"]),
                 key="universe", help="Bigger universes find more ideas but take longer to load.")
    has_ai = ai.available()
    st.toggle("Claude AI analyst", value=has_ai, key="use_ai", disabled=not has_ai,
              help="Uses your ANTHROPIC_API_KEY. Responses are cached for 12 hours to save credits.")
    if not has_ai:
        st.caption("Add ANTHROPIC_API_KEY to .env to enable AI analysis.")
    if st.button("🔄 Refresh data", use_container_width=True,
                 help="Re-download today's prices and rebuild the report"):
        st.session_state["bust"] = st.session_state.get("bust", 0) + 1
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    a = cfg["account"]
    st.caption(f"Capital {ui.inr(a['capital'])} · risk {a['risk_per_trade_pct']}%/trade · "
               f"{a['broker'].title()} costs · edit config.toml to change")
    ui.disclaimer()

pages = [
    st.Page(p_market.render, title="Market Today", icon="🧭", url_path="market", default=True),
    st.Page(p_swing.render, title="Swing Ideas", icon="🎯", url_path="swing"),
    st.Page(p_longterm.render, title="Long-Term Portfolio", icon="🏦", url_path="longterm"),
    st.Page(p_intraday.render, title="Intraday", icon="⚡", url_path="intraday"),
    st.Page(p_analyzer.render, title="Stock Analyzer", icon="🔍", url_path="analyzer"),
    st.Page(p_portfolio.render, title="My Portfolio", icon="💼", url_path="portfolio"),
    st.Page(p_backtest.render, title="Backtest Lab", icon="🧪", url_path="backtest"),
    st.Page(p_ml.render, title="ML Model", icon="🤖", url_path="ml"),
    st.Page(p_guide.render, title="Playbook & Tips", icon="📚", url_path="guide"),
]
st.navigation(pages).run()
