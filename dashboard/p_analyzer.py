"""Stock Analyzer: deep dive on any stock - chart, signals, ML view, fundamentals, news, AI."""

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import ui
from stockadvisor import ai, data, indicators, ml, news
from stockadvisor.fundamentals import days_to_results, get_fundamentals
from stockadvisor.strategies import swing
from stockadvisor.strategies.base import position_size


def _load_any(symbol: str, ctx):
    t = symbol.upper().strip()
    t = t if t.endswith(".NS") or t.startswith("^") else t + ".NS"
    if t in ctx.frames:
        return t, ctx.frames[t]
    raw = data.get_one(t, years=3)
    if raw.empty or len(raw) < 60:
        return t, None
    raw, _ = data.fix_corporate_actions(raw)
    return t, indicators.add_all(raw, ctx.nifty["close"])


def _state_table(r: pd.Series) -> list[tuple[str, str, str]]:
    c = r["close"]

    def tone(ok):
        return "green" if ok else "red"
    out = [
        ("Long-term trend", "Up" if c > r["sma200"] else "Down", tone(c > r["sma200"])),
        ("Medium trend (50 EMA)", "Up" if c > r["ema50"] else "Down", tone(c > r["ema50"])),
        ("Short trend (20 EMA)", "Up" if c > r["ema20"] else "Down", tone(c > r["ema20"])),
        ("Trend strength (ADX)", f"{r['adx']:.0f} - " + ("strong" if r["adx"] > 25 else "weak" if r["adx"] < 18 else "moderate"),
         "green" if r["adx"] > 25 else "amber"),
        ("RSI 14", f"{r['rsi14']:.0f} - " + ("overbought" if r["rsi14"] > 70 else "oversold" if r["rsi14"] < 30 else "neutral"),
         "amber" if r["rsi14"] > 70 or r["rsi14"] < 30 else "grey"),
        ("MACD", "Bullish" if r["hist"] > 0 else "Bearish", tone(r["hist"] > 0)),
        ("Supertrend", "Buy" if r["st_dir"] == 1 else "Sell", tone(r["st_dir"] == 1)),
        ("vs NIFTY (3M)", f"{r.get('rs63', np.nan) * 100:+.1f}%", tone(r.get("rs63", 0) > 0)),
        ("From 52W high", f"{r['dist_52wh'] * 100:.1f}%", "green" if r["dist_52wh"] > -0.1 else "grey"),
        ("Volume vs avg", f"{r['vol_ratio']:.1f}x", "blue" if r["vol_ratio"] > 1.5 else "grey"),
        ("Daily volatility (ATR)", f"{r['atr_pct']:.1f}%", "grey"),
    ]
    return out


def render():
    ctx, rep = ui.current()
    cfg = ui.settings()
    st.title("Stock Analyzer")
    opts = sorted(t.removesuffix(".NS") for t in ctx.frames)
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("Stock from universe", opts, index=opts.index("RELIANCE") if "RELIANCE" in opts else 0)
    other = c2.text_input("…or any NSE symbol", placeholder="e.g. IRCTC")
    symbol = other.strip().upper() or pick
    t, f = _load_any(symbol, ctx)
    if f is None:
        st.error(f"No price data for {symbol}. Check the NSE symbol.")
        return
    r = f.iloc[-1]
    name = ctx.name_of(t)
    st.subheader(f"{symbol} · {name}")
    k = st.columns(5)
    k[0].metric("Close", ui.inr(r["close"], 2), f"{f['close'].pct_change().iloc[-1] * 100:+.2f}%")
    k[1].metric("1 month", f"{r['ret21'] * 100:+.1f}%")
    k[2].metric("6 months", f"{r['ret126'] * 100:+.1f}%")
    k[3].metric("1 year", f"{r['ret252'] * 100:+.1f}%" if pd.notna(r["ret252"]) else "–")
    k[4].metric("52W range", f"{r['ll252']:,.0f} – {r['hh252']:,.0f}")

    months = st.select_slider("Chart period", [3, 6, 9, 12, 24], value=9, format_func=lambda m: f"{m}M")
    c1, c2, c3 = st.columns(3)
    bb = c1.toggle("Bollinger bands", value=False)
    stl = c2.toggle("Supertrend", value=True)
    macd_on = c3.toggle("MACD panel", value=True)
    panels = ("volume", "rsi", "macd") if macd_on else ("volume", "rsi")
    st.plotly_chart(ui.price_chart(f, "", months=months, panels=panels, show_bb=bb, show_st=stl),
                    use_container_width=True)

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("**Technical state**")
        st.markdown("".join(f'<div class="sa-level"><span>{k}</span>{ui.badge(v, tone)}</div>'
                            for k, v, tone in _state_table(r)), unsafe_allow_html=True)
        masks = swing.setup_masks(f, 0)
        active = [s for s, m in masks.items() if m.iloc[-1]]
        recent = [s for s, m in masks.items() if m.tail(5).any() and s not in active]
        st.markdown("**Setups**  " + (" ".join(ui.badge(s + " (today)", "green") for s in active)
                                      + " ".join(ui.badge(s + " (last 5 days)", "blue") for s in recent)
                                      or ui.badge("none active", "grey")), unsafe_allow_html=True)
        model, meta = ml.load()
        if model is not None and t in ctx.frames:
            mlp = ml.predict_latest(ctx)
            if t in mlp.index:
                rank = mlp.at[t, "ml_rank"]
                st.markdown(f"**ML rank:** {ui.badge(f'{rank:.0f} / 100', 'green' if rank >= 70 else 'red' if rank <= 30 else 'grey')}"
                            " - odds of beating the universe over the next 10 days", unsafe_allow_html=True)
                drivers = ml.explain_prediction(model, mlp.loc[[t]])
                st.markdown(ui.bullet_list([f"{n}: {'raises' if v > 0 else 'lowers'} the odds ({v * 100:+.1f} pts)"
                                            for n, v in drivers], "tip"), unsafe_allow_html=True)
        # sizing helper
        st.markdown("**Position size calculator**")
        a = cfg["account"]
        c1, c2 = st.columns(2)
        entry = c1.number_input("Entry", value=float(round(r["close"], 2)), step=0.05)
        stop = c2.number_input("Stop", value=float(round(r["close"] - 2 * r["atr14"], 2)), step=0.05)
        qty = position_size(a["capital"], a["risk_per_trade_pct"], entry, stop, a["max_position_pct"],
                            rep["market"]["exposure"])
        st.markdown(f"Buy **{qty:,}** shares · capital {ui.inr(qty * entry)} · max loss {ui.inr(qty * abs(entry - stop))} "
                    f"(regime size x{rep['market']['exposure']:.2f})")
    with right:
        fd = get_fundamentals([t]).get(t, {})
        dtr = days_to_results(fd)
        st.markdown("**Fundamentals**")
        fcols = st.columns(4)
        fcols[0].metric("P/E", f"{fd['trailingPE']:.1f}" if fd.get("trailingPE") else "–")
        fcols[1].metric("ROE", f"{fd['returnOnEquity'] * 100:.0f}%" if fd.get("returnOnEquity") else "–")
        fcols[2].metric("Debt/Equity", f"{fd['debtToEquity']:.0f}" if fd.get("debtToEquity") is not None else "–")
        fcols[3].metric("Earnings growth", f"{fd['earningsGrowth'] * 100:+.0f}%" if fd.get("earningsGrowth") is not None else "–")
        meta_line = []
        if fd.get("recommendationKey"):
            meta_line.append(f"Analysts: {fd['recommendationKey']} ({fd.get('numberOfAnalystOpinions') or '?'} opinions)")
        if fd.get("targetMeanPrice"):
            meta_line.append(f"avg target {ui.inr(fd['targetMeanPrice'])} ({(fd['targetMeanPrice'] / r['close'] - 1) * 100:+.0f}%)")
        if dtr is not None:
            meta_line.append(f"results in {dtr} days")
        st.markdown(f'<div class="sa-muted">{" · ".join(meta_line)}</div>', unsafe_allow_html=True)
        items = news.stock_news(symbol, name, cfg["news"]["lookback_days"], cfg["news"]["max_items"])
        summ = news.summarise(items)
        st.markdown(f"**News** {ui.sentiment_badge(summ['sentiment'] if summ['count'] else None)} "
                    + " ".join(ui.badge(f"{k} ×{v}", "blue") for k, v in summ["events"].items()),
                    unsafe_allow_html=True)
        st.markdown(ui.news_html(items, 8), unsafe_allow_html=True)

    st.divider()
    st.subheader("🤖 AI analyst")
    ai_ctx = {
        "symbol": symbol, "name": name, "sector": ctx.sector_of(t), "setup": ", ".join(active) or "none today",
        "close": float(r["close"]), "entry": float(entry), "stop": float(stop),
        "target1": float(entry + 1.5 * (entry - stop)), "target2": float(entry + 3 * (entry - stop)),
        "technical_state": {k: v for k, v, _ in _state_table(r)},
        "returns": {"1m_pct": round(r["ret21"] * 100, 1), "6m_pct": round(r["ret126"] * 100, 1)},
        "fundamentals": {k: fd.get(k) for k in ("trailingPE", "returnOnEquity", "debtToEquity", "earningsGrowth",
                                                  "revenueGrowth", "recommendationKey", "targetMeanPrice")},
        "days_to_results": dtr, "market": {"label": rep["market"]["label"], "score": rep["market"]["score"]},
        "news_summary": summ,
        "headlines": [{"title": n["title"], "source": n["source"], "date": n["published"][:10]} for n in items],
        "technical_score": 50, "reasons": [], "risks": [],
    }
    use_ai = st.session_state.get("use_ai", False)
    if st.button("Analyse with AI" if use_ai else "Analyse (rule-based)", type="primary"):
        st.session_state[f"ai_{t}"] = ai.stock_view(ai_ctx, use_ai=use_ai)
    v = st.session_state.get(f"ai_{t}")
    if v:
        st.markdown(f'<div class="sa-card">{ui.badge(v["verdict"], ui.tone_for_verdict(v["verdict"]))}'
                    f'{ui.badge("confidence " + str(v["confidence"]), "grey")}{ui.badge(v["news_view"], "blue")}'
                    f'<div style="margin-top:8px;line-height:1.5">{ui.esc(v["plain_english"])}</div>'
                    f'<div class="sa-muted" style="margin-top:6px">{ui.esc(v["news_summary"])}</div>'
                    f'<div class="sa-muted">{ui.esc(v.get("_source", ""))}</div></div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        a1.markdown("**Catalysts**" + ui.bullet_list(v["catalysts"], "good"), unsafe_allow_html=True)
        a2.markdown("**Risks**" + ui.bullet_list(v["risks"], "bad"), unsafe_allow_html=True)
        a3.markdown("**Watch / tips**" + ui.bullet_list(v["what_to_watch"] + v["tips"], "tip"), unsafe_allow_html=True)

    if use_ai:
        st.markdown("**Ask about this stock**")
        hist_key = f"chat_{t}"
        hist = st.session_state.setdefault(hist_key, [])
        for m in hist:
            st.chat_message(m["role"]).markdown(m["content"] if m["role"] == "assistant"
                                                else m["content"].split("</engine_data>")[-1].strip())
        q = st.chat_input(f"e.g. Is {symbol} a good buy before results?")
        if q:
            st.chat_message("user").markdown(q)
            with st.spinner("Thinking…"):
                ans = ai.ask(q, ai_ctx, [{"role": m["role"], "content": m["content"]} for m in hist[-6:]])
            hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
            st.chat_message("assistant").markdown(ans)
