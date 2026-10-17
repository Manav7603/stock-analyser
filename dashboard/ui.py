"""Shared look-and-feel, cached loaders and chart builders for the dashboard."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockadvisor import ai
from stockadvisor.advisor import build_report
from stockadvisor.config import load_config
from stockadvisor.market import load_context

UP, DOWN, ACCENT, MUTED = "#22c55e", "#ef4444", "#60a5fa", "#94a3b8"
AMBER, PURPLE = "#f59e0b", "#a78bfa"
GRID = "rgba(148,163,184,0.12)"

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
[data-testid="stMetricLabel"] p {color: #94a3b8; font-size: 0.82rem;}
.sa-card {background: #111a2e; border: 1px solid #1f2a44; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;}
.sa-hero {background: linear-gradient(135deg, #10203d 0%, #0d1628 60%); border: 1px solid #1f2a44;
          border-radius: 16px; padding: 18px 22px; margin-bottom: 14px;}
.sa-hero h3 {margin: 0 0 6px 0; font-size: 1.25rem;}
.sa-muted {color: #94a3b8; font-size: 0.88rem;}
.sa-badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
           font-weight: 600; margin-right: 6px; border: 1px solid transparent;}
.sa-green {background: rgba(34,197,94,.14); color: #4ade80; border-color: rgba(34,197,94,.35);}
.sa-red {background: rgba(239,68,68,.14); color: #f87171; border-color: rgba(239,68,68,.35);}
.sa-amber {background: rgba(245,158,11,.14); color: #fbbf24; border-color: rgba(245,158,11,.35);}
.sa-blue {background: rgba(96,165,250,.14); color: #93c5fd; border-color: rgba(96,165,250,.35);}
.sa-grey {background: rgba(148,163,184,.12); color: #cbd5e1; border-color: rgba(148,163,184,.3);}
.sa-list {margin: 4px 0 0 0; padding-left: 0; list-style: none;}
.sa-list li {padding: 3px 0 3px 22px; position: relative; font-size: 0.92rem; line-height: 1.4;}
.sa-list li:before {position: absolute; left: 0; top: 3px;}
.sa-good li:before {content: "✓"; color: #4ade80; font-weight: 700;}
.sa-bad li:before {content: "!"; color: #fbbf24; font-weight: 800; left: 5px;}
.sa-tip li:before {content: "→"; color: #93c5fd;}
.sa-news {padding: 8px 0; border-bottom: 1px solid #1f2a44; font-size: 0.9rem;}
.sa-news a {color: #e2e8f0; text-decoration: none;}
.sa-news a:hover {color: #93c5fd;}
.sa-level {display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed #1f2a44; font-size: 0.9rem;}
.sa-disclaimer {font-size: 0.75rem; color: #64748b; line-height: 1.35;}
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def badge(text: str, tone: str = "grey") -> str:
    return f'<span class="sa-badge sa-{tone}">{esc(text)}</span>'


def tone_for_grade(g: str) -> str:
    return {"A": "green", "B": "blue", "C": "grey"}.get(g, "grey")


def tone_for_verdict(v: str) -> str:
    return {"Strong setup": "green", "Good setup": "blue", "Wait": "amber", "Avoid": "red",
            "Proven edge": "green", "Marginal": "amber", "No edge - avoid": "red"}.get(v, "grey")


def tone_for_regime(label: str) -> str:
    return {"Bullish": "green", "Neutral-positive": "blue", "Cautious": "amber", "Bearish": "red"}.get(label, "grey")


def bullet_list(items, kind: str = "good") -> str:
    if not items:
        return ""
    return f'<ul class="sa-list sa-{kind}">' + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def inr(x, dec: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"₹{x:,.{dec}f}"


def sentiment_badge(s: float | None) -> str:
    if s is None:
        return badge("no news", "grey")
    if s > 0.15:
        return badge(f"positive {s:+.2f}", "green")
    if s < -0.15:
        return badge(f"negative {s:+.2f}", "red")
    return badge(f"neutral {s:+.2f}", "grey")


def news_html(items: list[dict], limit: int = 8) -> str:
    if not items:
        return '<div class="sa-muted">No relevant headlines in the lookback window.</div>'
    rows = []
    for n in items[:limit]:
        tags = "".join(badge(e, "blue") for e in n.get("events", []))
        rows.append(f'<div class="sa-news">{sentiment_badge(n.get("sentiment"))}{tags}'
                    f'<a href="{esc(n.get("link", "#"))}" target="_blank">{esc(n["title"])}</a>'
                    f'<div class="sa-muted">{esc(n.get("source", ""))} · {esc(n.get("published", "")[:10])}</div></div>')
    return "".join(rows)


# --------------------------------------------------------------- loaders

def settings() -> dict:
    return load_config()


@st.cache_resource(ttl=3600, show_spinner="Loading prices and computing indicators…")
def get_ctx(index: str, _bust: int = 0):
    return load_context(index)


@st.cache_resource(ttl=3600, show_spinner="Building today's report (signals, ML, news, AI)…")
def get_report(index: str, use_ai: bool, asof: str, _ctx, _bust: int = 0):
    return build_report(_ctx, use_ai=use_ai)


def current() -> tuple:
    """(ctx, report) for the universe chosen in the sidebar."""
    idx = st.session_state.get("universe", settings()["universe"]["index"])
    bust = st.session_state.get("bust", 0)
    ctx = get_ctx(idx, bust)
    rep = get_report(idx, st.session_state.get("use_ai", ai.available()), str(ctx.asof.date()), ctx, bust)
    return ctx, rep


# ---------------------------------------------------------------- charts

def _layout(fig: go.Figure, height: int = 520) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
                      hovermode="x unified", xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor=GRID, showspikes=True, spikemode="across", spikethickness=1)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def price_chart(f: pd.DataFrame, title: str = "", months: int = 9, levels: dict | None = None,
                panels: tuple = ("volume", "rsi"), show_bb: bool = False, show_st: bool = False) -> go.Figure:
    d = f.iloc[-int(months * 21):]
    rows = 1 + len(panels)
    heights = [0.62] + [0.38 / max(len(panels), 1)] * len(panels)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    intraday = bool((d.index.hour != 0).any())
    x = d.index.strftime("%d %b %H:%M" if intraday else "%Y-%m-%d")
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN,
                                 increasing_fillcolor=UP, decreasing_fillcolor=DOWN), 1, 1)
    for col, color, name in (("ema20", ACCENT, "EMA 20"), ("ema50", AMBER, "EMA 50"), ("sma200", PURPLE, "SMA 200")):
        if col in d:
            fig.add_trace(go.Scatter(x=x, y=d[col], name=name, line=dict(color=color, width=1.4)), 1, 1)
    if show_bb and "bb_up" in d:
        fig.add_trace(go.Scatter(x=x, y=d["bb_up"], name="BB upper", line=dict(color=MUTED, width=0.8, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=d["bb_lo"], name="BB lower", line=dict(color=MUTED, width=0.8, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(148,163,184,0.05)"), 1, 1)
    if show_st and "st_line" in d:
        up = d["st_line"].where(d["st_dir"] == 1)
        dn = d["st_line"].where(d["st_dir"] == -1)
        fig.add_trace(go.Scatter(x=x, y=up, name="Supertrend ↑", line=dict(color=UP, width=1, dash="dash")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=dn, name="Supertrend ↓", line=dict(color=DOWN, width=1, dash="dash")), 1, 1)
    if levels:
        styles = {"Entry": (ACCENT, "solid"), "Stop": (DOWN, "dash"), "Target 1": (UP, "dot"), "Target 2": (UP, "dash")}
        for k, v in levels.items():
            if v:
                c, dash = styles.get(k, (MUTED, "dot"))
                fig.add_hline(y=v, line=dict(color=c, width=1.3, dash=dash), row=1, col=1,
                              annotation_text=f"{k} {v:,.2f}", annotation_position="right",
                              annotation_font=dict(color=c, size=11))
    r = 2
    for p in panels:
        if p == "volume":
            colors = np.where(d["close"] >= d["open"], "rgba(34,197,94,.55)", "rgba(239,68,68,.55)")
            fig.add_trace(go.Bar(x=x, y=d["volume"], name="Volume", marker_color=colors, showlegend=False), r, 1)
            if "vol20" in d:
                fig.add_trace(go.Scatter(x=x, y=d["vol20"], name="Vol 20D avg", line=dict(color=MUTED, width=1),
                                         showlegend=False), r, 1)
            fig.update_yaxes(title_text="Vol", row=r, col=1)
        elif p == "rsi":
            fig.add_trace(go.Scatter(x=x, y=d["rsi14"], name="RSI 14", line=dict(color=PURPLE, width=1.3),
                                     showlegend=False), r, 1)
            for lvl, c in ((70, DOWN), (30, UP)):
                fig.add_hline(y=lvl, line=dict(color=c, width=0.8, dash="dot"), row=r, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)
        elif p == "macd":
            colors = np.where(d["hist"] >= 0, "rgba(34,197,94,.6)", "rgba(239,68,68,.6)")
            fig.add_trace(go.Bar(x=x, y=d["hist"], marker_color=colors, name="MACD hist", showlegend=False), r, 1)
            fig.add_trace(go.Scatter(x=x, y=d["macd"], line=dict(color=ACCENT, width=1), name="MACD", showlegend=False), r, 1)
            fig.add_trace(go.Scatter(x=x, y=d["signal"], line=dict(color=AMBER, width=1), name="Signal", showlegend=False), r, 1)
            fig.update_yaxes(title_text="MACD", row=r, col=1)
        r += 1
    fig.update_xaxes(type="category", nticks=10)
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    fig = _layout(fig, 560 if len(panels) >= 2 else 460)
    if levels:
        fig.update_layout(margin=dict(r=120))
    return fig


def equity_chart(eq: pd.Series, bench: pd.Series | None = None, title: str = "Equity curve") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy", line=dict(color=UP, width=2)), 1, 1)
    if bench is not None and len(bench):
        fig.add_trace(go.Scatter(x=bench.index, y=bench, name="NIFTY 50 (buy & hold)",
                                 line=dict(color=MUTED, width=1.5, dash="dot")), 1, 1)
    dd = (eq / eq.cummax() - 1) * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown %", fill="tozeroy",
                             line=dict(color=DOWN, width=1), fillcolor="rgba(239,68,68,.25)"), 2, 1)
    fig.update_yaxes(title_text="₹", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return _layout(fig, 460)


def gauge(score: float, label: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix="/100", font=dict(size=30)),
        title=dict(text=label, font=dict(size=16)),
        gauge=dict(axis=dict(range=[0, 100], tickwidth=0), bar=dict(color="#e2e8f0", thickness=0.25),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 30], color="rgba(239,68,68,.45)"),
                          dict(range=[30, 50], color="rgba(245,158,11,.45)"),
                          dict(range=[50, 70], color="rgba(96,165,250,.45)"),
                          dict(range=[70, 100], color="rgba(34,197,94,.45)")])))
    fig.update_layout(template="plotly_dark", height=230, margin=dict(l=20, r=20, t=50, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=0))
    return fig


def heatmap_monthly(pv: pd.DataFrame) -> go.Figure:
    z = pv.values.astype(float)
    text = [[("" if np.isnan(v) else f"{v:.1f}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(pv.columns), y=[str(i) for i in pv.index], text=text,
                               texttemplate="%{text}", colorscale=[[0, DOWN], [0.5, "#1e293b"], [1, UP]],
                               zmid=0, showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 60 + 34 * len(pv))


def disclaimer() -> None:
    st.markdown('<div class="sa-disclaimer">Decision support only - not investment advice and not a SEBI-registered '
                'adviser. Backtests use today\'s index members (survivorship bias) and past results do not '
                'guarantee future returns. You place and own every trade.</div>', unsafe_allow_html=True)
