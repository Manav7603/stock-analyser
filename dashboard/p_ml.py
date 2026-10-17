"""ML Model: honest out-of-sample evaluation, what the model looks at, retraining."""

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from dashboard import ui
from stockadvisor import ml
from stockadvisor.config import MODEL_DIR


def render():
    ctx, _ = ui.current()
    st.title("ML Model")
    st.caption("Gradient-boosted trees that rank stocks by their odds of beating the rest of the universe over "
               "the next 10 trading days. Every number below is out-of-sample: each year was predicted by a "
               "model trained only on earlier years.")
    model, meta = ml.load()
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("Retrain on current universe", type="primary", use_container_width=True):
            with st.spinner("Walk-forward training (about a minute for NIFTY 200)…"):
                ml.train(ctx)
            st.cache_resource.clear()
            st.rerun()
    if model is None:
        st.info("No model trained yet. Click Retrain.")
        return
    with c1:
        k = st.columns(4)
        k[0].metric("Mean out-of-sample AUC", f"{meta['mean_auc']:.3f}", "0.50 = coin flip", delta_color="off")
        k[1].metric("Top-10% excess return", f"{meta['mean_top10_excess_pct']:+.2f}%", "per 10 days vs median stock",
                    delta_color="off")
        k[2].metric("Years positive", f"{meta['years_positive']} / {meta['years_tested']}")
        k[3].metric("Training rows", f"{meta['rows']:,}", f"trained {meta['trained_at'][:10]}", delta_color="off")

    folds = pd.read_csv(MODEL_DIR / "swing_model_folds.csv")
    oos_p = MODEL_DIR / "swing_model_oos.parquet"
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=folds["year"].astype(str), y=folds["top10_excess_10d_pct"], name="Top 10% picks",
                    marker_color=ui.UP)
        fig.add_bar(x=folds["year"].astype(str), y=folds["bottom10_excess_10d_pct"], name="Bottom 10%",
                    marker_color=ui.DOWN)
        fig.update_layout(title="Excess 10-day return vs median stock, by test year (%)", barmode="group")
        st.plotly_chart(ui._layout(fig, 340), use_container_width=True)
    with c2:
        if oos_p.exists():
            dec = ml.decile_table(pd.read_parquet(oos_p))
            fig = go.Figure(go.Bar(x=dec["decile"].astype(str), y=dec["excess_10d_pct"],
                                   marker_color=[ui.UP if v > dec["excess_10d_pct"].median() else ui.MUTED
                                                 for v in dec["excess_10d_pct"]]))
            fig.update_layout(title="Excess return by model decile (1 = least likely, 10 = most likely)")
            st.plotly_chart(ui._layout(fig, 340), use_container_width=True)
    st.dataframe(folds.round(3), use_container_width=True, hide_index=True)

    imp = pd.read_csv(MODEL_DIR / "swing_model_importance.csv").head(15)
    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["label"], orientation="h", marker_color=ui.ACCENT))
    fig.update_layout(title="What the model pays attention to (permutation importance)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(ui._layout(fig, 460), use_container_width=True)

    st.markdown("**How to read this honestly**")
    st.markdown(ui.bullet_list([
        "An AUC of 0.53-0.56 sounds small, but in markets a small, consistent edge applied across many trades is "
        "what makes money. Anything claiming 70-80% accuracy on stock direction is overfit.",
        "The model is a ranking tool. Use it to prefer the best-ranked stocks among valid setups - not as a "
        "standalone buy signal.",
        "It does not try to time the market. An earlier version that did scored 0.50 out of sample, so market "
        "timing is left to the regime filter.",
        "Retrain monthly so it learns from the latest data. If the 'years positive' count drops, trust it less.",
    ], "tip"), unsafe_allow_html=True)
