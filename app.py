import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data.loader import load_market_data
from src.data.features import prepare_ml_dataset
from src.data.eda import class_balance
from src.models.train import load_model, predict_proba, get_feature_importance, cross_validate_models, THRESHOLD
from src.data.eda import crash_frequency_by_year

st.set_page_config(page_title="Market Anomaly Detector", page_icon="📉", layout="wide")


@st.cache_data(show_spinner="Downloading market data…")
def get_raw_data():
    return load_market_data(start="2000-01-01")


@st.cache_data(show_spinner="Engineering features…")
def get_ml_dataset(_raw):
    return prepare_ml_dataset(_raw)


@st.cache_resource(show_spinner="Loading model…")
def get_artifact():
    try:
        return load_model()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Computing predictions…")
def get_predictions(_df, _artifact):
    return predict_proba(_artifact, _df)


def render_sidebar(df, artifact):
    with st.sidebar:
        st.title("📉 Anomaly Detector")
        st.caption("S&P 500 Market Crash Early Warning")
        st.divider()
        st.subheader("Data")
        st.metric("Trading Days", f"{len(df):,}")
        bal = class_balance(df)
        st.metric("Crash Rate (hist.)", f"{bal['crash_rate']:.1%}")
        st.divider()
        st.subheader("Model")
        if artifact:
            st.success(f"Loaded: **{artifact['model_type']}**")
        else:
            st.error("No trained model found.")
            st.info("Run `python train_model.py` first.")
        st.divider()
        threshold = st.slider(
            "Crash probability threshold",
            min_value=0.10, max_value=0.70,
            value=float(artifact["threshold"]) if artifact else 0.35,
            step=0.05,
        )
    return threshold


def _gauge(value: float, title: str) -> go.Figure:
    color = "#e63946" if value >= 0.5 else "#f4a261" if value >= 0.3 else "#2a9d8f"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#d8f3dc"},
                {"range": [30, 60], "color": "#ffd166"},
                {"range": [60, 100], "color": "#f4846b"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 35},
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
    return fig


@st.cache_data(show_spinner="Running cross-validation (may take ~60s)…")
def _get_cv_results(_df):
    return cross_validate_models(_df)


def _tab_model_performance(df, artifact):
    if artifact is None:
        st.warning("Train the model first: `python train_model.py`")
        return

    st.subheader("Cross-Validation Results (5-fold TimeSeriesSplit)")
    st.caption("SMOTE applied only inside training folds — no data leakage.")

    cv = _get_cv_results(df)
    display = cv.copy()
    for col in display.columns:
        if display[col].dtype == float:
            display[col] = display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    st.dataframe(display, use_container_width=True)
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("XGBoost Feature Importance")
        fi = get_feature_importance(artifact)
        if fi is not None:
            fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                         labels={"x": "Importance", "y": "Feature"},
                         color=fi.values, color_continuous_scale="Blues")
            fig.update_layout(height=450, showlegend=False, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    with col_b:
        st.subheader("Crash Days per Year")
        freq = crash_frequency_by_year(df)
        fig2 = px.bar(x=freq.index, y=freq.values,
                      labels={"x": "Year", "y": "Crash Days"},
                      color=freq.values, color_continuous_scale="Reds")
        fig2.update_layout(height=450, showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.info("**ROC-AUC** measures overall discrimination. "
            "**Average Precision** (PR-AUC) is more informative for imbalanced classes.")


def _tab_overview(df, artifact, probs, threshold):
    if artifact is None:
        st.warning("Train the model first: `python train_model.py`")
        return

    latest = df.iloc[-1]
    latest_prob = float(probs[-1])
    signal = "CRASH WARNING" if latest_prob >= threshold else "LOW RISK"
    signal_color = "red" if latest_prob >= threshold else "green"

    st.markdown(f"## Latest Signal: :{signal_color}[{signal}]")
    st.caption(f"As of {df.index[-1].strftime('%A, %B %d %Y')}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.plotly_chart(_gauge(latest_prob, "Crash Probability"), use_container_width=True)
    with col2:
        st.metric("S&P 500 Close", f"${latest['Close']:,.0f}")
        st.metric("Today's Return", f"{latest['return_1d']:.2%}")
    with col3:
        vix_val = latest.get("VIX")
        st.metric("VIX", f"{vix_val:.1f}" if pd.notna(vix_val) else "N/A")
        vol_val = latest.get("vol_21d")
        st.metric("21d Volatility", f"{vol_val:.1%}" if pd.notna(vol_val) else "N/A")
    with col4:
        rsi_val = latest.get("rsi_14")
        st.metric("RSI (14)", f"{rsi_val:.1f}" if pd.notna(rsi_val) else "N/A")
        macd_val = latest.get("macd_hist")
        st.metric("MACD Hist", f"{macd_val:.2f}" if pd.notna(macd_val) else "N/A")
    with col5:
        sma_val = latest.get("price_sma200_ratio")
        st.metric("SMA200 Distance", f"{sma_val:.1%}" if pd.notna(sma_val) else "N/A")
        bb_val = latest.get("bb_pos")
        st.metric("BB Position", f"{bb_val:.2f}" if pd.notna(bb_val) else "N/A")

    st.divider()
    lookback = 252 * 3
    df_plot = df.iloc[-lookback:].copy()
    probs_plot = probs[-lookback:]
    crash_mask = df_plot["crash"] == 1
    high_risk_mask = probs_plot >= threshold

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], mode="lines",
                             name="S&P 500", line=dict(color="#1d3557", width=1.5)))
    fig.add_trace(go.Scatter(x=df_plot.index[crash_mask], y=df_plot["Close"][crash_mask],
                             mode="markers", name="Actual Crash",
                             marker=dict(color="red", size=6, symbol="x")))
    fig.add_trace(go.Scatter(x=df_plot.index[high_risk_mask], y=df_plot["Close"][high_risk_mask],
                             mode="markers", name="Predicted Crash",
                             marker=dict(color="orange", size=5, symbol="circle-open")))
    fig.update_layout(title="S&P 500 — Actual vs Predicted Crash Days (Last 3 Years)",
                      height=420, legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=60))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_plot.index, y=probs_plot, mode="lines",
                              name="Crash Probability", line=dict(color="#e63946", width=1),
                              fill="tozeroy", fillcolor="rgba(230,57,70,0.1)"))
    fig2.add_hline(y=threshold, line_dash="dash", line_color="orange",
                   annotation_text=f"Threshold ({threshold:.0%})")
    fig2.update_layout(title="Model Crash Probability (Last 3 Years)",
                       yaxis=dict(range=[0, 1]), height=280, margin=dict(t=50, b=40))
    st.plotly_chart(fig2, use_container_width=True)


def main():
    raw = get_raw_data()
    df = get_ml_dataset(raw)
    artifact = get_artifact()
    threshold = render_sidebar(df, artifact)
    probs = get_predictions(df, artifact) if artifact else None

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "🤖 Model Performance",
        "💼 Strategy Backtest", "💬 AI Assistant",
    ])

    with tab1:
        _tab_overview(df, artifact, probs, threshold)
    with tab2:
        _tab_model_performance(df, artifact)
    with tab3:
        st.info("Strategy Backtest tab — coming soon")
    with tab4:
        st.info("AI Assistant tab — coming soon")


if __name__ == "__main__":
    main()
