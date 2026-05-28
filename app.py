import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data.loader import load_market_data
from src.data.features import prepare_ml_dataset
from src.data.eda import class_balance
from src.models.train import load_model, predict_proba, THRESHOLD

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
        st.info("Overview tab — coming soon")
    with tab2:
        st.info("Model Performance tab — coming soon")
    with tab3:
        st.info("Strategy Backtest tab — coming soon")
    with tab4:
        st.info("AI Assistant tab — coming soon")


if __name__ == "__main__":
    main()
