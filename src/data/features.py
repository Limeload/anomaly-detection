import numpy as np
import pandas as pd

CRASH_THRESHOLD = -0.02

FEATURE_COLS = [
    "return_1d", "return_5d", "return_21d",
    "vol_5d", "vol_21d", "vol_63d",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]

    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)
    data["return_21d"] = close.pct_change(21)

    data["vol_5d"] = data["return_1d"].rolling(5).std() * np.sqrt(252)
    data["vol_21d"] = data["return_1d"].rolling(21).std() * np.sqrt(252)
    data["vol_63d"] = data["return_1d"].rolling(63).std() * np.sqrt(252)

    data["return_next"] = data["return_1d"].shift(-1)
    data["crash"] = (data["return_next"] < CRASH_THRESHOLD).astype(int)

    return data


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_features(df)
    available = get_feature_cols(df) + ["crash", "return_next"]
    return df.dropna(subset=available)
