"""
EDA helpers — called from train_model.py and the Streamlit app.
Returns plain dicts/DataFrames so callers can format output however they want.
"""

import numpy as np
import pandas as pd


def class_balance(df: pd.DataFrame) -> dict:
    """Return crash vs non-crash counts and ratio."""
    counts = df["crash"].value_counts()
    total = len(df)
    return {
        "total_days": total,
        "crash_days": int(counts.get(1, 0)),
        "normal_days": int(counts.get(0, 0)),
        "crash_rate": float(counts.get(1, 0) / total),
    }


def return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats for return_1d split by crash label."""
    return df.groupby("crash")["return_1d"].describe().rename(index={0: "Normal", 1: "Crash"})


def crash_frequency_by_year(df: pd.DataFrame) -> pd.Series:
    """Number of crash days per calendar year."""
    crash_df = df[df["crash"] == 1]
    return crash_df.groupby(crash_df.index.year).size().rename("crash_days")


def feature_correlation(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix for features + crash label."""
    cols = feature_cols + ["crash"]
    return df[cols].corr()
