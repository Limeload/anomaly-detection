import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from src.data.features import get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

N_SPLITS = 5
THRESHOLD = 0.35


def _fold_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    preds = (proba >= THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
    }


def cross_validate_models(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder — models added in subsequent commits."""
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["crash"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    return pd.DataFrame()
