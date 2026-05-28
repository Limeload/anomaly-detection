import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.features import get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

N_SPLITS = 5
THRESHOLD = 0.35
SMOTE_RANDOM_STATE = 42


def _make_pipelines() -> dict:
    return {
        "logistic_regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=SMOTE_RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "xgboost": ImbPipeline([
            ("smote", SMOTE(random_state=SMOTE_RANDOM_STATE)),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )),
        ]),
    }


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
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["crash"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    pipelines = _make_pipelines()
    records = []

    for name, pipe in pipelines.items():
        fold_results = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            if y_tr.sum() < 5:
                continue
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_val)[:, 1]
            fold_results.append(_fold_metrics(y_val, proba))

        agg = {k: [r[k] for r in fold_results] for k in fold_results[0]}
        records.append({
            "model": name,
            "roc_auc_mean": np.mean(agg["roc_auc"]),
            "roc_auc_std": np.std(agg["roc_auc"]),
            "avg_precision_mean": np.mean(agg["avg_precision"]),
            "f1_mean": np.mean(agg["f1"]),
            "precision_mean": np.mean(agg["precision"]),
            "recall_mean": np.mean(agg["recall"]),
        })

    # --- Isolation Forest (unsupervised) ---
    iso_aucs = []
    crash_rate = y.mean()
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_val = y[val_idx]
        iso = IsolationForest(
            contamination=float(np.clip(crash_rate, 0.01, 0.49)),
            n_estimators=200,
            random_state=42,
        )
        iso.fit(X_tr)
        scores = -iso.score_samples(X_val)
        iso_aucs.append(roc_auc_score(y_val, scores))

    records.append({
        "model": "isolation_forest",
        "roc_auc_mean": np.mean(iso_aucs),
        "roc_auc_std": np.std(iso_aucs),
        "avg_precision_mean": None,
        "f1_mean": None,
        "precision_mean": None,
        "recall_mean": None,
    })

    return pd.DataFrame(records).set_index("model")
