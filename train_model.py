#!/usr/bin/env python3
"""
Standalone training script — run this once before launching the Streamlit app.

    pip install -r requirements.txt
    python train_model.py

What it does
------------
1. Downloads S&P 500 + VIX data (2000-present) and caches to data/market_data.csv
2. Engineers features and builds the ML dataset
3. Runs TimeSeriesSplit cross-validation for:
      - Logistic Regression (with SMOTE)
      - XGBoost (with SMOTE)
      - Isolation Forest (unsupervised baseline)
4. Prints a CV comparison table
5. Trains final XGBoost model on full data and saves to models/crash_detector.joblib
6. Prints EDA summary
"""

import sys
from pathlib import Path

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.data.loader import load_market_data
from src.data.features import prepare_ml_dataset, get_feature_cols
from src.data.eda import class_balance, return_stats, crash_frequency_by_year
from src.models.train import cross_validate_models, train_final_model


def main():
    print("=" * 60)
    print("  Market Anomaly Detection — Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/4] Downloading market data (S&P 500 + VIX) …")
    raw = load_market_data(start="2000-01-01")
    print(f"      Loaded {len(raw):,} trading days  "
          f"({raw.index[0].date()} → {raw.index[-1].date()})")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/4] Engineering features …")
    df = prepare_ml_dataset(raw)
    feature_cols = get_feature_cols(df)
    print(f"      Dataset: {len(df):,} rows × {len(feature_cols)} features")

    # ------------------------------------------------------------------
    # 3. EDA summary
    # ------------------------------------------------------------------
    bal = class_balance(df)
    print(f"\n      Class balance:")
    print(f"        Normal days : {bal['normal_days']:,}  ({1 - bal['crash_rate']:.1%})")
    print(f"        Crash days  : {bal['crash_days']:,}  ({bal['crash_rate']:.1%})")

    print(f"\n      Crash days by year (sample):")
    freq = crash_frequency_by_year(df)
    for yr, cnt in freq.tail(10).items():
        print(f"        {yr}: {cnt}")

    print(f"\n      Return stats by class:")
    print(return_stats(df).to_string())

    # ------------------------------------------------------------------
    # 4. Cross-validation
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running {5}-fold TimeSeriesSplit cross-validation …")
    print("      (This may take a minute for XGBoost)\n")
    cv_results = cross_validate_models(df)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 100)
    print(cv_results.to_string())
    print()

    best_model = cv_results["roc_auc_mean"].idxmax()
    best_auc = cv_results.loc[best_model, "roc_auc_mean"]
    print(f"      Best model by ROC-AUC: {best_model}  (AUC = {best_auc:.4f})")

    # ------------------------------------------------------------------
    # 5. Train final model + save
    # ------------------------------------------------------------------
    print("\n[4/4] Training final XGBoost model on full dataset …")
    artifact = train_final_model(df, model_type="xgboost")
    print("      Done.")

    print("\n" + "=" * 60)
    print("  Training complete. Run the app with:")
    print("    streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
