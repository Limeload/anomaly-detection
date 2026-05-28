"""
Feature engineering for market crash detection.

Target definition
-----------------
crash[t] = 1  if  return[t+1] < -2%
           0  otherwise

Using shift(-1) means today's features predict TOMORROW's crash, so the
strategy signal is available at market close to act on before the open.
No future data touches the feature set — all indicators are backward-looking.
"""

import numpy as np
import pandas as pd

CRASH_THRESHOLD = -0.02  # -2% daily return

FEATURE_COLS = [
    "return_1d", "return_5d", "return_21d",
    "vol_5d", "vol_21d", "vol_63d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pos",
    "price_sma50_ratio", "price_sma200_ratio",
    "volume_ratio",
    "VIX", "vix_change", "vix_sma20_ratio",
]


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features and the crash label to a raw OHLCV+VIX frame."""
    data = df.copy()
    close = data["Close"]

    # --- Returns ---
    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)
    data["return_21d"] = close.pct_change(21)

    # --- Rolling volatility (annualised) ---
    data["vol_5d"] = data["return_1d"].rolling(5).std() * np.sqrt(252)
    data["vol_21d"] = data["return_1d"].rolling(21).std() * np.sqrt(252)
    data["vol_63d"] = data["return_1d"].rolling(63).std() * np.sqrt(252)

    # --- RSI ---
    data["rsi_14"] = _rsi(close, 14)

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # --- Bollinger Bands ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["bb_width"] = (2 * std20) / sma20
    data["bb_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    # --- Price vs moving averages ---
    data["price_sma50_ratio"] = close / close.rolling(50).mean() - 1
    data["price_sma200_ratio"] = close / close.rolling(200).mean() - 1

    # --- Volume ---
    if "Volume" in data.columns:
        vol_ma = data["Volume"].rolling(20).mean()
        data["volume_ratio"] = data["Volume"] / (vol_ma + 1)

    # --- VIX features ---
    if "VIX" in data.columns:
        data["vix_change"] = data["VIX"].pct_change()
        data["vix_sma20_ratio"] = data["VIX"] / data["VIX"].rolling(20).mean() - 1

    # --- Forward-looking target: will NEXT day be a crash? ---
    data["return_next"] = data["return_1d"].shift(-1)
    data["crash"] = (data["return_next"] < CRASH_THRESHOLD).astype(int)

    return data


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns that are actually present in df."""
    return [c for c in FEATURE_COLS if c in df.columns]


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing any feature or target; used for training."""
    df = compute_features(df)
    available = get_feature_cols(df) + ["crash", "return_next"]
    return df.dropna(subset=available)
