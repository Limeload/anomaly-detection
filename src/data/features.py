import numpy as np
import pandas as pd

CRASH_THRESHOLD = -0.02

FEATURE_COLS = [
    "return_1d", "return_5d", "return_21d",
    "vol_5d", "vol_21d", "vol_63d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pos",
    "price_sma50_ratio", "price_sma200_ratio",
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
    data = df.copy()
    close = data["Close"]

    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)
    data["return_21d"] = close.pct_change(21)

    data["vol_5d"] = data["return_1d"].rolling(5).std() * np.sqrt(252)
    data["vol_21d"] = data["return_1d"].rolling(21).std() * np.sqrt(252)
    data["vol_63d"] = data["return_1d"].rolling(63).std() * np.sqrt(252)

    data["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["bb_width"] = (2 * std20) / sma20
    data["bb_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    data["price_sma50_ratio"] = close / close.rolling(50).mean() - 1
    data["price_sma200_ratio"] = close / close.rolling(200).mean() - 1

    data["return_next"] = data["return_1d"].shift(-1)
    data["crash"] = (data["return_next"] < CRASH_THRESHOLD).astype(int)

    return data


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_features(df)
    available = get_feature_cols(df) + ["crash", "return_next"]
    return df.dropna(subset=available)
