import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CACHE_PATH = DATA_DIR / "market_data.csv"


def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tz-aware DatetimeIndex to naive UTC dates for consistent alignment.

    ^GSPC uses America/New_York and ^VIX uses America/Chicago; a naive
    tz_convert(None) produces different timestamps, breaking the join.
    Converting to UTC and then normalizing gives identical date keys.
    """
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").normalize().tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def load_market_data(start: str = "2000-01-01", end: str = None, use_cache: bool = True) -> pd.DataFrame:
    """Download S&P 500 + VIX data, merge on date, and cache locally."""
    DATA_DIR.mkdir(exist_ok=True)

    if use_cache and CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df

    sp500 = yf.Ticker("^GSPC").history(start=start, end=end)
    vix = yf.Ticker("^VIX").history(start=start, end=end)

    if sp500.empty:
        raise RuntimeError("yfinance returned empty data for ^GSPC. Check network connectivity.")

    sp500 = sp500[["Open", "High", "Low", "Close", "Volume"]].copy()
    sp500 = _to_date_index(sp500)
    vix = _to_date_index(vix)

    vix_close = vix[["Close"]].rename(columns={"Close": "VIX"})
    df = sp500.join(vix_close, how="left")
    df["VIX"] = df["VIX"].ffill()
    df.index.name = "Date"

    df.to_csv(CACHE_PATH)
    return df


def refresh_data(start: str = "2000-01-01") -> pd.DataFrame:
    """Force re-download, bypassing cache."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
    return load_market_data(start=start, use_cache=False)
