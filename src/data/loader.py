import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CACHE_PATH = DATA_DIR / "market_data.csv"


def load_market_data(start: str = "2000-01-01", end: str = None, use_cache: bool = True) -> pd.DataFrame:
    """Download S&P 500 OHLCV data and cache locally."""
    DATA_DIR.mkdir(exist_ok=True)

    if use_cache and CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df

    sp500 = yf.Ticker("^GSPC").history(start=start, end=end)
    sp500 = sp500[["Open", "High", "Low", "Close", "Volume"]].copy()
    sp500.index.name = "Date"

    sp500.to_csv(CACHE_PATH)
    return sp500
