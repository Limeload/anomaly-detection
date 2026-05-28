import numpy as np
import pandas as pd

RISK_FREE_DAILY = 0.02 / 252


def run_backtest(df: pd.DataFrame, crash_probs: np.ndarray, threshold: float = 0.35) -> pd.DataFrame:
    """
    Simulate cash-switching strategy.

    When crash_prob >= threshold: hold cash (return = 0).
    When crash_prob < threshold:  hold S&P 500 (return = market return).

    crash_prob[t] is computed from features at end of day t and applied to
    return[t+1], so there is no look-ahead bias.
    """
    bt = df[["return_next"]].copy()
    bt.columns = ["market_return"]
    bt["crash_prob"] = crash_probs
    bt["signal"] = (crash_probs >= threshold).astype(int)
    bt["strategy_return"] = bt["market_return"] * (1 - bt["signal"])
    bt["market_cumret"] = (1 + bt["market_return"]).cumprod()
    bt["strategy_cumret"] = (1 + bt["strategy_return"]).cumprod()
    return bt.dropna(subset=["market_return"])


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualised Sharpe ratio using daily returns."""
    excess = returns - RISK_FREE_DAILY
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())
