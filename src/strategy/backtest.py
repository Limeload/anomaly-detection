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


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum peak-to-trough decline."""
    rolling_peak = cumulative_returns.expanding().max()
    drawdown = cumulative_returns / rolling_peak - 1
    return float(drawdown.min())


def cagr(cumulative_returns: pd.Series) -> float:
    """Compound annual growth rate over the full series length."""
    n_years = len(cumulative_returns) / 252
    if n_years == 0:
        return 0.0
    final = cumulative_returns.dropna().iloc[-1]
    return float(final ** (1 / n_years) - 1)


def drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    """Full drawdown time series (0 to -1 range)."""
    rolling_peak = cumulative_returns.expanding().max()
    return cumulative_returns / rolling_peak - 1
