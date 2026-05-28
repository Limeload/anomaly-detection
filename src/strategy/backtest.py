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


def compute_metrics(bt: pd.DataFrame) -> dict:
    """Compute strategy vs buy-and-hold comparison metrics."""
    mkt_ret = bt["market_return"].dropna()
    strat_ret = bt["strategy_return"].dropna()
    mkt_cum = bt["market_cumret"].dropna()
    strat_cum = bt["strategy_cumret"].dropna()

    n_signals = int(bt["signal"].sum())
    n_days = len(bt)
    actual_crashes = (bt["market_return"] < -0.02).astype(int)
    signal_precision = float((bt["signal"] & actual_crashes).sum() / n_signals) if n_signals > 0 else 0.0

    mkt_cagr = cagr(mkt_cum)
    strat_cagr = cagr(strat_cum)

    return {
        "Market CAGR": mkt_cagr,
        "Strategy CAGR": strat_cagr,
        "CAGR Improvement": strat_cagr - mkt_cagr,
        "Market Sharpe": sharpe_ratio(mkt_ret),
        "Strategy Sharpe": sharpe_ratio(strat_ret),
        "Market Max Drawdown": max_drawdown(mkt_cum),
        "Strategy Max Drawdown": max_drawdown(strat_cum),
        "Days in Cash (Signals)": n_signals,
        "% Days in Cash": n_signals / n_days,
        "Signal Precision": signal_precision,
        "Total Trading Days": n_days,
    }


def format_metrics(metrics: dict) -> pd.DataFrame:
    """Human-readable single-column metrics table."""
    fmt = {
        "Market CAGR": "{:.1%}", "Strategy CAGR": "{:.1%}",
        "CAGR Improvement": "{:+.1%}",
        "Market Sharpe": "{:.2f}", "Strategy Sharpe": "{:.2f}",
        "Market Max Drawdown": "{:.1%}", "Strategy Max Drawdown": "{:.1%}",
        "Days in Cash (Signals)": "{:,.0f}", "% Days in Cash": "{:.1%}",
        "Signal Precision": "{:.1%}", "Total Trading Days": "{:,.0f}",
    }
    rows = [{"Metric": k, "Value": fmt.get(k, "{}").format(v)} for k, v in metrics.items()]
    return pd.DataFrame(rows).set_index("Metric")
