import numpy as np
import pandas as pd

def run_backtest(price, signal):
    """
    Runs a simple long-only backtest as in the notebook. Assumes signal is boolean Series (True=long, False=out).
    Returns equity curve, returns, drawdown, and metrics dict.
    """
    # Calculate weekly returns
    ret = price.pct_change().shift(-1)  # forward return
    strat_ret = ret * signal.astype(float)
    equity = (1 + strat_ret).cumprod()
    # Drawdown
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    # Metrics
    total_return = equity.iloc[-2] - 1  # -2 because last return is NaN
    cagr = (equity.iloc[-2]) ** (52/len(equity)) - 1
    max_dd = drawdown.min()
    vol = strat_ret.std() * np.sqrt(52)
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(52)
    metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Max Drawdown': max_dd,
        'Volatility': vol,
        'Sharpe': sharpe
    }
    print("[BACKTEST] Metrics:", metrics)
    return equity, strat_ret, drawdown, metrics
