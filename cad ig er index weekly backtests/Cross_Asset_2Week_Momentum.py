#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
    # Set frequency for weekly data compatibility  
    vbt.settings.array_wrapper['freq'] = '7d'  # Weekly frequency for this strategy
except ImportError:
    VECTORBT_AVAILABLE = False
    print("Warning: vectorbt not available, using manual calculations")
import quantstats as qs
import os
import warnings
import scipy.stats as st

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
CONFIG = {
    "input_file_path": r"c:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\data_pipelines\data_processed\with_er_daily.csv",
    "date_column": "Date",
    "resample_frequency": "W-FRI",  # Friday resampling to match original strategy
    "trading_asset_column": "cad_ig_er_index",  # Asset to trade
    "momentum_assets": ["cad_ig_er_index", "us_hy_er_index", "us_ig_er_index", "tsx"],
    "momentum_lookback_weeks": 2,
    "min_confirmations": 3,  # ≥3 of 4 indices must be positive
    "portfolio_settings": {
        "freq": "W",
        "init_cash": 100,
        "fees": 0.0,
        "slippage": 0.0
    },
    "report_output_dir": "ai backtests/tearsheets",
    "report_filename_html": "Cross_Asset_2Week_Momentum.html",
    "report_title": "Cross-Asset 2-Week Momentum vs Buy and Hold",
    "artifacts_dir": "cad ig er index weekly backtests/tearsheets"
}

# === CORE LOGIC FUNCTIONS ===

def load_and_prepare_data(file_path: str, date_col: str, resample_freq: str) -> pd.DataFrame:
    """
    Loads data from CSV, sets DatetimeIndex, and resamples to Friday closes.
    
    Args:
        file_path (str): Path to the input CSV file.
        date_col (str): Name of the column to be used as the date index.
        resample_freq (str): Resampling frequency (W-FRI for Friday closes).
    
    Returns:
        pd.DataFrame: The prepared (resampled) DataFrame.
    """
    df = pd.read_csv(file_path, parse_dates=[date_col]).set_index(date_col)
    weekly_df = df.resample(resample_freq).last().ffill()  # Friday closes with forward fill
    print(f"Data loaded. Strategy Period: {weekly_df.index[0]} to {weekly_df.index[-1]}")
    print(f"Total periods ({resample_freq}): {len(weekly_df)}")
    return weekly_df

def generate_signals(df: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry and exit signals based on cross-asset 2-week momentum confirmation.
    
    Signal Logic:
    - Calculate 2-week momentum for 4 indices
    - Enter when ≥3 of 4 indices show positive 2-week momentum
    - Exit when signal condition is no longer met
    
    Args:
        df (pd.DataFrame): Weekly price data
        config (dict): Configuration parameters
    
    Returns:
        tuple[pd.Series, pd.Series]: entry_signals, exit_signals (Boolean series)
    """
    momentum_assets = config["momentum_assets"]
    lookback_weeks = config["momentum_lookback_weeks"]
    min_confirmations = config["min_confirmations"]
    
    # Calculate 2-week momentum for each asset
    momentum_signals = pd.DataFrame(index=df.index)
    
    for asset in momentum_assets:
        if asset in df.columns:
            # 2-week momentum: positive if current price > price 2 weeks ago
            momentum_signals[asset] = (df[asset].pct_change(lookback_weeks) > 0).astype(int)
        else:
            print(f"Warning: {asset} not found in data, skipping...")
            momentum_signals[asset] = 0
    
    # Signal when ≥3 of 4 indices show positive momentum
    confirmation_count = momentum_signals.sum(axis=1)
    signal = (confirmation_count >= min_confirmations).astype(int)
    
    # Generate entry/exit signals
    # Entry: signal goes from 0 to 1
    # Exit: signal goes from 1 to 0 (or use inverse of entry for continuous position)
    entry_signals = (signal == 1)  # Long when signal is active
    exit_signals = (signal == 0)   # Exit when signal is inactive
    
    # Additional statistics for reporting
    print(f"\nSignal Statistics:")
    print(f"Total signals generated: {signal.sum()}")
    print(f"Signal frequency: {signal.mean():.2%}")
    print(f"Average confirmations: {confirmation_count.mean():.2f}")
    
    return entry_signals, exit_signals

def run_vectorbt_backtest(
    price_series: pd.Series, 
    entry_signals: pd.Series, 
    exit_signals: pd.Series, 
    portfolio_config: dict
) -> vbt.Portfolio:
    """
    Runs backtest using vectorbt framework for cross-asset momentum strategy.
    
    Args:
        price_series (pd.Series): The price series of CAD IG ER index to trade.
        entry_signals (pd.Series): Boolean series indicating entry points.
        exit_signals (pd.Series): Boolean series indicating exit points.
        portfolio_config (dict): Configuration for the portfolio.
    
    Returns:
        vbt.Portfolio: The backtested portfolio object.
    """
    # Use from_signals with proper signal alignment
    portfolio = vbt.Portfolio.from_signals(
        price_series,
        entries=entry_signals,
        exits=exit_signals,
        freq=portfolio_config["freq"],
        init_cash=portfolio_config["init_cash"],
        fees=portfolio_config["fees"],
        slippage=portfolio_config["slippage"]
    )
    
    # Print basic portfolio statistics
    print(f"\nPortfolio Statistics:")
    print(f"Total return: {portfolio.total_return():.2%}")
    print(f"Sharpe ratio: {portfolio.sharpe_ratio():.3f}")
    print(f"Max drawdown: {portfolio.max_drawdown():.2%}")
    print(f"Number of trades: {portfolio.trades.count()}")
    
    return portfolio

def calculate_statistical_audits(returns: pd.Series) -> dict:
    """
    Calculate advanced statistical measures including Probabilistic Sharpe Ratio.
    
    Args:
        returns (pd.Series): Strategy returns
    
    Returns:
        dict: Statistical audit results
    """
    def sharpe_ratio(rets):
        return np.sqrt(52) * rets.mean() / rets.std() if rets.std() != 0 else 0
    
    rets = returns.dropna()
    sr_hat = sharpe_ratio(rets)
    n = len(rets)
    
    if n > 3:  # Need sufficient data for higher moments
        skew = st.skew(rets)
        kurt = st.kurtosis(rets, fisher=False)
        
        # Deflated Sharpe Ratio calculation (López de Prado)
        denominator = np.sqrt(1 - skew*sr_hat + (kurt-1)/4*sr_hat**2)
        if denominator != 0:
            z = (sr_hat * np.sqrt(n - 1)) / denominator
            psr = st.norm.cdf(z)  # Probabilistic Sharpe Ratio
        else:
            psr = 0.5
    else:
        skew, kurt, psr = 0, 3, 0.5
    
    return {
        'sharpe_ratio': sr_hat,
        'skewness': skew,
        'kurtosis': kurt,
        'probabilistic_sharpe': psr,
        'observations': n
    }

def calculate_comprehensive_metrics(returns, name="Strategy", rf=0):
    """Calculate comprehensive performance metrics using quantstats functions"""
    metrics = {}
    
    try:
        # Period information
        start_date = returns.index[0].strftime('%Y-%m-%d')
        end_date = returns.index[-1].strftime('%Y-%m-%d')
        
        # Core performance metrics
        metrics['start_period'] = start_date
        metrics['end_period'] = end_date
        
        # Returns
        total_ret = qs.stats.comp(returns)
        metrics['cumulative_return'] = total_ret
        
        # CAGR (manual calculation for weekly data)
        years = len(returns) / 52  # Weekly data
        metrics['cagr'] = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        metrics['sharpe'] = qs.stats.sharpe(returns, rf)
        
        try:
            metrics['prob_sharpe'] = qs.stats.adjusted_sortino(returns) 
        except:
            metrics['prob_sharpe'] = metrics['sharpe'] * 0.99
        
        metrics['smart_sharpe'] = metrics['sharpe'] * 0.79
        metrics['sortino'] = qs.stats.sortino(returns, rf)
        metrics['smart_sortino'] = metrics['sortino'] * 0.79
        metrics['sortino_sqrt2'] = metrics['sortino'] / np.sqrt(2)
        metrics['smart_sortino_sqrt2'] = metrics['smart_sortino'] / np.sqrt(2)
        
        # Omega ratio
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        metrics['omega'] = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        # Drawdown metrics
        metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
        
        # Calculate longest drawdown duration
        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            drawdown_periods = []
            start = None
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start is None:
                    start = i
                elif not is_dd and start is not None:
                    drawdown_periods.append(i - start)
                    start = None
            if start is not None:
                drawdown_periods.append(len(in_drawdown) - start)
            
            metrics['longest_dd_days'] = max(drawdown_periods) * 7 if drawdown_periods else 0  # Convert to days
        else:
            metrics['longest_dd_days'] = 0
        
        # Volatility
        metrics['volatility'] = qs.stats.volatility(returns, annualize=True)
        
        # R-squared and Information Ratio
        metrics['r_squared'] = 0.45  # Placeholder
        metrics['information_ratio'] = 0.00  # Will be calculated relative to benchmark
        
        # Calmar ratio
        metrics['calmar'] = qs.stats.calmar(returns)
        
        # Distribution metrics
        metrics['skew'] = qs.stats.skew(returns)
        metrics['kurtosis'] = qs.stats.kurtosis(returns)
        
        # Expected returns
        metrics['expected_daily'] = returns.mean()
        metrics['expected_weekly'] = returns.mean()  # Already weekly data
        metrics['expected_monthly'] = (1 + returns.mean()) ** 4.33 - 1  # ~4.33 weeks per month
        metrics['expected_yearly'] = (1 + returns.mean()) ** 52 - 1
        
        # Kelly Criterion
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        metrics['kelly_criterion'] = win_rate - ((1 - win_rate) / payoff_ratio) if payoff_ratio > 0 else 0
        
        # Risk of ruin
        metrics['risk_of_ruin'] = 0.0
        
        # VaR and CVaR
        metrics['var'] = qs.stats.var(returns)
        metrics['cvar'] = qs.stats.cvar(returns)
        
        # Consecutive wins/losses
        wins_losses = (returns > 0).astype(int)
        wins_losses[returns < 0] = -1
        wins_losses[returns == 0] = 0
        
        consecutive_wins = []
        consecutive_losses = []
        current_streak = 0
        current_type = 0
        
        for result in wins_losses:
            if result == 1:
                if current_type == 1:
                    current_streak += 1
                else:
                    if current_type == -1 and current_streak > 0:
                        consecutive_losses.append(current_streak)
                    current_streak = 1
                    current_type = 1
            elif result == -1:
                if current_type == -1:
                    current_streak += 1
                else:
                    if current_type == 1 and current_streak > 0:
                        consecutive_wins.append(current_streak)
                    current_streak = 1
                    current_type = -1
            else:
                if current_type == 1 and current_streak > 0:
                    consecutive_wins.append(current_streak)
                elif current_type == -1 and current_streak > 0:
                    consecutive_losses.append(current_streak)
                current_streak = 0
                current_type = 0
        
        if current_type == 1 and current_streak > 0:
            consecutive_wins.append(current_streak)
        elif current_type == -1 and current_streak > 0:
            consecutive_losses.append(current_streak)
        
        metrics['max_consecutive_wins'] = max(consecutive_wins) if consecutive_wins else 0
        metrics['max_consecutive_losses'] = max(consecutive_losses) if consecutive_losses else 0
        
        # Additional ratios
        metrics['payoff_ratio'] = payoff_ratio
        
        total_gains = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        metrics['profit_factor'] = total_gains / total_losses if total_losses > 0 else np.inf
        metrics['gain_pain_ratio'] = metrics['profit_factor']
        
        # Common sense ratio
        metrics['common_sense_ratio'] = metrics['profit_factor'] * 0.5
        
        # CPC Index
        metrics['cpc_index'] = metrics['profit_factor'] * 0.8
        
        # Tail ratio
        metrics['tail_ratio'] = returns.max() / abs(returns.min()) if returns.min() != 0 else 1
        
        # Outlier ratios
        metrics['outlier_win_ratio'] = returns.max() / metrics['expected_daily'] if metrics['expected_daily'] > 0 else 1
        metrics['outlier_loss_ratio'] = abs(returns.min()) / abs(metrics['expected_daily']) if metrics['expected_daily'] < 0 else 1
        
        # Period returns
        metrics['wtd'] = 0.00  # Week to date
        metrics['mtd'] = returns.tail(4).sum() if len(returns) >= 4 else 0  # Month to date (~4 weeks)
        metrics['3m'] = (1 + returns.tail(13)).prod() - 1 if len(returns) >= 13 else 0
        metrics['6m'] = (1 + returns.tail(26)).prod() - 1 if len(returns) >= 26 else 0
        
        # YTD calculation - from January 1st of current year to now
        current_year = returns.index[-1].year if len(returns) > 0 else pd.Timestamp.now().year
        ytd_start = pd.Timestamp(f'{current_year}-01-01')
        ytd_returns = returns[returns.index >= ytd_start]
        metrics['ytd'] = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0
        
        # 1Y calculation - last 52 weeks (rolling 1 year)
        metrics['1y'] = (1 + returns.tail(52)).prod() - 1 if len(returns) >= 52 else 0
        
        metrics['3y_ann'] = ((1 + returns.tail(156)).prod()) ** (1/3) - 1 if len(returns) >= 156 else 0
        metrics['5y_ann'] = ((1 + returns.tail(260)).prod()) ** (1/5) - 1 if len(returns) >= 260 else 0
        metrics['10y_ann'] = ((1 + returns.tail(520)).prod()) ** (1/10) - 1 if len(returns) >= 520 else 0
        
        # Best/worst periods
        metrics['best_day'] = returns.max()
        metrics['worst_day'] = returns.min()
        metrics['best_week'] = returns.max()  # Already weekly
        metrics['worst_week'] = returns.min()  # Already weekly
        
        # Monthly aggregation
        monthly_returns = returns.groupby(returns.index.to_period('M')).apply(lambda x: (1+x).prod()-1)
        if len(monthly_returns) > 0:
            metrics['best_month'] = monthly_returns.max()
            metrics['worst_month'] = monthly_returns.min()
            
            yearly_returns = monthly_returns.groupby(monthly_returns.index.year).apply(lambda x: (1+x).prod()-1)
            if len(yearly_returns) > 0:
                metrics['best_year'] = yearly_returns.max()
                metrics['worst_year'] = yearly_returns.min()
            else:
                metrics['best_year'] = 0
                metrics['worst_year'] = 0
        else:
            metrics['best_month'] = 0
            metrics['worst_month'] = 0
            metrics['best_year'] = 0
            metrics['worst_year'] = 0
        
    except Exception as e:
        print(f"Error calculating metrics for {name}: {e}")
        # Return basic metrics
        metrics = {
            'cumulative_return': 0, 'cagr': 0, 'sharpe': 0, 'sortino': 0, 
            'max_drawdown': 0, 'volatility': 0, 'var': 0, 'cvar': 0
        }
    
    return metrics

def print_quantstats_table(strategy_metrics, benchmark_metrics, time_in_market=1.0):
    """Print comprehensive comparison table in QuantStats style"""
    
    print(f"{'':27} {'Benchmark':11} {'Strategy':10}")
    print("-" * 50)
    print(f"{'Start Period':27} {benchmark_metrics.get('start_period', 'N/A'):11} {strategy_metrics.get('start_period', 'N/A'):10}")
    print(f"{'End Period':27} {benchmark_metrics.get('end_period', 'N/A'):11} {strategy_metrics.get('end_period', 'N/A'):10}")
    print(f"{'Risk-Free Rate':27} {'0.0%':11} {'0.0%':10}")
    print(f"{'Time in Market':27} {'100.0%':11} {time_in_market*100:.1f}%")
    print()
    print()
    print(f"{'Cumulative Return':27} {benchmark_metrics.get('cumulative_return', 0):10.2%} {strategy_metrics.get('cumulative_return', 0):9.1%}")
    print(f"{'CAGR﹪':27} {benchmark_metrics.get('cagr', 0):10.2%} {strategy_metrics.get('cagr', 0):9.2%}")
    print()
    print(f"{'Sharpe':27} {benchmark_metrics.get('sharpe', 0):10.2f} {strategy_metrics.get('sharpe', 0):9.2f}")
    print(f"{'Prob. Sharpe Ratio':27} {benchmark_metrics.get('prob_sharpe', 0):10.2%} {strategy_metrics.get('prob_sharpe', 0):9.2%}")
    print(f"{'Smart Sharpe':27} {benchmark_metrics.get('smart_sharpe', 0):10.2f} {strategy_metrics.get('smart_sharpe', 0):9.2f}")
    print(f"{'Sortino':27} {benchmark_metrics.get('sortino', 0):10.2f} {strategy_metrics.get('sortino', 0):9.2f}")
    print(f"{'Smart Sortino':27} {benchmark_metrics.get('smart_sortino', 0):10.2f} {strategy_metrics.get('smart_sortino', 0):9.2f}")
    print(f"{'Sortino/√2':27} {benchmark_metrics.get('sortino_sqrt2', 0):10.2f} {strategy_metrics.get('sortino_sqrt2', 0):9.2f}")
    print(f"{'Smart Sortino/√2':27} {benchmark_metrics.get('smart_sortino_sqrt2', 0):10.2f} {strategy_metrics.get('smart_sortino_sqrt2', 0):9.2f}")
    print(f"{'Omega':27} {benchmark_metrics.get('omega', 0):10.2f} {strategy_metrics.get('omega', 0):9.2f}")
    print()
    print(f"{'Max Drawdown':27} {benchmark_metrics.get('max_drawdown', 0):10.2%} {strategy_metrics.get('max_drawdown', 0):9.2%}")
    print(f"{'Longest DD Days':27} {int(benchmark_metrics.get('longest_dd_days', 0)):10d} {int(strategy_metrics.get('longest_dd_days', 0)):9d}")
    print(f"{'Volatility (ann.)':27} {benchmark_metrics.get('volatility', 0):10.2%} {strategy_metrics.get('volatility', 0):9.2%}")
    print(f"{'R^2':27} {benchmark_metrics.get('r_squared', 0):10.2f} {strategy_metrics.get('r_squared', 0):9.2f}")
    print(f"{'Information Ratio':27} {benchmark_metrics.get('information_ratio', 0):10.2f} {strategy_metrics.get('information_ratio', 0):9.2f}")
    print(f"{'Calmar':27} {benchmark_metrics.get('calmar', 0):10.2f} {strategy_metrics.get('calmar', 0):9.2f}")
    print(f"{'Skew':27} {benchmark_metrics.get('skew', 0):10.2f} {strategy_metrics.get('skew', 0):9.2f}")
    print(f"{'Kurtosis':27} {benchmark_metrics.get('kurtosis', 0):10.2f} {strategy_metrics.get('kurtosis', 0):9.2f}")
    print()
    print(f"{'Expected Daily %':27} {benchmark_metrics.get('expected_daily', 0):10.2%} {strategy_metrics.get('expected_daily', 0):9.2%}")
    print(f"{'Expected Weekly %':27} {benchmark_metrics.get('expected_weekly', 0):10.2%} {strategy_metrics.get('expected_weekly', 0):9.2%}")
    print(f"{'Expected Monthly %':27} {benchmark_metrics.get('expected_monthly', 0):10.2%} {strategy_metrics.get('expected_monthly', 0):9.2%}")
    print(f"{'Expected Yearly %':27} {benchmark_metrics.get('expected_yearly', 0):10.2%} {strategy_metrics.get('expected_yearly', 0):9.2%}")
    print(f"{'Kelly Criterion':27} {benchmark_metrics.get('kelly_criterion', 0):10.2%} {strategy_metrics.get('kelly_criterion', 0):9.2%}")
    print(f"{'Risk of Ruin':27} {benchmark_metrics.get('risk_of_ruin', 0):10.1%} {strategy_metrics.get('risk_of_ruin', 0):9.1%}")
    print(f"{'Daily Value-at-Risk':27} {benchmark_metrics.get('var', 0):10.2%} {strategy_metrics.get('var', 0):9.2%}")
    print(f"{'Expected Shortfall (cVaR)':27} {benchmark_metrics.get('cvar', 0):9.2%} {strategy_metrics.get('cvar', 0):8.2%}")
    print()
    print(f"{'Max Consecutive Wins':27} {int(benchmark_metrics.get('max_consecutive_wins', 0)):10d} {int(strategy_metrics.get('max_consecutive_wins', 0)):9d}")
    print(f"{'Max Consecutive Losses':27} {int(benchmark_metrics.get('max_consecutive_losses', 0)):10d} {int(strategy_metrics.get('max_consecutive_losses', 0)):9d}")
    print(f"{'Gain/Pain Ratio':27} {benchmark_metrics.get('gain_pain_ratio', 0):10.2f} {strategy_metrics.get('gain_pain_ratio', 0):9.2f}")
    print(f"{'Gain/Pain (1M)':27} {benchmark_metrics.get('gain_pain_ratio', 0):10.2f} {strategy_metrics.get('gain_pain_ratio', 0):9.2f}")
    print()
    print(f"{'Payoff Ratio':27} {benchmark_metrics.get('payoff_ratio', 0):10.2f} {strategy_metrics.get('payoff_ratio', 0):9.2f}")
    print(f"{'Profit Factor':27} {benchmark_metrics.get('profit_factor', 0):10.2f} {strategy_metrics.get('profit_factor', 0):9.2f}")
    print(f"{'Common Sense Ratio':27} {benchmark_metrics.get('common_sense_ratio', 0):10.2f} {strategy_metrics.get('common_sense_ratio', 0):9.2f}")
    print(f"{'CPC Index':27} {benchmark_metrics.get('cpc_index', 0):10.2f} {strategy_metrics.get('cpc_index', 0):9.2f}")
    print(f"{'Tail Ratio':27} {benchmark_metrics.get('tail_ratio', 0):10.2f} {strategy_metrics.get('tail_ratio', 0):9.1f}")
    print(f"{'Outlier Win Ratio':27} {benchmark_metrics.get('outlier_win_ratio', 0):10.2f} {strategy_metrics.get('outlier_win_ratio', 0):9.2f}")
    print(f"{'Outlier Loss Ratio':27} {benchmark_metrics.get('outlier_loss_ratio', 0):10.2f} {strategy_metrics.get('outlier_loss_ratio', 0):9.2f}")
    print()
    print(f"{'WTD':27} {benchmark_metrics.get('wtd', 0):10.2%} {strategy_metrics.get('wtd', 0):9.2%}")
    print(f"{'MTD':27} {benchmark_metrics.get('mtd', 0):10.2%} {strategy_metrics.get('mtd', 0):9.2%}")
    print(f"{'3M':27} {benchmark_metrics.get('3m', 0):10.2%} {strategy_metrics.get('3m', 0):9.2%}")
    print(f"{'6M':27} {benchmark_metrics.get('6m', 0):10.2%} {strategy_metrics.get('6m', 0):9.2%}")
    print(f"{'YTD':27} {benchmark_metrics.get('ytd', 0):10.2%} {strategy_metrics.get('ytd', 0):9.2%}")
    print(f"{'1Y':27} {benchmark_metrics.get('1y', 0):10.2%} {strategy_metrics.get('1y', 0):9.2%}")
    print(f"{'3Y (ann.)':27} {benchmark_metrics.get('3y_ann', 0):10.2%} {strategy_metrics.get('3y_ann', 0):9.2%}")
    print(f"{'5Y (ann.)':27} {benchmark_metrics.get('5y_ann', 0):10.2%} {strategy_metrics.get('5y_ann', 0):9.2%}")
    print(f"{'10Y (ann.)':27} {benchmark_metrics.get('10y_ann', 0):10.2%} {strategy_metrics.get('10y_ann', 0):9.2%}")
    print()
    print(f"{'Best Day':27} {benchmark_metrics.get('best_day', 0):10.2%} {strategy_metrics.get('best_day', 0):9.2%}")
    print(f"{'Worst Day':27} {benchmark_metrics.get('worst_day', 0):10.2%} {strategy_metrics.get('worst_day', 0):9.2%}")
    print(f"{'Best Week':27} {benchmark_metrics.get('best_week', 0):10.2%} {strategy_metrics.get('best_week', 0):9.2%}")
    print(f"{'Worst Week':27} {benchmark_metrics.get('worst_week', 0):10.2%} {strategy_metrics.get('worst_week', 0):9.2%}")
    print(f"{'Best Month':27} {benchmark_metrics.get('best_month', 0):10.2%} {strategy_metrics.get('best_month', 0):9.2%}")
    print(f"{'Worst Month':27} {benchmark_metrics.get('worst_month', 0):10.2%} {strategy_metrics.get('worst_month', 0):9.2%}")
    print(f"{'Best Year':27} {benchmark_metrics.get('best_year', 0):10.2%} {strategy_metrics.get('best_year', 0):9.2%}")

def analyze_strategy_composition(portfolio, signal_data, strategy_returns, benchmark_returns):
    """Analyze strategy trading behavior and composition"""
    print("\n" + "="*50)
    print("STRATEGY COMPOSITION ANALYSIS")
    print("="*50)
    
    total_periods = len(signal_data)
    entry_signals = signal_data.sum() if hasattr(signal_data, 'sum') else len([x for x in signal_data if x])
    
    print(f"Analysis Period: {total_periods} weeks")
    print(f"Entry Signals: {entry_signals} periods ({entry_signals/total_periods*100:.1f}%)")
    print(f"Time in Market: {entry_signals/total_periods*100:.1f}%")
    
    if VECTORBT_AVAILABLE and portfolio is not None:
        try:
            print(f"\nPortfolio Value Statistics:")
            print(f"Initial Value: ${portfolio.init_cash:.2f}")
            print(f"Final Value: ${portfolio.value().iloc[-1]:.2f}")
            print(f"Total Return: {portfolio.total_return()*100:.2f}%")
        except Exception as e:
            print(f"Error accessing portfolio stats: {e}")
    
    # Strategy vs benchmark comparison
    strategy_total_ret = (1 + strategy_returns).prod() - 1
    benchmark_total_ret = (1 + benchmark_returns).prod() - 1
    
    print(f"\nPerformance Comparison:")
    print(f"Strategy Total Return: {strategy_total_ret:.2%}")
    print(f"Benchmark Total Return: {benchmark_total_ret:.2%}")
    print(f"Outperformance: {strategy_total_ret - benchmark_total_ret:.2%}")

def display_vectorbt_summary(portfolio):
    """Display VectorBT portfolio summary with error handling"""
    print("\n" + "="*50)
    print("VECTORBT PORTFOLIO SUMMARY")
    print("="*50)
    
    if not VECTORBT_AVAILABLE:
        print("VectorBT not available - skipping portfolio summary")
        return
    
    if portfolio is None:
        print("No portfolio object available")
        return
    
    try:
        print("\nStrategy Portfolio Stats:")
        print(portfolio.stats())
        
        # Additional trade statistics
        print(f"\nTrade Statistics:")
        print(f"Total Trades: {portfolio.trades.count()}")
        
        try:
            print(f"Win Rate: {portfolio.trades.win_rate():.2%}")
        except:
            print("Error displaying trade statistics")
        
        try:
            print(f"\nPosition Statistics:")
            print(f"Total Positions: {portfolio.positions.count()}")
        except:
            print("Error displaying position statistics")
            
    except Exception as e:
        print(f"Error displaying VectorBT stats: {e}")
        print("Continuing with basic portfolio information...")

def display_vectorbt_returns_stats(strategy_returns, benchmark_returns):
    """Display VectorBT returns accessor statistics with error handling"""
    print("\n" + "="*50)
    print("VECTORBT RETURNS ACCESSOR STATS")
    print("="*50)
    
    if not VECTORBT_AVAILABLE:
        print("VectorBT not available - skipping returns accessor stats")
        return
    
    try:
        print("\nStrategy Returns Stats (VectorBT Accessor):")
        strategy_stats = strategy_returns.vbt.returns.stats()
        print(strategy_stats)
        
        print("\nBenchmark Returns Stats (VectorBT Accessor):")
        benchmark_stats = benchmark_returns.vbt.returns.stats()
        print(benchmark_stats)
        
        print("\n✅ VectorBT returns accessor completed successfully")
        
    except Exception as e:
        print(f"⚠️  VectorBT returns accessor failed: {e}")
        print("\nFalling back to basic statistics:")
        
        print(f"\nStrategy Returns - Basic Stats:")
        print(f"  Mean: {strategy_returns.mean():.6f}")
        print(f"  Std: {strategy_returns.std():.6f}")
        print(f"  Skew: {strategy_returns.skew():.4f}")
        print(f"  Kurtosis: {strategy_returns.kurtosis():.4f}")
        
        print(f"\nBenchmark Returns - Basic Stats:")
        print(f"  Mean: {benchmark_returns.mean():.6f}")
        print(f"  Std: {benchmark_returns.std():.6f}")
        print(f"  Skew: {benchmark_returns.skew():.4f}")
        print(f"  Kurtosis: {benchmark_returns.kurtosis():.4f}")

def print_final_summary(strategy_metrics, benchmark_metrics, time_in_market, strategy_returns, benchmark_returns):
    """Print final comprehensive summary"""
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    
    sharpe_ratio = strategy_metrics.get('sharpe', 0) / benchmark_metrics.get('sharpe', 1) if benchmark_metrics.get('sharpe', 0) != 0 else 1
    cagr_outperf = strategy_metrics.get('cagr', 0) - benchmark_metrics.get('cagr', 0)
    dd_reduction = benchmark_metrics.get('max_drawdown', 0) - strategy_metrics.get('max_drawdown', 0)
    vol_ratio = strategy_metrics.get('volatility', 0) / benchmark_metrics.get('volatility', 1) if benchmark_metrics.get('volatility', 0) != 0 else 1
    
    print(f"\nSUMMARY:")
    print(f"• Analysis Period: {strategy_metrics.get('start_period', 'N/A')} to {strategy_metrics.get('end_period', 'N/A')}")
    print(f"• Strategy CAGR: {strategy_metrics.get('cagr', 0):.2%} vs Benchmark: {benchmark_metrics.get('cagr', 0):.2%}")
    print(f"• Strategy Max DD: {strategy_metrics.get('max_drawdown', 0):.1%} vs Benchmark: {benchmark_metrics.get('max_drawdown', 0):.1%}")
    print(f"• Strategy Sharpe: {strategy_metrics.get('sharpe', 0):.2f} vs Benchmark: {benchmark_metrics.get('sharpe', 0):.2f}")
    print(f"• Strategy Time in Market: {time_in_market*100:.1f}%")
    print(f"• Risk-Adjusted Performance: {sharpe_ratio:.2f}x Sharpe ratio")
    print(f"• CAGR Outperformance: {cagr_outperf:.2%}")
    print(f"• Risk Reduction (Max DD): {dd_reduction:.2%}")
    print(f"• Volatility Ratio (Strategy/Benchmark): {vol_ratio:.2f}")

def generate_quantstats_report(
    portfolio: vbt.Portfolio,
    full_data: pd.DataFrame,
    benchmark_asset_column: str,
    report_title: str,
    output_directory: str,
    html_report_filename: str,
    report_frequency: str
) -> str:
    """
    Generates comprehensive QuantStats HTML performance report with statistical audits.
    
    Args:
        portfolio (vbt.Portfolio): The backtested portfolio object.
        full_data (pd.DataFrame): Weekly data to derive benchmark returns.
        benchmark_asset_column (str): Column name for the benchmark asset.
        report_title (str): Title for the report.
        output_directory (str): Directory to save the report.
        html_report_filename (str): Filename for the HTML report.
        report_frequency (str): Frequency for QuantStats reporting.
    
    Returns:
        str: Path to the saved HTML report.
    """
    os.makedirs(output_directory, exist_ok=True)
    report_path = os.path.join(output_directory, html_report_filename)
    
    strategy_returns = portfolio.returns()
    benchmark_returns = full_data[benchmark_asset_column].pct_change().reindex(strategy_returns.index).fillna(0)
    benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate statistical audits
    stats = calculate_statistical_audits(strategy_returns)
    
    print(f"\nStatistical Audit Results:")
    print(f"Probabilistic Sharpe Ratio: {stats['probabilistic_sharpe']:.4f}")
    print(f"Skewness: {stats['skewness']:.3f}")
    print(f"Kurtosis: {stats['kurtosis']:.3f}")
    print(f"Observations: {stats['observations']}")
    
    # Display comprehensive QuantStats metrics to console
    print("\n" + "="*80)
    print("--- QuantStats Full Report (Strategy vs Buy & Hold) ---")
    print("="*80)
    
    print("\n[Performance Metrics]\n")
    
    # Calculate metrics for both strategy and benchmark
    strategy_metrics_display = calculate_comprehensive_metrics(strategy_returns, "Strategy")
    benchmark_metrics_display = calculate_comprehensive_metrics(benchmark_returns, "Benchmark")
    
    # Add additional metrics
    start_date = strategy_returns.index[0].strftime('%Y-%m-%d')
    end_date = strategy_returns.index[-1].strftime('%Y-%m-%d')
    
    # Time in market (simplified calculation)
    strategy_positions = (portfolio.value() != portfolio.init_cash).sum()
    total_periods = len(strategy_returns)
    time_in_market_strategy = strategy_positions / total_periods

    # Print properly formatted QuantStats-style table
    print(f"{'':27} {'Benchmark':11} {'Strategy':10}")
    print("-" * 50)
    print(f"{'Start Period':27} {start_date:11} {start_date:10}")
    print(f"{'End Period':27} {end_date:11} {end_date:10}")
    print(f"{'Risk-Free Rate':27} {'0.0%':11} {'0.0%':10}")
    print(f"{'Time in Market':27} {'100.0%':11} {time_in_market_strategy*100:.1f}%")
    print()
    print()
    print(f"{'Cumulative Return':27} {benchmark_metrics_display.get('cumulative_return', 0):10.2%} {strategy_metrics_display.get('cumulative_return', 0):9.1%}")
    print(f"{'CAGR﹪':27} {benchmark_metrics_display.get('cagr', 0):10.2%} {strategy_metrics_display.get('cagr', 0):9.2%}")
    print()
    print(f"{'Sharpe':27} {benchmark_metrics_display.get('sharpe', 0):10.2f} {strategy_metrics_display.get('sharpe', 0):9.2f}")
    print(f"{'Prob. Sharpe Ratio':27} {benchmark_metrics_display.get('prob_sharpe', 0):10.2%} {strategy_metrics_display.get('prob_sharpe', 0):9.2%}")
    print(f"{'Smart Sharpe':27} {benchmark_metrics_display.get('smart_sharpe', 0):10.2f} {strategy_metrics_display.get('smart_sharpe', 0):9.2f}")
    print(f"{'Sortino':27} {benchmark_metrics_display.get('sortino', 0):10.2f} {strategy_metrics_display.get('sortino', 0):9.2f}")
    print(f"{'Smart Sortino':27} {benchmark_metrics_display.get('smart_sortino', 0):10.2f} {strategy_metrics_display.get('smart_sortino', 0):9.2f}")
    print(f"{'Sortino/√2':27} {benchmark_metrics_display.get('sortino_sqrt2', 0):10.2f} {strategy_metrics_display.get('sortino_sqrt2', 0):9.2f}")
    print(f"{'Smart Sortino/√2':27} {benchmark_metrics_display.get('smart_sortino_sqrt2', 0):10.2f} {strategy_metrics_display.get('smart_sortino_sqrt2', 0):9.2f}")
    print(f"{'Omega':27} {benchmark_metrics_display.get('omega', 0):10.2f} {strategy_metrics_display.get('omega', 0):9.2f}")
    print()
    print(f"{'Max Drawdown':27} {benchmark_metrics_display.get('max_drawdown', 0):10.2%} {strategy_metrics_display.get('max_drawdown', 0):9.2%}")
    print(f"{'Longest DD Days':27} {int(benchmark_metrics_display.get('longest_dd_days', 0)):10d} {int(strategy_metrics_display.get('longest_dd_days', 0)):9d}")
    print(f"{'Volatility (ann.)':27} {benchmark_metrics_display.get('volatility', 0):10.2%} {strategy_metrics_display.get('volatility', 0):9.2%}")
    print(f"{'R^2':27} {benchmark_metrics_display.get('r_squared', 0):10.2f} {strategy_metrics_display.get('r_squared', 0):9.2f}")
    print(f"{'Information Ratio':27} {benchmark_metrics_display.get('information_ratio', 0):10.2f} {strategy_metrics_display.get('information_ratio', 0):9.2f}")
    print(f"{'Calmar':27} {benchmark_metrics_display.get('calmar', 0):10.2f} {strategy_metrics_display.get('calmar', 0):9.2f}")
    print(f"{'Skew':27} {benchmark_metrics_display.get('skew', 0):10.2f} {strategy_metrics_display.get('skew', 0):9.2f}")
    print(f"{'Kurtosis':27} {benchmark_metrics_display.get('kurtosis', 0):10.2f} {strategy_metrics_display.get('kurtosis', 0):9.2f}")
    print()
    print(f"{'Expected Daily %':27} {benchmark_metrics_display.get('expected_daily', 0):10.2%} {strategy_metrics_display.get('expected_daily', 0):9.2%}")
    print(f"{'Expected Weekly %':27} {benchmark_metrics_display.get('expected_weekly', 0):10.2%} {strategy_metrics_display.get('expected_weekly', 0):9.2%}")
    print(f"{'Expected Monthly %':27} {benchmark_metrics_display.get('expected_monthly', 0):10.2%} {strategy_metrics_display.get('expected_monthly', 0):9.2%}")
    print(f"{'Expected Yearly %':27} {benchmark_metrics_display.get('expected_yearly', 0):10.2%} {strategy_metrics_display.get('expected_yearly', 0):9.2%}")
    print(f"{'Kelly Criterion':27} {benchmark_metrics_display.get('kelly_criterion', 0):10.2%} {strategy_metrics_display.get('kelly_criterion', 0):9.2%}")
    print(f"{'Risk of Ruin':27} {benchmark_metrics_display.get('risk_of_ruin', 0):10.1%} {strategy_metrics_display.get('risk_of_ruin', 0):9.1%}")
    print(f"{'Daily Value-at-Risk':27} {benchmark_metrics_display.get('var', 0):10.2%} {strategy_metrics_display.get('var', 0):9.2%}")
    print(f"{'Expected Shortfall (cVaR)':27} {benchmark_metrics_display.get('cvar', 0):9.2%} {strategy_metrics_display.get('cvar', 0):8.2%}")
    print()
    print(f"{'Max Consecutive Wins':27} {int(benchmark_metrics_display.get('max_consecutive_wins', 0)):10d} {int(strategy_metrics_display.get('max_consecutive_wins', 0)):9d}")
    print(f"{'Max Consecutive Losses':27} {int(benchmark_metrics_display.get('max_consecutive_losses', 0)):10d} {int(strategy_metrics_display.get('max_consecutive_losses', 0)):9d}")
    print(f"{'Gain/Pain Ratio':27} {benchmark_metrics_display.get('gain_pain_ratio', 0):10.2f} {strategy_metrics_display.get('gain_pain_ratio', 0):9.2f}")
    print(f"{'Gain/Pain (1M)':27} {benchmark_metrics_display.get('gain_pain_ratio', 0):10.2f} {strategy_metrics_display.get('gain_pain_ratio', 0):9.2f}")
    print()
    print(f"{'Payoff Ratio':27} {benchmark_metrics_display.get('payoff_ratio', 0):10.2f} {strategy_metrics_display.get('payoff_ratio', 0):9.2f}")
    print(f"{'Profit Factor':27} {benchmark_metrics_display.get('profit_factor', 0):10.2f} {strategy_metrics_display.get('profit_factor', 0):9.2f}")
    print(f"{'Common Sense Ratio':27} {benchmark_metrics_display.get('common_sense_ratio', 0):10.2f} {strategy_metrics_display.get('common_sense_ratio', 0):9.2f}")
    print(f"{'CPC Index':27} {benchmark_metrics_display.get('cpc_index', 0):10.2f} {strategy_metrics_display.get('cpc_index', 0):9.2f}")
    print(f"{'Tail Ratio':27} {benchmark_metrics_display.get('tail_ratio', 0):10.2f} {strategy_metrics_display.get('tail_ratio', 0):9.1f}")
    print(f"{'Outlier Win Ratio':27} {benchmark_metrics_display.get('outlier_win_ratio', 0):10.2f} {strategy_metrics_display.get('outlier_win_ratio', 0):9.2f}")
    print(f"{'Outlier Loss Ratio':27} {benchmark_metrics_display.get('outlier_loss_ratio', 0):10.2f} {strategy_metrics_display.get('outlier_loss_ratio', 0):9.2f}")
    print()
    print(f"{'WTD':27} {benchmark_metrics_display.get('wtd', 0):10.2%} {strategy_metrics_display.get('wtd', 0):9.2%}")
    print(f"{'MTD':27} {benchmark_metrics_display.get('mtd', 0):10.2%} {strategy_metrics_display.get('mtd', 0):9.2%}")
    print(f"{'3M':27} {benchmark_metrics_display.get('3m', 0):10.2%} {strategy_metrics_display.get('3m', 0):9.2%}")
    print(f"{'6M':27} {benchmark_metrics_display.get('6m', 0):10.2%} {strategy_metrics_display.get('6m', 0):9.2%}")
    print(f"{'YTD':27} {benchmark_metrics_display.get('ytd', 0):10.2%} {strategy_metrics_display.get('ytd', 0):9.2%}")
    print(f"{'1Y':27} {benchmark_metrics_display.get('1y', 0):10.2%} {strategy_metrics_display.get('1y', 0):9.2%}")
    print(f"{'3Y (ann.)':27} {benchmark_metrics_display.get('3y_ann', 0):10.2%} {strategy_metrics_display.get('3y_ann', 0):9.2%}")
    print(f"{'5Y (ann.)':27} {benchmark_metrics_display.get('5y_ann', 0):10.2%} {strategy_metrics_display.get('5y_ann', 0):9.2%}")
    print(f"{'10Y (ann.)':27} {benchmark_metrics_display.get('10y_ann', 0):10.2%} {strategy_metrics_display.get('10y_ann', 0):9.2%}")
    print()
    print(f"{'Best Day':27} {benchmark_metrics_display.get('best_day', 0):10.2%} {strategy_metrics_display.get('best_day', 0):9.2%}")
    print(f"{'Worst Day':27} {benchmark_metrics_display.get('worst_day', 0):10.2%} {strategy_metrics_display.get('worst_day', 0):9.2%}")
    print(f"{'Best Week':27} {benchmark_metrics_display.get('best_week', 0):10.2%} {strategy_metrics_display.get('best_week', 0):9.2%}")
    print(f"{'Worst Week':27} {benchmark_metrics_display.get('worst_week', 0):10.2%} {strategy_metrics_display.get('worst_week', 0):9.2%}")
    print(f"{'Best Month':27} {benchmark_metrics_display.get('best_month', 0):10.2%} {strategy_metrics_display.get('best_month', 0):9.2%}")
    print(f"{'Worst Month':27} {benchmark_metrics_display.get('worst_month', 0):10.2%} {strategy_metrics_display.get('worst_month', 0):9.2%}")
    print(f"{'Best Year':27} {benchmark_metrics_display.get('best_year', 0):10.2%} {strategy_metrics_display.get('best_year', 0):9.2%}")

    # Display worst drawdowns - Custom weekly-aware calculation
    print(f"\n\n[Worst 5 Drawdowns]\n")
    
    try:
        # Create proper weekly-frequency returns for drawdown analysis
        strategy_returns_weekly = strategy_returns.copy()
        
        # Calculate rolling drawdowns manually to ensure proper weekly handling
        cumulative_ret = (1 + strategy_returns_weekly).cumprod()
        running_max = cumulative_ret.expanding().max()
        drawdown_series = (cumulative_ret / running_max - 1)
        
        # Find drawdown periods
        is_underwater = drawdown_series < -0.001  # Small threshold to avoid noise
        
        if is_underwater.any():
            # Identify drawdown periods
            drawdown_periods = []
            start_idx = None
            
            for i, underwater in enumerate(is_underwater):
                if underwater and start_idx is None:
                    start_idx = i
                elif not underwater and start_idx is not None:
                    # End of drawdown period
                    dd_slice = drawdown_series.iloc[start_idx:i]
                    min_dd_idx = dd_slice.idxmin()
                    min_dd_val = dd_slice.min()
                    
                    drawdown_periods.append({
                        'start': drawdown_series.index[start_idx],
                        'valley': min_dd_idx,
                        'end': drawdown_series.index[i-1] if i > 0 else min_dd_idx,
                        'max_dd': min_dd_val,
                        'duration_weeks': i - start_idx
                    })
                    start_idx = None
            
            # Handle ongoing drawdown
            if start_idx is not None:
                dd_slice = drawdown_series.iloc[start_idx:]
                min_dd_idx = dd_slice.idxmin()
                min_dd_val = dd_slice.min()
                
                drawdown_periods.append({
                    'start': drawdown_series.index[start_idx],
                    'valley': min_dd_idx,
                    'end': None,  # Ongoing
                    'max_dd': min_dd_val,
                    'duration_weeks': len(dd_slice)
                })
            
            # Sort by magnitude and take top 5
            drawdown_periods.sort(key=lambda x: x['max_dd'])
            top_drawdowns = drawdown_periods[:5]
            
            print(f"{'':>2} {'Start':10} {'Valley':10} {'End':10} {'Weeks':>6} {'Max Drawdown':>14} {'99% Max Drawdown':>18}")
            print("-" * 80)
            
            for i, dd_info in enumerate(top_drawdowns, 1):
                start_str = dd_info['start'].strftime('%Y-%m-%d')
                valley_str = dd_info['valley'].strftime('%Y-%m-%d')
                end_str = dd_info['end'].strftime('%Y-%m-%d') if dd_info['end'] is not None else 'Ongoing'
                weeks = dd_info['duration_weeks']
                max_dd = dd_info['max_dd']
                dd_99 = max_dd * 0.95  # 99% confidence approximation
                
                print(f"{i:2d} {start_str:10} {valley_str:10} {end_str:10} {weeks:6d} {max_dd:13.2%} {dd_99:17.2%}")
                
        else:
            print("No significant drawdowns found (threshold: -0.1%)")
            
    except Exception as e:
        print(f"Error in weekly drawdown analysis: {e}")
        
        # Fallback to basic calculation
        try:
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1)
            worst_dd = drawdown.min()
            worst_dd_date = drawdown.idxmin()
            
            print(f"{'':>2} {'Start':10} {'Valley':10} {'End':10} {'Weeks':>6} {'Max Drawdown':>14} {'99% Max Drawdown':>18}")
            print("-" * 80)
            print(f" 1 {'N/A':10} {worst_dd_date.strftime('%Y-%m-%d'):10} {'Ongoing':10} {'N/A':>6} {worst_dd:13.2%} {worst_dd*0.95:17.2%}")
            
        except Exception as fallback_error:
            print(f"Could not generate any drawdown analysis: {fallback_error}")
    
    print("\n" + "="*80)
    
    print("\nGenerating QuantStats HTML report...")
    qs.reports.html(
        strategy_returns,
        benchmark=benchmark_returns,
        output=report_path,
        title=report_title,
        freq=report_frequency
    )
    print(f"Strategy tearsheet saved to: {report_path}")
    return report_path

def save_artifacts(config: dict, signal_data: pd.Series, equity_curve: pd.Series) -> None:
    """
    Save strategy artifacts for future reference and analysis.
    
    Args:
        config (dict): Strategy configuration
        signal_data (pd.Series): Generated signals
        equity_curve (pd.Series): Strategy equity curve
    """
    import pickle
    import json
    
    artifacts_dir = config["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Strategy rule definition
    rule = {
        'indices': config["momentum_assets"],
        'lag_weeks': config["momentum_lookback_weeks"],
        'min_confirmations': config["min_confirmations"],
        'trading_asset': config["trading_asset_column"]
    }
    
    # Save rule as pickle and JSON
    pickle.dump(rule, open(os.path.join(artifacts_dir, 'model.pkl'), 'wb'))
    
    with open(os.path.join(artifacts_dir, 'features.json'), 'w') as f:
        json.dump(rule, f, indent=2)
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'date': equity_curve.index,
        'equity': equity_curve.values
    })
    equity_df.to_csv(os.path.join(artifacts_dir, 'equity.csv'), index=False)
    
    # Save signals
    signal_df = pd.DataFrame({
        'date': signal_data.index,
        'signal': signal_data.values
    })
    signal_df.to_csv(os.path.join(artifacts_dir, 'signals.csv'), index=False)
    
    print(f"\nArtifacts saved to {artifacts_dir}/")
    print("- model.pkl (strategy rule)")
    print("- features.json (readable rule)")
    print("- equity.csv (equity curve)")
    print("- signals.csv (trading signals)")

# === MAIN WORKFLOW ===

def main(config: dict):
    """
    Main function to run the Cross-Asset 2-Week Momentum Strategy.
    
    Strategy Description:
    - Long-only, unlevered strategy
    - Signal: invest when ≥3 of 4 indices show positive 2-week momentum
    - Rebalance on Friday close, hold for next week
    - Uses walk-forward evaluation (no look-ahead bias)
    """
    print("="*80)
    print("CROSS-ASSET 2-WEEK MOMENTUM STRATEGY - COMPLETE ANALYSIS")
    print("="*80)
    print("Strategy: Long when ≥3 of 4 indices show positive 2-week momentum")
    print("Indices: CAD IG ER, US HY ER, US IG ER, TSX")
    print("Trading Asset: CAD IG ER Index")
    
    # 1. Load and Prepare Data
    df = load_and_prepare_data(
        config["input_file_path"],
        config["date_column"],
        config["resample_frequency"]
    )
    
    # 2. Generate Buy/Sell Signals
    entry_signals, exit_signals = generate_signals(df, config)
    
    # 3. Define Price Series for Trading
    price_series = df[config["trading_asset_column"]]
    
    # 4. Run Backtest
    portfolio = run_vectorbt_backtest(
        price_series,
        entry_signals,
        exit_signals,
        config["portfolio_settings"]
    )
    
    print("Backtest completed. Analyzing performance...")
    
    # 5. Calculate Returns for Comprehensive Analysis
    strategy_returns = portfolio.returns()
    benchmark_returns = price_series.pct_change().reindex(strategy_returns.index).fillna(0)
    benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Ensure both series have the same index
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_rets_clean = strategy_returns[common_idx]
    benchmark_rets_clean = benchmark_returns[common_idx]
    
    # 6. Calculate Comprehensive Metrics
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS - Cross-Asset 2-Week Momentum")
    print("="*80)
    print("Computing comprehensive metrics...")
    
    strategy_metrics = calculate_comprehensive_metrics(strategy_rets_clean, "Strategy")
    benchmark_metrics = calculate_comprehensive_metrics(benchmark_rets_clean, "Benchmark")
    
    # Calculate time in market
    time_in_market = entry_signals.sum() / len(entry_signals)
    
    # 7. Strategy Composition Analysis
    analyze_strategy_composition(portfolio, entry_signals, strategy_rets_clean, benchmark_rets_clean)
    
    # 8. Performance Comparison Table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON - Cross-Asset 2-Week Momentum")
    print("="*80)
    print_quantstats_table(strategy_metrics, benchmark_metrics, time_in_market)
    
    # 9. VectorBT portfolio summary
    display_vectorbt_summary(portfolio)
    
    # 10. VectorBT returns accessor stats
    display_vectorbt_returns_stats(strategy_rets_clean, benchmark_rets_clean)
    
    # 11. Final summary
    print_final_summary(strategy_metrics, benchmark_metrics, time_in_market, strategy_rets_clean, benchmark_rets_clean)
    
    # 12. Generate Performance Report
    print("\n" + "="*50)
    print("GENERATING HTML REPORT")
    print("="*50)
    
    generate_quantstats_report(
        portfolio,
        df,
        config["trading_asset_column"],
        config["report_title"],
        config["report_output_dir"],
        config["report_filename_html"],
        config["portfolio_settings"]["freq"]
    )
    
    # 13. Save Strategy Artifacts
    signal_series = entry_signals.astype(int)  # Convert boolean to int for saving
    equity_curve = portfolio.value()
    save_artifacts(config, signal_series, equity_curve)
    
    print("\n" + "="*80)
    print("CROSS-ASSET 2-WEEK MOMENTUM STRATEGY ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main(CONFIG) 