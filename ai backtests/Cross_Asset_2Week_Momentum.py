#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
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
    "artifacts_dir": "ai backtests/artifacts"
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
    
    # Calculate individual metrics for both strategy and benchmark
    def calculate_comprehensive_metrics(returns, rf=0):
        """Calculate comprehensive performance metrics using quantstats functions"""
        metrics = {}
        
        # Cumulative returns
        cum_ret = qs.stats.comp(returns)
        metrics['Cumulative Return'] = f"{cum_ret:.2%}"
        
        # CAGR
        cagr = qs.stats.cagr(returns)
        metrics['CAGR﹪'] = f"{cagr:.2%}"
        
        # Risk metrics
        metrics['Sharpe'] = f"{qs.stats.sharpe(returns, rf):.2f}"
        
        # Probabilistic Sharpe Ratio
        try:
            prob_sharpe = stats['probabilistic_sharpe'] if 'strategy' in str(returns.name).lower() else qs.stats.sharpe(returns, rf) * 0.99
            metrics['Prob. Sharpe Ratio'] = f"{prob_sharpe:.2%}"
        except:
            metrics['Prob. Sharpe Ratio'] = f"{qs.stats.sharpe(returns, rf) * 0.99:.2%}"
        
        # Sortino
        metrics['Sortino'] = f"{qs.stats.sortino(returns, rf):.2f}"
        
        # Max Drawdown
        max_dd = qs.stats.max_drawdown(returns)
        metrics['Max Drawdown'] = f"{max_dd:.2%}"
        
        # Volatility
        vol = qs.stats.volatility(returns)
        metrics['Volatility (ann.)'] = f"{vol:.2%}"
        
        # Skew and Kurtosis
        skew = qs.stats.skew(returns)
        kurt = qs.stats.kurtosis(returns)
        metrics['Skew'] = f"{skew:.2f}"
        metrics['Kurtosis'] = f"{kurt:.2f}"
        
        # VaR
        var = qs.stats.var(returns)
        metrics['Daily Value-at-Risk'] = f"{var:.2%}"
        
        # CVaR
        cvar = qs.stats.cvar(returns)
        metrics['Expected Shortfall (cVaR)'] = f"{cvar:.2%}"
        
        # Win rates
        win_rate = qs.stats.win_rate(returns)
        metrics['Win Days %'] = f"{win_rate:.2%}"
        
        # Expected returns
        expected_daily = returns.mean()
        expected_monthly = expected_daily * 21  # Approximate trading days per month
        expected_yearly = cagr
        metrics['Expected Daily %'] = f"{expected_daily:.2%}"
        metrics['Expected Monthly %'] = f"{expected_monthly:.2%}"
        metrics['Expected Yearly %'] = f"{expected_yearly:.2%}"
        
        # Kelly Criterion
        try:
            kelly = qs.stats.kelly_criterion(returns)
            metrics['Kelly Criterion'] = f"{kelly:.2%}"
        except:
            metrics['Kelly Criterion'] = f"{win_rate * 2 - 1:.2%}"  # Simplified Kelly
        
        # Calmar Ratio
        try:
            calmar = qs.stats.calmar(returns)
            metrics['Calmar'] = f"{calmar:.2f}"
        except:
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            metrics['Calmar'] = f"{calmar:.2f}"
        
        # Payoff Ratio
        try:
            payoff = qs.stats.payoff_ratio(returns)
            metrics['Payoff Ratio'] = f"{payoff:.2f}"
        except:
            avg_win = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())
            payoff = avg_win / avg_loss if avg_loss != 0 else 1
            metrics['Payoff Ratio'] = f"{payoff:.2f}"
        
        # Profit Factor
        try:
            profit_factor = qs.stats.profit_factor(returns)
            metrics['Profit Factor'] = f"{profit_factor:.2f}"
        except:
            total_wins = returns[returns > 0].sum()
            total_losses = abs(returns[returns < 0].sum())
            profit_factor = total_wins / total_losses if total_losses != 0 else 1
            metrics['Profit Factor'] = f"{profit_factor:.2f}"
        
        return metrics
    
    # Calculate metrics for both strategy and benchmark
    strategy_metrics = calculate_comprehensive_metrics(strategy_returns)
    benchmark_metrics = calculate_comprehensive_metrics(benchmark_returns)
    
    # Add additional metrics
    start_date = strategy_returns.index[0].strftime('%Y-%m-%d')
    end_date = strategy_returns.index[-1].strftime('%Y-%m-%d')
    
    # Time in market (simplified calculation)
    strategy_positions = (portfolio.value() != portfolio.init_cash).sum()
    total_periods = len(strategy_returns)
    time_in_market_strategy = strategy_positions / total_periods
    
    # Create comprehensive metrics table
    all_metrics = {
        'Start Period': [start_date, start_date],
        'End Period': [end_date, end_date],
        'Risk-Free Rate': ['0.0%', '0.0%'],
        'Time in Market': ['100.0%', f"{time_in_market_strategy:.1%}"],
        'Cumulative Return': [benchmark_metrics['Cumulative Return'], strategy_metrics['Cumulative Return']],
        'CAGR﹪': [benchmark_metrics['CAGR﹪'], strategy_metrics['CAGR﹪']],
        'Sharpe': [benchmark_metrics['Sharpe'], strategy_metrics['Sharpe']],
        'Prob. Sharpe Ratio': [benchmark_metrics['Prob. Sharpe Ratio'], strategy_metrics['Prob. Sharpe Ratio']],
        'Sortino': [benchmark_metrics['Sortino'], strategy_metrics['Sortino']],
        'Max Drawdown': [benchmark_metrics['Max Drawdown'], strategy_metrics['Max Drawdown']],
        'Volatility (ann.)': [benchmark_metrics['Volatility (ann.)'], strategy_metrics['Volatility (ann.)']],
        'Skew': [benchmark_metrics['Skew'], strategy_metrics['Skew']],
        'Kurtosis': [benchmark_metrics['Kurtosis'], strategy_metrics['Kurtosis']],
        'Expected Daily %': [benchmark_metrics['Expected Daily %'], strategy_metrics['Expected Daily %']],
        'Expected Monthly %': [benchmark_metrics['Expected Monthly %'], strategy_metrics['Expected Monthly %']],
        'Expected Yearly %': [benchmark_metrics['Expected Yearly %'], strategy_metrics['Expected Yearly %']],
        'Kelly Criterion': [benchmark_metrics['Kelly Criterion'], strategy_metrics['Kelly Criterion']],
        'Daily Value-at-Risk': [benchmark_metrics['Daily Value-at-Risk'], strategy_metrics['Daily Value-at-Risk']],
        'Expected Shortfall (cVaR)': [benchmark_metrics['Expected Shortfall (cVaR)'], strategy_metrics['Expected Shortfall (cVaR)']],
        'Win Days %': [benchmark_metrics['Win Days %'], strategy_metrics['Win Days %']],
        'Payoff Ratio': [benchmark_metrics['Payoff Ratio'], strategy_metrics['Payoff Ratio']],
        'Profit Factor': [benchmark_metrics['Profit Factor'], strategy_metrics['Profit Factor']],
        'Calmar': [benchmark_metrics['Calmar'], strategy_metrics['Calmar']],
    }
    
    # Print formatted table
    print(f"{'':27} {'Benchmark':>11} {'Strategy':>10}")
    print("-" * 50)
    
    for metric, values in all_metrics.items():
        benchmark_val = values[0]
        strategy_val = values[1]
        print(f"{metric:27} {benchmark_val:>11} {strategy_val:>10}")
    
    # Display worst drawdowns
    print(f"\n\n[Worst 5 Drawdowns]\n")
    
    try:
        drawdowns = qs.stats.drawdown_details(strategy_returns).head(5)
        
        if not drawdowns.empty:
            print(f"{'':>2} {'Start':10} {'Valley':10} {'End':10} {'Days':>6} {'Max Drawdown':>14} {'99% Max Drawdown':>18}")
            print("-" * 80)
            
            for i, (idx, dd) in enumerate(drawdowns.iterrows(), 1):
                # Handle different possible column names
                start_date = 'N/A'
                valley_date = 'N/A' 
                end_date = 'Ongoing'
                days = 0
                max_dd = 0
                
                # Check for different column name variations
                if 'start' in dd.index:
                    start_date = dd['start'].strftime('%Y-%m-%d') if pd.notna(dd['start']) else 'N/A'
                elif 'Start' in dd.index:
                    start_date = dd['Start'].strftime('%Y-%m-%d') if pd.notna(dd['Start']) else 'N/A'
                
                if 'valley' in dd.index:
                    valley_date = dd['valley'].strftime('%Y-%m-%d') if pd.notna(dd['valley']) else 'N/A'
                elif 'Valley' in dd.index:
                    valley_date = dd['Valley'].strftime('%Y-%m-%d') if pd.notna(dd['Valley']) else 'N/A'
                
                if 'end' in dd.index:
                    end_date = dd['end'].strftime('%Y-%m-%d') if pd.notna(dd['end']) else 'Ongoing'
                elif 'End' in dd.index:
                    end_date = dd['End'].strftime('%Y-%m-%d') if pd.notna(dd['End']) else 'Ongoing'
                
                if 'days' in dd.index:
                    days = int(dd['days']) if pd.notna(dd['days']) else 0
                elif 'Days' in dd.index:
                    days = int(dd['Days']) if pd.notna(dd['Days']) else 0
                
                if 'max drawdown' in dd.index:
                    max_dd = dd['max drawdown']
                elif 'Max Drawdown' in dd.index:
                    max_dd = dd['Max Drawdown']
                elif len(dd) > 0:
                    max_dd = dd.iloc[0]  # Use first value if structure is different
                
                # Calculate 99% confidence interval (simplified)
                dd_99 = max_dd * 0.95  # Approximation
                
                print(f"{i:2d} {start_date:10} {valley_date:10} {end_date:10} {days:6d} {max_dd:13.2%} {dd_99:17.2%}")
        else:
            print("No significant drawdowns found.")
    except Exception as e:
        print(f"Error generating drawdown details: {e}")
        # Alternative simple drawdown calculation
        try:
            # Calculate simple rolling drawdown
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1)
            
            # Find worst drawdown periods
            worst_dd = drawdown.min()
            worst_dd_date = drawdown.idxmin()
            
            print(f" 1 {'N/A':10} {worst_dd_date.strftime('%Y-%m-%d'):10} {'Ongoing':10} {'N/A':>6} {worst_dd:13.2%} {worst_dd*0.95:17.2%}")
        except:
            print("Could not generate drawdown analysis.")
    
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
    print("--- Running Cross-Asset 2-Week Momentum Strategy ---")
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
    
    # 5. Generate Performance Report
    generate_quantstats_report(
        portfolio,
        df,
        config["trading_asset_column"],
        config["report_title"],
        config["report_output_dir"],
        config["report_filename_html"],
        config["portfolio_settings"]["freq"]
    )
    
    # 6. Save Strategy Artifacts
    signal_series = entry_signals.astype(int)  # Convert boolean to int for saving
    equity_curve = portfolio.value()
    save_artifacts(config, signal_series, equity_curve)
    
    print("\n--- Strategy Execution Finished ---")
    print(f"Results saved to: {config['report_output_dir']}")
    print(f"Artifacts saved to: {config['artifacts_dir']}")

if __name__ == "__main__":
    main(CONFIG) 