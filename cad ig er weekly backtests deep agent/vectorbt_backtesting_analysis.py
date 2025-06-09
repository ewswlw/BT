#!/usr/bin/env python3
"""
Comprehensive VectorBT Backtesting Analysis
==========================================

This script recreates the complete backtesting analysis using the vectorbt library
as specifically requested, with detailed portfolio statistics and comparisons.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set vectorbt settings for better performance and display
vbt.settings.set_theme("dark")
vbt.settings.portfolio.stats['incl_unrealized'] = True

print("=" * 80)
print("VECTORBT COMPREHENSIVE BACKTESTING ANALYSIS")
print("=" * 80)

# Load the data
print("\n1. LOADING DATA")
print("-" * 40)

# Load original market data
market_data = pd.read_csv('/home/ubuntu/Uploads/with_er_daily.csv')
market_data['Date'] = pd.to_datetime(market_data['Date'])
market_data.set_index('Date', inplace=True)

# Load trading signals
signals_data = pd.read_csv('/home/ubuntu/trading_signals.csv')
signals_data['date'] = pd.to_datetime(signals_data['date'])
signals_data.set_index('date', inplace=True)

print(f"Market data shape: {market_data.shape}")
print(f"Signals data shape: {signals_data.shape}")
print(f"Date range: {signals_data.index.min()} to {signals_data.index.max()}")
print(f"Total trading periods: {len(signals_data)}")

# Prepare data for vectorbt
prices = signals_data['price'].copy()
signals = signals_data['signal'].copy()
returns = signals_data['actual_return'].copy()

print(f"\nSignal distribution:")
print(f"Long signals (1): {(signals == 1).sum()} ({(signals == 1).mean()*100:.1f}%)")
print(f"Cash signals (0): {(signals == 0).sum()} ({(signals == 0).mean()*100:.1f}%)")

# 2. VECTORBT PORTFOLIO SETUP
print("\n2. VECTORBT PORTFOLIO SETUP")
print("-" * 40)

# Set up portfolio parameters
initial_cash = 100000  # $100,000 initial capital
freq = 'W'  # Weekly rebalancing

# Create entries and exits based on signals
# Long-only binary exposure: 100% invested when signal=1, 0% when signal=0
entries = signals == 1
exits = signals == 0

print(f"Entry signals: {entries.sum()}")
print(f"Exit signals: {exits.sum()}")

# Create vectorbt portfolio for the strategy
print("\nCreating strategy portfolio...")
strategy_pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits,
    init_cash=initial_cash,
    freq=freq,
    fees=0.0,  # No transaction fees for clean comparison
    size=np.inf,  # Use all available cash (100% invested when signal=1)
    accumulate=False  # Don't accumulate positions
)

# Create buy-and-hold benchmark portfolio
print("Creating buy-and-hold benchmark portfolio...")
bnh_entries = pd.Series(False, index=prices.index)
bnh_entries.iloc[0] = True  # Buy at the beginning
bnh_exits = pd.Series(False, index=prices.index)  # Never sell

benchmark_pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=bnh_entries,
    exits=bnh_exits,
    init_cash=initial_cash,
    freq=freq,
    fees=0.0,
    size=np.inf,
    size_type='amount',
    accumulate=False
)

print("Portfolio objects created successfully!")

# 3. COMPREHENSIVE PORTFOLIO STATISTICS
print("\n3. COMPREHENSIVE PORTFOLIO STATISTICS")
print("=" * 60)

print("\n3.1 STRATEGY PORTFOLIO STATISTICS (pf.stats())")
print("-" * 50)
strategy_stats = strategy_pf.stats()
print(strategy_stats)

print("\n3.2 BENCHMARK PORTFOLIO STATISTICS (pf.stats())")
print("-" * 50)
benchmark_stats = benchmark_pf.stats()
print(benchmark_stats)

# 4. RETURNS ANALYSIS
print("\n4. RETURNS ANALYSIS")
print("=" * 60)

print("\n4.1 STRATEGY RETURNS STATISTICS")
print("-" * 55)
strategy_returns = strategy_pf.returns()
print(f"Strategy Returns - Mean: {strategy_returns.mean():.6f}")
print(f"Strategy Returns - Std: {strategy_returns.std():.6f}")
print(f"Strategy Returns - Skewness: {strategy_returns.skew():.6f}")
print(f"Strategy Returns - Kurtosis: {strategy_returns.kurtosis():.6f}")
print(f"Strategy Returns - Min: {strategy_returns.min():.6f}")
print(f"Strategy Returns - Max: {strategy_returns.max():.6f}")

print("\n4.2 BENCHMARK RETURNS STATISTICS")
print("-" * 55)
benchmark_returns = benchmark_pf.returns()
print(f"Benchmark Returns - Mean: {benchmark_returns.mean():.6f}")
print(f"Benchmark Returns - Std: {benchmark_returns.std():.6f}")
print(f"Benchmark Returns - Skewness: {benchmark_returns.skew():.6f}")
print(f"Benchmark Returns - Kurtosis: {benchmark_returns.kurtosis():.6f}")
print(f"Benchmark Returns - Min: {benchmark_returns.min():.6f}")
print(f"Benchmark Returns - Max: {benchmark_returns.max():.6f}")

# 5. DETAILED PERFORMANCE COMPARISON
print("\n5. DETAILED PERFORMANCE COMPARISON")
print("=" * 60)

# Extract key metrics for comparison
strategy_total_return = strategy_pf.total_return()
benchmark_total_return = benchmark_pf.total_return()
strategy_annual_return = strategy_pf.annualized_return()
benchmark_annual_return = benchmark_pf.annualized_return()
strategy_sharpe = strategy_pf.sharpe_ratio()
benchmark_sharpe = benchmark_pf.sharpe_ratio()
strategy_max_dd = strategy_pf.max_drawdown()
benchmark_max_dd = benchmark_pf.max_drawdown()
strategy_calmar = strategy_pf.calmar_ratio()
benchmark_calmar = benchmark_pf.calmar_ratio()
strategy_volatility = strategy_pf.annualized_volatility()
benchmark_volatility = benchmark_pf.annualized_volatility()

# Create comparison table
comparison_data = {
    'Strategy': [
        f"{strategy_total_return:.2%}",
        f"{strategy_annual_return:.2%}",
        f"{strategy_sharpe:.3f}",
        f"{strategy_max_dd:.2%}",
        f"{strategy_calmar:.3f}",
        f"{strategy_volatility:.2%}",
        f"{strategy_pf.trades.win_rate():.2%}",
        f"{strategy_pf.trades.profit_factor():.3f}",
        f"{len(strategy_pf.trades.records_readable)}"
    ],
    'Buy & Hold': [
        f"{benchmark_total_return:.2%}",
        f"{benchmark_annual_return:.2%}",
        f"{benchmark_sharpe:.3f}",
        f"{benchmark_max_dd:.2%}",
        f"{benchmark_calmar:.3f}",
        f"{benchmark_volatility:.2%}",
        f"{benchmark_pf.trades.win_rate():.2%}",
        f"{benchmark_pf.trades.profit_factor():.3f}",
        f"{len(benchmark_pf.trades.records_readable)}"
    ]
}

comparison_df = pd.DataFrame(comparison_data, index=[
    'Total Return',
    'Annualized Return',
    'Sharpe Ratio',
    'Max Drawdown',
    'Calmar Ratio',
    'Annualized Volatility',
    'Win Rate',
    'Profit Factor',
    'Number of Trades'
])

print("\nKEY PERFORMANCE METRICS COMPARISON:")
print("=" * 50)
print(comparison_df)

# Calculate outperformance
outperformance = strategy_total_return - benchmark_total_return
annual_outperformance = strategy_annual_return - benchmark_annual_return

print(f"\nOUTPERFORMANCE ANALYSIS:")
print(f"Total Outperformance: {outperformance:.2%}")
print(f"Annualized Outperformance: {annual_outperformance:.2%}")
print(f"Sharpe Ratio Improvement: {strategy_sharpe - benchmark_sharpe:.3f}")

# 6. DRAWDOWN ANALYSIS
print("\n6. DRAWDOWN ANALYSIS")
print("=" * 60)

strategy_dd = strategy_pf.drawdown()
benchmark_dd = benchmark_pf.drawdown()

print(f"Strategy Max Drawdown: {strategy_max_dd:.2%}")
print(f"Benchmark Max Drawdown: {benchmark_max_dd:.2%}")
print(f"Drawdown Improvement: {benchmark_max_dd - strategy_max_dd:.2%}")

# Drawdown duration analysis
try:
    strategy_dd_info = strategy_pf.drawdowns.records_readable
    benchmark_dd_info = benchmark_pf.drawdowns.records_readable
    
    print(f"\nDrawdown Duration Analysis:")
    print(f"Strategy drawdown records columns: {list(strategy_dd_info.columns) if len(strategy_dd_info) > 0 else 'No drawdowns'}")
    print(f"Benchmark drawdown records columns: {list(benchmark_dd_info.columns) if len(benchmark_dd_info) > 0 else 'No drawdowns'}")
    
    if len(strategy_dd_info) > 0:
        # Check available columns and use appropriate one
        duration_col = None
        for col in ['Duration', 'duration', 'Days', 'days']:
            if col in strategy_dd_info.columns:
                duration_col = col
                break
        
        if duration_col:
            print(f"Strategy Avg DD Duration: {strategy_dd_info[duration_col].mean():.1f} periods")
            print(f"Strategy Max DD Duration: {strategy_dd_info[duration_col].max():.0f} periods")
        print(f"Strategy Number of Drawdowns: {len(strategy_dd_info)}")
    else:
        print("Strategy: No significant drawdowns")
    
    if len(benchmark_dd_info) > 0:
        # Check available columns and use appropriate one
        duration_col = None
        for col in ['Duration', 'duration', 'Days', 'days']:
            if col in benchmark_dd_info.columns:
                duration_col = col
                break
        
        if duration_col:
            print(f"Benchmark Avg DD Duration: {benchmark_dd_info[duration_col].mean():.1f} periods")
            print(f"Benchmark Max DD Duration: {benchmark_dd_info[duration_col].max():.0f} periods")
        print(f"Benchmark Number of Drawdowns: {len(benchmark_dd_info)}")
    else:
        print("Benchmark: No significant drawdowns")
        
except Exception as e:
    print(f"Drawdown analysis error: {e}")
    print("Skipping detailed drawdown duration analysis")

# 7. RISK-ADJUSTED RETURNS ANALYSIS
print("\n7. RISK-ADJUSTED RETURNS ANALYSIS")
print("=" * 60)

# Calculate additional risk metrics
strategy_sortino = strategy_pf.sortino_ratio()
benchmark_sortino = benchmark_pf.sortino_ratio()
strategy_omega = strategy_pf.omega_ratio()
benchmark_omega = benchmark_pf.omega_ratio()

print(f"Sortino Ratio - Strategy: {strategy_sortino:.3f}, Benchmark: {benchmark_sortino:.3f}")
print(f"Omega Ratio - Strategy: {strategy_omega:.3f}, Benchmark: {benchmark_omega:.3f}")

# Value at Risk analysis
strategy_var_95 = np.percentile(strategy_returns.dropna(), 5)
benchmark_var_95 = np.percentile(benchmark_returns.dropna(), 5)

print(f"95% VaR - Strategy: {strategy_var_95:.2%}, Benchmark: {benchmark_var_95:.2%}")

# 8. STATISTICAL SIGNIFICANCE TESTING
print("\n8. STATISTICAL SIGNIFICANCE TESTING")
print("=" * 60)

from scipy import stats

# Get returns for statistical testing (already calculated above)
strategy_returns_clean = strategy_returns.dropna()
benchmark_returns_clean = benchmark_returns.dropna()

# Align returns for proper comparison
aligned_returns = pd.concat([strategy_returns_clean, benchmark_returns_clean], axis=1, keys=['Strategy', 'Benchmark']).dropna()

# T-test for difference in means
t_stat, p_value = stats.ttest_rel(aligned_returns['Strategy'], aligned_returns['Benchmark'])

print(f"Paired T-Test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Statistically significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")

# Correlation analysis
correlation = aligned_returns['Strategy'].corr(aligned_returns['Benchmark'])
print(f"\nCorrelation between Strategy and Benchmark: {correlation:.4f}")

# 9. TRADE ANALYSIS
print("\n9. TRADE ANALYSIS")
print("=" * 60)

strategy_trades = strategy_pf.trades.records_readable
print(f"Total number of trades: {len(strategy_trades)}")

if len(strategy_trades) > 0:
    print(f"Average trade return: {strategy_trades['Return'].mean():.2%}")
    print(f"Best trade: {strategy_trades['Return'].max():.2%}")
    print(f"Worst trade: {strategy_trades['Return'].min():.2%}")
    print(f"Win rate: {(strategy_trades['Return'] > 0).mean():.2%}")
    
    # Trade duration analysis
    duration_col = None
    for col in ['Duration', 'duration', 'Days', 'days']:
        if col in strategy_trades.columns:
            duration_col = col
            break
    
    if duration_col:
        print(f"Average trade duration: {strategy_trades[duration_col].mean():.1f} periods")
        print(f"Longest trade: {strategy_trades[duration_col].max():.0f} periods")
        print(f"Shortest trade: {strategy_trades[duration_col].min():.0f} periods")
    else:
        print("Trade duration information not available in this format")

# 10. PERFORMANCE ATTRIBUTION
print("\n10. PERFORMANCE ATTRIBUTION")
print("=" * 60)

# Calculate monthly returns for attribution
strategy_monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
benchmark_monthly_returns = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

monthly_outperformance = strategy_monthly_returns - benchmark_monthly_returns
positive_months = (monthly_outperformance > 0).sum()
total_months = len(monthly_outperformance)

print(f"Monthly outperformance analysis:")
print(f"Months with positive outperformance: {positive_months}/{total_months} ({positive_months/total_months:.1%})")
print(f"Average monthly outperformance: {monthly_outperformance.mean():.2%}")
print(f"Best month outperformance: {monthly_outperformance.max():.2%}")
print(f"Worst month outperformance: {monthly_outperformance.min():.2%}")

# 11. SAVE DETAILED RESULTS
print("\n11. SAVING DETAILED RESULTS")
print("=" * 60)

# Save comprehensive results
results_summary = {
    'Metric': [
        'Total Return (%)',
        'Annualized Return (%)',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Calmar Ratio',
        'Max Drawdown (%)',
        'Volatility (%)',
        'Win Rate (%)',
        'Profit Factor',
        'Number of Trades',
        'Avg Trade Duration',
        'VaR 95% (%)',
        'Correlation with Benchmark'
    ],
    'Strategy': [
        f"{strategy_total_return:.2f}",
        f"{strategy_annual_return:.2f}",
        f"{strategy_sharpe:.3f}",
        f"{strategy_sortino:.3f}",
        f"{strategy_calmar:.3f}",
        f"{strategy_max_dd:.2f}",
        f"{strategy_volatility:.2f}",
        f"{strategy_pf.trades.win_rate():.2f}",
        f"{strategy_pf.trades.profit_factor():.3f}",
        f"{len(strategy_pf.trades.records_readable)}",
        "N/A",
        f"{strategy_var_95:.2f}",
        f"{correlation:.3f}"
    ],
    'Benchmark': [
        f"{benchmark_total_return:.2f}",
        f"{benchmark_annual_return:.2f}",
        f"{benchmark_sharpe:.3f}",
        f"{benchmark_sortino:.3f}",
        f"{benchmark_calmar:.3f}",
        f"{benchmark_max_dd:.2f}",
        f"{benchmark_volatility:.2f}",
        f"{benchmark_pf.trades.win_rate():.2f}",
        f"{benchmark_pf.trades.profit_factor():.3f}",
        f"{len(benchmark_pf.trades.records_readable)}",
        "N/A",
        f"{benchmark_var_95:.2f}",
        "1.000"
    ],
    'Difference': [
        f"{outperformance:.2f}",
        f"{annual_outperformance:.2f}",
        f"{strategy_sharpe - benchmark_sharpe:.3f}",
        f"{strategy_sortino - benchmark_sortino:.3f}",
        f"{strategy_calmar - benchmark_calmar:.3f}",
        f"{benchmark_max_dd - strategy_max_dd:.2f}",
        f"{strategy_volatility - benchmark_volatility:.2f}",
        f"{strategy_pf.trades.win_rate() - benchmark_pf.trades.win_rate():.2f}",
        f"{strategy_pf.trades.profit_factor() - benchmark_pf.trades.profit_factor():.3f}",
        f"{len(strategy_pf.trades.records_readable) - len(benchmark_pf.trades.records_readable)}",
        "N/A",
        f"{strategy_var_95 - benchmark_var_95:.2f}",
        "N/A"
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('/home/ubuntu/vectorbt_comprehensive_results.csv', index=False)

print("Comprehensive results saved to: vectorbt_comprehensive_results.csv")

# Save trade details if available
if len(strategy_trades) > 0:
    strategy_trades.to_csv('/home/ubuntu/vectorbt_trade_details.csv', index=False)
    print("Trade details saved to: vectorbt_trade_details.csv")

print("\n" + "=" * 80)
print("VECTORBT BACKTESTING ANALYSIS COMPLETED")
print("=" * 80)

# Store portfolio objects for visualization
print("\nStoring portfolio objects for visualization...")
strategy_pf_global = strategy_pf
benchmark_pf_global = benchmark_pf
