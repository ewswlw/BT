#!/usr/bin/env python3
"""
VectorBT Comprehensive Visualizations
====================================

This script creates comprehensive visualizations using vectorbt's built-in plotting capabilities
and additional custom charts for the backtesting analysis.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VECTORBT COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Load the data and recreate portfolios
print("\n1. LOADING DATA AND RECREATING PORTFOLIOS")
print("-" * 50)

# Load data
signals_data = pd.read_csv('/home/ubuntu/trading_signals.csv')
signals_data['date'] = pd.to_datetime(signals_data['date'])
signals_data.set_index('date', inplace=True)

prices = signals_data['price'].copy()
signals = signals_data['signal'].copy()

# Portfolio parameters
initial_cash = 100000
freq = 'W'

# Create entries and exits
entries = signals == 1
exits = signals == 0

# Create portfolios
strategy_pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits,
    init_cash=initial_cash,
    freq=freq,
    fees=0.0,
    size=np.inf,
    accumulate=False
)

bnh_entries = pd.Series(False, index=prices.index)
bnh_entries.iloc[0] = True
bnh_exits = pd.Series(False, index=prices.index)

benchmark_pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=bnh_entries,
    exits=bnh_exits,
    init_cash=initial_cash,
    freq=freq,
    fees=0.0,
    size=np.inf,
    accumulate=False
)

print("Portfolios recreated successfully!")

# 2. PORTFOLIO VALUE EVOLUTION
print("\n2. PORTFOLIO VALUE EVOLUTION")
print("-" * 50)

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Portfolio Value Evolution', 'Cumulative Returns Comparison'),
    vertical_spacing=0.1,
    row_heights=[0.7, 0.3]
)

# Portfolio values
strategy_value = strategy_pf.value()
benchmark_value = benchmark_pf.value()

fig.add_trace(
    go.Scatter(
        x=strategy_value.index,
        y=strategy_value.values,
        name='Strategy',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=benchmark_value.index,
        y=benchmark_value.values,
        name='Buy & Hold',
        line=dict(color='red', width=2),
        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ),
    row=1, col=1
)

# Cumulative returns
strategy_cumret = strategy_pf.total_return()
benchmark_cumret = benchmark_pf.total_return()

strategy_returns = strategy_pf.returns()
benchmark_returns = benchmark_pf.returns()
strategy_cumulative = (1 + strategy_returns).cumprod()
benchmark_cumulative = (1 + benchmark_returns).cumprod()

fig.add_trace(
    go.Scatter(
        x=strategy_cumulative.index,
        y=strategy_cumulative.values,
        name='Strategy Cumulative',
        line=dict(color='blue', width=2),
        showlegend=False,
        hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.3f}<extra></extra>'
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=benchmark_cumulative.index,
        y=benchmark_cumulative.values,
        name='Benchmark Cumulative',
        line=dict(color='red', width=2),
        showlegend=False,
        hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.3f}<extra></extra>'
    ),
    row=2, col=1
)

fig.update_layout(
    title='Portfolio Performance Analysis',
    height=800,
    template='plotly_dark'
)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)

fig.write_html('/home/ubuntu/portfolio_evolution.html')
print("Portfolio evolution chart saved to: portfolio_evolution.html")

# 3. DRAWDOWN ANALYSIS
print("\n3. DRAWDOWN ANALYSIS")
print("-" * 50)

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Strategy Drawdown', 'Benchmark Drawdown'),
    vertical_spacing=0.1
)

strategy_dd = strategy_pf.drawdown()
benchmark_dd = benchmark_pf.drawdown()

fig.add_trace(
    go.Scatter(
        x=strategy_dd.index,
        y=strategy_dd.values * 100,
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=1),
        name='Strategy Drawdown',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=benchmark_dd.index,
        y=benchmark_dd.values * 100,
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=1),
        name='Benchmark Drawdown',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ),
    row=2, col=1
)

fig.update_layout(
    title='Drawdown Analysis',
    height=600,
    template='plotly_dark'
)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

fig.write_html('/home/ubuntu/drawdown_analysis.html')
print("Drawdown analysis chart saved to: drawdown_analysis.html")

# 4. RETURNS DISTRIBUTION
print("\n4. RETURNS DISTRIBUTION")
print("-" * 50)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Strategy Returns Distribution', 'Benchmark Returns Distribution'),
    horizontal_spacing=0.1
)

# Strategy returns histogram
fig.add_trace(
    go.Histogram(
        x=strategy_returns.values * 100,
        nbinsx=50,
        name='Strategy',
        opacity=0.7,
        marker_color='blue',
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ),
    row=1, col=1
)

# Benchmark returns histogram
fig.add_trace(
    go.Histogram(
        x=benchmark_returns.values * 100,
        nbinsx=50,
        name='Benchmark',
        opacity=0.7,
        marker_color='red',
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ),
    row=1, col=2
)

fig.update_layout(
    title='Returns Distribution Analysis',
    height=500,
    template='plotly_dark'
)

fig.update_xaxes(title_text="Daily Returns (%)", row=1, col=1)
fig.update_xaxes(title_text="Daily Returns (%)", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

fig.write_html('/home/ubuntu/returns_distribution.html')
print("Returns distribution chart saved to: returns_distribution.html")

# 5. ROLLING PERFORMANCE METRICS
print("\n5. ROLLING PERFORMANCE METRICS")
print("-" * 50)

# Calculate rolling metrics
window = 252  # 1 year rolling window

strategy_rolling_sharpe = strategy_returns.rolling(window).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
)
benchmark_rolling_sharpe = benchmark_returns.rolling(window).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
)

strategy_rolling_vol = strategy_returns.rolling(window).std() * np.sqrt(252) * 100
benchmark_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(252) * 100

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Rolling Sharpe Ratio (1-Year)', 'Rolling Volatility (1-Year)'),
    vertical_spacing=0.1
)

# Rolling Sharpe
fig.add_trace(
    go.Scatter(
        x=strategy_rolling_sharpe.index,
        y=strategy_rolling_sharpe.values,
        name='Strategy Sharpe',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=benchmark_rolling_sharpe.index,
        y=benchmark_rolling_sharpe.values,
        name='Benchmark Sharpe',
        line=dict(color='red', width=2),
        hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# Rolling Volatility
fig.add_trace(
    go.Scatter(
        x=strategy_rolling_vol.index,
        y=strategy_rolling_vol.values,
        name='Strategy Vol',
        line=dict(color='blue', width=2),
        showlegend=False,
        hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=benchmark_rolling_vol.index,
        y=benchmark_rolling_vol.values,
        name='Benchmark Vol',
        line=dict(color='red', width=2),
        showlegend=False,
        hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
    ),
    row=2, col=1
)

fig.update_layout(
    title='Rolling Performance Metrics',
    height=700,
    template='plotly_dark'
)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

fig.write_html('/home/ubuntu/rolling_metrics.html')
print("Rolling metrics chart saved to: rolling_metrics.html")

# 6. TRADE ANALYSIS
print("\n6. TRADE ANALYSIS")
print("-" * 50)

trades = strategy_pf.trades.records_readable

if len(trades) > 0:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trade Returns Distribution', 'Trade Returns Over Time', 
                       'Cumulative Trade PnL', 'Win/Loss Analysis'),
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    # Trade returns distribution
    fig.add_trace(
        go.Histogram(
            x=trades['Return'].values * 100,
            nbinsx=20,
            name='Trade Returns',
            marker_color='green',
            opacity=0.7,
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Trade returns over time
    fig.add_trace(
        go.Scatter(
            x=trades['Exit Timestamp'],
            y=trades['Return'].values * 100,
            mode='markers',
            name='Trade Returns',
            marker=dict(
                color=trades['Return'].values,
                colorscale='RdYlGn',
                size=8,
                colorbar=dict(title="Return (%)")
            ),
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Cumulative trade PnL
    cumulative_pnl = trades['PnL'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=trades['Exit Timestamp'],
            y=cumulative_pnl.values,
            name='Cumulative PnL',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Cumulative PnL: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Win/Loss analysis
    wins = (trades['Return'] > 0).sum()
    losses = (trades['Return'] <= 0).sum()
    
    fig.add_trace(
        go.Bar(
            x=['Wins', 'Losses'],
            y=[wins, losses],
            name='Win/Loss Count',
            marker_color=['green', 'red'],
            hovertemplate='Type: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Trade Analysis',
        height=800,
        template='plotly_dark'
    )
    
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Trade Type", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.write_html('/home/ubuntu/trade_analysis.html')
    print("Trade analysis chart saved to: trade_analysis.html")

# 7. MONTHLY PERFORMANCE HEATMAP
print("\n7. MONTHLY PERFORMANCE HEATMAP")
print("-" * 50)

# Calculate monthly returns
strategy_monthly = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
benchmark_monthly = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

# Create monthly performance matrix
strategy_monthly_matrix = strategy_monthly.groupby([strategy_monthly.index.year, strategy_monthly.index.month]).first().unstack()
benchmark_monthly_matrix = benchmark_monthly.groupby([benchmark_monthly.index.year, benchmark_monthly.index.month]).first().unstack()

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Strategy Monthly Returns (%)', 'Benchmark Monthly Returns (%)'),
    vertical_spacing=0.1
)

# Strategy heatmap
fig.add_trace(
    go.Heatmap(
        z=strategy_monthly_matrix.values * 100,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=strategy_monthly_matrix.index,
        colorscale='RdYlGn',
        name='Strategy',
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ),
    row=1, col=1
)

# Benchmark heatmap
fig.add_trace(
    go.Heatmap(
        z=benchmark_monthly_matrix.values * 100,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=benchmark_monthly_matrix.index,
        colorscale='RdYlGn',
        name='Benchmark',
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ),
    row=2, col=1
)

fig.update_layout(
    title='Monthly Performance Heatmap',
    height=800,
    template='plotly_dark'
)

fig.write_html('/home/ubuntu/monthly_heatmap.html')
print("Monthly performance heatmap saved to: monthly_heatmap.html")

# 8. COMPREHENSIVE DASHBOARD
print("\n8. COMPREHENSIVE DASHBOARD")
print("-" * 50)

# Create a comprehensive dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Portfolio Value', 'Drawdown', 'Rolling Sharpe', 
                   'Returns Distribution', 'Monthly Outperformance', 'Key Metrics'),
    vertical_spacing=0.08,
    horizontal_spacing=0.1,
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"type": "table"}]]
)

# Portfolio value
fig.add_trace(
    go.Scatter(x=strategy_value.index, y=strategy_value.values, name='Strategy', line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=benchmark_value.index, y=benchmark_value.values, name='Benchmark', line=dict(color='red')),
    row=1, col=1
)

# Drawdown
fig.add_trace(
    go.Scatter(x=strategy_dd.index, y=strategy_dd.values * 100, fill='tonexty', name='Strategy DD', line=dict(color='red')),
    row=1, col=2
)

# Rolling Sharpe
fig.add_trace(
    go.Scatter(x=strategy_rolling_sharpe.index, y=strategy_rolling_sharpe.values, name='Strategy Sharpe', line=dict(color='blue')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=benchmark_rolling_sharpe.index, y=benchmark_rolling_sharpe.values, name='Benchmark Sharpe', line=dict(color='red')),
    row=2, col=1
)

# Returns distribution
fig.add_trace(
    go.Histogram(x=strategy_returns.values * 100, nbinsx=30, name='Strategy Returns', opacity=0.7, marker_color='blue'),
    row=2, col=2
)

# Monthly outperformance
monthly_outperf = strategy_monthly - benchmark_monthly
fig.add_trace(
    go.Bar(x=monthly_outperf.index, y=monthly_outperf.values * 100, name='Monthly Outperformance', 
           marker_color=['green' if x > 0 else 'red' for x in monthly_outperf.values]),
    row=3, col=1
)

# Key metrics table
metrics_data = [
    ['Total Return', f"{strategy_pf.total_return():.1%}", f"{benchmark_pf.total_return():.1%}"],
    ['Sharpe Ratio', f"{strategy_pf.sharpe_ratio():.2f}", f"{benchmark_pf.sharpe_ratio():.2f}"],
    ['Max Drawdown', f"{strategy_pf.max_drawdown():.1%}", f"{benchmark_pf.max_drawdown():.1%}"],
    ['Volatility', f"{strategy_pf.annualized_volatility():.1%}", f"{benchmark_pf.annualized_volatility():.1%}"]
]

fig.add_trace(
    go.Table(
        header=dict(values=['Metric', 'Strategy', 'Benchmark'], fill_color='darkblue', font_color='white'),
        cells=dict(values=list(zip(*metrics_data)), fill_color='lightgrey', font_color='black')
    ),
    row=3, col=2
)

fig.update_layout(
    title='Comprehensive Trading Strategy Dashboard',
    height=1200,
    template='plotly_dark',
    showlegend=True
)

fig.write_html('/home/ubuntu/comprehensive_dashboard.html')
print("Comprehensive dashboard saved to: comprehensive_dashboard.html")

print("\n" + "=" * 80)
print("VECTORBT VISUALIZATIONS COMPLETED")
print("=" * 80)
print("\nGenerated Files:")
print("- portfolio_evolution.html")
print("- drawdown_analysis.html") 
print("- returns_distribution.html")
print("- rolling_metrics.html")
print("- trade_analysis.html")
print("- monthly_heatmap.html")
print("- comprehensive_dashboard.html")
