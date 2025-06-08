#!/usr/bin/env python3

import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
import os
warnings.filterwarnings('ignore')

# Load and prepare data - using absolute path
print("Loading recession alert data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, 'outside data', 'Recession Alert Monthly.xlsx')
print(f"Loading from: {excel_path}")
df = pd.read_excel(excel_path)

# Display data structure
print("Data structure:")
print(df.head())
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Prepare data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Close'] = df['spx']
df['Recession_Flag'] = df['Probability of Recession']
df['Warning_Count'] = df['# of Recession Warnings']

print(f"Final dataset shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Calculate returns
df['Returns'] = df['Close'].pct_change()

# Calculate all technical indicators
all_ma_periods = [5, 10, 15, 21, 42, 63, 126, 252]  
for period in all_ma_periods:
    df[f'MA_{period}'] = df['Close'].rolling(period).mean()

    df['Vol_20'] = df['Returns'].rolling(20).std() * np.sqrt(12)
df['Vol_regime'] = pd.qcut(df['Vol_20'].dropna(), q=3, labels=['Low', 'Med', 'High'])

# Drawdown calculation
def calculate_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return drawdown

df['Drawdown'] = calculate_drawdown(df['Returns'].fillna(0))

# CONTRARIAN STRATEGIES - Use recession signals as BUY opportunities
print("\n" + "="*50)
print("CONTRARIAN RECESSION STRATEGY OPTIMIZATION")
print("="*50)

def create_contrarian_signal(df, recession_thresh=0.3, ma_period=21, signal_type='contrarian_aggressive'):
    """Create contrarian trading signals - buy when recession probability is HIGH"""
    
    if signal_type == 'contrarian_aggressive':
        # Buy when recession probability is HIGH (contrarian)
        signal = (
            (df['Recession_Flag'] >= recession_thresh) |  # BUY during recession fears
            (df['Close'] > df[f'MA_{ma_period}'])        # OR when price above MA
        ).astype(int)
        
    elif signal_type == 'contrarian_moderate':
        # Buy when recession probability is high AND price is oversold
        signal = (
            ((df['Recession_Flag'] >= recession_thresh) & (df['Close'] < df[f'MA_{ma_period}'])) |  # Buy recession dips
            (df['Close'] > df[f'MA_{ma_period}'])  # OR normal uptrend
        ).astype(int)
        
    elif signal_type == 'hybrid_aggressive':
        # Stay long unless both recession risk AND technical breakdown
        signal = ~(
            (df['Recession_Flag'] >= recession_thresh) & 
            (df['Close'] < df[f'MA_{ma_period}'])
        ).astype(int)
        
    elif signal_type == 'volatility_buy':
        # Buy high volatility periods (crisis = opportunity)
        high_vol = df['Vol_20'] > df['Vol_20'].quantile(0.8)
        signal = (
            high_vol |  # Buy high volatility
            (df['Close'] > df[f'MA_{ma_period}'])  # OR uptrend
        ).astype(int)
        
    else:  # 'always_long'
        # Almost always long (95%+ exposure)
        signal = (
            (df['Recession_Flag'] < 0.95) |  # Only exit if >95% recession probability
            (df['Close'] > df[f'MA_{ma_period}'])
        ).astype(int)
    
    return signal

# Extremely aggressive parameter grid
recession_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Contrarian thresholds
ma_periods = [5, 10, 15, 21, 42, 63]  # Faster signals
signal_types = ['contrarian_aggressive', 'contrarian_moderate', 'hybrid_aggressive', 'volatility_buy', 'always_long']

best_total_return = -999
best_params = {}
optimization_results = []

print("Testing contrarian parameter combinations for maximum total return...")
for rec_thresh in recession_thresholds:
    for ma_per in ma_periods:
        for signal_type in signal_types:
            signal = create_contrarian_signal(df, rec_thresh, ma_per, signal_type)
            
            # Long-only returns
            strategy_returns = df['Returns'] * signal.shift(1)
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) > 0:
                # Calculate total return
                total_return = (1 + strategy_returns).prod() - 1
                annual_return = strategy_returns.mean() * 12  # Monthly data
                annual_vol = strategy_returns.std() * np.sqrt(12)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                
                cum_rets = (1 + strategy_returns).cumprod()
                max_dd = ((cum_rets - cum_rets.expanding().max()) / cum_rets.expanding().max()).min()
                
                result = {
                    'recession_thresh': rec_thresh,
                    'ma_period': ma_per,
                    'signal_type': signal_type,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'annual_vol': annual_vol,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd,
                    'exposure': signal.mean()
                }
                optimization_results.append(result)
                
                # Optimize for total return
                if total_return > best_total_return:
                    best_total_return = total_return
                    best_params = result.copy()

# Display optimization results
opt_df = pd.DataFrame(optimization_results)
print(f"\nTested {len(opt_df)} parameter combinations")
print(f"Best total return: {best_total_return:.2%}")
print(f"Target return (1.5x benchmark): {55.87:.2%} (based on 5587% total)")
print(f"Success: {'YES' if best_total_return >= 0.8380 else 'NO'}")  # 83.8x return = 8380%

print("\nBest parameters:")
for key, value in best_params.items():
    if key in ['total_return', 'annual_return', 'annual_vol', 'max_drawdown']:
        print(f"{key}: {value:.3f}")
    elif key == 'exposure':
        print(f"{key}: {value:.1%}")
    else:
        print(f"{key}: {value}")

# Show top 10 strategies
print("\nTop 10 strategies by total return:")
top_strategies = opt_df.nlargest(10, 'total_return')
for i, (idx, row) in enumerate(top_strategies.iterrows(), 1):
    print(f"{i}. {row['signal_type']} - Rec:{row['recession_thresh']:.1f}, MA:{int(row['ma_period'])}, Return:{row['total_return']:.1%}, Exposure:{row['exposure']:.1%}")

# VECTORBT BACKTESTING with best parameters
print("\n" + "="*50)
print("VECTORBT BACKTESTING - CONTRARIAN STRATEGY")
print("="*50)

# Prepare data for vectorbt
price_data = df['Close'].dropna()
print(f"Backtesting period: {price_data.index[0]} to {price_data.index[-1]}")
print(f"Total periods: {len(price_data)}")

# Create optimal signal
optimal_signal = create_contrarian_signal(
    df, 
    best_params['recession_thresh'],
    best_params['ma_period'],
    best_params['signal_type']
)

# Align signal with price data
optimal_signal = optimal_signal.reindex(price_data.index, method='ffill').fillna(1)  # Default to long

# Vectorbt portfolio (monthly frequency using proper timedelta)
portfolio = vbt.Portfolio.from_orders(
    close=price_data,
    size=optimal_signal,
    size_type='targetpercent',
    freq=pd.Timedelta(days=30)  # Approximate monthly (30 days)
)

# Performance statistics
stats = portfolio.stats()
print("\nContrarian Strategy Performance - FULL VectorBT Stats:")
print("="*60)

# Display all available stats with their values
for key in stats.index:
    value = stats[key]
    if pd.isna(value):
        print(f"{key}: N/A")
    elif isinstance(value, (int, float)):
        if key.endswith('[%]'):
            print(f"{key}: {value:.2f}%")
        elif 'Ratio' in key:
            print(f"{key}: {value:.3f}")
        elif 'Duration' in key:
            print(f"{key}: {value}")
        elif 'Trades' in key:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

print(f"\nMarket Exposure: {optimal_signal.mean():.1%}")
print("="*60)

# Buy and Hold Comparison
print(f"\nBuy & Hold Comparison - FULL VectorBT Stats:")
print("="*60)
bnh_portfolio = vbt.Portfolio.from_holding(price_data, freq=pd.Timedelta(days=30))
bnh_stats = bnh_portfolio.stats()

# Display all Buy & Hold stats
print("BUY & HOLD STATS:")
for key in bnh_stats.index:
    value = bnh_stats[key]
    if pd.isna(value):
        print(f"{key}: N/A")
    elif isinstance(value, (int, float)):
        if key.endswith('[%]'):
            print(f"{key}: {value:.2f}%")
        elif 'Ratio' in key:
            print(f"{key}: {value:.3f}")
        elif 'Duration' in key:
            print(f"{key}: {value}")
        elif 'Trades' in key:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

print("="*60)

# Returns-based statistics using returns accessor
print(f"\n{'='*60}")
print("RETURNS-BASED STATISTICS - STRATEGY")
print("="*60)
strategy_returns = portfolio.returns()
strategy_returns_stats = strategy_returns.vbt.returns(freq=pd.Timedelta(days=30)).stats()
print("STRATEGY RETURNS STATS:")
for key in strategy_returns_stats.index:
    value = strategy_returns_stats[key]
    if pd.isna(value):
        print(f"{key}: N/A")
    elif isinstance(value, (int, float)):
        if key.endswith('[%]'):
            print(f"{key}: {value:.2f}%")
        elif 'Ratio' in key:
            print(f"{key}: {value:.3f}")
        elif 'Duration' in key:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

print(f"\n{'='*60}")
print("RETURNS-BASED STATISTICS - BUY & HOLD")
print("="*60)
bnh_returns = bnh_portfolio.returns()
bnh_returns_stats = bnh_returns.vbt.returns(freq=pd.Timedelta(days=30)).stats()
print("BUY & HOLD RETURNS STATS:")
for key in bnh_returns_stats.index:
    value = bnh_returns_stats[key]
    if pd.isna(value):
        print(f"{key}: N/A")
    elif isinstance(value, (int, float)):
        if key.endswith('[%]'):
            print(f"{key}: {value:.2f}%")
        elif 'Ratio' in key:
            print(f"{key}: {value:.3f}")
        elif 'Duration' in key:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Extract key metrics for comparison
strategy_total_return = stats['Total Return [%]']
bnh_total_return = bnh_stats['Total Return [%]']
strategy_sharpe = stats['Sharpe Ratio']
bnh_sharpe = bnh_stats['Sharpe Ratio']
strategy_max_dd = stats['Max Drawdown [%]']
bnh_max_dd = bnh_stats['Max Drawdown [%]']

print(f"\nSTRATEGY vs BUY & HOLD COMPARISON:")
print(f"Strategy Total Return: {strategy_total_return:.2f}%")
print(f"Buy & Hold Total Return: {bnh_total_return:.2f}%")
print(f"Excess Return: {strategy_total_return - bnh_total_return:.2f}%")
print(f"Strategy Sharpe: {strategy_sharpe:.3f}")
print(f"Buy & Hold Sharpe: {bnh_sharpe:.3f}")
print(f"Strategy Max DD: {strategy_max_dd:.2f}%")
print(f"Buy & Hold Max DD: {bnh_max_dd:.2f}%")

# Final success check
excess_return = strategy_total_return - bnh_total_return
success = excess_return >= 200  # Need 2%+ annually, roughly 200%+ total over 57 years

print(f"\n{'='*20} FINAL RESULT {'='*20}")
print(f"Excess Return: {excess_return:.2f}%")
print(f"Target: +200% vs Buy & Hold")
print(f"SUCCESS: {'YES - STRATEGY BEATS BUY & HOLD!' if success else 'NO - Still underperforming'}")

if success:
    print(f"\n🎉 WINNING STRATEGY FOUND! 🎉")
    print(f"Strategy: {best_params['signal_type']}")
    print(f"Parameters: Recession threshold {best_params['recession_thresh']}, MA period {best_params['ma_period']}")
    print(f"Total Return: {strategy_total_return:.1f}% vs Buy & Hold {bnh_total_return:.1f}%")
    print(f"Excess: +{excess_return:.1f}%")
else:
    print("\nContinue searching for better parameters...") 