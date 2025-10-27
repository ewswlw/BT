import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('/home/user/BT/data_pipelines/data_processed/with_er_daily.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total days: {len(df)}")
print(f"Years: {(df['Date'].max() - df['Date'].min()).days / 365.25:.2f}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows of cad_ig_er_index:")
print(df[['Date', 'cad_ig_er_index']].head(10))
print(f"\nLast few rows of cad_ig_er_index:")
print(df[['Date', 'cad_ig_er_index']].tail(10))

# Calculate returns
df['returns'] = df['cad_ig_er_index'].pct_change()

# Function to calculate performance metrics
def calculate_performance(df, signal_col='signal'):
    """Calculate performance metrics for a strategy"""
    df = df.copy()

    # Strategy returns (only invested when signal = 1, else 0% return)
    df['strategy_returns'] = df['returns'] * df[signal_col]

    # Calculate cumulative returns
    df['strategy_cum_returns'] = (1 + df['strategy_returns']).cumprod()
    df['bh_cum_returns'] = (1 + df['returns']).cumprod()

    # Calculate total return
    total_return = df['strategy_cum_returns'].iloc[-1] - 1
    bh_total_return = df['bh_cum_returns'].iloc[-1] - 1

    # Calculate annualized returns
    years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1
    bh_ann_return = (1 + bh_total_return) ** (1 / years) - 1

    # Calculate volatility
    strategy_vol = df['strategy_returns'].std() * np.sqrt(252)
    bh_vol = df['returns'].std() * np.sqrt(252)

    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    sharpe = (ann_return / strategy_vol) if strategy_vol > 0 else 0
    bh_sharpe = (bh_ann_return / bh_vol) if bh_vol > 0 else 0

    # Calculate max drawdown
    cum_returns = df['strategy_cum_returns']
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    bh_cum_returns = df['bh_cum_returns']
    bh_running_max = bh_cum_returns.expanding().max()
    bh_drawdown = (bh_cum_returns - bh_running_max) / bh_running_max
    bh_max_dd = bh_drawdown.min()

    # Time in market
    time_in_market = df[signal_col].mean()

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'volatility': strategy_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'time_in_market': time_in_market,
        'bh_total_return': bh_total_return,
        'bh_ann_return': bh_ann_return,
        'bh_volatility': bh_vol,
        'bh_sharpe': bh_sharpe,
        'bh_max_dd': bh_max_dd,
        'years': years
    }

def print_performance(name, metrics):
    """Print performance metrics in a nice format"""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    print(f"Strategy:")
    print(f"  Annualized Return: {metrics['ann_return']*100:.2f}%")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Volatility: {metrics['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown: {metrics['max_dd']*100:.2f}%")
    print(f"  Time in Market: {metrics['time_in_market']*100:.1f}%")
    print(f"\nBuy & Hold:")
    print(f"  Annualized Return: {metrics['bh_ann_return']*100:.2f}%")
    print(f"  Total Return: {metrics['bh_total_return']*100:.2f}%")
    print(f"  Volatility: {metrics['bh_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['bh_sharpe']:.3f}")
    print(f"  Max Drawdown: {metrics['bh_max_dd']*100:.2f}%")
    print(f"\nOutperformance:")
    print(f"  Ann. Return Diff: {(metrics['ann_return'] - metrics['bh_ann_return'])*100:.2f}%")
    print(f"  ✓ TARGET ACHIEVED!" if metrics['ann_return'] > 0.04 else f"  ✗ Need {(0.04 - metrics['ann_return'])*100:.2f}% more")

# Buy and hold baseline
print("\n" + "=" * 80)
print("BUY AND HOLD BASELINE")
print("=" * 80)
df_bh = df.copy()
df_bh['signal'] = 1  # Always invested
metrics_bh = calculate_performance(df_bh)
print(f"Annualized Return: {metrics_bh['bh_ann_return']*100:.2f}%")
print(f"Total Return: {metrics_bh['bh_total_return']*100:.2f}%")
print(f"Volatility: {metrics_bh['bh_volatility']*100:.2f}%")
print(f"Sharpe Ratio: {metrics_bh['bh_sharpe']:.3f}")
print(f"Max Drawdown: {metrics_bh['bh_max_dd']*100:.2f}%")
print(f"Period: {metrics_bh['years']:.2f} years")

print("\n" + "=" * 80)
print("TESTING STRATEGIES")
print("=" * 80)

results = []

# Strategy 1: Simple Momentum (various lookbacks)
print("\n\nSTRATEGY 1: SIMPLE MOMENTUM")
print("-" * 80)
for lookback in [20, 40, 60, 90, 120, 180, 252]:
    df_test = df.copy()
    df_test['momentum'] = df_test['cad_ig_er_index'].pct_change(lookback)
    df_test['signal'] = (df_test['momentum'] > 0).astype(int)
    df_test = df_test.dropna()

    metrics = calculate_performance(df_test)
    results.append({
        'strategy': f'Momentum {lookback}d',
        'ann_return': metrics['ann_return'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_dd'],
        'time_in_market': metrics['time_in_market']
    })

    if metrics['ann_return'] > 0.04:
        print_performance(f"Momentum Strategy (lookback={lookback} days)", metrics)

# Strategy 2: Moving Average Crossover
print("\n\nSTRATEGY 2: MOVING AVERAGE CROSSOVER")
print("-" * 80)
for short_ma in [20, 30, 50]:
    for long_ma in [100, 150, 200]:
        if short_ma >= long_ma:
            continue
        df_test = df.copy()
        df_test['short_ma'] = df_test['cad_ig_er_index'].rolling(short_ma).mean()
        df_test['long_ma'] = df_test['cad_ig_er_index'].rolling(long_ma).mean()
        df_test['signal'] = (df_test['short_ma'] > df_test['long_ma']).astype(int)
        df_test = df_test.dropna()

        metrics = calculate_performance(df_test)
        results.append({
            'strategy': f'MA {short_ma}/{long_ma}',
            'ann_return': metrics['ann_return'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_dd'],
            'time_in_market': metrics['time_in_market']
        })

        if metrics['ann_return'] > 0.04:
            print_performance(f"MA Crossover ({short_ma}/{long_ma})", metrics)

# Strategy 3: VIX Filter
print("\n\nSTRATEGY 3: VIX FILTER")
print("-" * 80)
for threshold in [15, 20, 25, 30]:
    df_test = df.copy()
    df_test['signal'] = (df_test['vix'] < threshold).astype(int)
    df_test = df_test.dropna()

    metrics = calculate_performance(df_test)
    results.append({
        'strategy': f'VIX < {threshold}',
        'ann_return': metrics['ann_return'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_dd'],
        'time_in_market': metrics['time_in_market']
    })

    if metrics['ann_return'] > 0.04:
        print_performance(f"VIX Filter (threshold={threshold})", metrics)

# Strategy 4: OAS Spread Signal
print("\n\nSTRATEGY 4: OAS SPREAD SIGNALS")
print("-" * 80)
for lookback in [20, 40, 60, 90]:
    df_test = df.copy()
    df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(lookback).mean()
    # Buy when spreads are tightening (below MA)
    df_test['signal'] = (df_test['cad_oas'] < df_test['cad_oas_ma']).astype(int)
    df_test = df_test.dropna()

    metrics = calculate_performance(df_test)
    results.append({
        'strategy': f'OAS < MA{lookback}',
        'ann_return': metrics['ann_return'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_dd'],
        'time_in_market': metrics['time_in_market']
    })

    if metrics['ann_return'] > 0.04:
        print_performance(f"OAS Spread Strategy (lookback={lookback})", metrics)

print("\n\n" + "=" * 80)
print("SUMMARY OF ALL STRATEGIES")
print("=" * 80)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ann_return', ascending=False)
print(results_df.to_string(index=False))

print("\n\n" + "=" * 80)
print("TOP 5 STRATEGIES (by Annualized Return)")
print("=" * 80)
print(results_df.head(5).to_string(index=False))

# Check if any strategy beat 4%
best_strategy = results_df.iloc[0]
print(f"\n\nBest Strategy: {best_strategy['strategy']}")
print(f"Annualized Return: {best_strategy['ann_return']*100:.2f}%")
if best_strategy['ann_return'] > 0.04:
    print("✓ TARGET ACHIEVED!")
else:
    print(f"✗ Still need {(0.04 - best_strategy['ann_return'])*100:.2f}% more")
    print("\nTrying more advanced strategies...")
