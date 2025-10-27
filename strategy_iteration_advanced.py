import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('/home/user/BT/data_pipelines/data_processed/with_er_daily.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['returns'] = df['cad_ig_er_index'].pct_change()

def calculate_performance(df, signal_col='signal'):
    """Calculate performance metrics for a strategy"""
    df = df.copy()
    df['strategy_returns'] = df['returns'] * df[signal_col]
    df['strategy_cum_returns'] = (1 + df['strategy_returns']).cumprod()
    df['bh_cum_returns'] = (1 + df['returns']).cumprod()

    total_return = df['strategy_cum_returns'].iloc[-1] - 1
    bh_total_return = df['bh_cum_returns'].iloc[-1] - 1

    years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1
    bh_ann_return = (1 + bh_total_return) ** (1 / years) - 1

    strategy_vol = df['strategy_returns'].std() * np.sqrt(252)
    bh_vol = df['returns'].std() * np.sqrt(252)

    sharpe = (ann_return / strategy_vol) if strategy_vol > 0 else 0
    bh_sharpe = (bh_ann_return / bh_vol) if bh_vol > 0 else 0

    cum_returns = df['strategy_cum_returns']
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    bh_cum_returns = df['bh_cum_returns']
    bh_running_max = bh_cum_returns.expanding().max()
    bh_drawdown = (bh_cum_returns - bh_running_max) / bh_running_max
    bh_max_dd = bh_drawdown.min()

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
    if metrics['ann_return'] > 0.04:
        print(f"  TARGET ACHIEVED!")
    else:
        print(f"  Need {(0.04 - metrics['ann_return'])*100:.2f}% more")

print("=" * 80)
print("TESTING ADVANCED COMBINATION STRATEGIES")
print("=" * 80)

results = []

# Strategy 1: OAS + Momentum Combination
print("\n\nSTRATEGY 1: OAS + MOMENTUM COMBINATION")
print("-" * 80)
for oas_lb in [15, 20, 30]:
    for mom_lb in [15, 20, 30, 40]:
        df_test = df.copy()
        df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(oas_lb).mean()
        df_test['momentum'] = df_test['cad_ig_er_index'].pct_change(mom_lb)
        # Both conditions must be true
        df_test['signal'] = ((df_test['cad_oas'] < df_test['cad_oas_ma']) &
                             (df_test['momentum'] > 0)).astype(int)
        df_test = df_test.dropna()

        metrics = calculate_performance(df_test)
        results.append({
            'strategy': f'OAS<MA{oas_lb} & Mom{mom_lb}d',
            'ann_return': metrics['ann_return'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_dd'],
            'time_in_market': metrics['time_in_market']
        })

        if metrics['ann_return'] > 0.04:
            print_performance(f"OAS < MA{oas_lb} AND Momentum {mom_lb}d", metrics)

# Strategy 2: OAS + VIX Combination
print("\n\nSTRATEGY 2: OAS + VIX COMBINATION")
print("-" * 80)
for oas_lb in [15, 20, 30]:
    for vix_thresh in [20, 25, 30]:
        df_test = df.copy()
        df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(oas_lb).mean()
        df_test['signal'] = ((df_test['cad_oas'] < df_test['cad_oas_ma']) &
                             (df_test['vix'] < vix_thresh)).astype(int)
        df_test = df_test.dropna()

        metrics = calculate_performance(df_test)
        results.append({
            'strategy': f'OAS<MA{oas_lb} & VIX<{vix_thresh}',
            'ann_return': metrics['ann_return'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_dd'],
            'time_in_market': metrics['time_in_market']
        })

        if metrics['ann_return'] > 0.04:
            print_performance(f"OAS < MA{oas_lb} AND VIX < {vix_thresh}", metrics)

# Strategy 3: Triple Combination
print("\n\nSTRATEGY 3: OAS + MOMENTUM + VIX (TRIPLE FILTER)")
print("-" * 80)
for oas_lb in [15, 20]:
    for mom_lb in [20, 30]:
        for vix_thresh in [25, 30]:
            df_test = df.copy()
            df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(oas_lb).mean()
            df_test['momentum'] = df_test['cad_ig_er_index'].pct_change(mom_lb)
            df_test['signal'] = ((df_test['cad_oas'] < df_test['cad_oas_ma']) &
                                 (df_test['momentum'] > 0) &
                                 (df_test['vix'] < vix_thresh)).astype(int)
            df_test = df_test.dropna()

            metrics = calculate_performance(df_test)
            results.append({
                'strategy': f'OAS<MA{oas_lb} & Mom{mom_lb} & VIX<{vix_thresh}',
                'ann_return': metrics['ann_return'],
                'sharpe': metrics['sharpe'],
                'max_dd': metrics['max_dd'],
                'time_in_market': metrics['time_in_market']
            })

            if metrics['ann_return'] > 0.04:
                print_performance(f"Triple: OAS<MA{oas_lb} & Mom{mom_lb}d & VIX<{vix_thresh}", metrics)

# Strategy 4: Economic Regime Filter
print("\n\nSTRATEGY 4: OAS WITH ECONOMIC REGIME")
print("-" * 80)
for oas_lb in [15, 20, 25, 30]:
    df_test = df.copy()
    df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(oas_lb).mean()
    # Only trade when economic regime is positive (1.0)
    df_test['signal'] = ((df_test['cad_oas'] < df_test['cad_oas_ma']) &
                         (df_test['us_economic_regime'] == 1.0)).astype(int)
    df_test = df_test.dropna()

    metrics = calculate_performance(df_test)
    results.append({
        'strategy': f'OAS<MA{oas_lb} & Regime+',
        'ann_return': metrics['ann_return'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_dd'],
        'time_in_market': metrics['time_in_market']
    })

    if metrics['ann_return'] > 0.04:
        print_performance(f"OAS < MA{oas_lb} with Positive Regime", metrics)

# Strategy 5: Mean Reversion on OAS (opposite signal)
print("\n\nSTRATEGY 5: OAS MEAN REVERSION (CONTRARIAN)")
print("-" * 80)
for oas_lb in [15, 20, 30, 40]:
    df_test = df.copy()
    df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(oas_lb).mean()
    # Buy when spreads are widening (above MA) - contrarian
    df_test['signal'] = (df_test['cad_oas'] > df_test['cad_oas_ma']).astype(int)
    df_test = df_test.dropna()

    metrics = calculate_performance(df_test)
    results.append({
        'strategy': f'OAS>MA{oas_lb} (contrarian)',
        'ann_return': metrics['ann_return'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_dd'],
        'time_in_market': metrics['time_in_market']
    })

    if metrics['ann_return'] > 0.04:
        print_performance(f"OAS > MA{oas_lb} (Contrarian)", metrics)

# Strategy 6: Momentum + VIX (no OAS)
print("\n\nSTRATEGY 6: MOMENTUM + VIX COMBINATION")
print("-" * 80)
for mom_lb in [15, 20, 30, 40]:
    for vix_thresh in [20, 25, 30]:
        df_test = df.copy()
        df_test['momentum'] = df_test['cad_ig_er_index'].pct_change(mom_lb)
        df_test['signal'] = ((df_test['momentum'] > 0) &
                             (df_test['vix'] < vix_thresh)).astype(int)
        df_test = df_test.dropna()

        metrics = calculate_performance(df_test)
        results.append({
            'strategy': f'Mom{mom_lb}d & VIX<{vix_thresh}',
            'ann_return': metrics['ann_return'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_dd'],
            'time_in_market': metrics['time_in_market']
        })

        if metrics['ann_return'] > 0.04:
            print_performance(f"Momentum {mom_lb}d AND VIX < {vix_thresh}", metrics)

print("\n\n" + "=" * 80)
print("ALL STRATEGIES SUMMARY (SORTED BY ANNUALIZED RETURN)")
print("=" * 80)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ann_return', ascending=False)
print(results_df.head(20).to_string(index=False))

print("\n\n" + "=" * 80)
print("STRATEGIES EXCEEDING 4% TARGET")
print("=" * 80)
winners = results_df[results_df['ann_return'] > 0.04]
if len(winners) > 0:
    print(f"\nFound {len(winners)} strategies that beat 4% annualized!")
    print(winners.to_string(index=False))

    best = winners.iloc[0]
    print(f"\n\nBEST STRATEGY: {best['strategy']}")
    print(f"Annualized Return: {best['ann_return']*100:.2f}%")
    print(f"Sharpe Ratio: {best['sharpe']:.3f}")
    print(f"Max Drawdown: {best['max_dd']*100:.2f}%")
    print(f"Time in Market: {best['time_in_market']*100:.1f}%")
    print(f"\nTARGET ACHIEVED!")
else:
    print("\nNo strategies exceeded 4% target yet.")
    print("Best strategy found:")
    best = results_df.iloc[0]
    print(f"  {best['strategy']}: {best['ann_return']*100:.2f}% annualized")
    print(f"  Still need {(0.04 - best['ann_return'])*100:.2f}% more")
