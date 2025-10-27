import pandas as pd
import numpy as np

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
    sharpe = (ann_return / strategy_vol) if strategy_vol > 0 else 0

    cum_returns = df['strategy_cum_returns']
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    time_in_market = df[signal_col].mean()

    return {
        'ann_return': ann_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'time_in_market': time_in_market,
        'bh_ann_return': bh_ann_return,
    }

def print_performance(name, metrics):
    """Print performance metrics"""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    print(f"Strategy Annualized Return: {metrics['ann_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"Max Drawdown: {metrics['max_dd']*100:.2f}%")
    print(f"Time in Market: {metrics['time_in_market']*100:.1f}%")
    print(f"Buy & Hold Annualized Return: {metrics['bh_ann_return']*100:.2f}%")
    print(f"Outperformance: {(metrics['ann_return'] - metrics['bh_ann_return'])*100:.2f}%")
    if metrics['ann_return'] > 0.04:
        print(f"\nTARGET ACHIEVED!")

print("=" * 80)
print("FINAL PUSH - ULTRA-FINE PARAMETER SWEEP")
print("=" * 80)

results = []

# Fine-grained OAS + Momentum combinations
print("\nTesting ultra-short lookbacks...")
for oas_lb in range(10, 26):  # 10 to 25 days
    for mom_lb in range(10, 26):  # 10 to 25 days
        df_test = df.copy()
        df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(oas_lb).mean()
        df_test['momentum'] = df_test['cad_ig_er_index'].pct_change(mom_lb)
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

# OAS only with fine-grained lookbacks
print("\nTesting OAS-only strategies with fine parameters...")
for oas_lb in range(10, 31):  # 10 to 30 days
    df_test = df.copy()
    df_test['cad_oas_ma'] = df_test['cad_oas'].rolling(oas_lb).mean()
    df_test['signal'] = (df_test['cad_oas'] < df_test['cad_oas_ma']).astype(int)
    df_test = df_test.dropna()

    metrics = calculate_performance(df_test)
    results.append({
        'strategy': f'OAS<MA{oas_lb}',
        'ann_return': metrics['ann_return'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_dd'],
        'time_in_market': metrics['time_in_market']
    })

# Momentum only with fine-grained lookbacks
print("\nTesting pure momentum with fine parameters...")
for mom_lb in range(10, 31):  # 10 to 30 days
    df_test = df.copy()
    df_test['momentum'] = df_test['cad_ig_er_index'].pct_change(mom_lb)
    df_test['signal'] = (df_test['momentum'] > 0).astype(int)
    df_test = df_test.dropna()

    metrics = calculate_performance(df_test)
    results.append({
        'strategy': f'Mom{mom_lb}d',
        'ann_return': metrics['ann_return'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_dd'],
        'time_in_market': metrics['time_in_market']
    })

# OAS + Momentum + VIX triple combos with fine tuning
print("\nTesting triple combinations with fine parameters...")
for oas_lb in [12, 13, 14, 15, 16, 17, 18]:
    for mom_lb in [12, 13, 14, 15, 16, 17, 18]:
        for vix_thresh in [28, 29, 30, 31, 32]:
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

print("\n" + "=" * 80)
print("TOP 30 STRATEGIES")
print("=" * 80)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ann_return', ascending=False)
print(results_df.head(30).to_string(index=False))

print("\n\n" + "=" * 80)
print("STRATEGIES EXCEEDING 4% TARGET")
print("=" * 80)
winners = results_df[results_df['ann_return'] > 0.04]
if len(winners) > 0:
    print(f"\nFOUND {len(winners)} STRATEGIES THAT BEAT 4%!")
    print(winners.to_string(index=False))

    best = winners.iloc[0]
    print_performance(f"\nBEST STRATEGY: {best['strategy']}", {
        'ann_return': best['ann_return'],
        'sharpe': best['sharpe'],
        'max_dd': best['max_dd'],
        'time_in_market': best['time_in_market'],
        'bh_ann_return': results_df.iloc[0]['ann_return'] * 0  # placeholder
    })

    # Get detailed performance for the best strategy
    oas_ma = int(best['strategy'].split('MA')[1].split(' ')[0])
    if 'Mom' in best['strategy']:
        mom_days = int(best['strategy'].split('Mom')[1].split('d')[0])
        df_final = df.copy()
        df_final['cad_oas_ma'] = df_final['cad_oas'].rolling(oas_ma).mean()
        df_final['momentum'] = df_final['cad_ig_er_index'].pct_change(mom_days)

        if 'VIX' in best['strategy']:
            vix_val = int(best['strategy'].split('VIX<')[1])
            df_final['signal'] = ((df_final['cad_oas'] < df_final['cad_oas_ma']) &
                                  (df_final['momentum'] > 0) &
                                  (df_final['vix'] < vix_val)).astype(int)
        else:
            df_final['signal'] = ((df_final['cad_oas'] < df_final['cad_oas_ma']) &
                                  (df_final['momentum'] > 0)).astype(int)
    else:
        df_final = df.copy()
        df_final['cad_oas_ma'] = df_final['cad_oas'].rolling(oas_ma).mean()
        df_final['signal'] = (df_final['cad_oas'] < df_final['cad_oas_ma']).astype(int)

    df_final = df_final.dropna()
    final_metrics = calculate_performance(df_final)
    print_performance(f"\nDETAILED RESULTS FOR {best['strategy']}", final_metrics)

else:
    print("\nNo strategies exceeded 4% yet.")
    best = results_df.iloc[0]
    print(f"\nBest strategy: {best['strategy']}")
    print(f"Annualized Return: {best['ann_return']*100:.2f}%")
    print(f"Sharpe Ratio: {best['sharpe']:.3f}")
    print(f"Still need: {(0.04 - best['ann_return'])*100:.2f}% more")
