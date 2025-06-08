import pandas as pd
import numpy as np
import vectorbt as vbt
import os
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the recession alert data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'outside data', 'Recession Alert Monthly.xlsx')
    
    print("Loading recession alert data...")
    print(f"Loading from: {file_path}")
    
    df = pd.read_excel(file_path)
    print("Data structure:")
    print(df.head())
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Rename columns for easier access
    df = df.rename(columns={
        'Probability of Recession': 'RecessionProb',
        '# of Recession Warnings': 'RecessionWarnings'
    })
    
    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Create close prices and returns
    df['Close'] = df['spx']
    df['Returns'] = df['Close'].pct_change()
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def create_features(df):
    """Create various technical and fundamental features"""
    features_df = df.copy()
    
    # Price-based features
    for period in [3, 6, 9, 12, 15, 18, 21, 24]:
        features_df[f'MA_{period}'] = features_df['Close'].rolling(period).mean()
        features_df[f'Price_above_MA_{period}'] = (features_df['Close'] > features_df[f'MA_{period}']).astype(int)
    
    # Volatility features
    for period in [6, 12, 20, 24]:
        features_df[f'Vol_{period}'] = features_df['Returns'].rolling(period).std() * np.sqrt(12)
        features_df[f'Vol_{period}_percentile'] = features_df[f'Vol_{period}'].rolling(60).rank(pct=True)
    
    # Momentum features
    for period in [3, 6, 9, 12]:
        features_df[f'Momentum_{period}'] = features_df['Close'].pct_change(period)
        features_df[f'Momentum_{period}_positive'] = (features_df[f'Momentum_{period}'] > 0).astype(int)
    
    # Recession-based features
    features_df['RecessionProb_MA_3'] = features_df['RecessionProb'].rolling(3).mean()
    features_df['RecessionProb_MA_6'] = features_df['RecessionProb'].rolling(6).mean()
    features_df['RecessionWarnings_MA_3'] = features_df['RecessionWarnings'].rolling(3).mean()
    
    # High/Low recession probability periods
    features_df['HighRecessionProb'] = (features_df['RecessionProb'] > features_df['RecessionProb'].quantile(0.8)).astype(int)
    features_df['LowRecessionProb'] = (features_df['RecessionProb'] < features_df['RecessionProb'].quantile(0.2)).astype(int)
    
    # Market regime features
    features_df['BullMarket'] = (features_df['Close'] > features_df['MA_12']).astype(int)
    features_df['BearMarket'] = (features_df['Close'] < features_df['MA_12']).astype(int)
    
    # Volatility regime
    features_df['HighVol_Regime'] = (features_df['Vol_20'] > features_df['Vol_20'].quantile(0.7)).astype(int)
    features_df['LowVol_Regime'] = (features_df['Vol_20'] < features_df['Vol_20'].quantile(0.3)).astype(int)
    
    return features_df

def generate_signals(df, strategy_config):
    """Generate buy/sell signals based on strategy configuration"""
    
    strategy_type = strategy_config['type']
    params = strategy_config['params']
    
    if strategy_type == 'recession_contrarian':
        # Buy during high recession probability
        recession_signal = df['RecessionProb'] > params['recession_thresh']
        trend_signal = df['Close'] > df[f'MA_{params["ma_period"]}']
        signal = recession_signal | trend_signal
        
    elif strategy_type == 'volatility_momentum':
        # Buy during high volatility + positive momentum
        vol_signal = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        momentum_signal = df[f'Momentum_{params["momentum_period"]}'] > params['momentum_thresh']
        signal = vol_signal & momentum_signal
        
    elif strategy_type == 'trend_following':
        # Classic trend following
        short_ma = df[f'MA_{params["short_ma"]}']
        long_ma = df[f'MA_{params["long_ma"]}']
        signal = short_ma > long_ma
        
    elif strategy_type == 'mean_reversion':
        # Buy when price is below MA and volatility is high
        below_ma = df['Close'] < df[f'MA_{params["ma_period"]}']
        high_vol = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        signal = below_ma & high_vol
        
    elif strategy_type == 'regime_switch':
        # Switch between trend following and mean reversion based on volatility regime
        high_vol_regime = df['HighVol_Regime'] == 1
        trend_signal = df['Close'] > df[f'MA_{params["trend_ma"]}']
        mean_rev_signal = df['Close'] < df[f'MA_{params["mean_rev_ma"]}']
        signal = np.where(high_vol_regime, mean_rev_signal, trend_signal)
        
    elif strategy_type == 'multi_factor':
        # Combine multiple factors
        factors = []
        
        if 'recession' in params['factors']:
            factors.append(df['RecessionProb'] > params['recession_thresh'])
        if 'trend' in params['factors']:
            factors.append(df['Close'] > df[f'MA_{params["ma_period"]}'])
        if 'volatility' in params['factors']:
            factors.append(df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh']))
        if 'momentum' in params['factors']:
            factors.append(df[f'Momentum_{params["momentum_period"]}'] > 0)
            
        # Combine factors based on logic
        if params['logic'] == 'OR':
            signal = factors[0]
            for factor in factors[1:]:
                signal = signal | factor
        else:  # AND
            signal = factors[0]
            for factor in factors[1:]:
                signal = signal & factor
                
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return signal.fillna(False)

def backtest_strategy(df, signal, strategy_name):
    """Backtest a strategy using VectorBT"""
    
    # Create entries and exits
    entries = signal
    exits = ~signal
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=df['Close'],
        entries=entries,
        exits=exits,
        freq=pd.Timedelta(days=30),
        init_cash=100
    )
    
    # Get basic stats
    stats = portfolio.stats()
    
    # Get returns-based statistics
    returns_stats = portfolio.returns().vbt.returns(freq=pd.Timedelta(days=30)).stats()
    
    return {
        'strategy_name': strategy_name,
        'total_return': stats['Total Return [%]'],
        'annual_return': returns_stats['Annualized Return [%]'],
        'sharpe_ratio': stats['Sharpe Ratio'],
        'max_drawdown': stats['Max Drawdown [%]'],
        'exposure': (signal.sum() / len(signal)) * 100,
        'num_trades': stats['Total Trades'],
        'win_rate': stats['Win Rate [%]'] if pd.notna(stats['Win Rate [%]']) else 0,
        'portfolio': portfolio,
        'signal': signal
    }

def run_optimization():
    """Run comprehensive strategy optimization"""
    
    # Load data
    df = load_data()
    df = create_features(df)
    
    # Create buy-and-hold benchmark
    bnh_signal = pd.Series(True, index=df.index)
    bnh_result = backtest_strategy(df, bnh_signal, 'Buy_and_Hold')
    
    print(f"\n{'='*60}")
    print("BUY & HOLD BENCHMARK")
    print(f"{'='*60}")
    print(f"Total Return: {bnh_result['total_return']:.2f}%")
    print(f"Annualized Return: {bnh_result['annual_return']:.2f}%")
    print(f"Sharpe Ratio: {bnh_result['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {bnh_result['max_drawdown']:.2f}%")
    
    target_annual_return = bnh_result['annual_return'] + 2.0  # Target: +2% above B&H
    
    print(f"\nTARGET ANNUAL RETURN: {target_annual_return:.2f}%")
    print(f"{'='*60}")
    
    # Define strategy configurations to test
    strategies = []
    
    # 1. Recession Contrarian strategies
    for recession_thresh in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for ma_period in [6, 9, 12, 15, 18]:
            strategies.append({
                'type': 'recession_contrarian',
                'params': {
                    'recession_thresh': recession_thresh,
                    'ma_period': ma_period
                }
            })
    
    # 2. Volatility + Momentum strategies
    for vol_period in [12, 20, 24]:
        for vol_thresh in [0.6, 0.7, 0.8]:
            for momentum_period in [3, 6, 9]:
                strategies.append({
                    'type': 'volatility_momentum',
                    'params': {
                        'vol_period': vol_period,
                        'vol_thresh': vol_thresh,
                        'momentum_period': momentum_period,
                        'momentum_thresh': 0.0
                    }
                })
    
    # 3. Trend Following strategies
    for short_ma in [3, 6, 9]:
        for long_ma in [12, 15, 18, 21]:
            if short_ma < long_ma:
                strategies.append({
                    'type': 'trend_following',
                    'params': {
                        'short_ma': short_ma,
                        'long_ma': long_ma
                    }
                })
    
    # 4. Mean Reversion strategies
    for ma_period in [12, 15, 18]:
        for vol_period in [12, 20]:
            for vol_thresh in [0.6, 0.7, 0.8]:
                strategies.append({
                    'type': 'mean_reversion',
                    'params': {
                        'ma_period': ma_period,
                        'vol_period': vol_period,
                        'vol_thresh': vol_thresh
                    }
                })
    
    # 5. Multi-factor strategies
    factor_combinations = [
        ['recession', 'trend'],
        ['volatility', 'momentum'],
        ['recession', 'volatility'],
        ['trend', 'momentum'],
        ['recession', 'trend', 'volatility'],
        ['trend', 'volatility', 'momentum']
    ]
    
    for factors in factor_combinations:
        for logic in ['OR', 'AND']:
            strategies.append({
                'type': 'multi_factor',
                'params': {
                    'factors': factors,
                    'logic': logic,
                    'recession_thresh': 0.1,
                    'ma_period': 12,
                    'vol_period': 20,
                    'vol_thresh': 0.7,
                    'momentum_period': 6
                }
            })
    
    # Test all strategies
    results = []
    winning_strategies = []
    
    print(f"\nTesting {len(strategies)} strategy configurations...")
    
    for i, strategy_config in enumerate(strategies):
        try:
            strategy_name = f"{strategy_config['type']}_{i+1}"
            signal = generate_signals(df, strategy_config)
            result = backtest_strategy(df, signal, strategy_name)
            result['config'] = strategy_config
            results.append(result)
            
            # Check if strategy beats target
            if result['annual_return'] > target_annual_return:
                winning_strategies.append(result)
                print(f"✓ WINNER #{len(winning_strategies)}: {strategy_name}")
                print(f"  Annual Return: {result['annual_return']:.2f}% (vs {bnh_result['annual_return']:.2f}%)")
                print(f"  Excess: +{result['annual_return'] - bnh_result['annual_return']:.2f}%")
                print(f"  Sharpe: {result['sharpe_ratio']:.3f}, Max DD: {result['max_drawdown']:.2f}%")
                
        except Exception as e:
            print(f"Error testing strategy {i+1}: {e}")
            continue
    
    # Sort results by annualized return
    results.sort(key=lambda x: x['annual_return'], reverse=True)
    winning_strategies.sort(key=lambda x: x['annual_return'], reverse=True)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Total strategies tested: {len(results)}")
    print(f"Strategies beating target (+{target_annual_return - bnh_result['annual_return']:.1f}%): {len(winning_strategies)}")
    
    if winning_strategies:
        print(f"\nTOP 10 WINNING STRATEGIES:")
        print("-" * 80)
        for i, result in enumerate(winning_strategies[:10]):
            excess = result['annual_return'] - bnh_result['annual_return']
            print(f"{i+1:2d}. {result['strategy_name']:<25} | "
                  f"Annual: {result['annual_return']:6.2f}% | "
                  f"Excess: +{excess:5.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.3f} | "
                  f"DD: {result['max_drawdown']:.2f}%")
        
        # Show best strategy details
        best = winning_strategies[0]
        print(f"\n{'='*60}")
        print("BEST STRATEGY DETAILS")
        print(f"{'='*60}")
        print(f"Strategy Type: {best['config']['type']}")
        print(f"Parameters: {best['config']['params']}")
        print(f"Annual Return: {best['annual_return']:.2f}%")
        print(f"Buy & Hold Annual Return: {bnh_result['annual_return']:.2f}%")
        print(f"Excess Annual Return: +{best['annual_return'] - bnh_result['annual_return']:.2f}%")
        print(f"Total Return: {best['total_return']:.2f}%")
        print(f"Sharpe Ratio: {best['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {best['max_drawdown']:.2f}%")
        print(f"Market Exposure: {best['exposure']:.1f}%")
        print(f"Number of Trades: {best['num_trades']}")
        print(f"Win Rate: {best['win_rate']:.1f}%")
        
        return best, bnh_result
    else:
        print("\nNo strategies found that beat the target return.")
        print(f"\nTOP 10 BEST PERFORMING STRATEGIES:")
        print("-" * 80)
        for i, result in enumerate(results[:10]):
            excess = result['annual_return'] - bnh_result['annual_return']
            print(f"{i+1:2d}. {result['strategy_name']:<25} | "
                  f"Annual: {result['annual_return']:6.2f}% | "
                  f"Excess: {excess:+6.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.3f} | "
                  f"DD: {result['max_drawdown']:.2f}%")
        
        return results[0] if results else None, bnh_result

if __name__ == "__main__":
    best_strategy, bnh_result = run_optimization() 