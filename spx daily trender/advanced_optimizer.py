import pandas as pd
import numpy as np
import vectorbt as vbt
import os
from itertools import product, combinations
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the recession alert data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'outside data', 'Recession Alert Monthly.xlsx')
    
    print("Loading recession alert data...")
    print(f"Loading from: {file_path}")
    
    df = pd.read_excel(file_path)
    
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
    
    return df

def create_features(df):
    """Create comprehensive technical and fundamental features"""
    features_df = df.copy()
    
    # Price-based features (Moving Averages)
    for period in [3, 6, 9, 12, 15, 18, 21, 24, 30, 36]:
        features_df[f'MA_{period}'] = features_df['Close'].rolling(period).mean()
        features_df[f'Price_above_MA_{period}'] = (features_df['Close'] > features_df[f'MA_{period}']).astype(int)
    
    # Volatility features (corrected for monthly data)
    for period in [6, 12, 20, 24, 30, 36]:
        features_df[f'Vol_{period}'] = features_df['Returns'].rolling(period).std() * np.sqrt(12)
        features_df[f'Vol_{period}_percentile'] = features_df[f'Vol_{period}'].rolling(60).rank(pct=True)
    
    # Momentum features
    for period in [1, 3, 6, 9, 12, 18, 24]:
        features_df[f'Momentum_{period}'] = features_df['Close'].pct_change(period)
        features_df[f'Momentum_{period}_positive'] = (features_df[f'Momentum_{period}'] > 0).astype(int)
    
    # RSI-like features
    for period in [6, 12, 24]:
        delta = features_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        features_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        features_df[f'RSI_{period}_oversold'] = (features_df[f'RSI_{period}'] < 30).astype(int)
        features_df[f'RSI_{period}_overbought'] = (features_df[f'RSI_{period}'] > 70).astype(int)
    
    # Bollinger Band-like features
    for period in [12, 20, 24]:
        features_df[f'BB_middle_{period}'] = features_df['Close'].rolling(period).mean()
        features_df[f'BB_std_{period}'] = features_df['Close'].rolling(period).std()
        features_df[f'BB_upper_{period}'] = features_df[f'BB_middle_{period}'] + (2 * features_df[f'BB_std_{period}'])
        features_df[f'BB_lower_{period}'] = features_df[f'BB_middle_{period}'] - (2 * features_df[f'BB_std_{period}'])
        features_df[f'BB_above_upper_{period}'] = (features_df['Close'] > features_df[f'BB_upper_{period}']).astype(int)
        features_df[f'BB_below_lower_{period}'] = (features_df['Close'] < features_df[f'BB_lower_{period}']).astype(int)
    
    # Recession-based features
    features_df['RecessionProb_MA_3'] = features_df['RecessionProb'].rolling(3).mean()
    features_df['RecessionProb_MA_6'] = features_df['RecessionProb'].rolling(6).mean()
    features_df['RecessionWarnings_MA_3'] = features_df['RecessionWarnings'].rolling(3).mean()
    
    # Recession quantile features
    for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        features_df[f'RecessionProb_above_q{int(q*100)}'] = (features_df['RecessionProb'] > features_df['RecessionProb'].quantile(q)).astype(int)
    
    # Market regime features
    for ma_period in [6, 12, 18, 24]:
        features_df[f'BullMarket_MA{ma_period}'] = (features_df['Close'] > features_df[f'MA_{ma_period}']).astype(int)
        features_df[f'BearMarket_MA{ma_period}'] = (features_df['Close'] < features_df[f'MA_{ma_period}']).astype(int)
    
    # Volatility regime features
    for vol_period in [12, 20, 24]:
        for q in [0.3, 0.5, 0.7, 0.8, 0.9]:
            features_df[f'HighVol_{vol_period}_q{int(q*100)}'] = (features_df[f'Vol_{vol_period}'] > features_df[f'Vol_{vol_period}'].quantile(q)).astype(int)
            features_df[f'LowVol_{vol_period}_q{int(q*100)}'] = (features_df[f'Vol_{vol_period}'] < features_df[f'Vol_{vol_period}'].quantile(1-q)).astype(int)
    
    return features_df

def generate_signals(df, strategy_config):
    """Generate buy/sell signals based on strategy configuration"""
    
    strategy_type = strategy_config['type']
    params = strategy_config['params']
    
    if strategy_type == 'volatility_buy_original':
        # Recreate the original winning strategy
        vol_signal = df['Vol_20'] > df['Vol_20'].quantile(0.8)
        trend_signal = df['Close'] > df[f'MA_{params["ma_period"]}']
        signal = vol_signal | trend_signal
        
    elif strategy_type == 'volatility_buy_enhanced':
        # Enhanced volatility buy with different thresholds
        vol_signal = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        trend_signal = df['Close'] > df[f'MA_{params["ma_period"]}']
        signal = vol_signal | trend_signal
        
    elif strategy_type == 'recession_volatility_combo':
        # Combine recession alerts with volatility
        recession_signal = df['RecessionProb'] > params['recession_thresh']
        vol_signal = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        trend_signal = df['Close'] > df[f'MA_{params["ma_period"]}']
        
        if params['logic'] == 'OR_ALL':
            signal = recession_signal | vol_signal | trend_signal
        elif params['logic'] == 'RECESSION_OR_VOL_AND_TREND':
            signal = recession_signal | (vol_signal & trend_signal)
        else:  # AND_ALL
            signal = recession_signal & vol_signal & trend_signal
            
    elif strategy_type == 'triple_momentum':
        # Multiple timeframe momentum
        short_mom = df[f'Momentum_{params["short_mom"]}'] > 0
        med_mom = df[f'Momentum_{params["med_mom"]}'] > 0
        long_mom = df[f'Momentum_{params["long_mom"]}'] > 0
        
        if params['logic'] == 'ALL':
            signal = short_mom & med_mom & long_mom
        elif params['logic'] == 'MAJORITY':
            signal = (short_mom.astype(int) + med_mom.astype(int) + long_mom.astype(int)) >= 2
        else:  # ANY
            signal = short_mom | med_mom | long_mom
            
    elif strategy_type == 'bollinger_reversion':
        # Bollinger band mean reversion
        bb_period = params['bb_period']
        below_lower = df[f'BB_below_lower_{bb_period}'] == 1
        above_ma = df['Close'] > df[f'MA_{params["ma_period"]}']
        high_vol = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        
        signal = below_lower | (high_vol & above_ma)
        
    elif strategy_type == 'rsi_volatility':
        # RSI oversold + high volatility
        rsi_period = params['rsi_period']
        rsi_oversold = df[f'RSI_{rsi_period}'] < params['rsi_level']
        high_vol = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        trend_ok = df['Close'] > df[f'MA_{params["ma_period"]}']
        
        signal = rsi_oversold | (high_vol & trend_ok)
        
    elif strategy_type == 'adaptive_ma':
        # Adaptive moving average crossover based on volatility
        short_ma = params['short_ma']
        long_ma = params['long_ma']
        
        # Normal trend following
        trend_signal = df[f'MA_{short_ma}'] > df[f'MA_{long_ma}']
        
        # High volatility override
        high_vol = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        vol_signal = df['Close'] > df[f'MA_{params["ma_period"]}']
        
        signal = trend_signal | (high_vol & vol_signal)
        
    elif strategy_type == 'regime_aware':
        # Different strategies for different market regimes
        bull_market = df[f'BullMarket_MA{params["regime_ma"]}'] == 1
        high_vol = df[f'Vol_{params["vol_period"]}'] > df[f'Vol_{params["vol_period"]}'].quantile(params['vol_thresh'])
        
        # In bull markets: momentum
        bull_signal = df[f'Momentum_{params["momentum_period"]}'] > 0
        
        # In bear markets: contrarian + high vol
        bear_signal = high_vol & (df['RecessionProb'] > params['recession_thresh'])
        
        signal = np.where(bull_market, bull_signal, bear_signal)
        
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

def run_advanced_optimization():
    """Run advanced strategy optimization with sophisticated combinations"""
    
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
    
    # Define advanced strategy configurations
    strategies = []
    
    # 1. Original winning strategy variations
    for ma_period in [12, 15, 18, 21, 24]:
        strategies.append({
            'type': 'volatility_buy_original',
            'params': {'ma_period': ma_period}
        })
    
    # 2. Enhanced volatility buy strategies
    for vol_period in [12, 20, 24]:
        for vol_thresh in [0.7, 0.75, 0.8, 0.85, 0.9]:
            for ma_period in [9, 12, 15, 18]:
                strategies.append({
                    'type': 'volatility_buy_enhanced',
                    'params': {
                        'vol_period': vol_period,
                        'vol_thresh': vol_thresh,
                        'ma_period': ma_period
                    }
                })
    
    # 3. Recession + Volatility combinations
    for recession_thresh in [0.05, 0.1, 0.15, 0.2]:
        for vol_period in [12, 20, 24]:
            for vol_thresh in [0.7, 0.8, 0.9]:
                for ma_period in [12, 15, 18]:
                    for logic in ['OR_ALL', 'RECESSION_OR_VOL_AND_TREND']:
                        strategies.append({
                            'type': 'recession_volatility_combo',
                            'params': {
                                'recession_thresh': recession_thresh,
                                'vol_period': vol_period,
                                'vol_thresh': vol_thresh,
                                'ma_period': ma_period,
                                'logic': logic
                            }
                        })
    
    # 4. Triple momentum strategies
    momentum_combos = [(1, 3, 6), (1, 6, 12), (3, 6, 12), (3, 9, 18)]
    for short_mom, med_mom, long_mom in momentum_combos:
        for logic in ['ALL', 'MAJORITY', 'ANY']:
            strategies.append({
                'type': 'triple_momentum',
                'params': {
                    'short_mom': short_mom,
                    'med_mom': med_mom,
                    'long_mom': long_mom,
                    'logic': logic
                }
            })
    
    # 5. Bollinger + volatility reversion
    for bb_period in [12, 20, 24]:
        for ma_period in [12, 15, 18]:
            for vol_period in [12, 20]:
                for vol_thresh in [0.7, 0.8]:
                    strategies.append({
                        'type': 'bollinger_reversion',
                        'params': {
                            'bb_period': bb_period,
                            'ma_period': ma_period,
                            'vol_period': vol_period,
                            'vol_thresh': vol_thresh
                        }
                    })
    
    # 6. RSI + Volatility
    for rsi_period in [6, 12, 24]:
        for rsi_level in [20, 25, 30, 35]:
            for vol_period in [12, 20]:
                for vol_thresh in [0.7, 0.8]:
                    for ma_period in [12, 15]:
                        strategies.append({
                            'type': 'rsi_volatility',
                            'params': {
                                'rsi_period': rsi_period,
                                'rsi_level': rsi_level,
                                'vol_period': vol_period,
                                'vol_thresh': vol_thresh,
                                'ma_period': ma_period
                            }
                        })
    
    # 7. Adaptive MA
    ma_combos = [(3, 12), (6, 15), (9, 18), (3, 18), (6, 21)]
    for short_ma, long_ma in ma_combos:
        for vol_period in [12, 20]:
            for vol_thresh in [0.7, 0.8]:
                for ma_period in [12, 15]:
                    strategies.append({
                        'type': 'adaptive_ma',
                        'params': {
                            'short_ma': short_ma,
                            'long_ma': long_ma,
                            'vol_period': vol_period,
                            'vol_thresh': vol_thresh,
                            'ma_period': ma_period
                        }
                    })
    
    # 8. Regime-aware strategies
    for regime_ma in [12, 18, 24]:
        for vol_period in [12, 20]:
            for vol_thresh in [0.7, 0.8]:
                for momentum_period in [3, 6]:
                    for recession_thresh in [0.1, 0.15]:
                        strategies.append({
                            'type': 'regime_aware',
                            'params': {
                                'regime_ma': regime_ma,
                                'vol_period': vol_period,
                                'vol_thresh': vol_thresh,
                                'momentum_period': momentum_period,
                                'recession_thresh': recession_thresh
                            }
                        })
    
    # Test all strategies
    results = []
    winning_strategies = []
    
    print(f"\nTesting {len(strategies)} advanced strategy configurations...")
    
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
            
        # Progress update every 100 strategies
        if (i + 1) % 100 == 0:
            print(f"Tested {i+1}/{len(strategies)} strategies... Winners so far: {len(winning_strategies)}")
    
    # Sort results by annualized return
    results.sort(key=lambda x: x['annual_return'], reverse=True)
    winning_strategies.sort(key=lambda x: x['annual_return'], reverse=True)
    
    print(f"\n{'='*80}")
    print("FINAL ADVANCED OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Total strategies tested: {len(results)}")
    print(f"Strategies beating target (+{target_annual_return - bnh_result['annual_return']:.1f}%): {len(winning_strategies)}")
    
    if winning_strategies:
        print(f"\nTOP 15 WINNING STRATEGIES:")
        print("-" * 90)
        for i, result in enumerate(winning_strategies[:15]):
            excess = result['annual_return'] - bnh_result['annual_return']
            config_type = result['config']['type']
            print(f"{i+1:2d}. {config_type:<25} | "
                  f"Annual: {result['annual_return']:6.2f}% | "
                  f"Excess: +{excess:5.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.3f} | "
                  f"DD: {result['max_drawdown']:.2f}% | "
                  f"Exp: {result['exposure']:.1f}%")
        
        # Show best strategy details
        best = winning_strategies[0]
        print(f"\n{'='*70}")
        print("🏆 BEST WINNING STRATEGY DETAILS")
        print(f"{'='*70}")
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
        print(f"\nTOP 15 BEST PERFORMING STRATEGIES:")
        print("-" * 90)
        for i, result in enumerate(results[:15]):
            excess = result['annual_return'] - bnh_result['annual_return']
            config_type = result['config']['type']
            print(f"{i+1:2d}. {config_type:<25} | "
                  f"Annual: {result['annual_return']:6.2f}% | "
                  f"Excess: {excess:+6.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.3f} | "
                  f"DD: {result['max_drawdown']:.2f}% | "
                  f"Exp: {result['exposure']:.1f}%")
        
        return results[0] if results else None, bnh_result

if __name__ == "__main__":
    best_strategy, bnh_result = run_advanced_optimization() 