"""
Optimize multi-timeframe strategy to reach 2.5% target.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader
from core.feature_engineering import AdvancedFeatureEngineer
from core.backtest_engine import BacktestEngine
from core.metrics import MetricsCalculator
from core.validation import ProbabilisticSharpeRatio
from strategies.multi_timeframe_strategy import MultiTimeframeStrategy


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_buy_hold_metrics(data: pd.DataFrame, asset: str) -> Dict:
    returns = data[asset].pct_change().dropna()
    total_return = (data[asset].iloc[-1] / data[asset].iloc[0] - 1)
    years = (data.index[-1] - data.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    annual_factor = 252
    mean_return = returns.mean() * annual_factor
    std_return = returns.std() * np.sqrt(annual_factor)
    sharpe = mean_return / std_return if std_return > 0 else 0.0
    
    return {'cagr': cagr, 'total_return': total_return, 'sharpe_ratio': sharpe, 'years': years}


def test_multi_timeframe_strategy(data: pd.DataFrame, features: pd.DataFrame, 
                                 config: Dict, strategy_config: Dict, 
                                 buy_hold_cagr: float, trading_asset: str) -> Dict:
    """Test multi-timeframe strategy."""
    
    strategy_config['trading_asset'] = trading_asset
    strategy_config['holding_period_days'] = config['portfolio']['holding_period_days']
    
    try:
        strategy = MultiTimeframeStrategy(strategy_config)
        backtest_engine = BacktestEngine()
        result = strategy.backtest(data, features, trading_asset, backtest_engine)
        
        benchmark_returns = data[trading_asset].pct_change().dropna()
        strategy_returns = result.returns[result.returns.index.isin(benchmark_returns.index)]
        
        if len(strategy_returns) < 100:
            return None
        
        years = (data.index[-1] - data.index[0]).days / 365.25
        total_return = (1 + strategy_returns).prod() - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        outperformance = cagr - buy_hold_cagr
        
        annual_factor = 252
        mean_return = strategy_returns.mean() * annual_factor
        std_return = strategy_returns.std() * np.sqrt(annual_factor)
        sharpe = mean_return / std_return if std_return > 0 else 0.0
        
        equity = result.equity_curve
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        
        psr_calc = ProbabilisticSharpeRatio()
        psr = psr_calc.calculate(strategy_returns, benchmark_sr=buy_hold_cagr)
        
        return {
            'config': strategy_config,
            'cagr': cagr,
            'outperformance': outperformance,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'psr': psr,
            'time_in_market': result.time_in_market,
            'num_trades': result.trades_count,
            'meets_target': outperformance >= 0.025
        }
    except Exception as e:
        print(f"  Error: {str(e)}")
        return None


def optimize_multi_timeframe(config_path: str):
    print("=" * 80)
    print("MULTI-TIMEFRAME STRATEGY OPTIMIZATION")
    print("=" * 80)
    
    config = load_config(config_path)
    
    print("\nLoading data...")
    script_dir = Path(__file__).parent
    data_file = Path(config['data']['file_path'])
    if not data_file.exists():
        data_file = script_dir.parent / config['data']['file_path']
    if not data_file.exists():
        data_file = script_dir.parent / "data_pipelines" / "data_processed" / "cdx_related.csv"
    
    data_loader = DataLoader()
    data = data_loader.load_and_prepare(
        str(data_file),
        config['data']['date_column'],
        config['data'].get('start_date'),
        config['data'].get('end_date')
    )
    
    trading_asset = config['data']['trading_asset']
    print(f"Data loaded: {data.index[0]} to {data.index[-1]}")
    
    buy_hold = calculate_buy_hold_metrics(data, trading_asset)
    print(f"Buy-and-Hold CAGR: {buy_hold['cagr']:.4f}")
    print(f"Target CAGR: {buy_hold['cagr'] + 0.025:.4f} (outperformance: 2.5%)")
    
    print("\nEngineering features...")
    feature_engineer = AdvancedFeatureEngineer()
    features = feature_engineer.create_features(data, config['features'])
    print(f"Created {len(features.columns)} features")
    
    data = data[~data.index.duplicated(keep='first')]
    features = features[~features.index.duplicated(keep='first')]
    common_index = data.index.intersection(features.index)
    data = data.loc[common_index]
    features = features.loc[common_index]
    
    # Test different configurations
    print("\nTesting multi-timeframe configurations...")
    results = []
    
    # Test different timeframe combinations
    timeframe_combos = [
        ([5, 7, 10], [0.3, 0.4, 0.3]),
        ([5, 7, 10], [0.25, 0.5, 0.25]),
        ([5, 7, 10], [0.33, 0.33, 0.34]),
        ([7, 10, 14], [0.3, 0.4, 0.3]),
        ([7, 10, 14], [0.25, 0.5, 0.25]),
    ]
    
    probability_thresholds = [0.38, 0.40, 0.42, 0.45]
    top_k_features_list = [120, 150]
    
    base_ml_config = {
        'use_random_forest': True,
        'use_lightgbm': True,
        'use_xgboost': False,
        'use_catboost': False,
        'ensemble_weights': [0.5, 0.5, 0.0, 0.0]
    }
    
    for timeframes, weights in timeframe_combos:
        for prob_thresh in probability_thresholds:
            for top_k in top_k_features_list:
                strategy_config = {
                    'name': 'Multi_Timeframe_ML',
                    'timeframes': timeframes,
                    'timeframe_weights': weights,
                    'probability_threshold': prob_thresh,
                    'ml_config': base_ml_config,
                    'use_feature_selection': True,
                    'feature_selection_method': 'combined',
                    'top_k_features': top_k
                }
                
                print(f"\nTesting: timeframes={timeframes}, prob={prob_thresh:.2f}, top_k={top_k}")
                
                result = test_multi_timeframe_strategy(
                    data, features, config, strategy_config, 
                    buy_hold['cagr'], trading_asset
                )
                
                if result:
                    results.append(result)
                    if result['meets_target']:
                        print(f"  ✓✓✓ MEETS TARGET! CAGR: {result['cagr']:.4f}, Outperformance: {result['outperformance']:.4f}")
                    else:
                        print(f"  CAGR: {result['cagr']:.4f}, Outperformance: {result['outperformance']:.4f}")
    
    if results:
        results.sort(key=lambda x: x['outperformance'], reverse=True)
        
        print("\n" + "=" * 80)
        print("TOP RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(results[:10]):
            print(f"\n{i+1}. Outperformance: {result['outperformance']:.4f} {'✓ MEETS TARGET' if result['meets_target'] else ''}")
            print(f"   CAGR: {result['cagr']:.4f}")
            print(f"   Sharpe: {result['sharpe_ratio']:.4f}")
            print(f"   Timeframes: {result['config']['timeframes']}")
            print(f"   Prob Threshold: {result['config']['probability_threshold']:.2f}")
            print(f"   Top K Features: {result['config']['top_k_features']}")
        
        best = results[0]
        print("\n" + "=" * 80)
        if best['meets_target']:
            print("SUCCESS! FOUND STRATEGY MEETING TARGET")
            print("=" * 80)
            print(f"Outperformance: {best['outperformance']:.4f}")
            print(f"CAGR: {best['cagr']:.4f}")
            print(f"\nBest Configuration:")
            print(f"  timeframes: {best['config']['timeframes']}")
            print(f"  timeframe_weights: {best['config']['timeframe_weights']}")
            print(f"  probability_threshold: {best['config']['probability_threshold']}")
            print(f"  top_k_features: {best['config']['top_k_features']}")
        else:
            print("BEST CONFIGURATION (below target)")
            print("=" * 80)
            print(f"Outperformance: {best['outperformance']:.4f} (need 0.0250)")
            print(f"CAGR: {best['cagr']:.4f}")
            print(f"Gap: {0.025 - best['outperformance']:.4f}")
    else:
        print("\nNo valid results found!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    optimize_multi_timeframe(args.config)

