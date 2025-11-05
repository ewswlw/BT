"""
Optimization script to find ML strategy configuration that beats buy-and-hold by 2.5%.

Tests different:
- Probability thresholds
- Forward return periods  
- Ensemble weights
- Model combinations
- Target variable definitions
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from itertools import product

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader
from core.feature_engineering import AdvancedFeatureEngineer
from core.backtest_engine import BacktestEngine
from core.metrics import MetricsCalculator
from core.validation import ProbabilisticSharpeRatio, DeflatedSharpeRatio
from strategies.ml_strategy import MLStrategy

def load_config(config_path: str) -> Dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_buy_hold_metrics(data: pd.DataFrame, asset: str) -> Dict:
    """Calculate buy-and-hold baseline metrics."""
    returns = data[asset].pct_change().dropna()
    metrics_calc = MetricsCalculator()
    
    total_return = (data[asset].iloc[-1] / data[asset].iloc[0] - 1)
    years = (data.index[-1] - data.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate Sharpe ratio manually
    annual_factor = 252
    mean_return = returns.mean() * annual_factor
    std_return = returns.std() * np.sqrt(annual_factor)
    sharpe = mean_return / std_return if std_return > 0 else 0.0
    
    # Calculate max drawdown manually
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'cagr': cagr,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'years': years
    }

def test_strategy_config(data: pd.DataFrame, features: pd.DataFrame, 
                         config: Dict, ml_config: Dict, 
                         buy_hold_cagr: float, trading_asset: str) -> Dict:
    """Test a specific ML strategy configuration."""
    
    # Create strategy config
    strategy_config = config['ml_strategy'].copy()
    strategy_config.update(ml_config)
    strategy_config['trading_asset'] = trading_asset
    strategy_config['holding_period_days'] = config['portfolio']['holding_period_days']
    
    try:
        # Create strategy
        strategy = MLStrategy(strategy_config)
        
        # Run backtest
        backtest_engine = BacktestEngine()
        result = strategy.backtest(data, features, trading_asset, backtest_engine)
        
        # Calculate metrics
        benchmark_returns = data[trading_asset].pct_change().dropna()
        strategy_returns = result.returns[result.returns.index.isin(benchmark_returns.index)]
        
        if len(strategy_returns) < 100:
            return None
        
        years = (data.index[-1] - data.index[0]).days / 365.25
        total_return = (1 + strategy_returns).prod() - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        outperformance = cagr - buy_hold_cagr
        
        # Calculate Sharpe ratio manually
        annual_factor = 252
        mean_return = strategy_returns.mean() * annual_factor
        std_return = strategy_returns.std() * np.sqrt(annual_factor)
        sharpe = mean_return / std_return if std_return > 0 else 0.0
        
        # Calculate max drawdown manually
        equity = result.equity_curve
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        
        psr_calc = ProbabilisticSharpeRatio()
        psr = psr_calc.calculate(strategy_returns, benchmark_sr=buy_hold_cagr)
        
        return {
            'config': ml_config,
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
        print(f"  Error with config {ml_config}: {str(e)}")
        return None

def optimize_strategy(config_path: str, max_configs: int = 100):
    """Optimize strategy by testing different configurations."""
    
    print("=" * 80)
    print("STRATEGY OPTIMIZATION")
    print("=" * 80)
    
    # Load config
    config = load_config(config_path)
    
    # Load data
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
    print(f"Total periods: {len(data)}")
    
    # Calculate buy-and-hold baseline
    print("\nCalculating buy-and-hold baseline...")
    buy_hold = calculate_buy_hold_metrics(data, trading_asset)
    print(f"Buy-and-Hold CAGR: {buy_hold['cagr']:.4f}")
    print(f"Target CAGR: {buy_hold['cagr'] + 0.025:.4f} (outperformance: 2.5%)")
    
    # Engineer features
    print("\nEngineering features...")
    feature_engineer = AdvancedFeatureEngineer()
    features = feature_engineer.create_features(data, config['features'])
    print(f"Created {len(features.columns)} features")
    
    # Remove duplicate indices
    data = data[~data.index.duplicated(keep='first')]
    features = features[~features.index.duplicated(keep='first')]
    common_index = data.index.intersection(features.index)
    data = data.loc[common_index]
    features = features.loc[common_index]
    
    # Define optimization space
    probability_thresholds = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
    forward_periods = [5, 7, 10, 14, 20]
    ensemble_configs = [
        {'use_random_forest': True, 'use_lightgbm': True, 'use_xgboost': False, 'use_catboost': False,
         'ensemble_weights': [0.5, 0.5, 0.0, 0.0]},
        {'use_random_forest': True, 'use_lightgbm': False, 'use_xgboost': True, 'use_catboost': False,
         'ensemble_weights': [0.5, 0.0, 0.5, 0.0]},
        {'use_random_forest': False, 'use_lightgbm': True, 'use_xgboost': True, 'use_catboost': False,
         'ensemble_weights': [0.0, 0.5, 0.5, 0.0]},
        {'use_random_forest': True, 'use_lightgbm': True, 'use_xgboost': True, 'use_catboost': False,
         'ensemble_weights': [0.4, 0.3, 0.3, 0.0]},
        {'use_random_forest': True, 'use_lightgbm': True, 'use_xgboost': True, 'use_catboost': False,
         'ensemble_weights': [0.33, 0.33, 0.34, 0.0]},
        {'use_random_forest': True, 'use_lightgbm': True, 'use_xgboost': True, 'use_catboost': False,
         'ensemble_weights': [0.25, 0.5, 0.25, 0.0]},
    ]
    
    # Test configurations
    print("\nTesting configurations...")
    results = []
    
    configs_to_test = list(product(probability_thresholds, forward_periods, ensemble_configs))
    np.random.shuffle(configs_to_test)  # Randomize order
    
    for i, (prob_thresh, fwd_period, ensemble_cfg) in enumerate(configs_to_test[:max_configs]):
        ml_config = {
            'probability_threshold': prob_thresh,
            'forward_return_periods': fwd_period,
            **ensemble_cfg
        }
        
        print(f"\n[{i+1}/{min(max_configs, len(configs_to_test))}] Testing: prob={prob_thresh:.2f}, fwd={fwd_period}, ensemble={ensemble_cfg['ensemble_weights']}")
        
        result = test_strategy_config(data, features, config, ml_config, 
                                      buy_hold['cagr'], trading_asset)
        
        if result:
            results.append(result)
            if result['meets_target']:
                print(f"  ✓ MEETS TARGET! CAGR: {result['cagr']:.4f}, Outperformance: {result['outperformance']:.4f}")
            else:
                print(f"  CAGR: {result['cagr']:.4f}, Outperformance: {result['outperformance']:.4f}")
    
    # Sort by outperformance
    results.sort(key=lambda x: x['outperformance'], reverse=True)
    
    # Print top results
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    
    for i, result in enumerate(results[:10]):
        print(f"\n{i+1}. Outperformance: {result['outperformance']:.4f} ({'MEETS TARGET' if result['meets_target'] else 'below target'})")
        print(f"   CAGR: {result['cagr']:.4f}")
        print(f"   Sharpe: {result['sharpe_ratio']:.4f}")
        print(f"   Max DD: {result['max_drawdown']:.4f}")
        print(f"   PSR: {result['psr']:.4f}")
        print(f"   Config: {result['config']}")
    
    # Save best configuration
    if results:
        best = results[0]
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"Outperformance: {best['outperformance']:.4f}")
        print(f"CAGR: {best['cagr']:.4f}")
        print(f"Meets Target: {best['meets_target']}")
        print(f"\nConfiguration to add to config.yaml:")
        print(f"  probability_threshold: {best['config']['probability_threshold']}")
        print(f"  forward_return_periods: {best['config']['forward_return_periods']}")
        print(f"  use_random_forest: {best['config']['use_random_forest']}")
        print(f"  use_lightgbm: {best['config']['use_lightgbm']}")
        print(f"  use_xgboost: {best['config']['use_xgboost']}")
        print(f"  use_catboost: {best['config']['use_catboost']}")
        print(f"  ensemble_weights: {best['config']['ensemble_weights']}")
        
        # Save to file
        output_dir = Path(config.get('reporting', {}).get('output_dir', 'outputs'))
        output_dir.mkdir(exist_ok=True)
        results_file = output_dir / 'optimization_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("BEST CONFIGURATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Outperformance: {best['outperformance']:.4f}\n")
            f.write(f"CAGR: {best['cagr']:.4f}\n")
            f.write(f"Meets Target: {best['meets_target']}\n\n")
            f.write("Configuration:\n")
            for key, value in best['config'].items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return best
    else:
        print("\nNo valid results found!")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize ML strategy configuration')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--max-configs', type=int, default=100,
                       help='Maximum number of configurations to test')
    
    args = parser.parse_args()
    
    optimize_strategy(args.config, args.max_configs)
