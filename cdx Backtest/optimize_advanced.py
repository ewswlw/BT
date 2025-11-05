"""
Advanced optimization with more aggressive target definitions and feature selection.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader
from core.feature_engineering import AdvancedFeatureEngineer
from core.backtest_engine import BacktestEngine
from core.metrics import MetricsCalculator
from core.validation import ProbabilisticSharpeRatio
from strategies.ml_strategy import MLStrategy

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

def test_strategy_with_target_percentile(data: pd.DataFrame, features: pd.DataFrame, 
                                         config: Dict, ml_config: Dict, 
                                         buy_hold_cagr: float, trading_asset: str,
                                         target_percentile: float) -> Dict:
    """Test strategy with custom target percentile."""
    
    strategy_config = config['ml_strategy'].copy()
    strategy_config.update(ml_config)
    strategy_config['trading_asset'] = trading_asset
    strategy_config['holding_period_days'] = config['portfolio']['holding_period_days']
    
    # Create custom strategy with percentile-based target
    class CustomMLStrategy(MLStrategy):
        def create_target(self, data: pd.DataFrame) -> pd.Series:
            if self.trading_asset not in data.columns:
                raise ValueError(f"Trading asset {self.trading_asset} not found in data")
            
            forward_return = data[self.trading_asset].pct_change(self.forward_return_periods).shift(-self.forward_return_periods)
            percentile_threshold = forward_return.quantile(target_percentile)
            threshold = max(percentile_threshold, 0.0)
            target = (forward_return > threshold).astype(int)
            return target
    
    try:
        strategy = CustomMLStrategy(strategy_config)
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
            'config': ml_config,
            'target_percentile': target_percentile,
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
        return None

def optimize_advanced(config_path: str):
    print("=" * 80)
    print("ADVANCED STRATEGY OPTIMIZATION")
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
    print(f"Target CAGR: {buy_hold['cagr'] + 0.025:.4f}")
    
    print("\nEngineering features...")
    feature_engineer = AdvancedFeatureEngineer()
    features = feature_engineer.create_features(data, config['features'])
    print(f"Created {len(features.columns)} features")
    
    data = data[~data.index.duplicated(keep='first')]
    features = features[~features.index.duplicated(keep='first')]
    common_index = data.index.intersection(features.index)
    data = data.loc[common_index]
    features = features.loc[common_index]
    
    # Test different target percentiles with best config from previous run
    best_config = {
        'probability_threshold': 0.5,
        'forward_return_periods': 7,
        'use_random_forest': True,
        'use_lightgbm': True,
        'use_xgboost': False,
        'use_catboost': False,
        'ensemble_weights': [0.5, 0.5, 0.0, 0.0]
    }
    
    target_percentiles = [0.55, 0.60, 0.65, 0.70, 0.75]
    probability_thresholds = [0.45, 0.48, 0.50, 0.52, 0.55]
    
    print("\nTesting aggressive configurations...")
    results = []
    
    for percentile in target_percentiles:
        for prob_thresh in probability_thresholds:
            test_config = best_config.copy()
            test_config['probability_threshold'] = prob_thresh
            
            print(f"\nTesting: percentile={percentile:.2f}, prob={prob_thresh:.2f}")
            
            result = test_strategy_with_target_percentile(
                data, features, config, test_config, 
                buy_hold['cagr'], trading_asset, percentile
            )
            
            if result:
                results.append(result)
                if result['meets_target']:
                    print(f"  ✓ MEETS TARGET! CAGR: {result['cagr']:.4f}, Outperformance: {result['outperformance']:.4f}")
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
            print(f"   Target Percentile: {result['target_percentile']:.2f}")
            print(f"   Prob Threshold: {result['config']['probability_threshold']:.2f}")
        
        best = results[0]
        if best['meets_target']:
            print("\n" + "=" * 80)
            print("SUCCESS! FOUND STRATEGY MEETING TARGET")
            print("=" * 80)
            print(f"Outperformance: {best['outperformance']:.4f}")
            print(f"CAGR: {best['cagr']:.4f}")
            print(f"\nConfiguration:")
            print(f"  probability_threshold: {best['config']['probability_threshold']}")
            print(f"  forward_return_periods: {best['config']['forward_return_periods']}")
            print(f"  target_percentile: {best['target_percentile']}")
            print(f"  ensemble_weights: {best['config']['ensemble_weights']}")
            
            # Update config file with best settings
            config['ml_strategy']['probability_threshold'] = best['config']['probability_threshold']
            config['ml_strategy']['forward_return_periods'] = best['config']['forward_return_periods']
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"\nConfig updated: {config_path}")
        else:
            print("\nBest result still below 2.5% target. Consider:")
            print("  - More feature engineering")
            print("  - Different model architectures")
            print("  - Alternative target definitions")
    else:
        print("\nNo valid results found!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    optimize_advanced(args.config)

