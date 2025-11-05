"""
Optimize regime-based strategy to reach 2.5% target.
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
from strategies.regime_ml_strategy import RegimeMLStrategy


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


def test_regime_strategy(data: pd.DataFrame, features: pd.DataFrame, 
                         config: Dict, regime_config: Dict, 
                         buy_hold_cagr: float, trading_asset: str) -> Dict:
    """Test regime-based strategy."""
    
    strategy_config = {
        'name': 'Regime_ML_Strategy',
        'trading_asset': trading_asset,
        'holding_period_days': config['portfolio']['holding_period_days'],
        'vol_regime_config': regime_config
    }
    
    try:
        strategy = RegimeMLStrategy(strategy_config)
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
            'config': regime_config,
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


def optimize_regime(config_path: str):
    print("=" * 80)
    print("REGIME-BASED STRATEGY OPTIMIZATION")
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
    
    # Test different regime configurations
    print("\nTesting regime configurations...")
    results = []
    
    # Test different configurations for high_vol and low_vol regimes
    high_vol_configs = [
        {'probability_threshold': 0.50, 'forward_return_periods': 7, 'top_k_features': 120},
        {'probability_threshold': 0.52, 'forward_return_periods': 7, 'top_k_features': 100},
        {'probability_threshold': 0.48, 'forward_return_periods': 7, 'top_k_features': 150},
    ]
    
    low_vol_configs = [
        {'probability_threshold': 0.40, 'forward_return_periods': 7, 'top_k_features': 150},
        {'probability_threshold': 0.38, 'forward_return_periods': 7, 'top_k_features': 150},
        {'probability_threshold': 0.42, 'forward_return_periods': 7, 'top_k_features': 120},
    ]
    
    for high_vol_config in high_vol_configs:
        for low_vol_config in low_vol_configs:
            high_vol_config['ensemble_weights'] = [0.5, 0.5, 0.0, 0.0]
            low_vol_config['ensemble_weights'] = [0.5, 0.5, 0.0, 0.0]
            
            regime_config = {
                'high_vol': high_vol_config,
                'low_vol': low_vol_config
            }
            
            print(f"\nTesting: high_vol_prob={high_vol_config['probability_threshold']:.2f}, "
                  f"low_vol_prob={low_vol_config['probability_threshold']:.2f}")
            
            result = test_regime_strategy(
                data, features, config, regime_config, 
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
            print(f"   Config: {result['config']}")
        
        best = results[0]
        print("\n" + "=" * 80)
        if best['meets_target']:
            print("SUCCESS! FOUND STRATEGY MEETING TARGET")
        else:
            print("BEST CONFIGURATION (below target)")
        print("=" * 80)
        print(f"Outperformance: {best['outperformance']:.4f} (need 0.0250)")
        print(f"CAGR: {best['cagr']:.4f}")
        print(f"Gap: {0.025 - best['outperformance']:.4f}")
        print(f"\nBest Configuration:")
        for regime, config in best['config'].items():
            print(f"  {regime}: {config}")
    else:
        print("\nNo valid results found!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    optimize_regime(args.config)

