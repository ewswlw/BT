#!/usr/bin/env python3
"""
Main execution script for CDX backtesting framework.
"""

import argparse
import sys
import yaml
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime

# Setup paths
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))
os.chdir(script_dir)

# Import core modules
from core import (
    DataLoader, CSVDataProvider,
    AdvancedFeatureEngineer,
    BacktestEngine,
    WalkForwardValidator, BiasChecker, ProbabilisticSharpeRatio, DeflatedSharpeRatio,
    MetricsCalculator
)

# Import strategies
from strategies import MLStrategy, RuleBasedStrategy, HybridStrategy


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def calculate_buy_and_hold(data: pd.DataFrame, asset: str, initial_capital: float = 100000) -> dict:
    """Calculate buy-and-hold baseline."""
    if asset not in data.columns:
        raise ValueError(f"Asset {asset} not found in data")
    
    price_series = data[asset]
    returns = price_series.pct_change().dropna()
    
    # Calculate metrics
    total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
    
    # Drawdown
    equity_curve = (1 + returns).cumprod() * initial_capital
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'start_price': price_series.iloc[0],
        'end_price': price_series.iloc[-1],
        'start_date': price_series.index[0],
        'end_date': price_series.index[-1],
        'years': years
    }


def run_strategy_walk_forward(data: pd.DataFrame, features: pd.DataFrame,
                              strategy, price_column: str, backtest_engine: BacktestEngine,
                              walk_forward_validator: WalkForwardValidator) -> dict:
    """Run strategy with walk-forward validation."""
    
    def strategy_func(train_data, test_data):
        """Wrapper function for walk-forward validation."""
        # Remove duplicate indices
        train_data = train_data[~train_data.index.duplicated(keep='first')]
        test_data = test_data[~test_data.index.duplicated(keep='first')]
        features = features[~features.index.duplicated(keep='first')]
        
        # Get features for train and test
        train_features = features.reindex(train_data.index).ffill().fillna(0)
        test_features = features.reindex(test_data.index).ffill().fillna(0)
        
        # Generate signals
        entry_signals, exit_signals = strategy.generate_signals(
            train_data, test_data, train_features, test_features
        )
        
        return entry_signals, exit_signals
    
    # Run walk-forward validation
    wf_results = walk_forward_validator.run_walk_forward(
        data, price_column, strategy_func, backtest_engine
    )
    
    return wf_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='CDX Backtesting Framework')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--strategy', type=str, choices=['ml', 'rule', 'hybrid', 'all'],
                       default='all', help='Strategy to run')
    parser.add_argument('--no-walk-forward', action='store_true',
                       help='Skip walk-forward validation')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CDX BACKTESTING FRAMEWORK")
    print("="*80)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script directory (before chdir)
        config_path = script_dir / args.config
    if not config_path.exists():
        # Try relative to current directory (after chdir)
        config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        print(f"  Checked: {Path(args.config)}")
        print(f"  Checked: {script_dir / args.config}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    print(f"\nLoaded configuration from: {config_path}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    data_loader = DataLoader(CSVDataProvider())
    
    data_file = Path(config['data']['file_path'])
    if not data_file.exists():
        # Try relative to script directory (before chdir)
        data_file = script_dir.parent / config['data']['file_path']
    if not data_file.exists():
        # Try relative to current directory (after chdir)
        data_file = Path(config['data']['file_path'])
    if not data_file.exists():
        # Try absolute path from project root
        data_file = script_dir.parent / "data_pipelines" / "data_processed" / "cdx_related.csv"
    
    data = data_loader.load_and_prepare(
        str(data_file),
        date_column=config['data']['date_column'],
        start_date=config['data'].get('start_date'),
        end_date=config['data'].get('end_date')
    )
    
    data_info = data_loader.get_data_info(data)
    print(f"Data loaded: {data_info['start_date']} to {data_info['end_date']}")
    print(f"Total periods: {data_info['total_periods']}")
    print(f"Columns: {len(data_info['columns'])}")
    
    # Calculate buy-and-hold baseline
    print("\n" + "="*80)
    print("CALCULATING BUY-AND-HOLD BASELINE")
    print("="*80)
    trading_asset = config['data']['trading_asset']
    benchmark_asset = config['data']['benchmark_asset']
    
    buy_hold = calculate_buy_and_hold(data, benchmark_asset, config['portfolio']['initial_capital'])
    print(f"Buy-and-Hold Results:")
    print(f"  Total Return: {buy_hold['total_return']:.2%}")
    print(f"  CAGR: {buy_hold['cagr']:.2%}")
    print(f"  Sharpe Ratio: {buy_hold['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {buy_hold['max_drawdown']:.2%}")
    print(f"  Period: {buy_hold['years']:.2f} years")
    
    target_cagr = buy_hold['cagr'] + config['target']['min_outperformance_annualized']
    print(f"\nTarget CAGR: {target_cagr:.2%} (beat buy-and-hold by {config['target']['min_outperformance_annualized']:.2%})")
    
    # Engineer features
    print("\n" + "="*80)
    print("ENGINEERING FEATURES")
    print("="*80)
    feature_engineer = AdvancedFeatureEngineer()
    features = feature_engineer.create_features(data, config['features'])
    print(f"Created {len(features.columns)} features")
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=config['portfolio']['initial_capital'],
        fees=config['portfolio']['fees'],
        slippage=config['portfolio']['slippage'],
        holding_period_days=config['portfolio']['holding_period_days']
    )
    
    # Initialize walk-forward validator
    walk_forward_validator = WalkForwardValidator(
        initial_train_ratio=config['validation']['initial_train_ratio'],
        test_periods=config['validation']['test_periods'],
        min_test_period_size=config['validation']['min_test_period_size']
    )
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(risk_free_rate=0.0, frequency='D')
    
    # Run strategies
    results = {}
    strategies_to_run = []
    
    if args.strategy == 'all':
        strategies_to_run = ['ml', 'rule', 'hybrid']
    else:
        strategies_to_run = [args.strategy]
    
    for strategy_type in strategies_to_run:
        print("\n" + "="*80)
        print(f"RUNNING {strategy_type.upper()} STRATEGY")
        print("="*80)
        
        # Initialize strategy
        if strategy_type == 'ml':
            strategy_config = config['ml_strategy'].copy()
            strategy_config['trading_asset'] = trading_asset
            strategy_config['holding_period_days'] = config['portfolio']['holding_period_days']
            strategy = MLStrategy(strategy_config)
        elif strategy_type == 'rule':
            strategy_config = config['rule_based_strategy'].copy()
            strategy_config['trading_asset'] = trading_asset
            strategy_config['holding_period_days'] = config['portfolio']['holding_period_days']
            strategy = RuleBasedStrategy(strategy_config)
        elif strategy_type == 'hybrid':
            strategy_config = config['hybrid_strategy'].copy()
            strategy_config['trading_asset'] = trading_asset
            strategy_config['holding_period_days'] = config['portfolio']['holding_period_days']
            strategy_config['ml_config'] = config['ml_strategy'].copy()
            strategy_config['rule_config'] = config['rule_based_strategy'].copy()
            strategy = HybridStrategy(strategy_config)
        else:
            continue
        
        # Run full backtest
        print("\nRunning full backtest...")
        result = strategy.backtest(data, features, trading_asset, backtest_engine)
        
        # Calculate comprehensive metrics
        benchmark_returns = data[trading_asset].pct_change().dropna()
        strategy_returns = result.returns[result.returns.index.isin(benchmark_returns.index)]
        comparison = metrics_calc.calculate_benchmark_comparison(strategy_returns, benchmark_returns)
        
        # Calculate PSR and DSR
        psr_calc = ProbabilisticSharpeRatio()
        psr = psr_calc.calculate(strategy_returns, benchmark_sr=buy_hold['sharpe_ratio'])
        
        dsr_calc = DeflatedSharpeRatio()
        num_strategies = len(strategies_to_run)
        dsr = dsr_calc.calculate(strategy_returns, num_tests=num_strategies)
        
        # Run walk-forward validation if requested
        wf_results = None
        if not args.no_walk_forward:
            print("\nRunning walk-forward validation...")
            wf_results = run_strategy_walk_forward(
                data, features, strategy, trading_asset, backtest_engine, walk_forward_validator
            )
        
        # Store results
        results[strategy_type] = {
            'result': result,
            'comparison': comparison,
            'psr': psr,
            'dsr': dsr,
            'walk_forward': wf_results
        }
        
        # Print results
        print(f"\n{strategy.name} Results:")
        print(f"  CAGR: {result.metrics['cagr']:.2%}")
        print(f"  Outperformance: {comparison['outperformance_abs']:.2%}")
        print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
        print(f"  Time in Market: {result.time_in_market:.2%}")
        print(f"  Trades: {result.trades_count}")
        print(f"  PSR: {psr:.3f}")
        print(f"  DSR: {dsr:.3f}")
        
        if wf_results:
            summary = wf_results['summary']
            print(f"\nWalk-Forward Summary:")
            print(f"  Avg CAGR: {summary['avg_cagr']:.2%}")
            print(f"  Win Rate: {summary['win_rate']:.2%}")
            print(f"  Consistency: {summary['consistency_score']:.2%}")
    
    # Compare all strategies and identify best
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    best_strategy = None
    best_outperformance = -np.inf
    
    for strategy_type, result_dict in results.items():
        outperformance = result_dict['comparison']['outperformance_abs']
        cagr = result_dict['result'].metrics['cagr']
        
        print(f"\n{strategy_type.upper()}:")
        print(f"  CAGR: {cagr:.2%}")
        print(f"  Outperformance: {outperformance:.2%}")
        print(f"  Meets Target: {'YES' if outperformance >= config['target']['min_outperformance_annualized'] else 'NO'}")
        
        if outperformance > best_outperformance:
            best_outperformance = outperformance
            best_strategy = strategy_type
    
    print(f"\n{'='*80}")
    print(f"BEST STRATEGY: {best_strategy.upper()}")
    print(f"Outperformance: {best_outperformance:.2%}")
    print(f"Meets 2.5% target: {'YES' if best_outperformance >= config['target']['min_outperformance_annualized'] else 'NO'}")
    print(f"{'='*80}")
    
    # Save results
    output_dir = Path(config['reporting']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        'buy_and_hold': buy_hold,
        'target_cagr': target_cagr,
        'strategies': {}
    }
    
    for strategy_type, result_dict in results.items():
        summary['strategies'][strategy_type] = {
            'cagr': result_dict['result'].metrics['cagr'],
            'outperformance': result_dict['comparison']['outperformance_abs'],
            'sharpe_ratio': result_dict['result'].metrics['sharpe_ratio'],
            'max_drawdown': result_dict['result'].metrics['max_drawdown'],
            'psr': result_dict['psr'],
            'dsr': result_dict['dsr']
        }
    
    summary['best_strategy'] = best_strategy
    summary['best_outperformance'] = best_outperformance
    
    import json
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

