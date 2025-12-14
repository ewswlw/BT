#!/usr/bin/env python3
"""
Iterative ML Strategy Optimization Script
Runs multiple iterations until target outperformance is achieved.
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))
os.chdir(script_dir)

from core import (
    DataLoader, CSVDataProvider,
    PortfolioEngine, MetricsCalculator, ReportGenerator,
    create_config_from_dict
)
from strategies import StrategyFactory


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_annualized_return(returns: pd.Series) -> float:
    """Calculate annualized return."""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252  # Trading days per year
    if years > 0:
        return (1 + total_return) ** (1/years) - 1
    return 0.0


def run_iterative_optimization(config_path: str = 'configs/iterative_ml_config.yaml'):
    """Run iterative optimization until target is met."""
    
    print("="*100)
    print("ITERATIVE ML STRATEGY OPTIMIZATION")
    print("="*100)
    print(f"Goal: Beat buy-and-hold by at least 2.5% annualized return")
    print("="*100)
    
    # Load configuration
    config_dict = load_config(config_path)
    base_config = create_config_from_dict(config_dict)
    
    # Load data
    print("\nLoading data...")
    data_loader = DataLoader(CSVDataProvider())
    data = data_loader.load_and_prepare(base_config.data)
    
    print(f"Data loaded. Period: {data.index[0]} to {data.index[-1]}")
    print(f"Total periods: {len(data)}")
    
    # Calculate benchmark return
    benchmark_data = data[base_config.data.benchmark_asset]
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cagr = calculate_annualized_return(benchmark_returns)
    
    print(f"\nBenchmark (Buy-and-Hold) CAGR: {benchmark_cagr:.4%}")
    print(f"Target Strategy CAGR: {benchmark_cagr + config_dict['iterative_ml_strategy']['min_outperformance']:.4%}")
    
    # Iterative optimization loop
    max_iterations = config_dict['iterative_ml_strategy'].get('max_iterations', 20)
    min_outperformance = config_dict['iterative_ml_strategy'].get('min_outperformance', 0.025)
    
    best_result = None
    best_cagr = -np.inf
    best_outperformance = -np.inf
    
    for iteration in range(max_iterations):
        print("\n" + "="*100)
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print("="*100)
        
        # Update iteration number in config
        config_dict['iterative_ml_strategy']['iteration'] = iteration
        
        # Create strategy
        strategy_config = {
            'type': 'iterative_ml_strategy',
            'name': f'iterative_ml_strategy_iter{iteration+1}',
            **config_dict.get('iterative_ml_strategy', {}),
            'random_seed': config_dict.get('random_seed', 42)
        }
        
        strategy = StrategyFactory.create_strategy(strategy_config)
        
        # Create features (empty, strategy generates its own)
        features = pd.DataFrame(index=data.index)
        
        # Run backtest
        print("\nRunning backtest...")
        try:
            result = strategy.backtest(data, features, base_config.portfolio, base_config.data.benchmark_asset)
            
            # Calculate metrics
            strategy_returns = result.returns.dropna()
            strategy_cagr = calculate_annualized_return(strategy_returns)
            outperformance = strategy_cagr - benchmark_cagr
            
            # Calculate additional metrics
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            
            equity_curve = (1 + strategy_returns).cumprod()
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1} RESULTS")
            print(f"{'='*60}")
            print(f"Strategy CAGR: {strategy_cagr:.4%}")
            print(f"Benchmark CAGR: {benchmark_cagr:.4%}")
            print(f"Outperformance: {outperformance:.4%}")
            print(f"Sharpe Ratio: {sharpe:.3f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Trades: {result.trades_count}")
            print(f"Time in Market: {result.time_in_market:.2%}")
            
            # Check if target met
            target_met = outperformance >= min_outperformance
            
            if target_met:
                print(f"\n{'='*60}")
                print(f"✓ TARGET ACHIEVED!")
                print(f"  Outperformance: {outperformance:.4%} >= {min_outperformance:.4%}")
                print(f"{'='*60}")
            
            # Update best result
            if outperformance > best_outperformance:
                best_result = result
                best_cagr = strategy_cagr
                best_outperformance = outperformance
                print(f"\n✓ New best outperformance: {best_outperformance:.4%}")
            
            # Validate performance
            if hasattr(strategy, 'validate_performance'):
                validation = strategy.validate_performance(result, benchmark_data)
                print(f"\nValidation Results:")
                print(f"  Meets outperformance target: {validation.get('meets_outperformance_target', False)}")
                print(f"  Meets Sharpe target: {validation.get('meets_sharpe_target', False)}")
                print(f"  Meets drawdown limit: {validation.get('meets_drawdown_limit', False)}")
                print(f"  Meets min trades: {validation.get('meets_min_trades', False)}")
            
            # Save iteration results
            if hasattr(strategy, 'save_iteration_results'):
                validation = strategy.validate_performance(result, benchmark_data) if hasattr(strategy, 'validate_performance') else {}
                strategy.save_iteration_results(result, validation)
            
            # Generate report for this iteration
            report_generator = ReportGenerator(base_config.reporting.output_dir)
            report_text, html_path = report_generator.generate_strategy_report(
                result, benchmark_data, base_config.reporting.generate_html
            )
            
            # Save artifacts
            artifacts = report_generator.save_artifacts(result)
            
            # Stop if target met
            if target_met:
                print(f"\n{'='*100}")
                print("OPTIMIZATION COMPLETE - TARGET ACHIEVED!")
                print(f"{'='*100}")
                print(f"Final Strategy CAGR: {strategy_cagr:.4%}")
                print(f"Benchmark CAGR: {benchmark_cagr:.4%}")
                print(f"Outperformance: {outperformance:.4%}")
                print(f"\nResults saved to: {base_config.reporting.output_dir}")
                break
            
        except Exception as e:
            print(f"\n✗ Error in iteration {iteration + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")
    
    if best_result is not None:
        print(f"Best Strategy CAGR: {best_cagr:.4%}")
        print(f"Benchmark CAGR: {benchmark_cagr:.4%}")
        print(f"Best Outperformance: {best_outperformance:.4%}")
        print(f"Target: {min_outperformance:.4%}")
        
        if best_outperformance >= min_outperformance:
            print(f"\n✓ SUCCESS: Target outperformance achieved!")
        else:
            print(f"\n⚠ Target not fully met, but best result: {best_outperformance:.4%}")
    else:
        print("No successful iterations completed.")
    
    print(f"{'='*100}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run iterative ML strategy optimization")
    parser.add_argument('--config', type=str, default='configs/iterative_ml_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    run_iterative_optimization(args.config)
