#!/usr/bin/env python3
"""
Main execution script for the modular backtesting framework.
This script can be run from any directory.
"""

import argparse
import sys
import yaml
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths for running from any directory
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Change working directory to the framework directory
os.chdir(script_dir)

from core import (
    DataLoader, CSVDataProvider, 
    TechnicalFeatureEngineer, CrossAssetFeatureEngineer, MultiAssetFeatureEngineer,
    PortfolioEngine, MetricsCalculator, ReportGenerator,
    create_config_from_dict
)
from strategies import StrategyFactory


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def run_single_strategy(strategy_name: str, config_dict: dict):
    """Run a single strategy."""
    print(f"\n{'='*80}")
    print(f"RUNNING STRATEGY: {strategy_name.upper()}")
    print(f"{'='*80}")
    
    # Set random seed (matching reference implementations)
    import random
    import numpy as np
    random_seed = config_dict.get('random_seed', 7)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Random seed set to: {random_seed}")
    
    try:
        # Create configuration objects
        base_config = create_config_from_dict(config_dict)
        
        # Load and prepare data
        print("Loading data...")
        data_loader = DataLoader(CSVDataProvider())
        data = data_loader.load_and_prepare(base_config.data)
        
        print(f"Data loaded. Period: {data.index[0]} to {data.index[-1]}")
        print(f"Total periods: {len(data)}")
        
        # Create features based on strategy type
        print("Engineering features...")
        if strategy_name.lower() == 'cross_asset_momentum':
            feature_engineer = CrossAssetFeatureEngineer()
            feature_config = {
                **config_dict.get('features', {}),
                **config_dict.get('cross_asset_momentum', {})
            }
        elif strategy_name.lower() == 'multi_asset_momentum':
            feature_engineer = MultiAssetFeatureEngineer()
            feature_config = {
                **config_dict.get('features', {}),
                **config_dict.get('multi_asset_momentum', {})
            }
        else:
            feature_engineer = TechnicalFeatureEngineer()
            feature_config = config_dict.get('features', {})
        
        features = feature_engineer.create_features(data, feature_config)
        print(f"Features created: {len(features.columns)} features")
        
        # Create strategy
        strategy_config = {
            'type': strategy_name,
            'name': strategy_name,
            **config_dict.get(strategy_name.lower(), {}),
            **config_dict.get('features', {}),
            'random_seed': config_dict.get('random_seed', 7)
        }
        
        strategy = StrategyFactory.create_strategy(strategy_config)
        print(f"Strategy created: {strategy.name}")
        
        # Run backtest
        print("Running backtest...")
        result = strategy.backtest(data, features, base_config.portfolio, base_config.data.benchmark_asset)
        
        # Generate report
        print("Generating report...")
        report_generator = ReportGenerator(base_config.reporting.output_dir)
        benchmark_data = data[base_config.data.benchmark_asset]
        
        report_text, html_path = report_generator.generate_strategy_report(
            result, benchmark_data, base_config.reporting.generate_html
        )
        
        # Print report
        print(report_text)
        
        # Save artifacts
        artifacts = report_generator.save_artifacts(result)
        print(f"\nArtifacts saved:")
        for artifact_type, path in artifacts.items():
            print(f"  {artifact_type}: {path}")
        
        if html_path:
            print(f"\nHTML Report: {html_path}")
            
    except Exception as e:
        print(f"Error running strategy {strategy_name}: {e}")
        import traceback
        traceback.print_exc()


def run_strategy_comparison(strategy_names: list, config_dict: dict):
    """Run and compare multiple strategies."""
    print(f"\n{'='*100}")
    print(f"RUNNING STRATEGY COMPARISON")
    print(f"Strategies: {', '.join(strategy_names)}")
    print(f"{'='*100}")
    
    try:
        # Create configuration objects
        base_config = create_config_from_dict(config_dict)
        
        # Load and prepare data
        print("Loading data...")
        data_loader = DataLoader(CSVDataProvider())
        data = data_loader.load_and_prepare(base_config.data)
        
        print(f"Data loaded. Period: {data.index[0]} to {data.index[-1]}")
        print(f"Total periods: {len(data)}")
        
        results = {}
        
        for strategy_name in strategy_names:
            print(f"\n{'-'*60}")
            print(f"Running {strategy_name}...")
            print(f"{'-'*60}")
            
            try:
                # Create features based on strategy type
                if strategy_name.lower() == 'cross_asset_momentum':
                    feature_engineer = CrossAssetFeatureEngineer()
                    feature_config = {
                        **config_dict.get('features', {}),
                        **config_dict.get('cross_asset_momentum', {})
                    }
                elif strategy_name.lower() == 'multi_asset_momentum':
                    feature_engineer = MultiAssetFeatureEngineer()
                    feature_config = {
                        **config_dict.get('features', {}),
                        **config_dict.get('multi_asset_momentum', {})
                    }
                else:
                    feature_engineer = TechnicalFeatureEngineer()
                    feature_config = config_dict.get('features', {})
                
                features = feature_engineer.create_features(data, feature_config)
                
                # Create strategy
                strategy_config = {
                    'type': strategy_name,
                    'name': strategy_name,
                    **config_dict.get(strategy_name.lower(), {}),
                    **config_dict.get('features', {}),
                    'random_seed': config_dict.get('random_seed', 7)
                }
                
                strategy = StrategyFactory.create_strategy(strategy_config)
                
                # Run backtest
                result = strategy.backtest(data, features, base_config.portfolio, base_config.data.benchmark_asset)
                results[strategy_name] = result
                
                print(f"✓ {strategy_name} completed")
                print(f"  Total Return: {result.metrics['total_return']:.2%}")
                print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
                
            except Exception as e:
                print(f"✗ Error running {strategy_name}: {e}")
                continue
        
        if results:
            # Generate comparison report
            print(f"\n{'='*80}")
            print("STRATEGY COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            report_generator = ReportGenerator(base_config.reporting.output_dir)
            benchmark_data = data[base_config.data.benchmark_asset]
            
            comparison_report = report_generator.generate_strategy_comparison_report(
                results, benchmark_data
            )
            print(comparison_report)
            
            # Save individual reports
            for strategy_name, result in results.items():
                try:
                    report_text, html_path = report_generator.generate_strategy_report(
                        result, benchmark_data, base_config.reporting.generate_html
                    )
                    artifacts = report_generator.save_artifacts(result)
                    print(f"\n✓ {strategy_name} reports saved")
                except Exception as e:
                    print(f"✗ Error saving {strategy_name} reports: {e}")
                    
    except Exception as e:
        print(f"Error in strategy comparison: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Modular Backtesting Framework")
    parser.add_argument("--config", default="configs/base_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--strategy", help="Single strategy to run")
    parser.add_argument("--compare", nargs="+", 
                       help="Compare multiple strategies")
    parser.add_argument("--list-strategies", action="store_true",
                       help="List available strategies")
    
    args = parser.parse_args()
    
    # List available strategies
    if args.list_strategies:
        print("Available Strategies:")
        for name, strategy_class in StrategyFactory.get_available_strategies().items():
            print(f"  - {name}: {strategy_class.__name__}")
        return
    
    # Load configuration file (we're now always in the framework directory)
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Config file not found: {config_file.absolute()}")
        print("Please ensure the config file exists or provide a valid path.")
        print(f"Current working directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"Using config file: {config_file}")
    
    config_dict = load_config(str(config_file))
    
    # Validate data path (we're now always in the framework directory)
    data_path = Path(config_dict['data']['file_path'])
    if not data_path.exists():
        # Try the original data location as fallback
        fallback_path = Path("../../cad ig er index weekly backtests/data_pipelines/data_processed/with_er_daily.csv")
        if fallback_path.exists():
            config_dict['data']['file_path'] = str(fallback_path)
            data_path = fallback_path
        else:
            print(f"Data file not found: {data_path.absolute()}")
            print("Please update the file_path in the config file or ensure data exists.")
            sys.exit(1)
    
    print(f"Using data file: {data_path}")
    
    # Run based on arguments
    if args.strategy:
        run_single_strategy(args.strategy, config_dict)
    elif args.compare:
        run_strategy_comparison(args.compare, config_dict)
    else:
        # Default: run all strategies
        available_strategies = list(StrategyFactory.get_available_strategies().keys())
        print(f"No specific strategy selected. Running all available strategies:")
        print(f"Strategies: {', '.join(available_strategies)}")
        run_strategy_comparison(available_strategies, config_dict)


if __name__ == "__main__":
    main() 