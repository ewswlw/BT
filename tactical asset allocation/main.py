"""
Main Runner Script

Executes all 4 TAA strategies, generates reports, and creates comparison.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoader
from strategies import (
    VIXTimingStrategy,
    DefenseFirstBaseStrategy,
    DefenseFirstLeveredStrategy,
    SectorRotationStrategy
)
from backtests import TAABacktestEngine
from reporting import ReportGenerator


def main():
    """Main execution function."""
    
    print("\n" + "="*100)
    print(" TACTICAL ASSET ALLOCATION BACKTEST SUITE")
    print("="*100)
    print("\nReplicating 4 academic TAA strategies:")
    print("  1. VIX Timing (2013-2025)")
    print("  2. Defense First Base - SPY (2008-2025)")
    print("  3. Defense First Leveraged - SPXL (2008-2025)")
    print("  4. Sector Rotation (1999-2025)")
    print("\n" + "="*100 + "\n")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Initialize report generator
    output_dir = Path(__file__).parent / 'outputs'
    report_gen = ReportGenerator(output_dir)
    
    # Store all results
    all_results = {}
    
    # -------------------------------------------------------------------------
    # STRATEGY 1: VIX TIMING
    # -------------------------------------------------------------------------
    print("\n" + "#"*100)
    print("# STRATEGY 1: VIX TIMING")
    print("#"*100 + "\n")
    
    try:
        # Load data
        vix_data = data_loader.load_all_strategy_data(
            strategy_name='vix_timing',
            end_date='2025-07-31',  # Match VIX Timing paper end date
            use_cache=True
        )
        
        # Initialize strategy with enhanced parameters
        vix_strategy = VIXTimingStrategy(
            target_volatility=0.10,  # 10% vol target per paper
            vix_averaging=True  # Use 10-day average VIX (per paper recommendation)
        )
        vix_strategy.set_data(
            prices=vix_data['prices'],
            vix=vix_data['vix'],
            tbill=vix_data['tbill']
        )
        
        # Run backtest with volatility normalization
        vix_engine = TAABacktestEngine(
            strategy=vix_strategy,
            prices=vix_data['prices'],
            benchmark_ticker='SPY',
            target_volatility=0.10  # 10% annualized vol target (matches paper)
        )
        vix_portfolio = vix_engine.run()
        
        # Generate metrics
        vix_metrics = vix_engine.get_metrics()
        
        # Generate reports
        report_gen.generate_html_tearsheet(
            portfolio=vix_portfolio,
            strategy_name='VIX_TIMING',
            benchmark_returns=vix_data['prices']['SPY'].pct_change()
        )
        
        vix_report = report_gen.generate_text_report(
            strategy_name='VIX_TIMING',
            metrics=vix_metrics,
            strategy_info=vix_strategy.get_strategy_info(),
            weights=vix_engine.weights
        )
        
        report_gen.save_text_report(vix_report, 'VIX_TIMING')
        
        # Save results
        vix_engine.save_results(output_dir / 'results')
        
        # Store for comparison
        all_results['VIX_TIMING'] = {
            'metrics': vix_metrics,
            'strategy_info': vix_strategy.get_strategy_info()
        }
        
        print("\n✓ VIX Timing strategy completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Error running VIX Timing strategy: {e}\n")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # STRATEGY 2: DEFENSE FIRST BASE (SPY)
    # -------------------------------------------------------------------------
    print("\n" + "#"*100)
    print("# STRATEGY 2: DEFENSE FIRST BASE (SPY)")
    print("#"*100 + "\n")
    
    try:
        # Load data
        df_base_data = data_loader.load_all_strategy_data(
            strategy_name='defense_first_base',
            end_date='2025-07-31',  # Match Defense First paper end date
            use_cache=True
        )
        
        # Initialize strategy
        df_base_strategy = DefenseFirstBaseStrategy()
        df_base_strategy.set_data(
            prices=df_base_data['prices'],
            tbill=df_base_data['tbill']
        )
        
        # Run backtest
        df_base_engine = TAABacktestEngine(
            strategy=df_base_strategy,
            prices=df_base_data['prices'],
            benchmark_ticker='SPY'
        )
        df_base_portfolio = df_base_engine.run()
        
        # Generate metrics
        df_base_metrics = df_base_engine.get_metrics()
        
        # Generate reports
        report_gen.generate_html_tearsheet(
            portfolio=df_base_portfolio,
            strategy_name='DEFENSE_FIRST_BASE',
            benchmark_returns=df_base_data['prices']['SPY'].pct_change()
        )
        
        df_base_report = report_gen.generate_text_report(
            strategy_name='DEFENSE_FIRST_BASE',
            metrics=df_base_metrics,
            strategy_info=df_base_strategy.get_strategy_info(),
            weights=df_base_engine.weights
        )
        
        report_gen.save_text_report(df_base_report, 'DEFENSE_FIRST_BASE')
        
        # Save results
        df_base_engine.save_results(output_dir / 'results')
        
        # Store for comparison
        all_results['DEFENSE_FIRST_BASE'] = {
            'metrics': df_base_metrics,
            'strategy_info': df_base_strategy.get_strategy_info()
        }
        
        print("\n✓ Defense First Base strategy completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Error running Defense First Base strategy: {e}\n")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # STRATEGY 3: DEFENSE FIRST LEVERAGED (SPXL)
    # -------------------------------------------------------------------------
    print("\n" + "#"*100)
    print("# STRATEGY 3: DEFENSE FIRST LEVERAGED (SPXL)")
    print("#"*100 + "\n")
    
    try:
        # Load data
        df_lev_data = data_loader.load_all_strategy_data(
            strategy_name='defense_first_levered',
            end_date='2025-07-31',  # Match Defense First paper end date
            use_cache=True
        )
        
        # Initialize strategy
        df_lev_strategy = DefenseFirstLeveredStrategy()
        df_lev_strategy.set_data(
            prices=df_lev_data['prices'],
            tbill=df_lev_data['tbill']
        )
        
        # Run backtest
        df_lev_engine = TAABacktestEngine(
            strategy=df_lev_strategy,
            prices=df_lev_data['prices'],
            benchmark_ticker='SPY'
        )
        df_lev_portfolio = df_lev_engine.run()
        
        # Generate metrics
        df_lev_metrics = df_lev_engine.get_metrics()
        
        # Generate reports
        report_gen.generate_html_tearsheet(
            portfolio=df_lev_portfolio,
            strategy_name='DEFENSE_FIRST_LEVERED',
            benchmark_returns=df_lev_data['prices']['SPY'].pct_change()
        )
        
        df_lev_report = report_gen.generate_text_report(
            strategy_name='DEFENSE_FIRST_LEVERED',
            metrics=df_lev_metrics,
            strategy_info=df_lev_strategy.get_strategy_info(),
            weights=df_lev_engine.weights
        )
        
        report_gen.save_text_report(df_lev_report, 'DEFENSE_FIRST_LEVERED')
        
        # Save results
        df_lev_engine.save_results(output_dir / 'results')
        
        # Store for comparison
        all_results['DEFENSE_FIRST_LEVERED'] = {
            'metrics': df_lev_metrics,
            'strategy_info': df_lev_strategy.get_strategy_info()
        }
        
        print("\n✓ Defense First Leveraged strategy completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Error running Defense First Leveraged strategy: {e}\n")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # STRATEGY 4: SECTOR ROTATION
    # -------------------------------------------------------------------------
    print("\n" + "#"*100)
    print("# STRATEGY 4: SECTOR ROTATION")
    print("#"*100 + "\n")
    
    try:
        # Load data
        sector_data = data_loader.load_all_strategy_data(
            strategy_name='sector_rotation',
            end_date='2025-08-31',  # Match Sector Rotation paper end date
            use_cache=True
        )
        
        # Initialize strategy
        sector_strategy = SectorRotationStrategy()
        sector_strategy.set_data(
            prices=sector_data['prices']
        )
        
        # Run backtest
        sector_engine = TAABacktestEngine(
            strategy=sector_strategy,
            prices=sector_data['prices'],
            benchmark_ticker='SPY'
        )
        sector_portfolio = sector_engine.run()
        
        # Generate metrics
        sector_metrics = sector_engine.get_metrics()
        
        # Generate reports
        report_gen.generate_html_tearsheet(
            portfolio=sector_portfolio,
            strategy_name='SECTOR_ROTATION',
            benchmark_returns=sector_data['prices']['SPY'].pct_change()
        )
        
        sector_report = report_gen.generate_text_report(
            strategy_name='SECTOR_ROTATION',
            metrics=sector_metrics,
            strategy_info=sector_strategy.get_strategy_info(),
            weights=sector_engine.weights
        )
        
        report_gen.save_text_report(sector_report, 'SECTOR_ROTATION')
        
        # Save results
        sector_engine.save_results(output_dir / 'results')
        
        # Store for comparison
        all_results['SECTOR_ROTATION'] = {
            'metrics': sector_metrics,
            'strategy_info': sector_strategy.get_strategy_info()
        }
        
        print("\n✓ Sector Rotation strategy completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Error running Sector Rotation strategy: {e}\n")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # GENERATE COMPARISON REPORT
    # -------------------------------------------------------------------------
    print("\n" + "#"*100)
    print("# GENERATING COMPREHENSIVE COMPARISON REPORT")
    print("#"*100 + "\n")
    
    if len(all_results) > 0:
        comparison_report = report_gen.generate_comparison_report(all_results)
        report_gen.save_comparison_report(comparison_report)
        
        print("\n✓ Comparison report generated!\n")
    else:
        print("\n✗ No strategies completed successfully - skipping comparison report\n")
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*100)
    print(" BACKTEST SUITE COMPLETED")
    print("="*100)
    print(f"\nTotal Strategies Run: {len(all_results)}/4")
    print(f"Reports Location: {output_dir}")
    print("\nGenerated Files:")
    print("  - HTML Tearsheets: outputs/reports/*.html")
    print("  - Individual Reports: outputs/results/*_report.txt")
    print("  - Comparison Report: outputs/results/strategy_comparison.txt")
    print("  - Weight Files: outputs/results/*_weights.csv")
    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    main()

