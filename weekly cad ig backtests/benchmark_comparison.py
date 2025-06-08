"""
Benchmark Comparison Script - CAD IG Buy and Hold vs Strategy Performance
======================================================================

This script calculates the buy-and-hold return of the CAD IG index over the full
history and compares our strategy performance against the >2x target requirement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the full dataset"""
    try:
        data_path = '../data_pipelines/data_processed/with_er_daily.csv'
        logger.info(f"Loading data from {data_path}")
        
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        logger.info(f"Data loaded: {len(data)} rows")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def calculate_buy_and_hold_return(data):
    """
    Calculate the total return of buy-and-hold CAD IG index over the full period
    """
    try:
        logger.info("Calculating buy-and-hold return for CAD IG index")
        
        # Get CAD IG excess return index
        cad_ig_index = data['cad_ig_er_index'].dropna()
        
        # Calculate total return over full period
        start_value = cad_ig_index.iloc[0]
        end_value = cad_ig_index.iloc[-1]
        
        total_return = (end_value / start_value - 1) * 100
        
        # Calculate annualized return
        years = len(cad_ig_index) / 252  # Approximate trading days per year
        annualized_return = ((end_value / start_value) ** (1/years) - 1) * 100
        
        logger.info(f"Buy-and-Hold Performance:")
        logger.info(f"  Start Date: {cad_ig_index.index[0]}")
        logger.info(f"  End Date: {cad_ig_index.index[-1]}")
        logger.info(f"  Start Value: {start_value:.4f}")
        logger.info(f"  End Value: {end_value:.4f}")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Annualized Return: {annualized_return:.2f}%")
        logger.info(f"  Period: {years:.1f} years")
        
        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'start_date': cad_ig_index.index[0],
            'end_date': cad_ig_index.index[-1],
            'years': years,
            'start_value': start_value,
            'end_value': end_value
        }
        
    except Exception as e:
        logger.error(f"Error calculating buy-and-hold return: {str(e)}")
        raise

def calculate_weekly_buy_and_hold(data):
    """
    Calculate buy-and-hold return using weekly resampling to match our strategy
    """
    try:
        logger.info("Calculating weekly resampled buy-and-hold return")
        
        # Resample to weekly frequency (matching our strategy)
        weekly_data = data.resample('W-FRI').last()
        cad_ig_weekly = weekly_data['cad_ig_er_index'].dropna()
        
        # Calculate total return over full period
        start_value = cad_ig_weekly.iloc[0]
        end_value = cad_ig_weekly.iloc[-1]
        
        total_return = (end_value / start_value - 1) * 100
        
        # Calculate annualized return (52 weeks per year)
        weeks = len(cad_ig_weekly)
        years = weeks / 52
        annualized_return = ((end_value / start_value) ** (1/years) - 1) * 100
        
        logger.info(f"Weekly Buy-and-Hold Performance:")
        logger.info(f"  Start Date: {cad_ig_weekly.index[0]}")
        logger.info(f"  End Date: {cad_ig_weekly.index[-1]}")
        logger.info(f"  Start Value: {start_value:.4f}")
        logger.info(f"  End Value: {end_value:.4f}")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Annualized Return: {annualized_return:.2f}%")
        logger.info(f"  Period: {weeks} weeks ({years:.1f} years)")
        
        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'start_date': cad_ig_weekly.index[0],
            'end_date': cad_ig_weekly.index[-1],
            'weeks': weeks,
            'years': years,
            'start_value': start_value,
            'end_value': end_value
        }
        
    except Exception as e:
        logger.error(f"Error calculating weekly buy-and-hold return: {str(e)}")
        raise

def compare_strategy_performance(buy_hold_return, strategy_total_return):
    """
    Compare strategy performance against buy-and-hold and check if >2x target is met
    """
    try:
        logger.info("="*80)
        logger.info("STRATEGY vs BUY-AND-HOLD COMPARISON")
        logger.info("="*80)
        
        buy_hold_total = buy_hold_return['total_return_pct']
        target_return = buy_hold_total * 2  # 2x target
        
        # Calculate the ratio
        performance_ratio = strategy_total_return / buy_hold_total if buy_hold_total != 0 else 0
        
        logger.info(f"Buy-and-Hold Total Return:     {buy_hold_total:.2f}%")
        logger.info(f"Strategy Total Return:         {strategy_total_return:.2f}%")
        logger.info(f"Target Return (2x Buy-Hold):   {target_return:.2f}%")
        logger.info(f"Performance Ratio:             {performance_ratio:.2f}x")
        
        if strategy_total_return >= target_return:
            logger.info("🎯 TARGET ACHIEVED! Strategy achieves >2x buy-and-hold return!")
            target_met = True
        else:
            shortfall = target_return - strategy_total_return
            logger.info(f"❌ TARGET NOT MET. Shortfall: {shortfall:.2f}%")
            logger.info(f"   Need additional return of {shortfall:.2f}% to reach 2x target")
            target_met = False
            
        return {
            'buy_hold_return': buy_hold_total,
            'strategy_return': strategy_total_return,
            'target_return': target_return,
            'performance_ratio': performance_ratio,
            'target_met': target_met,
            'shortfall': target_return - strategy_total_return if not target_met else 0
        }
        
    except Exception as e:
        logger.error(f"Error in performance comparison: {str(e)}")
        raise

def analyze_current_results():
    """
    Analyze our current best strategy results against the benchmark
    """
    try:
        logger.info("="*80)
        logger.info("ANALYSIS OF CURRENT STRATEGY RESULTS")
        logger.info("="*80)
        
        # Our best strategy results from the previous run
        # Best strategy: momentum_4_2 with total return of 18.10%
        best_strategy_total_return = 18.10
        
        logger.info(f"Current Best Strategy: momentum_4_2")
        logger.info(f"Strategy Total Return: {best_strategy_total_return:.2f}%")
        logger.info(f"Strategy Sharpe Ratio: 1.484")
        logger.info(f"Strategy CAGR: 0.11%")
        logger.info(f"Strategy Max Drawdown: -0.79%")
        
        return best_strategy_total_return
        
    except Exception as e:
        logger.error(f"Error analyzing current results: {str(e)}")
        raise

def main():
    """
    Main function to perform benchmark comparison and target analysis
    """
    try:
        logger.info("="*80)
        logger.info("CAD IG BENCHMARK COMPARISON & TARGET ANALYSIS")
        logger.info("="*80)
        
        # Load data
        data = load_data()
        
        # Calculate buy-and-hold returns (both daily and weekly)
        daily_buy_hold = calculate_buy_and_hold_return(data)
        weekly_buy_hold = calculate_weekly_buy_and_hold(data)
        
        # Analyze current strategy results
        strategy_return = analyze_current_results()
        
        # Compare against weekly buy-and-hold (matches our strategy frequency)
        comparison = compare_strategy_performance(weekly_buy_hold, strategy_return)
        
        logger.info("\n" + "="*80)
        logger.info("SUMMARY & NEXT STEPS")
        logger.info("="*80)
        
        if not comparison['target_met']:
            logger.info("CURRENT STATUS: Target not achieved")
            logger.info("REQUIRED ACTION: Continue strategy iteration and optimization")
            logger.info("\nSuggested improvements:")
            logger.info("1. Test more aggressive parameter combinations")
            logger.info("2. Combine multiple signals/strategies")
            logger.info("3. Add volatility timing overlays")
            logger.info("4. Test regime-switching approaches")
            logger.info("5. Implement ensemble methods")
            logger.info("6. Test longer-term momentum strategies")
            logger.info("7. Add economic indicator timing")
            
            # Calculate what return we need
            needed_multiple = comparison['target_return'] / comparison['strategy_return']
            logger.info(f"\nTo reach target, strategy needs {needed_multiple:.2f}x current performance")
        else:
            logger.info("🎉 CONGRATULATIONS! Target achieved!")
            
        return comparison
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 