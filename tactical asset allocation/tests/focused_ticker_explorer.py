"""
Focused Ticker Explorer Script

This script provides a focused approach to exploring tickers and generating
the specific summary table requested: ticker, name, start date, end date,
frequency, CAGR since inception, and max drawdown since inception.

Usage:
    poetry run python tactical\ asset\ allocation/tests/focused_ticker_explorer.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from ticker_explorer import TickerExplorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedTickerExplorer(TickerExplorer):
    """
    Focused version of TickerExplorer for specific use cases.
    """
    
    def __init__(self, start_date: str = "1970-01-01", end_date: str = None):
        super().__init__(start_date, end_date)
        
        # Focused ticker list for initial exploration
        self.focused_tickers = [
            # Major Equity Indices
            'SPX Index',           # S&P 500
            'SPTSX Index',         # S&P/TSX Composite (Canada)
            'UKX Index',           # FTSE 100
            'DAX Index',           # DAX (Germany)
            'NKY Index',           # Nikkei 225
            'HSI Index',          # Hang Seng
            'ASX Index',          # ASX 200 (Australia)
            'MXEF Index',         # MSCI Emerging Markets
            'MXWO Index',         # MSCI World
            'RTY Index',          # Russell 2000
            'NDX Index',          # NASDAQ 100
            
            # Fixed Income
            'USGG10YR Index',      # US 10Y Treasury
            'USGG30YR Index',      # US 30Y Treasury
            'LUACTRUU Index',     # US Investment Grade Corporate
            'LF98TRUU Index',     # US High Yield Corporate
            'I05510CA Index',     # Canadian Investment Grade
            
            # Commodities
            'GOLDS Comdty',        # Gold
            'SILVER Comdty',       # Silver
            'CRUDE Comdty',        # WTI Crude Oil
            'BRENT Comdty',        # Brent Crude Oil
            
            # Currency
            'DXY Curncy',         # US Dollar Index
            'EURUSD Curncy',       # EUR/USD
            
            # Real Estate
            'FTSEEPRA Index',     # FTSE EPRA/NAREIT Global
            'FTSEEPUS Index',     # FTSE EPRA/NAREIT US
            
            # Total Return Indices
            'SPXT Index',         # S&P 500 Total Return
            'SPTSXT Index',       # S&P/TSX Total Return
            'UKXT Index',         # FTSE 100 Total Return
        ]
    
    def explore_focused_tickers(self, field: str = 'PX_LAST') -> List[Dict]:
        """
        Explore the focused list of tickers.
        
        Args:
            field: Bloomberg field to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        
        logger.info(f"Exploring {len(self.focused_tickers)} focused tickers")
        
        for i, ticker in enumerate(self.focused_tickers, 1):
            logger.info(f"Processing {i}/{len(self.focused_tickers)}: {ticker}")
            
            try:
                result = self.analyze_ticker(ticker, field)
                results.append(result)
                
                # Store in results
                self.results[f"{ticker}_{field}"] = result
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {str(e)}")
                results.append({
                    'ticker': ticker,
                    'field': field,
                    'available': False,
                    'error': str(e)
                })
        
        return results
    
    def generate_focused_summary(self, results: List[Dict] = None) -> pd.DataFrame:
        """
        Generate focused summary table with requested columns.
        
        Args:
            results: List of analysis results
            
        Returns:
            DataFrame with focused summary
        """
        if results is None:
            results = self.summary_data
        
        if not results:
            logger.warning("No results to summarize")
            return pd.DataFrame()
        
        # Filter for available tickers only
        available_results = [r for r in results if r.get('available', False)]
        
        if not available_results:
            logger.warning("No available tickers found")
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(available_results)
        
        # Select only the requested columns
        columns = [
            'ticker', 'name', 'start_date', 'end_date', 
            'frequency', 'cagr_since_inception', 'max_drawdown_since_inception'
        ]
        
        # Ensure all columns exist
        for col in columns:
            if col not in summary_df.columns:
                summary_df[col] = None
        
        summary_df = summary_df[columns]
        
        # Sort by CAGR descending
        summary_df = summary_df.sort_values('cagr_since_inception', ascending=False)
        
        return summary_df
    
    def save_focused_results(self, output_path: str = None) -> str:
        """
        Save focused results to CSV file.
        
        Args:
            output_path: Path to save results
            
        Returns:
            Path to saved file
        """
        if not self.summary_data:
            logger.warning("No data to save")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"tactical asset allocation/tests/focused_ticker_results_{timestamp}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate and save summary table
        summary_df = self.generate_focused_summary()
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Focused results saved to: {output_path}")
        return output_path


def explore_custom_tickers(tickers: List[str], field: str = 'PX_LAST') -> pd.DataFrame:
    """
    Explore a custom list of tickers.
    
    Args:
        tickers: List of Bloomberg tickers to explore
        field: Bloomberg field to analyze
        
    Returns:
        DataFrame with results
    """
    explorer = FocusedTickerExplorer()
    
    logger.info(f"Exploring custom ticker list: {tickers}")
    
    results = []
    for ticker in tickers:
        try:
            result = explorer.analyze_ticker(ticker, field)
            results.append(result)
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            results.append({
                'ticker': ticker,
                'field': field,
                'available': False,
                'error': str(e)
            })
    
    # Generate summary
    explorer.summary_data = results
    summary_df = explorer.generate_focused_summary()
    
    return summary_df


def main():
    """
    Main function to run focused ticker exploration.
    """
    print("=== Focused Ticker Explorer ===")
    print("Generating summary table with: ticker, name, start date, end date,")
    print("frequency, CAGR since inception, max drawdown since inception")
    
    # Initialize focused explorer
    explorer = FocusedTickerExplorer()
    
    print(f"Date range: {explorer.start_date} to {explorer.end_date}")
    print(f"Exploring {len(explorer.focused_tickers)} focused tickers")
    
    try:
        # Explore focused tickers
        results = explorer.explore_focused_tickers()
        
        # Generate focused summary
        summary_df = explorer.generate_focused_summary(results)
        
        if not summary_df.empty:
            print(f"\n=== FOCUSED SUMMARY TABLE ===")
            print(f"Found {len(summary_df)} available tickers")
            print("\nAll Results:")
            print(summary_df.to_string(index=False))
            
            # Save results
            output_path = explorer.save_focused_results()
            print(f"\nResults saved to: {output_path}")
            
            # Show some statistics
            print(f"\n=== STATISTICS ===")
            print(f"Average CAGR: {summary_df['cagr_since_inception'].mean():.2f}%")
            print(f"Median CAGR: {summary_df['cagr_since_inception'].median():.2f}%")
            print(f"Average Max Drawdown: {summary_df['max_drawdown_since_inception'].mean():.2f}%")
            print(f"Median Max Drawdown: {summary_df['max_drawdown_since_inception'].median():.2f}%")
            
        else:
            print("No available tickers found")
    
    except Exception as e:
        print(f"Error during exploration: {str(e)}")
        logger.error(f"Exploration failed: {str(e)}")


if __name__ == "__main__":
    main()
