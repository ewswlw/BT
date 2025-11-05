"""
Analyze data frequency for successful tickers to identify which ones
need forward fill for lower frequency data (monthly/weekly).
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_pipelines.fetch_data import DataPipeline

# Load successful tickers
successful_tickers_file = Path(project_root) / "tests" / "successful_tickers.csv"
df_successful = pd.read_csv(successful_tickers_file)

# Load validation results for more details
validation_results_file = Path(project_root) / "tests" / "ticker_validation_results.json"
with open(validation_results_file, 'r') as f:
    validation_results = json.load(f)

# Configuration
TEST_CONFIG = {
    "start_date": "2006-01-01",
    "end_date": datetime.now().strftime('%Y-%m-%d'),
    "periodicity": "D",
    "pipeline_params": {
        "align_start": True,
        "fill": "ffill",
        "start_date_align": "yes",
    },
}

# Expected number of trading days from 2006-01-01 to 2025-11-05
# Approximate: ~20 years * 252 trading days/year = ~5040 days
# But actual will vary, so we'll use a threshold
EXPECTED_MIN_DAILY_ROWS = 4000  # Minimum expected for daily data
MIN_DAYS_BETWEEN_ROWS_FOR_MONTHLY = 25  # If average gap > 25 days, likely monthly
MIN_DAYS_BETWEEN_ROWS_FOR_WEEKLY = 4  # If average gap > 4 days, likely weekly

def analyze_ticker_frequency(ticker: str, field: str, column_name: str) -> dict:
    """
    Analyze the actual data frequency for a ticker.
    
    Returns:
        dict with frequency analysis results
    """
    print(f"\nAnalyzing: {ticker} | {field} -> {column_name}")
    
    try:
        # Fetch data
        pipeline = DataPipeline(
            start_date=TEST_CONFIG['start_date'],
            end_date=TEST_CONFIG['end_date'],
            periodicity=TEST_CONFIG['periodicity'],
            align_start=TEST_CONFIG['pipeline_params']['align_start'],
            fill=None,  # Don't fill initially to see actual gaps
            start_date_align="no",  # Don't align to see actual start date
            ohlc_mapping={(ticker, field): column_name},
            er_ytd_mapping={},
            bloomberg_overrides_mapping={},
            bad_dates={},
            enable_ohlc_data=True,
            enable_er_ytd_data=False,
            enable_bloomberg_overrides_data=False
        )
        
        # Get raw data
        df = pipeline.fetch_bloomberg_data(mapping={(ticker, field): column_name})
        
        if df.empty or column_name not in df.columns:
            return {
                "ticker": ticker,
                "field": field,
                "column": column_name,
                "error": "Empty DataFrame or column not found"
            }
        
        # Get non-null values
        series = df[column_name].dropna()
        
        if len(series) == 0:
            return {
                "ticker": ticker,
                "field": field,
                "column": column_name,
                "error": "No non-null values"
            }
        
        # Calculate date gaps
        dates = series.index.sort_values()
        date_gaps = dates.to_series().diff().dt.days.dropna()
        
        # Calculate statistics
        total_rows = len(series)
        date_range_days = (dates.max() - dates.min()).days
        avg_gap_days = date_gaps.mean() if len(date_gaps) > 0 else None
        median_gap_days = date_gaps.median() if len(date_gaps) > 0 else None
        min_gap_days = date_gaps.min() if len(date_gaps) > 0 else None
        max_gap_days = date_gaps.max() if len(date_gaps) > 0 else None
        
        # Determine frequency type
        frequency_type = "UNKNOWN"
        if total_rows >= EXPECTED_MIN_DAILY_ROWS:
            frequency_type = "DAILY"
        elif avg_gap_days and avg_gap_days >= MIN_DAYS_BETWEEN_ROWS_FOR_MONTHLY:
            frequency_type = "MONTHLY"
        elif avg_gap_days and avg_gap_days >= MIN_DAYS_BETWEEN_ROWS_FOR_WEEKLY:
            frequency_type = "WEEKLY"
        elif avg_gap_days and avg_gap_days < MIN_DAYS_BETWEEN_ROWS_FOR_WEEKLY:
            frequency_type = "DAILY"  # Likely daily but shorter history
        else:
            frequency_type = "SPARSE"
        
        # Check if forward fill is needed
        needs_fill = frequency_type in ["MONTHLY", "WEEKLY", "SPARSE"]
        
        result = {
            "ticker": ticker,
            "field": field,
            "column": column_name,
            "total_rows": total_rows,
            "start_date": str(dates.min()),
            "end_date": str(dates.max()),
            "date_range_days": date_range_days,
            "avg_gap_days": float(avg_gap_days) if avg_gap_days is not None else None,
            "median_gap_days": float(median_gap_days) if median_gap_days is not None else None,
            "min_gap_days": float(min_gap_days) if min_gap_days is not None else None,
            "max_gap_days": float(max_gap_days) if max_gap_days is not None else None,
            "frequency_type": frequency_type,
            "needs_fill": needs_fill,
            "expected_daily_rows": EXPECTED_MIN_DAILY_ROWS,
        }
        
        print(f"  Frequency: {frequency_type}")
        print(f"  Total rows: {total_rows}")
        print(f"  Avg gap: {avg_gap_days:.2f} days" if avg_gap_days else "  Avg gap: N/A")
        print(f"  Needs fill: {needs_fill}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "ticker": ticker,
            "field": field,
            "column": column_name,
            "error": str(e)
        }

def main():
    """Analyze frequency for all successful tickers."""
    print("="*80)
    print("DATA FREQUENCY ANALYSIS FOR SUCCESSFUL TICKERS")
    print("="*80)
    
    results = []
    
    for _, row in df_successful.iterrows():
        result = analyze_ticker_frequency(
            ticker=row['ticker'],
            field=row['field'],
            column_name=row['column_name']
        )
        results.append(result)
    
    # Create DataFrame of results
    df_results = pd.DataFrame(results)
    
    # Separate by frequency type
    needs_fill = df_results[df_results['needs_fill'] == True].copy()
    daily_data = df_results[df_results['needs_fill'] == False].copy()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal tickers analyzed: {len(df_results)}")
    print(f"Daily data (no fill needed): {len(daily_data)}")
    print(f"Lower frequency (needs fill): {len(needs_fill)}")
    
    if len(needs_fill) > 0:
        print("\n" + "="*80)
        print("TICKERS THAT NEED FORWARD FILL (Lower Frequency Data)")
        print("="*80)
        print(f"\n{'Ticker':<40} {'Field':<30} {'Frequency':<15} {'Rows':<10} {'Avg Gap':<10}")
        print("-"*80)
        for _, row in needs_fill.iterrows():
            if 'error' not in row or pd.isna(row.get('error')):
                print(f"{row['ticker']:<40} {row['field']:<30} {row['frequency_type']:<15} {int(row['total_rows']):<10} {row['avg_gap_days']:.1f} days" if row['avg_gap_days'] else f"{row['ticker']:<40} {row['field']:<30} {row['frequency_type']:<15} {int(row['total_rows']):<10} N/A")
    
    # Save results
    output_file = Path(project_root) / "tests" / "data_frequency_analysis.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Save tickers that need fill
    if len(needs_fill) > 0:
        needs_fill_file = Path(project_root) / "tests" / "tickers_needing_fill.csv"
        needs_fill.to_csv(needs_fill_file, index=False)
        print(f"✅ Tickers needing fill saved to: {needs_fill_file}")
    
    # Save JSON results
    json_file = Path(project_root) / "tests" / "data_frequency_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ JSON results saved to: {json_file}")
    
    return df_results

if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

