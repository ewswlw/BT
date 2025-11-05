## This script fetches the CDX related data from Bloomberg
## The data is used to backtest the CDX related strategies


import os
import sys
from datetime import datetime

# Add the project root to the Python path to import from data_pipelines
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_pipelines.fetch_data import DataPipeline

#############################################################
# SCRIPT CONFIGURATION
#############################################################
# Use this section to easily configure the data fetching script.
CONFIG = {
    # --- Data Request Mappings ---
    # Define the main securities and fields you want to fetch.
    # Format: {('TICKER', 'FIELD'): 'your_column_name'}
    # NOTE: Some fields have lower frequency data (monthly/weekly) and will be forward-filled:
    #   - LEI YOY Index (MONTHLY)
    #   - .ECONREGI G Index (MONTHLY)
    #   - CGERGLOB Index (WEEKLY)
    "ohlc_mapping": {
        # CDX Related Data
        ('ERIXCDIG Index', 'PX_LAST'): 'us_ig_cdx_er_index',
        ('UISYMH5S Index', 'PX_LAST'): 'us_hy_cdx_er_index',
        ('CDX IG CDSI GEN 5Y MKIT Corp', 'ROLL_ADJUSTED_MID_PRICE'): 'ig_cdx',
        ('CDX HY CDSI GEN 5Y SPRD MKIT Corp', 'ROLL_ADJUSTED_MID_PRICE'): 'hy_cdx',
        ('MOVE Index', 'PX_LAST'): 'rate_vol',
        
        # Treasury Yields & Yield Curve
        ('USGG10YR Index', 'PX_LAST'): 'us_10y_yield',
        ('USGG2YR Index', 'PX_LAST'): 'us_2y_yield',
        ('USYC2Y10 Index', 'PX_LAST'): 'us_2s10s_spread',
        ('USGG30YR Index', 'PX_LAST'): 'us_30y_yield',
        ('USGG3M Index', 'PX_LAST'): 'us_3m_tbill',
        ('USYC3M30 Index', 'PX_LAST'): 'us_3m_30y_spread',
        ('USYC5Y30 Index', 'PX_LAST'): 'us_5y30y_spread',
        ('USYC3M10 Index', 'PX_LAST'): 'us_3m10y_spread',
        
        # Macro Indicators
        ('FDTR Index', 'PX_LAST'): 'fed_funds_rate',
        ('BCMPUSGR Index', 'PX_LAST'): 'us_growth_surprises',
        ('BCMPUSIF Index', 'PX_LAST'): 'us_inflation_surprises',
        ('.HARDATA G Index', 'PX_LAST'): 'us_hard_data_surprises',
        ('LEI YOY  Index', 'PX_LAST'): 'us_lei_yoy',  # MONTHLY - forward filled
        ('.ECONREGI G Index', 'PX_LAST'): 'us_economic_regime',  # MONTHLY - forward filled
        ('USGGBE10 Index', 'PX_LAST'): 'us_10y_breakeven',
        
        # Equity & Volatility
        ('SPX Index', 'TOT_RETURN_INDEX_GROSS_DVDS'): 'spx_tr',
        ('NDX Index', 'PX_LAST'): 'nasdaq100',
        ('VIX Index', 'PX_LAST'): 'vix',
        ('VVIX Index', 'PX_LAST'): 'vvix',
        
        # Commodities & FX
        ('XAU Curncy', 'PX_LAST'): 'gold_spot',
        ('DXY Curncy', 'PX_LAST'): 'dollar_index',
        ('CL1 Comdty', 'PX_LAST'): 'wti_crude',
        
        # Financial Conditions & Credit
        ('BFCIUS Index', 'PX_LAST'): 'us_bloomberg_fci',
        ('CGERGLOB Index', 'PX_LAST'): 'us_equity_revisions',  # WEEKLY - forward filled
    },

    # --- Excess Return Data (Optional) ---
    # Format: {('TICKER', 'FIELD'): 'your_column_name'}
    "er_ytd_mapping": {
        # Example: ('LUACTRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_ig_er',
    },

    # --- Bloomberg Overrides Data (Optional) ---
    # Format: {('TICKER', 'FIELD', 'OVERRIDE'): 'your_column_name'}
    "bloomberg_overrides_mapping": {
        # Example: ('SPX Index', 'BEST_EPS', '1BF'): 'spx_1bf_eps',
    },

    # --- Time Period ---
    "start_date": "2006-01-01",
    "end_date": datetime.now().strftime('%Y-%m-%d'),
    "periodicity": "D",  # Options: 'D' (daily), 'W' (weekly), 'M' (monthly)

    # --- Data Pipeline Parameters ---
    "pipeline_params": {
        "align_start": True,        # Align all series to start on the same date
        "fill": "ffill",            # Options: 'ffill', 'bfill', or None
        "start_date_align": "yes",  # Options: 'yes' or 'no'
    },
    
    # --- Data Category Toggles ---
    # Set to True to enable fetching for a category.
    "data_toggles": {
        "enable_ohlc_data": True,
        "enable_er_ytd_data": False,
        "enable_bloomberg_overrides_data": False,
    },
    
    # --- Bad Dates (Optional) ---
    # Define known bad data points to be cleaned.
    # See `data_pipelines/fetch_data.py` for format details.
    "bad_dates": {},

    # --- Output ---
    "output_csv_path": "data_pipelines/data_processed/cdx_related.csv",
}
#############################################################
# END OF CONFIGURATION
#############################################################


def run_pipeline():
    """
    Fetch and process generic data from Bloomberg using the configured settings.
    """
    print("--- Starting Data Pipeline Script ---")
    print(f"Fetching data from {CONFIG['start_date']} to {CONFIG['end_date']}")
    
    # Initialize the DataPipeline with settings from the CONFIG dictionary
    pipeline = DataPipeline(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        periodicity=CONFIG['periodicity'],
        align_start=CONFIG['pipeline_params']['align_start'],
        fill=CONFIG['pipeline_params']['fill'],
        start_date_align=CONFIG['pipeline_params']['start_date_align'],
        ohlc_mapping=CONFIG['ohlc_mapping'],
        er_ytd_mapping=CONFIG['er_ytd_mapping'],
        bloomberg_overrides_mapping=CONFIG['bloomberg_overrides_mapping'],
        bad_dates=CONFIG['bad_dates'],
        enable_ohlc_data=CONFIG['data_toggles']['enable_ohlc_data'],
        enable_er_ytd_data=CONFIG['data_toggles']['enable_er_ytd_data'],
        enable_bloomberg_overrides_data=CONFIG['data_toggles']['enable_bloomberg_overrides_data']
    )
    
    try:
        # Fetch and process the data
        print("Processing data...")
        fetched_data = pipeline.process_data()
        
        if fetched_data.empty:
            print("\nWarning: The pipeline returned an empty dataset. Check your configuration.")
            return None

        # Display basic information about the dataset
        print(f"\n--- API Data Information ---")
        print(f"Date Range: {fetched_data.index.min()} to {fetched_data.index.max()}")
        print(f"Number of rows: {len(fetched_data)}")
        print(f"Number of columns: {len(fetched_data.columns)}")
        print(f"Columns: {fetched_data.columns.tolist()}")
        
        # Display first and last few rows
        print(f"\nFirst 5 rows:")
        print(fetched_data.head())
        
        print(f"\nLast 5 rows:")
        print(fetched_data.tail())

        print("\n--- API Data Info ---")
        fetched_data.info()
        
       # Display some basic statistics
        print(f"\n--- Basic Statistics (from API) ---")
        print(fetched_data.describe())
        
        # Save the data to CSV
        output_path = CONFIG.get("output_csv_path")
        if output_path:
            print(f"\n--- Saving data to {output_path} ---")
            saved_path = pipeline.save_dataset(fetched_data, output_path)
            print(f"Data saved successfully to {saved_path}")

            # Load data from CSV and verify
            print("\n--- Verifying saved CSV data ---")
            csv_data = pipeline.load_dataset(saved_path)
            
            if csv_data.empty:
                print("\nWarning: The loaded CSV data is empty.")
                return None
            
            print(f"\n--- CSV Data Information ---")
            print(f"Date Range: {csv_data.index.min()} to {csv_data.index.max()}")
            print(f"Number of rows: {len(csv_data)}")
            print(f"Number of columns: {len(csv_data.columns)}")
            print(f"Columns: {csv_data.columns.tolist()}")
            
            print(f"\nFirst 5 rows (from CSV):")
            print(csv_data.head())
            
            print(f"\nLast 5 rows (from CSV):")
            print(csv_data.tail())

            print("\n--- CSV Data Info ---")
            csv_data.info()

            print(f"\n--- Basic Statistics (from CSV) ---")
            print(csv_data.describe())

        print("\n--- Script Finished Successfully ---")
        return fetched_data
        
    except Exception as e:
        print(f"\n--- An error occurred during pipeline execution ---")
        print(f"Error: {str(e)}")
        # raise
        return None

if __name__ == "__main__":
    # Run the data fetching pipeline
    run_pipeline() 