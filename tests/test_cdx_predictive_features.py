"""
Test script to validate all suggested Bloomberg tickers for CDX predictive features.

This script tests all the tickers suggested for the US IG CDX Excess Return Index strategy
using the same specifications as fetch cdx related.py.

Specifications:
- start_date: 2006-01-01
- periodicity: "D" (daily)
- align_start: True
- fill: "ffill" (forward fill for lower frequency data)
- start_date_align: "yes"

NOTE: Some tickers have lower frequency data (monthly/weekly) and require forward fill:
- LEI YOY Index (MONTHLY) -> us_lei_yoy
- .ECONREGI G Index (MONTHLY) -> us_economic_regime
- CGERGLOB Index (WEEKLY) -> us_equity_revisions

These are automatically forward-filled to daily frequency using the global fill: "ffill" setting.
"""

import os
import sys
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_pipelines.fetch_data import DataPipeline

#############################################################
# TEST CONFIGURATION (Same as fetch cdx related.py)
#############################################################
TEST_CONFIG = {
    "start_date": "2006-01-01",
    "end_date": datetime.now().strftime('%Y-%m-%d'),
    "periodicity": "D",
    "pipeline_params": {
        "align_start": True,
        "fill": "ffill",  # Forward fill for lower frequency data (monthly/weekly)
        "start_date_align": "yes",
    },
}

# Lower frequency tickers that need forward fill
# These are identified as having monthly or weekly data instead of daily
LOWER_FREQUENCY_TICKERS = {
    ('LEI YOY  Index', 'PX_LAST'): {
        'frequency': 'MONTHLY',
        'avg_gap_days': 30.4,
        'column': 'us_lei_yoy'
    },
    ('.ECONREGI G Index', 'PX_LAST'): {
        'frequency': 'MONTHLY',
        'avg_gap_days': 30.4,
        'column': 'us_economic_regime'
    },
    ('CGERGLOB Index', 'PX_LAST'): {
        'frequency': 'WEEKLY',
        'avg_gap_days': 6.9,
        'column': 'us_equity_revisions'
    },
}

#############################################################
# ALL SUGGESTED TICKERS ORGANIZED BY CATEGORY
#############################################################

# 1. CREDIT SPREADS & RELATIVE VALUE
CREDIT_SPREADS = {
    ('LUACTRUU Index', 'INDEX_OAS_TSY_BP'): 'us_ig_oas',
    ('LF98TRUU Index', 'INDEX_OAS_TSY_BP'): 'us_hy_oas',
    ('USGG10YR Index', 'PX_LAST'): 'us_10y_yield',
    ('USGG2YR Index', 'PX_LAST'): 'us_2y_yield',
    ('USYC2Y10 Index', 'PX_LAST'): 'us_2s10s_spread',
    ('USGG30YR Index', 'PX_LAST'): 'us_30y_yield',
    ('USGG3M Index', 'PX_LAST'): 'us_3m_tbill',
    ('USYC3M30 Index', 'PX_LAST'): 'us_3m_30y_spread',
    ('USYC5Y30 Index', 'PX_LAST'): 'us_5y30y_spread',
    ('USYC3M10 Index', 'PX_LAST'): 'us_3m10y_spread',
}

# 2. MACRO INDICATORS
MACRO_INDICATORS = {
    ('FDTR Index', 'PX_LAST'): 'fed_funds_rate',
    ('BCMPUSGR Index', 'PX_LAST'): 'us_growth_surprises',
    ('BCMPUSIF Index', 'PX_LAST'): 'us_inflation_surprises',
    ('.HARDATA G Index', 'PX_LAST'): 'us_hard_data_surprises',
    ('LEI YOY  Index', 'PX_LAST'): 'us_lei_yoy',
    ('.ECONREGI G Index', 'PX_LAST'): 'us_economic_regime',
    ('USGGBE10 Index', 'PX_LAST'): 'us_10y_breakeven',
    ('USGGBE5 Index', 'PX_LAST'): 'us_5y_breakeven',
    ('USGGR10 Index', 'PX_LAST'): 'us_10y_real_rate',
}

# 3. EQUITY & VOLATILITY
EQUITY_VOLATILITY = {
    ('SPX Index', 'PX_LAST'): 'spx_close',
    ('SPX Index', 'TOT_RETURN_INDEX_GROSS_DVDS'): 'spx_tr',
    ('NDX Index', 'PX_LAST'): 'nasdaq100',
    ('VIX Index', 'PX_LAST'): 'vix',
    ('VVIX Index', 'PX_LAST'): 'vvix',
    ('SPXSKEW Index', 'PX_LAST'): 'spx_skew',
    ('SPX Index', 'VOLUME'): 'spx_volume',
}

# 4. CROSS-ASSET MOMENTUM
CROSS_ASSET = {
    ('HYG US Equity', 'PX_LAST'): 'hyg_etf',
    ('LQD US Equity', 'PX_LAST'): 'lqd_etf',
    ('TLT US Equity', 'PX_LAST'): 'tlt_etf',
    ('SPY US Equity', 'PX_LAST'): 'spy_etf',
    ('XAU Curncy', 'PX_LAST'): 'gold_spot',
    ('DXY Curncy', 'PX_LAST'): 'dollar_index',
    ('EURUSD Curncy', 'PX_LAST'): 'eurusd',
    ('JPYUSD Curncy', 'PX_LAST'): 'jpyusd',
    ('CL1 Comdty', 'PX_LAST'): 'wti_crude',
}

# 5. FINANCIAL CONDITIONS
FINANCIAL_CONDITIONS = {
    ('NFCI Index', 'PX_LAST'): 'us_financial_conditions',
    ('STLFSI2 Index', 'PX_LAST'): 'us_stlouis_financial_stress',
    ('BFCIUS Index', 'PX_LAST'): 'us_bloomberg_fci',
}

# 6. CREDIT MARKET SPECIFIC
CREDIT_MARKET = {
    ('USCORPIG Index', 'TOT_RETURN_INDEX_GROSS_DVDS'): 'us_ig_corp_tr',
    ('USCORPHY Index', 'TOT_RETURN_INDEX_GROSS_DVDS'): 'us_hy_corp_tr',
    ('LUACTRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_ig_er_ytd',
    ('LF98TRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_hy_er_ytd',
    ('CGERGLOB Index', 'PX_LAST'): 'us_equity_revisions',
}

# 7. SWAP SPREADS
SWAP_SPREADS = {
    ('USSW2 Curncy', 'PX_LAST'): 'us_2y_swap_rate',
    ('USSW5 Curncy', 'PX_LAST'): 'us_5y_swap_rate',
    ('USSW10 Curncy', 'PX_LAST'): 'us_10y_swap_rate',
    ('USSW30 Curncy', 'PX_LAST'): 'us_30y_swap_rate',
}

# 8. OIS RATES
OIS_RATES = {
    ('USOIS3M Index', 'PX_LAST'): 'us_ois_3m',
    ('USOIS1Y Index', 'PX_LAST'): 'us_ois_1y',
}

# 9. CREDIT SPREAD TERM STRUCTURE
CDX_TERM_STRUCTURE = {
    ('CDX IG CDSI GEN 3Y SPRD MKIT Corp', 'ROLL_ADJUSTED_MID_PRICE'): 'ig_cdx_3y_spread',
    ('CDX IG CDSI GEN 7Y SPRD MKIT Corp', 'ROLL_ADJUSTED_MID_PRICE'): 'ig_cdx_7y_spread',
    ('CDX IG CDSI GEN 10Y SPRD MKIT Corp', 'ROLL_ADJUSTED_MID_PRICE'): 'ig_cdx_10y_spread',
}

# 10. ADVANCED CREDIT METRICS
ADVANCED_CREDIT = {
    ('LUACTRUU Index', 'MODIFIED_DURATION'): 'us_ig_duration',
    ('LUACTRUU Index', 'CONVEXITY'): 'us_ig_convexity',
    ('LF98TRUU Index', 'MODIFIED_DURATION'): 'us_hy_duration',
    ('LF98TRUU Index', 'CONVEXITY'): 'us_hy_convexity',
    ('LUACTRUU Index', 'OAS_SPREAD_STD_DEV'): 'us_ig_oas_volatility',
    ('LUACTRUU Index', 'OAS_SPREAD_SKEW'): 'us_ig_oas_skew',
    ('LUACTRUU Index', 'INDEX_AVG_CREDIT_RATING'): 'us_ig_avg_rating',
    ('LUACTRUU Index', 'NUM_CONSTITUENTS'): 'us_ig_num_constituents',
}

# 11. LIQUIDITY & ISSUANCE
LIQUIDITY_ISSUANCE = {
    ('LQD US Equity', 'VOLUME'): 'lqd_volume',
    ('HYG US Equity', 'VOLUME'): 'hyg_volume',
    ('LQD US Equity', 'FUND_TOTAL_ASSETS'): 'lqd_assets',
    ('HYG US Equity', 'FUND_TOTAL_ASSETS'): 'hyg_assets',
    ('USIGISSUE Index', 'PX_LAST'): 'us_ig_issuance',
    ('USHYISSUE Index', 'PX_LAST'): 'us_hy_issuance',
    ('USIGLIQ Index', 'PX_LAST'): 'us_ig_liquidity_index',
    ('USHYLIQ Index', 'PX_LAST'): 'us_hy_liquidity_index',
}

# 12. DEFAULT & RATING METRICS
DEFAULT_RATING = {
    ('USCORPDEF Index', 'PX_LAST'): 'us_corp_default_rate',
    ('USIGDEF Index', 'PX_LAST'): 'us_ig_default_rate',
    ('USHYDEF Index', 'PX_LAST'): 'us_hy_default_rate',
    ('USCORPREC Index', 'PX_LAST'): 'us_corp_recovery_rate',
    ('USIGDOWNG Index', 'PX_LAST'): 'us_ig_downgrade_rate',
    ('USIGUPG Index', 'PX_LAST'): 'us_ig_upgrade_rate',
}

# 13. SLOOS & CREDIT CONDITIONS
SLOOS_CONDITIONS = {
    ('SLOOSCIL Index', 'PX_LAST'): 'sloos_ci_standards',
    ('SLOOSCRE Index', 'PX_LAST'): 'sloos_cre_standards',
    ('BKCSCREU Index', 'PX_LAST'): 'us_bank_credit_conditions',
}

# 14. SWAPTION VOLATILITY
SWAPTION_VOL = {
    ('USSV010Y Index', 'PX_LAST'): 'us_10y_swaption_vol',
    ('USSV030Y Index', 'PX_LAST'): 'us_30y_swaption_vol',
}

# 15. FED FUNDS FUTURES
FED_FUTURES = {
    ('FF1 Comdty', 'PX_LAST'): 'fed_funds_fut_1m',
    ('FF2 Comdty', 'PX_LAST'): 'fed_funds_fut_2m',
    ('FF3 Comdty', 'PX_LAST'): 'fed_funds_fut_3m',
}

# 16. EURODOLLAR FUTURES
EURODOLLAR_FUTURES = {
    ('ED1 Comdty', 'PX_LAST'): 'eurodollar_fut_1m',
    ('ED2 Comdty', 'PX_LAST'): 'eurodollar_fut_2m',
    ('ED3 Comdty', 'PX_LAST'): 'eurodollar_fut_3m',
}

# 17. ADDITIONAL ECONOMIC SURPRISES
ECON_SURPRISES = {
    ('BCMPUSUR Index', 'PX_LAST'): 'us_unemployment_surprises',
    ('BCMPUSEM Index', 'PX_LAST'): 'us_employment_surprises',
    ('BCMPUSPMI Index', 'PX_LAST'): 'us_pmi_surprises',
    ('BCMPUSRS Index', 'PX_LAST'): 'us_retail_sales_surprises',
}

# 18. SECTOR & ENERGY
SECTOR_ENERGY = {
    ('XLE US Equity', 'PX_LAST'): 'xle_energy_etf',
    ('XLE US Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'): 'xle_tr',
    ('CO1 Comdty', 'PX_LAST'): 'brent_crude',
}

# Combine all categories
ALL_TEST_TICKERS = {
    "Credit Spreads & Relative Value": CREDIT_SPREADS,
    "Macro Indicators": MACRO_INDICATORS,
    "Equity & Volatility": EQUITY_VOLATILITY,
    "Cross-Asset Momentum": CROSS_ASSET,
    "Financial Conditions": FINANCIAL_CONDITIONS,
    "Credit Market Specific": CREDIT_MARKET,
    "Swap Spreads": SWAP_SPREADS,
    "OIS Rates": OIS_RATES,
    "CDX Term Structure": CDX_TERM_STRUCTURE,
    "Advanced Credit Metrics": ADVANCED_CREDIT,
    "Liquidity & Issuance": LIQUIDITY_ISSUANCE,
    "Default & Rating Metrics": DEFAULT_RATING,
    "SLOOS & Credit Conditions": SLOOS_CONDITIONS,
    "Swaption Volatility": SWAPTION_VOL,
    "Fed Funds Futures": FED_FUTURES,
    "Eurodollar Futures": EURODOLLAR_FUTURES,
    "Economic Surprises": ECON_SURPRISES,
    "Sector & Energy": SECTOR_ENERGY,
}

#############################################################
# TEST FUNCTIONS
#############################################################

def test_ticker_category(category_name: str, ticker_mapping: Dict[Tuple[str, str], str]) -> Dict:
    """
    Test a category of tickers.
    
    Args:
        category_name: Name of the category
        ticker_mapping: Dictionary of {(ticker, field): column_name} mappings
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing Category: {category_name}")
    print(f"Number of tickers: {len(ticker_mapping)}")
    print(f"{'='*80}")
    
    results = {
        "category": category_name,
        "total_tickers": len(ticker_mapping),
        "successful": [],
        "failed": [],
        "partial_data": [],
        "errors": {}
    }
    
    # Test each ticker individually to isolate failures
    for (ticker, field), column_name in ticker_mapping.items():
        print(f"\nTesting: {ticker} | {field} -> {column_name}")
        
        try:
            # Create a pipeline with just this ticker
            pipeline = DataPipeline(
                start_date=TEST_CONFIG['start_date'],
                end_date=TEST_CONFIG['end_date'],
                periodicity=TEST_CONFIG['periodicity'],
                align_start=TEST_CONFIG['pipeline_params']['align_start'],
                fill=TEST_CONFIG['pipeline_params']['fill'],
                start_date_align=TEST_CONFIG['pipeline_params']['start_date_align'],
                ohlc_mapping={(ticker, field): column_name},
                er_ytd_mapping={},
                bloomberg_overrides_mapping={},
                bad_dates={},
                enable_ohlc_data=True,
                enable_er_ytd_data=False,
                enable_bloomberg_overrides_data=False
            )
            
            # Try to fetch data
            df = pipeline.process_data()
            
            if df.empty:
                print(f"  ❌ FAILED: Empty DataFrame returned")
                results["failed"].append({
                    "ticker": ticker,
                    "field": field,
                    "column": column_name,
                    "reason": "Empty DataFrame"
                })
            else:
                # Check data quality
                non_null_count = df[column_name].notna().sum()
                total_count = len(df)
                pct_non_null = (non_null_count / total_count * 100) if total_count > 0 else 0
                
                if pct_non_null < 50:
                    print(f"  ⚠️  PARTIAL: Only {pct_non_null:.1f}% non-null data ({non_null_count}/{total_count} rows)")
                    results["partial_data"].append({
                        "ticker": ticker,
                        "field": field,
                        "column": column_name,
                        "non_null_pct": pct_non_null,
                        "non_null_count": non_null_count,
                        "total_count": total_count,
                        "date_range": f"{df.index.min()} to {df.index.max()}" if len(df) > 0 else "N/A"
                    })
                else:
                    # Check if this is a lower frequency ticker
                    is_lower_freq = (ticker, field) in LOWER_FREQUENCY_TICKERS
                    freq_note = f" [{LOWER_FREQUENCY_TICKERS[(ticker, field)]['frequency']}]" if is_lower_freq else ""
                    
                    print(f"  ✅ SUCCESS: {pct_non_null:.1f}% non-null data ({non_null_count}/{total_count} rows){freq_note}")
                    print(f"     Date range: {df.index.min()} to {df.index.max()}")
                    print(f"     First value: {df[column_name].iloc[0] if non_null_count > 0 else 'N/A'}")
                    print(f"     Last value: {df[column_name].iloc[-1] if non_null_count > 0 else 'N/A'}")
                    
                    # Verify forward fill worked for lower frequency data
                    if is_lower_freq:
                        # Check that data is actually filled (should have many more rows than original)
                        freq_info = LOWER_FREQUENCY_TICKERS[(ticker, field)]
                        original_expected_rows = int((df.index.max() - df.index.min()).days / freq_info['avg_gap_days'])
                        if total_count > original_expected_rows * 0.8:  # Allow some margin
                            print(f"     ✓ Forward fill verified: {total_count} daily rows (expected ~{original_expected_rows} original data points)")
                        else:
                            print(f"     ⚠ Forward fill may not be working correctly")
                    
                    results["successful"].append({
                        "ticker": ticker,
                        "field": field,
                        "column": column_name,
                        "non_null_pct": pct_non_null,
                        "non_null_count": non_null_count,
                        "total_count": total_count,
                        "is_lower_frequency": is_lower_freq,
                        "frequency_type": LOWER_FREQUENCY_TICKERS[(ticker, field)]['frequency'] if is_lower_freq else "DAILY",
                        "date_range": {
                            "start": str(df.index.min()),
                            "end": str(df.index.max())
                        },
                        "sample_values": {
                            "first": float(df[column_name].iloc[0]) if non_null_count > 0 else None,
                            "last": float(df[column_name].iloc[-1]) if non_null_count > 0 else None,
                            "mean": float(df[column_name].mean()) if non_null_count > 0 else None,
                        }
                    })
                    
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ FAILED: {error_msg}")
            results["failed"].append({
                "ticker": ticker,
                "field": field,
                "column": column_name,
                "reason": error_msg
            })
            results["errors"][f"{ticker}|{field}"] = error_msg
    
    return results


def run_all_tests():
    """
    Run tests for all ticker categories.
    """
    print("\n" + "="*80)
    print("BLOOMBERG TICKER VALIDATION TEST")
    print(f"Start Date: {TEST_CONFIG['start_date']}")
    print(f"End Date: {TEST_CONFIG['end_date']}")
    print(f"Periodicity: {TEST_CONFIG['periodicity']}")
    print("="*80)
    
    all_results = {}
    summary = {
        "total_categories": len(ALL_TEST_TICKERS),
        "total_tickers": sum(len(mapping) for mapping in ALL_TEST_TICKERS.values()),
        "total_successful": 0,
        "total_failed": 0,
        "total_partial": 0,
    }
    
    # Test each category
    for category_name, ticker_mapping in ALL_TEST_TICKERS.items():
        category_results = test_ticker_category(category_name, ticker_mapping)
        all_results[category_name] = category_results
        
        # Update summary
        summary["total_successful"] += len(category_results["successful"])
        summary["total_failed"] += len(category_results["failed"])
        summary["total_partial"] += len(category_results["partial_data"])
    
    # Count lower frequency tickers
    lower_freq_count = 0
    lower_freq_list = []
    for category_name, results in all_results.items():
        for success in results['successful']:
            if (success['ticker'], success['field']) in LOWER_FREQUENCY_TICKERS:
                lower_freq_count += 1
                lower_freq_list.append({
                    'category': category_name,
                    'ticker': success['ticker'],
                    'field': success['field'],
                    'column': success['column'],
                    'frequency': LOWER_FREQUENCY_TICKERS[(success['ticker'], success['field'])]['frequency']
                })
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Categories Tested: {summary['total_categories']}")
    print(f"Total Tickers Tested: {summary['total_tickers']}")
    print(f"✅ Successful: {summary['total_successful']}")
    print(f"⚠️  Partial Data: {summary['total_partial']}")
    print(f"❌ Failed: {summary['total_failed']}")
    if lower_freq_count > 0:
        print(f"\n📊 Lower Frequency Tickers (with forward fill): {lower_freq_count}")
        for item in lower_freq_list:
            print(f"   - {item['ticker']} | {item['field']} -> {item['column']} [{item['frequency']}]")
    print("="*80)
    
    # Print detailed results by category
    print("\n" + "="*80)
    print("DETAILED RESULTS BY CATEGORY")
    print("="*80)
    
    for category_name, results in all_results.items():
        print(f"\n{category_name}:")
        print(f"  Successful: {len(results['successful'])}/{results['total_tickers']}")
        print(f"  Partial: {len(results['partial_data'])}/{results['total_tickers']}")
        print(f"  Failed: {len(results['failed'])}/{results['total_tickers']}")
        
        if results['failed']:
            print(f"  Failed tickers:")
            for fail in results['failed']:
                print(f"    - {fail['ticker']} | {fail['field']}: {fail['reason']}")
    
    # Save results to JSON
    output_file = os.path.join(project_root, "tests", "ticker_validation_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "summary": summary,
            "config": TEST_CONFIG,
            "results": all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to: {output_file}")
    
    # Create a summary CSV of successful tickers
    successful_list = []
    for category_name, results in all_results.items():
        for success in results['successful']:
            successful_list.append({
                "category": category_name,
                "ticker": success['ticker'],
                "field": success['field'],
                "column_name": success['column'],
                "non_null_pct": success['non_null_pct'],
                "non_null_count": success['non_null_count'],
                "total_count": success['total_count'],
                "start_date": success['date_range']['start'],
                "end_date": success['date_range']['end'],
            })
    
    if successful_list:
        successful_df = pd.DataFrame(successful_list)
        csv_file = os.path.join(project_root, "tests", "successful_tickers.csv")
        successful_df.to_csv(csv_file, index=False)
        print(f"✅ Successful tickers saved to: {csv_file}")
    
    return all_results, summary


if __name__ == "__main__":
    try:
        results, summary = run_all_tests()
        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

