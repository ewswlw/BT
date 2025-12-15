# XBBG Expert Guide - Complete Bloomberg Python Library Reference

## When to Use

- Use this reference whenever you or an agent must interact with Bloomberg data via xbbg and need authoritative syntax, parameters, and troubleshooting tips.
- Apply it before automating any Bloomberg workflows so credential prerequisites, session handling, and field mappings are explicitly confirmed.
- Reference it while debugging data pulls; the function-specific sections outline edge cases, overrides, and error diagnostics.
- Consult it when onboarding team members to Bloomberg integration; the guide covers installation, ticker conventions, and advanced usage patterns.
- If your project relies exclusively on free APIs, the free-source guides may suffice; once Bloomberg access is available, treat this document as your primary xbbg manual.

## Table of Contents

- [Introduction](#introduction)
- [Installation & Setup](#installation--setup)
- [Core Concepts](#core-concepts)
- [Main Functions Overview](#main-functions-overview)
- [bdh() - Historical Data Retrieval](#bdh---historical-data-retrieval)
- [bdp() - Point-in-Time Reference Data](#bdp---point-in-time-reference-data)
- [bds() - Bulk Data Retrieval](#bds---bulk-data-retrieval)
- [bdib() - Intraday Bar Data](#bdib---intraday-bar-data)
- [bdtick() - Tick Data](#bdtick---tick-data)
- [beqs() - Equity Screening](#beqs---equity-screening)
- [Ticker Mapping & Conventions](#ticker-mapping--conventions)
- [Field Reference Guide](#field-reference-guide)
- [Advanced Usage Patterns](#advanced-usage-patterns)
- [Error Handling & Troubleshooting](#error-handling--troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Real-World Use Cases](#real-world-use-cases)
- [Best Practices](#best-practices)

---

## Introduction

### What is XBBG?

**XBBG** is a Python library that provides a clean, Pythonic interface to Bloomberg's data API. It wraps the complex Bloomberg API calls into simple, intuitive functions that return pandas DataFrames.

### Key Benefits

- **Pandas Integration**: Returns DataFrames for easy data manipulation
- **Simplified Syntax**: Cleaner than the raw Bloomberg API
- **Automatic Connection Management**: Handles Bloomberg session lifecycle
- **Error Handling**: Better error messages than raw API
- **Pythonic Design**: Works naturally with Python data science stack

### Prerequisites

1. **Bloomberg Terminal** installed on your machine
2. **Bloomberg Terminal License** (must be logged in)
3. **Python 3.7+**
4. **xbbg library** installed

---

## Installation & Setup

### Installation

```python
# Install via pip
pip install xbbg

# Or with conda
conda install -c conda-forge xbbg

# Upgrade to latest version
pip install --upgrade xbbg
```

### Basic Import

```python
from xbbg import blp

# That's it! No need to manage connections manually
```

### Verify Installation

```python
from xbbg import blp
import pandas as pd

# Simple test query
df = blp.bdp(tickers='SPY US Equity', flds='PX_LAST')
print(df)

# If this works, you're connected to Bloomberg
```

### Common Import Pattern

```python
from xbbg import blp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
```

---

## Core Concepts

### 1. Tickers (Securities)

Bloomberg uses specific ticker formats that combine:
- **Security identifier** (e.g., AAPL)
- **Market identifier** (e.g., US)  
- **Security type** (e.g., Equity)

**Format**: `IDENTIFIER MARKET TYPE`

### 2. Fields

Fields are the data points you want to retrieve. Each field has a specific Bloomberg field mnemonic (e.g., `PX_LAST`, `PX_VOLUME`).

### 3. Data Types

XBBG provides several data retrieval functions:
- **Historical**: Time series data (bdh)
- **Reference**: Single point data (bdp)
- **Bulk**: Structured datasets (bds)
- **Intraday**: Bar/tick data (bdib, bdtick)
- **Screening**: Universe selection (beqs)

---

## Main Functions Overview

### Quick Reference Table

| Function | Purpose | Returns | Typical Use |
|----------|---------|---------|-------------|
| `bdh()` | Historical time series | DataFrame with dates as index | Price history, volume, fundamentals over time |
| `bdp()` | Current/reference data | DataFrame with tickers as index | Latest prices, static data, company info |
| `bds()` | Bulk reference data | DataFrame with structured data | Earnings history, dividend schedule, members |
| `bdib()` | Intraday bars | DataFrame with datetime index | OHLCV intraday data |
| `bdtick()` | Tick-by-tick data | DataFrame with trade/quote ticks | Microstructure analysis |
| `beqs()` | Equity screening | DataFrame of tickers | Universe construction |

---

## bdh() - Historical Data Retrieval

### Function Signature

```python
blp.bdh(
    tickers,           # Single ticker string or list of tickers
    flds,              # Single field string or list of fields
    start_date,        # Start date (string or datetime)
    end_date=None,     # End date (defaults to today)
    Per='D',           # Periodicity: D, W, M, Q, S, Y
    Fill=None,         # Fill method for missing data
    Days='A',          # Days: A (all), T (trading), W (weekdays)
    **kwargs
)
```

### Basic Usage

```python
# Single ticker, single field
df = blp.bdh(
    tickers='SPY US Equity',
    flds='PX_LAST',
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### Multiple Tickers

```python
# Multiple tickers, single field
tickers = ['SPY US Equity', 'QQQ US Equity', 'IWM US Equity']
df = blp.bdh(
    tickers=tickers,
    flds='PX_LAST',
    start_date='2023-01-01'
)

# Returns MultiIndex DataFrame: (date, ticker)
print(df)
```

### Multiple Fields

```python
# Single ticker, multiple fields
fields = ['PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_VOLUME']
df = blp.bdh(
    tickers='AAPL US Equity',
    flds=fields,
    start_date='2023-01-01'
)

# Returns columns for each field
print(df.columns)
```

### Multiple Tickers and Fields

```python
# Multiple tickers and fields
tickers = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']
fields = ['PX_LAST', 'PX_VOLUME', 'EQY_SH_OUT']

df = blp.bdh(
    tickers=tickers,
    flds=fields,
    start_date='2023-01-01'
)

# Returns MultiIndex: (date, ticker) with columns for each field
```

### Periodicity Options

```python
# Daily (default)
df = blp.bdh('SPY US Equity', 'PX_LAST', '2023-01-01', Per='D')

# Weekly
df = blp.bdh('SPY US Equity', 'PX_LAST', '2023-01-01', Per='W')

# Monthly
df = blp.bdh('SPY US Equity', 'PX_LAST', '2023-01-01', Per='M')

# Quarterly
df = blp.bdh('SPY US Equity', 'PX_LAST', '2020-01-01', Per='Q')

# Semi-Annual
df = blp.bdh('SPY US Equity', 'PX_LAST', '2020-01-01', Per='S')

# Annual/Yearly
df = blp.bdh('SPY US Equity', 'PX_LAST', '2020-01-01', Per='Y')
```

### Fill Options for Missing Data

```python
# Forward fill
df = blp.bdh(
    'AAPL US Equity',
    'PX_LAST',
    '2023-01-01',
    Fill='P'  # Previous value
)

# No fill (default)
df = blp.bdh(
    'AAPL US Equity',
    'PX_LAST',
    '2023-01-01',
    Fill='NA'  # Leave as NaN
)

# Bloomberg's native handling
df = blp.bdh(
    'AAPL US Equity',
    'PX_LAST',
    '2023-01-01',
    Fill='B'  # Bloomberg interpolation
)
```

### Days Filter

```python
# All days (including non-trading days)
df = blp.bdh('SPY US Equity', 'PX_LAST', '2023-01-01', Days='A')

# Trading days only (default)
df = blp.bdh('SPY US Equity', 'PX_LAST', '2023-01-01', Days='T')

# Weekdays
df = blp.bdh('SPY US Equity', 'PX_LAST', '2023-01-01', Days='W')
```

### Advanced Options

```python
# Adjust for splits and dividends
df = blp.bdh(
    'AAPL US Equity',
    ['PX_LAST', 'PX_VOLUME'],
    '2020-01-01',
    adjustment_normal=True,      # Adjust for splits
    adjustment_abnormal=True,    # Adjust for special dividends
    adjustment_split=True,       # Adjust for splits
    currency='USD'               # Convert to specific currency
)

# Calendar type
df = blp.bdh(
    'AAPL US Equity',
    'PX_LAST',
    '2023-01-01',
    CshAdjNormal=True,    # Cash adjustment
    CshAdjAbnormal=True    # Abnormal cash adjustment
)
```

### Common Field Combinations

```python
# OHLCV data
ohlcv_fields = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
df = blp.bdh('SPY US Equity', ohlcv_fields, '2023-01-01')

# Price and fundamental data
fields = ['PX_LAST', 'EQY_SH_OUT', 'CUR_MKT_CAP', 'VOLUME_AVG_20D']
df = blp.bdh('AAPL US Equity', fields, '2023-01-01')

# Returns and volatility
fields = ['PX_LAST', 'VOLATILITY_30D', 'SHARPE_RATIO_1YR']
df = blp.bdh('SPY US Equity', fields, '2023-01-01')
```

---

## bdp() - Point-in-Time Reference Data

### Function Signature

```python
blp.bdp(
    tickers,     # Single ticker or list of tickers
    flds,        # Single field or list of fields
    **kwargs
)
```

### Basic Usage

```python
# Single ticker, single field
df = blp.bdp('AAPL US Equity', 'PX_LAST')
print(df)
```

### Multiple Tickers

```python
# Get current price for multiple stocks
tickers = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']
df = blp.bdp(tickers, 'PX_LAST')
print(df)
```

### Multiple Fields

```python
# Get multiple data points for one ticker
fields = ['PX_LAST', 'CUR_MKT_CAP', 'PE_RATIO', 'DIVIDEND_YIELD']
df = blp.bdp('AAPL US Equity', fields)
print(df)
```

### Multiple Tickers and Fields

```python
# Matrix of data
tickers = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']
fields = ['PX_LAST', 'CUR_MKT_CAP', 'PE_RATIO', 'EPS_CUR_FY']

df = blp.bdp(tickers, fields)
print(df)
# Returns: DataFrame with tickers as index, fields as columns
```

### Common Use Cases

#### Company Information

```python
fields = [
    'NAME',
    'COUNTRY',
    'INDUSTRY_SECTOR',
    'SECURITY_TYP',
    'EXCHANGE_CODE',
    'ID_ISIN',
    'ID_CUSIP',
    'ID_SEDOL'
]

df = blp.bdp('AAPL US Equity', fields)
```

#### Current Valuation Metrics

```python
fields = [
    'PX_LAST',
    'CUR_MKT_CAP',
    'PE_RATIO',
    'PX_TO_BOOK_RATIO',
    'EV_TO_T12M_EBITDA',
    'DIVIDEND_YIELD',
    'SHORT_INT_RATIO'
]

df = blp.bdp('AAPL US Equity', fields)
```

#### Fundamental Data

```python
fields = [
    'SALES_REV_TURN',
    'RETURN_ON_ASSET',
    'RETURN_ON_EQUITY',
    'DEBT_TO_EBITDA',
    'CASH_RATIO',
    'QUICK_RATIO',
    'CURRENT_RATIO'
]

df = blp.bdp('AAPL US Equity', fields)
```

#### Analyst Data

```python
fields = [
    'TARGET_PRICE_12M',
    'NUM_OF_ANALYSTS',
    'BEST_ANALYST_RATING',
    'AVERAGE_RATING',
    'RATING_BUY',
    'RATING_HOLD',
    'RATING_SELL'
]

df = blp.bdp('AAPL US Equity', fields)
```

#### Technical Indicators

```python
fields = [
    'VOLATILITY_30D',
    'VOLATILITY_90D',
    'BETA_RAW_1YR',
    'SHARPE_RATIO_1YR',
    'RSI_14D',
    'VOLUME_AVG_20D',
    'PX_TO_BOOK_RATIO'
]

df = blp.bdp('SPY US Equity', fields)
```

### Options Override

```python
# Get fiscal year specific data
df = blp.bdp(
    'AAPL US Equity',
    'SALES_REV_TURN',
    BEST_FPERIOD_OVERRIDE='2023'
)

# Currency conversion
df = blp.bdp(
    'AAPL US Equity',
    'CUR_MKT_CAP',
    EQY_FUND_CRNCY='EUR'
)
```

---

## bds() - Bulk Data Retrieval

### Function Signature

```python
blp.bds(
    tickers,     # Single ticker or list
    flds,        # Bulk field (usually singular)
    **kwargs
)
```

### What is Bulk Data?

Bulk fields return structured datasets rather than single values. Examples:
- Earnings history
- Dividend schedule
- Index members
- Ratings history
- Estimates

### Common Bulk Fields

#### 1. Earnings History

```python
# Get historical earnings
df = blp.bds('AAPL US Equity', 'EARN_ANN')
print(df)

# Columns typically include:
# - Earnings Per Share
# - EPS Actual
# - EPS Estimate
# - Announcement Date
# - Surprise %
```

#### 2. Dividend History

```python
# Dividend schedule
df = blp.bds('AAPL US Equity', 'DVD_HIST_ALL')
print(df)

# Returns:
# - Declared Date
# - Ex-Date
# - Record Date
# - Payable Date
# - Dividend Amount
# - Dividend Type
```

#### 3. Index Members

```python
# Get all members of an index
df = blp.bds('SPX Index', 'INDX_MEMBERS')
print(df)

# Returns list of constituent tickers
```

#### 4. Analyst Ratings History

```python
# Ratings changes
df = blp.bds('AAPL US Equity', 'ANALYST_RATINGS')
print(df)

# Returns:
# - Date
# - Firm
# - Rating
# - Previous Rating
```

#### 5. Corporate Actions

```python
# Stock splits
df = blp.bds('AAPL US Equity', 'DVD_HIST_ALL')

# Mergers and acquisitions
df = blp.bds('AAPL US Equity', 'COMPANY_TRANSACTIONS')
```

#### 6. Top Holders

```python
# Major shareholders
df = blp.bds('AAPL US Equity', 'TOP_20_HOLDERS_PUBLIC_FILINGS')
print(df)
```

#### 7. Estimates

```python
# Analyst estimates
df = blp.bds('AAPL US Equity', 'BEST_EPS_GAAP')

# Revenue estimates
df = blp.bds('AAPL US Equity', 'BEST_SALES')
```

### Multiple Tickers with Bulk Data

```python
# Get index members for multiple indices
indices = ['SPX Index', 'NDX Index', 'RTY Index']
df = blp.bds(indices, 'INDX_MEMBERS')
print(df)
```

### Overrides for Bulk Data

```python
# Get historical index members as of specific date
df = blp.bds(
    'SPX Index',
    'INDX_MEMBERS',
    INDX_MEMBERS_AS_OF_DATE='20230101'
)

# Get earnings for specific periods
df = blp.bds(
    'AAPL US Equity',
    'EARN_ANN',
    START_DATE='20200101',
    END_DATE='20231231'
)
```

### Bulk Field Categories

#### Financial Statement Fields

```python
# Income statement items
df = blp.bds('AAPL US Equity', 'IS_INC_STMT')

# Balance sheet items
df = blp.bds('AAPL US Equity', 'BS_BALANCE_SHEET')

# Cash flow items
df = blp.bds('AAPL US Equity', 'CF_CASH_FLOW')
```

#### Market Data Fields

```python
# Options chain
df = blp.bds('AAPL US Equity', 'OPT_CHAIN')

# Futures chain
df = blp.bds('CL1 Comdty', 'FUT_CHAIN')
```

---

## bdib() - Intraday Bar Data

### Function Signature

```python
blp.bdib(
    ticker,         # Single ticker (not list)
    dt,             # Date or datetime
    start_time=None,
    end_time=None,
    interval=1,     # Minutes
    typ='TRADE',    # TRADE, BID, ASK, BID_BEST, ASK_BEST, BEST_BID, BEST_ASK
    **kwargs
)
```

### Basic Usage

```python
# Get full day intraday data
df = blp.bdib(
    ticker='SPY US Equity',
    dt='2023-12-01'
)

print(df)
# Returns: DataFrame with open, high, low, close, volume, numEvents
```

### Specific Time Range

```python
# Get data for specific time window
from datetime import datetime

df = blp.bdib(
    ticker='AAPL US Equity',
    dt='2023-12-01',
    start_time='09:30:00',
    end_time='16:00:00'
)
```

### Different Intervals

```python
# 1-minute bars (default)
df = blp.bdib('SPY US Equity', dt='2023-12-01', interval=1)

# 5-minute bars
df = blp.bdib('SPY US Equity', dt='2023-12-01', interval=5)

# 15-minute bars
df = blp.bdib('SPY US Equity', dt='2023-12-01', interval=15)

# 60-minute bars
df = blp.bdib('SPY US Equity', dt='2023-12-01', interval=60)
```

### Different Data Types

```python
# Trade data (default)
df = blp.bdib('AAPL US Equity', dt='2023-12-01', typ='TRADE')

# Bid data
df = blp.bdib('AAPL US Equity', dt='2023-12-01', typ='BID')

# Ask data
df = blp.bdib('AAPL US Equity', dt='2023-12-01', typ='ASK')

# Best bid
df = blp.bdib('AAPL US Equity', dt='2023-12-01', typ='BEST_BID')

# Best ask
df = blp.bdib('AAPL US Equity', dt='2023-12-01', typ='BEST_ASK')
```

### Multiple Days

```python
import pandas as pd
from datetime import datetime, timedelta

# Loop through multiple days
start_date = datetime(2023, 12, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.bdate_range(start_date, end_date)

all_data = []
for date in date_range:
    try:
        df = blp.bdib('SPY US Equity', dt=date.strftime('%Y-%m-%d'))
        all_data.append(df)
    except:
        continue

combined_df = pd.concat(all_data)
```

### Market Microstructure Analysis

```python
# Get bid-ask spread data
bid_df = blp.bdib('AAPL US Equity', dt='2023-12-01', typ='BID', interval=1)
ask_df = blp.bdib('AAPL US Equity', dt='2023-12-01', typ='ASK', interval=1)

# Calculate spread
spread = ask_df['close'] - bid_df['close']
```

---

## bdtick() - Tick Data

### Function Signature

```python
blp.bdtick(
    ticker,
    dt,
    start_time=None,
    end_time=None,
    types=['TRADE'],  # TRADE, BID, ASK, AT_TRADE, BEST_BID, BEST_ASK
    **kwargs
)
```

### Basic Usage

```python
# Get all trades for a day
df = blp.bdtick(
    ticker='AAPL US Equity',
    dt='2023-12-01'
)

print(df)
# Returns: DataFrame with timestamp, price, size for each tick
```

### Specific Time Window

```python
# First hour of trading
df = blp.bdtick(
    ticker='AAPL US Equity',
    dt='2023-12-01',
    start_time='09:30:00',
    end_time='10:30:00'
)
```

### Different Tick Types

```python
# Trade ticks
df = blp.bdtick('AAPL US Equity', dt='2023-12-01', types=['TRADE'])

# Bid and Ask ticks
df = blp.bdtick('AAPL US Equity', dt='2023-12-01', types=['BID', 'ASK'])

# All tick types
df = blp.bdtick('AAPL US Equity', dt='2023-12-01', 
                types=['TRADE', 'BID', 'ASK', 'BEST_BID', 'BEST_ASK'])
```

### Tick Analysis Examples

```python
# Volume profile
ticks = blp.bdtick('SPY US Equity', dt='2023-12-01')
volume_by_minute = ticks.groupby(ticks.index.floor('T'))['size'].sum()

# Trade size distribution
trade_size_hist = ticks['size'].hist(bins=50)

# VWAP calculation
ticks['dollar_volume'] = ticks['value'] * ticks['size']
vwap = ticks['dollar_volume'].sum() / ticks['size'].sum()
```

---

## beqs() - Equity Screening

### Function Signature

```python
blp.beqs(
    screen_name,    # Name of saved screen
    **kwargs
)
```

### Basic Usage

```python
# Run a saved equity screen
df = blp.beqs('My Saved Screen')
print(df)

# Returns: List of tickers matching screen criteria
```

### Programmatic Screening

While `beqs()` uses saved screens, you can combine `bdp()` for custom screening:

```python
# Get universe
universe = blp.bds('SPX Index', 'INDX_MEMBERS')['Member Ticker']

# Get screening data
fields = ['PX_LAST', 'PE_RATIO', 'DIVIDEND_YIELD', 'CUR_MKT_CAP']
df = blp.bdp(universe.tolist(), fields)

# Apply filters
filtered = df[
    (df['PE_RATIO'] < 20) &
    (df['DIVIDEND_YIELD'] > 2) &
    (df['CUR_MKT_CAP'] > 10000)  # > $10B
]

print(filtered)
```

---

## Ticker Mapping & Conventions

### Equity Tickers

```python
# US Stocks
'AAPL US Equity'
'MSFT US Equity'
'GOOGL US Equity'

# International stocks
'7203 JT Equity'        # Toyota (Japan)
'VOD LN Equity'         # Vodafone (London)
'RIO AU Equity'         # Rio Tinto (Australia)
'NESN SE Equity'        # Nestle (Switzerland)
'SAP GY Equity'         # SAP (Germany)
```

### Market Codes

| Code | Market |
|------|--------|
| US | United States |
| LN | London |
| JP or JT | Japan |
| HK | Hong Kong |
| AU | Australia |
| GY | Germany |
| FP | France |
| IM | Italy |
| SM | Spain |
| SE | Switzerland |
| CN | China |

### Security Types

```python
# Equities
'AAPL US Equity'

# Indices
'SPX Index'
'NDX Index'
'INDU Index'         # Dow Jones
'RTY Index'          # Russell 2000

# ETFs
'SPY US Equity'
'QQQ US Equity'
'IWM US Equity'

# Bonds
'T 2 1/4 08/15/27 Govt'
'912828ZG8 Corp'

# Commodities
'CL1 Comdty'         # Crude Oil
'GC1 Comdty'         # Gold
'HG1 Comdty'         # Copper

# Currencies
'EURUSD Curncy'
'GBPUSD Curncy'
'USDJPY Curncy'

# Mutual Funds
'VFINX US Equity'

# Options (use OPT_CHAIN for options)
```

### Index Codes

```python
# US Indices
'SPX Index'          # S&P 500
'NDX Index'          # Nasdaq 100
'INDU Index'         # Dow Jones Industrial
'RTY Index'          # Russell 2000
'RIY Index'          # Russell 1000
'RAY Index'          # Russell 3000
'VIX Index'          # VIX Volatility

# International Indices
'UKX Index'          # FTSE 100
'DAX Index'          # German DAX
'CAC Index'          # French CAC 40
'NKY Index'          # Nikkei 225
'HSCEI Index'        # Hang Seng China
'SENSEX Index'       # India Sensex
```

### ETF Ticker Examples

```python
# US Broad Market
'SPY US Equity'      # S&P 500
'QQQ US Equity'      # Nasdaq 100
'IWM US Equity'      # Russell 2000
'VTI US Equity'      # Total US Market

# Sector ETFs
'XLF US Equity'      # Financials
'XLE US Equity'      # Energy
'XLK US Equity'      # Technology
'XLV US Equity'      # Healthcare

# Bond ETFs
'AGG US Equity'      # Aggregate Bond
'TLT US Equity'      # 20+ Year Treasury
'HYG US Equity'      # High Yield Corp

# International ETFs
'EFA US Equity'      # MSCI EAFE
'EEM US Equity'      # Emerging Markets
'VEA US Equity'      # Developed Markets
```

### Fixed Income Tickers

```python
# US Treasuries
'USGG10YR Index'     # 10-Year Yield
'USGG2YR Index'      # 2-Year Yield
'USGG30YR Index'     # 30-Year Yield

# Corporate Spreads
'LF98TRUU Index'     # US Agg
'H0A0 Index'         # IG Corp
```

### Commodity Tickers

```python
# Energy
'CL1 Comdty'         # WTI Crude Oil (front month)
'CO1 Comdty'         # Brent Crude
'NG1 Comdty'         # Natural Gas
'HO1 Comdty'         # Heating Oil

# Metals
'GC1 Comdty'         # Gold
'SI1 Comdty'         # Silver
'HG1 Comdty'         # Copper
'PL1 Comdty'         # Platinum

# Agriculture
'C 1 Comdty'         # Corn
'S 1 Comdty'         # Soybeans
'W 1 Comdty'         # Wheat
'CT1 Comdty'         # Cotton
```

### Currency Pairs

```python
# Major Pairs
'EURUSD Curncy'
'GBPUSD Curncy'
'USDJPY Curncy'
'AUDUSD Curncy'
'USDCAD Curncy'
'USDCHF Curncy'

# Cross Pairs
'EURGBP Curncy'
'EURJPY Curncy'
'GBPJPY Curncy'
```

---

## Field Reference Guide

### Price Fields

```python
# Current/Last Price
'PX_LAST'            # Last price
'LAST_PRICE'         # Same as PX_LAST
'PX_BID'             # Bid price
'PX_ASK'             # Ask price
'PX_MID'             # Mid price

# OHLC
'PX_OPEN'            # Open price
'PX_HIGH'            # High price
'PX_LOW'             # Low price
'PX_CLOSE'           # Close price (for historical)

# Previous
'PX_LAST_EOD'        # Previous close
'PX_LAST_BID'        # Last bid
'PX_LAST_ASK'        # Last ask

# Special
'PX_OFFICIAL'        # Official close
'PX_SETTLE'          # Settlement price (futures)
```

### Volume Fields

```python
'PX_VOLUME'          # Volume
'VOLUME_AVG_5D'      # 5-day average volume
'VOLUME_AVG_10D'     # 10-day average volume
'VOLUME_AVG_20D'     # 20-day average volume
'VOLUME_AVG_30D'     # 30-day average volume
'TURNOVER'           # Dollar volume
'TRADE_SIZE_AVG_1D'  # Average trade size
```

### Market Cap & Shares

```python
'CUR_MKT_CAP'        # Current market cap
'EQY_SH_OUT'         # Shares outstanding
'EQY_FLOAT'          # Float shares
'EQY_FREE_FLOAT_PCT' # Free float %
'SHORT_INT'          # Short interest
'SHORT_INT_RATIO'    # Short interest ratio
```

### Valuation Ratios

```python
'PE_RATIO'           # P/E ratio
'PX_TO_BOOK_RATIO'   # P/B ratio
'PX_TO_SALES_RATIO'  # P/S ratio
'PX_TO_CASH_FLOW'    # P/CF ratio
'PX_TO_FREE_CASH_FLOW' # P/FCF ratio
'EV_TO_T12M_EBITDA'  # EV/EBITDA
'EV_TO_T12M_SALES'   # EV/Sales
'ENTERPRISE_VALUE'   # Enterprise value
```

### Earnings & Revenue

```python
'SALES_REV_TURN'     # Revenue (trailing 12 months)
'EBITDA'             # EBITDA
'EBIT'               # EBIT  
'NET_INCOME'         # Net income
'EARN_YLD'           # Earnings yield
'EARN_YLD_HIST'      # Historical earnings yield
'IS_EPS'             # EPS
'EPS_CUR_FY'         # Current FY EPS estimate
'TRAIL_12M_EPS'      # Trailing 12M EPS
```

### Profitability & Returns

```python
'RETURN_ON_ASSET'    # ROA
'RETURN_ON_EQUITY'   # ROE
'RETURN_COM_EQY'     # Return on common equity
'PROFIT_MARGIN'      # Profit margin
'GROSS_MARGIN'       # Gross margin
'EBITDA_MARGIN'      # EBITDA margin
'OPER_MARGIN'        # Operating margin
```

### Financial Health

```python
'TOT_DEBT_TO_TOT_EQY' # Debt/Equity
'DEBT_TO_EBITDA'     # Net Debt/EBITDA
'BS_TOT_ASSET'       # Total assets
'BS_TOT_LIAB2'       # Total liabilities
'CURRENT_RATIO'      # Current ratio
'QUICK_RATIO'        # Quick ratio
'CASH_RATIO'         # Cash ratio
'FREE_CASH_FLOW'     # Free cash flow
```

### Dividends

```python
'DIVIDEND_YIELD'     # Dividend yield
'DVD_HIST_ALL'       # All dividend history (bulk)
'DVD_PAYOUT_RATIO'   # Payout ratio
'DVD_INDICATED_GROSS' # Indicated annual dividend
'EQY_DVD_YLD_EST'    # Estimated div yield
```

### Growth Rates

```python
'SALES_GROWTH'       # Revenue growth
'EBITDA_GROWTH'      # EBITDA growth
'EBIT_GROWTH'        # EBIT growth
'NET_INCOME_GROWTH'  # Net income growth
'EPS_GROWTH'         # EPS growth
'BOOK_VAL_PER_SH_GROWTH' # Book value growth
```

### Volatility & Risk

```python
'VOLATILITY_10D'     # 10-day volatility
'VOLATILITY_20D'     # 20-day volatility
'VOLATILITY_30D'     # 30-day volatility
'VOLATILITY_60D'     # 60-day volatility
'VOLATILITY_90D'     # 90-day volatility
'VOLATILITY_180D'    # 180-day volatility
'VOLATILITY_260D'    # 260-day (1Y) volatility
'BETA_RAW_1YR'       # 1-year beta
'BETA_ADJ_RAW_1YR'   # Adjusted beta
```

### Performance & Returns

```python
'DAY_TO_DAY_TOT_RETURN_GROSS_DVDS' # Daily return
'TOT_RETURN_1D'      # 1-day total return
'TOT_RETURN_1WK'     # 1-week total return
'TOT_RETURN_1MO'     # 1-month total return
'TOT_RETURN_3MO'     # 3-month total return
'TOT_RETURN_6MO'     # 6-month total return
'TOT_RETURN_1YR'     # 1-year total return
'TOT_RETURN_YTD'     # YTD return
'SHARPE_RATIO_1YR'   # Sharpe ratio
```

### Technical Indicators

```python
'RSI_14D'            # 14-day RSI
'MACD'               # MACD
'MACD_SIGNAL'        # MACD signal line
'BB_UPPER_20D'       # Bollinger upper band
'BB_LOWER_20D'       # Bollinger lower band
'MA_5D'              # 5-day moving average
'MA_20D'             # 20-day moving average
'MA_50D'             # 50-day moving average
'MA_200D'            # 200-day moving average
```

### Analyst Data

```python
'TARGET_PRICE_12M'   # 12-month target price
'NUM_OF_ANALYSTS'    # Number of analysts
'AVERAGE_RATING'     # Average rating
'RATING_BUY'         # Number of buy ratings
'RATING_HOLD'        # Number of hold ratings
'RATING_SELL'        # Number of sell ratings
'BEST_ANALYST_RATING' # Best analyst rating
'WORST_ANALYST_RATING' # Worst analyst rating
```

### Reference Data

```python
'NAME'               # Company name
'LONG_COMP_NAME'     # Long company name
'COUNTRY'            # Country
'INDUSTRY_SECTOR'    # Sector
'INDUSTRY_GROUP'     # Industry group
'GICS_SECTOR_NAME'   # GICS sector
'GICS_INDUSTRY_NAME' # GICS industry
'SECURITY_TYP'       # Security type
'EXCHANGE_CODE'      # Exchange
'ID_ISIN'            # ISIN
'ID_CUSIP'           # CUSIP
'ID_SEDOL'           # SEDOL
'TICKER'             # Ticker symbol
```

### Estimates

```python
'BEST_EPS'           # Best EPS estimate
'BEST_SALES'         # Best sales estimate
'BEST_EBITDA'        # Best EBITDA estimate
'ESTIMATE_MEAN_EPS'  # Mean EPS estimate
'ESTIMATE_HIGH_EPS'  # High EPS estimate
'ESTIMATE_LOW_EPS'   # Low EPS estimate
'NUM_EST_EPS'        # Number of EPS estimates
```

---

## Advanced Usage Patterns

### Pattern 1: Building a Complete Dataset

```python
from xbbg import blp
import pandas as pd

# Define universe
tickers = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']

# Step 1: Get reference data
ref_fields = ['NAME', 'COUNTRY', 'GICS_SECTOR_NAME', 'CUR_MKT_CAP']
ref_df = blp.bdp(tickers, ref_fields)

# Step 2: Get valuation data
val_fields = ['PE_RATIO', 'PX_TO_BOOK_RATIO', 'DIVIDEND_YIELD']
val_df = blp.bdp(tickers, val_fields)

# Step 3: Get historical prices
price_df = blp.bdh(
    tickers,
    ['PX_LAST', 'PX_VOLUME'],
    start_date='2023-01-01'
)

# Step 4: Combine
master_df = pd.concat([ref_df, val_df], axis=1)
print(master_df)
```

### Pattern 2: Time Series Analysis

```python
# Get multi-year data
df = blp.bdh(
    'SPY US Equity',
    ['PX_LAST', 'VOLATILITY_30D'],
    start_date='2010-01-01',
    end_date='2023-12-31'
)

# Calculate returns
df['returns'] = df['PX_LAST'].pct_change()
df['log_returns'] = np.log(df['PX_LAST'] / df['PX_LAST'].shift(1))

# Rolling metrics
df['MA_50'] = df['PX_LAST'].rolling(50).mean()
df['MA_200'] = df['PX_LAST'].rolling(200).mean()
df['rolling_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
```

### Pattern 3: Cross-Sectional Analysis

```python
# Get S&P 500 members
sp500 = blp.bds('SPX Index', 'INDX_MEMBERS')['Member Ticker'].tolist()

# Get screening data
fields = [
    'PX_LAST',
    'CUR_MKT_CAP',
    'PE_RATIO',
    'RETURN_ON_EQUITY',
    'SALES_GROWTH',
    'VOLATILITY_90D'
]

data = blp.bdp(sp500, fields)

# Calculate percentile ranks
for field in fields[1:]:  # Skip PX_LAST
    data[f'{field}_rank'] = data[field].rank(pct=True)

# Create composite score
data['composite_score'] = (
    data['PE_RATIO_rank'] * -1 +  # Lower is better
    data['RETURN_ON_EQUITY_rank'] +
    data['SALES_GROWTH_rank'] +
    data['VOLATILITY_90D_rank'] * -1  # Lower is better
) / 4

# Top 20 stocks
top_20 = data.nlargest(20, 'composite_score')
```

### Pattern 4: Fundamental Data Over Time

```python
# Quarterly financials over time
ticker = 'AAPL US Equity'
fields = ['SALES_REV_TURN', 'EBITDA', 'NET_INCOME']

# Get quarterly data for 5 years
df = blp.bdh(
    ticker,
    fields,
    start_date='2019-01-01',
    Per='Q'  # Quarterly
)

# Calculate margins
df['EBITDA_Margin'] = df['EBITDA'] / df['SALES_REV_TURN']
df['Net_Margin'] = df['NET_INCOME'] / df['SALES_REV_TURN']

# YoY growth
df['Revenue_YoY'] = df['SALES_REV_TURN'].pct_change(4)
df['EBITDA_YoY'] = df['EBITDA'].pct_change(4)
```

### Pattern 5: Multi-Asset Portfolio

```python
# Define multi-asset universe
portfolio = {
    'US_Stocks': ['SPY US Equity', 'QQQ US Equity', 'IWM US Equity'],
    'Intl_Stocks': ['EFA US Equity', 'EEM US Equity'],
    'Bonds': ['AGG US Equity', 'TLT US Equity'],
    'Commodities': ['GLD US Equity', 'USO US Equity'],
    'Real_Estate': ['VNQ US Equity']
}

# Flatten
all_tickers = [ticker for tickers in portfolio.values() for ticker in tickers]

# Get prices
prices = blp.bdh(
    all_tickers,
    'PX_LAST',
    start_date='2020-01-01'
)

# Calculate returns
returns = prices.pct_change()

# Correlation matrix
corr_matrix = returns.corr()

# Optimal weights (simple equal risk contribution)
import scipy.optimize as sco

def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(len(all_tickers)))
initial_weights = np.array([1/len(all_tickers)] * len(all_tickers))

result = sco.minimize(
    portfolio_volatility,
    initial_weights,
    args=(returns,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
```

### Pattern 6: Earnings Analysis

```python
ticker = 'AAPL US Equity'

# Get earnings history
earnings = blp.bds(ticker, 'EARN_ANN')

# Get estimates
estimates = blp.bds(ticker, 'BEST_EPS_GAAP')

# Merge and analyze surprise %
earnings['surprise_pct'] = (
    (earnings['EPS Actual'] - earnings['EPS Estimate']) / 
    earnings['EPS Estimate'] * 100
)

# Get price reaction
announcement_dates = earnings['Announcement Date'].tolist()

reactions = []
for date in announcement_dates:
    try:
        # Get price 1 day before and after
        df = blp.bdh(
            ticker,
            'PX_LAST',
            start_date=(pd.to_datetime(date) - pd.Timedelta(days=5)).strftime('%Y%m%d'),
            end_date=(pd.to_datetime(date) + pd.Timedelta(days=5)).strftime('%Y%m%d')
        )
        
        reaction = (df.iloc[-1] / df.iloc[0] - 1) * 100
        reactions.append(reaction)
    except:
        reactions.append(None)

earnings['price_reaction'] = reactions
```

### Pattern 7: Factor Analysis

```python
# Define factor exposures
def get_factor_data(tickers):
    fields = {
        'Value': ['PE_RATIO', 'PX_TO_BOOK_RATIO', 'DIVIDEND_YIELD'],
        'Quality': ['RETURN_ON_EQUITY', 'EBITDA_MARGIN', 'DEBT_TO_EBITDA'],
        'Momentum': ['TOT_RETURN_3MO', 'TOT_RETURN_6MO', 'TOT_RETURN_1YR'],
        'Size': ['CUR_MKT_CAP'],
        'Volatility': ['VOLATILITY_90D', 'BETA_RAW_1YR']
    }
    
    all_fields = [f for factor_fields in fields.values() for f in factor_fields]
    
    data = blp.bdp(tickers, all_fields)
    
    # Calculate factor scores
    for factor, factor_fields in fields.items():
        # Z-score normalize
        for field in factor_fields:
            data[f'{field}_z'] = (data[field] - data[field].mean()) / data[field].std()
        
        # Composite factor score
        z_fields = [f'{f}_z' for f in factor_fields]
        
        # Handle direction (low PE is good, high ROE is good, etc.)
        if factor == 'Value':
            data[f'{factor}_score'] = -data[[f'{factor_fields[0]}_z', f'{factor_fields[1]}_z']].mean(axis=1) + data[f'{factor_fields[2]}_z']
        elif factor == 'Volatility':
            data[f'{factor}_score'] = -data[z_fields].mean(axis=1)  # Low vol is good
        else:
            data[f'{factor}_score'] = data[z_fields].mean(axis=1)
    
    return data

# Use it
sp500 = blp.bds('SPX Index', 'INDX_MEMBERS')['Member Ticker'].tolist()
factor_data = get_factor_data(sp500)
```

---

## Error Handling & Troubleshooting

### Common Errors

#### 1. Connection Error

```python
try:
    df = blp.bdp('AAPL US Equity', 'PX_LAST')
except Exception as e:
    print(f"Connection error: {e}")
    print("Make sure Bloomberg Terminal is running and you're logged in")
```

#### 2. Invalid Ticker

```python
# Wrong ticker format
try:
    df = blp.bdp('AAPL', 'PX_LAST')  # Missing "US Equity"
except:
    print("Invalid ticker format")
    
# Correct format
df = blp.bdp('AAPL US Equity', 'PX_LAST')
```

#### 3. Invalid Field

```python
# Check if field is valid
try:
    df = blp.bdp('AAPL US Equity', 'INVALID_FIELD')
    if df.empty or df.isna().all().all():
        print("Field returned no data - may be invalid")
except:
    print("Field not recognized")
```

#### 4. No Data Available

```python
# Handle missing data gracefully
df = blp.bdh('AAPL US Equity', 'PX_LAST', '2023-01-01')

if df.empty:
    print("No data returned - check date range and ticker")
elif df.isna().all().all():
    print("Data returned but all NaN - field may not be available for this security")
```

#### 5. Date Range Issues

```python
from datetime import datetime, timedelta

# Ensure date is in the past
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=365)

df = blp.bdh(
    'SPY US Equity',
    'PX_LAST',
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)
```

### Robust Data Retrieval Pattern

```python
def safe_bdp(tickers, fields, max_retries=3):
    """
    Safely retrieve reference data with retries
    """
    import time
    
    for attempt in range(max_retries):
        try:
            df = blp.bdp(tickers, fields)
            
            if df.empty:
                print(f"No data returned on attempt {attempt + 1}")
                time.sleep(2)
                continue
            
            return df
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    
    return None

def safe_bdh(tickers, fields, start_date, end_date=None, max_retries=3):
    """
    Safely retrieve historical data with retries
    """
    import time
    
    for attempt in range(max_retries):
        try:
            df = blp.bdh(
                tickers=tickers,
                flds=fields,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                print(f"No data returned on attempt {attempt + 1}")
                time.sleep(2)
                continue
            
            return df
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    
    return None
```

### Batch Processing with Error Handling

```python
def batch_bdp(tickers, fields, batch_size=50):
    """
    Process large lists in batches
    """
    results = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} tickers)")
        
        try:
            df = blp.bdp(batch, fields)
            results.append(df)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    if results:
        return pd.concat(results)
    else:
        return pd.DataFrame()

# Usage
sp500 = blp.bds('SPX Index', 'INDX_MEMBERS')['Member Ticker'].tolist()
data = batch_bdp(sp500, ['PX_LAST', 'CUR_MKT_CAP'], batch_size=100)
```

---

## Performance Optimization

### 1. Batch Requests

```python
# SLOW - Multiple individual requests
prices = {}
for ticker in tickers:
    prices[ticker] = blp.bdp(ticker, 'PX_LAST')

# FAST - Single batch request
prices = blp.bdp(tickers, 'PX_LAST')
```

### 2. Request Only Needed Fields

```python
# Don't request everything
all_fields = ['PX_LAST', 'PX_OPEN', 'PX_HIGH', ...]  # 50+ fields

# Request only what you need
needed_fields = ['PX_LAST', 'CUR_MKT_CAP']
df = blp.bdp(tickers, needed_fields)
```

### 3. Use Appropriate Periodicity

```python
# For long-term analysis, use monthly data
df = blp.bdh('SPY US Equity', 'PX_LAST', '2000-01-01', Per='M')

# Not daily if you don't need it
df = blp.bdh('SPY US Equity', 'PX_LAST', '2000-01-01', Per='D')  # Much slower
```

### 4. Cache Results

```python
import pickle
from pathlib import Path

def cached_bdh(tickers, fields, start_date, cache_dir='cache'):
    """
    Cache historical data to avoid repeated Bloomberg calls
    """
    Path(cache_dir).mkdir(exist_ok=True)
    
    cache_key = f"{tickers}_{fields}_{start_date}".replace(' ', '_')
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    
    # Check cache
    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Fetch from Bloomberg
    print(f"Fetching from Bloomberg...")
    df = blp.bdh(tickers, fields, start_date)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    
    return df
```

### 5. Parallel Processing (Use with Caution)

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def get_ticker_data(ticker):
    """Get data for a single ticker"""
    try:
        return blp.bdp(ticker, ['PX_LAST', 'CUR_MKT_CAP'])
    except:
        return None

# Split large universe into chunks
sp500 = blp.bds('SPX Index', 'INDX_MEMBERS')['Member Ticker'].tolist()
chunks = np.array_split(sp500, 10)

results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    for chunk in chunks:
        future = executor.submit(blp.bdp, chunk.tolist(), ['PX_LAST', 'CUR_MKT_CAP'])
        results.append(future.result())

combined = pd.concat(results)
```

---

## Real-World Use Cases

### Use Case 1: Portfolio Monitoring

```python
def monitor_portfolio(holdings, benchmark='SPX Index'):
    """
    Monitor portfolio performance vs benchmark
    """
    # Get current prices and returns
    fields = ['PX_LAST', 'TOT_RETURN_1D', 'TOT_RETURN_1MO', 
              'TOT_RETURN_YTD', 'CUR_MKT_CAP']
    
    portfolio_data = blp.bdp(holdings, fields)
    benchmark_data = blp.bdp(benchmark, fields)
    
    # Calculate portfolio metrics
    print("=== Portfolio Performance ===")
    print(f"1-Day Return: {portfolio_data['TOT_RETURN_1D'].mean():.2f}%")
    print(f"1-Month Return: {portfolio_data['TOT_RETURN_1MO'].mean():.2f}%")
    print(f"YTD Return: {portfolio_data['TOT_RETURN_YTD'].mean():.2f}%")
    
    print("\n=== Benchmark Performance ===")
    print(f"1-Day Return: {benchmark_data['TOT_RETURN_1D'].iloc[0]:.2f}%")
    print(f"1-Month Return: {benchmark_data['TOT_RETURN_1MO'].iloc[0]:.2f}%")
    print(f"YTD Return: {benchmark_data['TOT_RETURN_YTD'].iloc[0]:.2f}%")
    
    return portfolio_data

# Usage
my_holdings = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']
monitor_portfolio(my_holdings)
```

### Use Case 2: Value Screening

```python
def value_screen(universe, top_n=20):
    """
    Screen for undervalued stocks
    """
    # Get valuation metrics
    fields = [
        'PE_RATIO',
        'PX_TO_BOOK_RATIO',
        'DIVIDEND_YIELD',
        'FREE_CASH_FLOW_YIELD',
        'EV_TO_T12M_EBITDA',
        'CUR_MKT_CAP'
    ]
    
    data = blp.bdp(universe, fields)
    
    # Remove stocks with missing data
    data = data.dropna()
    
    # Calculate composite value score
    data['PE_rank'] = data['PE_RATIO'].rank()
    data['PB_rank'] = data['PX_TO_BOOK_RATIO'].rank()
    data['DY_rank'] = data['DIVIDEND_YIELD'].rank(ascending=False)
    data['FCF_rank'] = data['FREE_CASH_FLOW_YIELD'].rank(ascending=False)
    data['EV_EBITDA_rank'] = data['EV_TO_T12M_EBITDA'].rank()
    
    # Composite (lower is better for value)
    data['value_score'] = (
        data['PE_rank'] +
        data['PB_rank'] +
        data['DY_rank'] +
        data['FCF_rank'] +
        data['EV_EBITDA_rank']
    ) / 5
    
    # Return top N
    return data.nsmallest(top_n, 'value_score')

# Usage
sp500 = blp.bds('SPX Index', 'INDX_MEMBERS')['Member Ticker'].tolist()
value_stocks = value_screen(sp500, top_n=20)
print(value_stocks)
```

### Use Case 3: Momentum Strategy

```python
def momentum_strategy(universe, lookback='6MO', top_n=20):
    """
    Identify momentum stocks
    """
    # Get momentum metrics
    fields = [
        f'TOT_RETURN_{lookback}',
        'VOLATILITY_90D',
        'CUR_MKT_CAP',
        'VOLUME_AVG_20D'
    ]
    
    data = blp.bdp(universe, fields)
    data = data.dropna()
    
    # Filter for liquidity (>$1B market cap, >$10M daily volume)
    data = data[
        (data['CUR_MKT_CAP'] > 1000) &
        (data['VOLUME_AVG_20D'] > 1000000)
    ]
    
    # Calculate risk-adjusted momentum
    data['risk_adj_momentum'] = (
        data[f'TOT_RETURN_{lookback}'] / data['VOLATILITY_90D']
    )
    
    # Get top N
    top_momentum = data.nlargest(top_n, 'risk_adj_momentum')
    
    return top_momentum

# Usage
sp500 = blp.bds('SPX Index', 'INDX_MEMBERS')['Member Ticker'].tolist()
momentum_stocks = momentum_strategy(sp500, lookback='6MO', top_n=20)
```

### Use Case 4: Risk Management

```python
def portfolio_risk_analysis(holdings, weights=None):
    """
    Analyze portfolio risk
    """
    if weights is None:
        weights = np.array([1/len(holdings)] * len(holdings))
    
    # Get historical prices
    prices = blp.bdh(
        holdings,
        'PX_LAST',
        start_date='2020-01-01'
    )
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Portfolio returns
    port_returns = (returns * weights).sum(axis=1)
    
    # Risk metrics
    vol = port_returns.std() * np.sqrt(252)
    sharpe = (port_returns.mean() * 252) / vol
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(port_returns, 5)
    
    # Conditional VaR (CVaR)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    
    # Maximum drawdown
    cum_returns = (1 + port_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    print("=== Portfolio Risk Metrics ===")
    print(f"Annualized Volatility: {vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"95% VaR (daily): {var_95*100:.2f}%")
    print(f"95% CVaR (daily): {cvar_95*100:.2f}%")
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    
    # Correlation matrix
    corr = returns.corr()
    print(f"\nAverage Correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].mean():.2f}")
    
    return {
        'volatility': vol,
        'sharpe': sharpe,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown': max_dd,
        'correlation': corr
    }

# Usage
my_portfolio = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity', 'AMZN US Equity']
risk_metrics = portfolio_risk_analysis(my_portfolio)
```

### Use Case 5: Earnings Calendar

```python
def earnings_calendar(tickers, days_ahead=30):
    """
    Get upcoming earnings announcements
    """
    from datetime import datetime, timedelta
    
    # Get next earnings dates
    fields = ['EARN_ANN_DT_TIME_HIST_1', 'EPS_CUR_FY', 'NAME']
    data = blp.bdp(tickers, fields)
    
    # Get earnings history to estimate next date
    earnings_list = []
    for ticker in tickers:
        try:
            earn_hist = blp.bds(ticker, 'EARN_ANN')
            if not earn_hist.empty:
                last_date = earn_hist['Announcement Date'].max()
                # Estimate next earnings (typically quarterly, ~90 days)
                next_date = last_date + timedelta(days=90)
                
                if next_date <= datetime.now() + timedelta(days=days_ahead):
                    earnings_list.append({
                        'Ticker': ticker,
                        'Next_Earnings_Est': next_date,
                        'EPS_Estimate': data.loc[ticker, 'EPS_CUR_FY']
                    })
        except:
            continue
    
    return pd.DataFrame(earnings_list).sort_values('Next_Earnings_Est')

# Usage
watch_list = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']
upcoming_earnings = earnings_calendar(watch_list)
print(upcoming_earnings)
```

### Use Case 6: Sector Rotation

```python
def sector_rotation_analysis():
    """
    Analyze sector performance for rotation strategy
    """
    # Sector ETFs
    sectors = {
        'Technology': 'XLK US Equity',
        'Financials': 'XLF US Equity',
        'Healthcare': 'XLV US Equity',
        'Energy': 'XLE US Equity',
        'Consumer Discretionary': 'XLY US Equity',
        'Consumer Staples': 'XLP US Equity',
        'Industrials': 'XLI US Equity',
        'Materials': 'XLB US Equity',
        'Real Estate': 'XLRE US Equity',
        'Utilities': 'XLU US Equity',
        'Communication Services': 'XLC US Equity'
    }
    
    tickers = list(sectors.values())
    
    # Get performance metrics
    fields = [
        'TOT_RETURN_1MO',
        'TOT_RETURN_3MO',
        'TOT_RETURN_6MO',
        'VOLATILITY_90D',
        'SHARPE_RATIO_1YR'
    ]
    
    data = blp.bdp(tickers, fields)
    data['Sector'] = [k for k in sectors.keys()]
    data = data.set_index('Sector')
    
    # Calculate momentum score
    data['momentum_score'] = (
        data['TOT_RETURN_1MO'] * 0.3 +
        data['TOT_RETURN_3MO'] * 0.4 +
        data['TOT_RETURN_6MO'] * 0.3
    )
    
    # Rank sectors
    data['rank'] = data['momentum_score'].rank(ascending=False)
    
    print("=== Sector Rankings ===")
    print(data.sort_values('rank')[['momentum_score', 'SHARPE_RATIO_1YR', 'rank']])
    
    return data.sort_values('rank')

# Usage
sector_analysis = sector_rotation_analysis()
```

---

## Best Practices

### 1. Always Use Proper Ticker Format

```python
# CORRECT
'AAPL US Equity'
'SPX Index'
'EURUSD Curncy'

# WRONG
'AAPL'
'SPX'
'EUR/USD'
```

### 2. Handle Missing Data

```python
# Check for missing data
df = blp.bdp(tickers, fields)

# Remove rows with any missing data
df_clean = df.dropna()

# Or fill missing data
df_filled = df.fillna(method='ffill')

# Or remove columns with too much missing data
threshold = 0.5
df_clean = df.loc[:, df.isnull().mean() < threshold]
```

### 3. Use Appropriate Date Formats

```python
# Bloomberg accepts multiple formats
'2023-01-01'
'20230101'
'01/01/2023'

# Python datetime objects work too
from datetime import datetime
dt = datetime(2023, 1, 1)
df = blp.bdh('SPY US Equity', 'PX_LAST', start_date=dt)
```

### 4. Validate Data Before Analysis

```python
def validate_data(df):
    """
    Validate Bloomberg data
    """
    issues = []
    
    # Check for empty
    if df.empty:
        issues.append("DataFrame is empty")
    
    # Check for all NaN
    if df.isna().all().all():
        issues.append("All values are NaN")
    
    # Check for duplicates
    if df.index.duplicated().any():
        issues.append("Duplicate index values found")
    
    # Check for outliers (example: 3 standard deviations)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        outliers = np.abs((df[col] - df[col].mean()) / df[col].std()) > 3
        if outliers.sum() > 0:
            issues.append(f"Column '{col}' has {outliers.sum()} outliers")
    
    if issues:
        print("Data validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Data validation passed")
        return True

# Usage
df = blp.bdh('SPY US Equity', 'PX_LAST', '2023-01-01')
validate_data(df)
```

### 5. Document Your Field Choices

```python
# Good practice: Document why you chose specific fields
VALUATION_FIELDS = {
    'PE_RATIO': 'Price-to-Earnings ratio',
    'PX_TO_BOOK_RATIO': 'Price-to-Book ratio',
    'DIVIDEND_YIELD': 'Current dividend yield',
    'EV_TO_T12M_EBITDA': 'Enterprise Value to EBITDA'
}

# Use documented fields
df = blp.bdp(tickers, list(VALUATION_FIELDS.keys()))
```

### 6. Set Up Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use in your code
def get_data_with_logging(tickers, fields):
    logger.info(f"Requesting data for {len(tickers)} tickers, {len(fields)} fields")
    
    try:
        df = blp.bdp(tickers, fields)
        logger.info(f"Successfully retrieved {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        raise
```

### 7. Create Reusable Functions

```python
def get_stock_fundamentals(ticker):
    """
    Get comprehensive fundamental data for a stock
    """
    fields = {
        'identification': ['NAME', 'COUNTRY', 'GICS_SECTOR_NAME'],
        'valuation': ['PE_RATIO', 'PX_TO_BOOK_RATIO', 'DIVIDEND_YIELD'],
        'profitability': ['RETURN_ON_EQUITY', 'RETURN_ON_ASSET', 'PROFIT_MARGIN'],
        'financial_health': ['CURRENT_RATIO', 'DEBT_TO_EBITDA', 'FREE_CASH_FLOW'],
        'growth': ['SALES_GROWTH', 'EPS_GROWTH', 'EBITDA_GROWTH']
    }
    
    all_fields = [f for category in fields.values() for f in category]
    
    data = blp.bdp(ticker, all_fields)
    
    return data

# Usage
aapl_fundamentals = get_stock_fundamentals('AAPL US Equity')
```

### 8. Version Control Your Queries

```python
# Keep a version history of your important queries
QUERIES_V1 = {
    'value_screen': {
        'fields': ['PE_RATIO', 'PX_TO_BOOK_RATIO', 'DIVIDEND_YIELD'],
        'filters': {'PE_RATIO': '<', 15, 'DIVIDEND_YIELD': '>', 2}
    }
}

QUERIES_V2 = {
    'value_screen': {
        'fields': ['PE_RATIO', 'PX_TO_BOOK_RATIO', 'DIVIDEND_YIELD', 'FREE_CASH_FLOW_YIELD'],
        'filters': {'PE_RATIO': '<', 15, 'DIVIDEND_YIELD': '>', 2, 'FREE_CASH_FLOW_YIELD': '>', 5}
    }
}

# Use current version
CURRENT_QUERIES = QUERIES_V2
```

---

## Summary

This guide provides comprehensive coverage of the XBBG library for Bloomberg data access in Python:

1. **Core Functions**: bdh(), bdp(), bds(), bdib(), bdtick(), beqs()
2. **Ticker Formats**: Proper syntax for equities, indices, bonds, commodities, currencies
3. **Field Reference**: 200+ Bloomberg fields organized by category
4. **Advanced Patterns**: Real-world implementations for common financial analysis tasks
5. **Error Handling**: Robust data retrieval with retries and validation
6. **Performance**: Optimization techniques for large-scale data requests
7. **Best Practices**: Professional patterns for production code

### Quick Start Template

```python
from xbbg import blp
import pandas as pd
import numpy as np

# 1. Get reference data
tickers = ['AAPL US Equity', 'MSFT US Equity']
ref_data = blp.bdp(tickers, ['PX_LAST', 'CUR_MKT_CAP'])

# 2. Get historical data
hist_data = blp.bdh(tickers, 'PX_LAST', start_date='2023-01-01')

# 3. Get bulk data
index_members = blp.bds('SPX Index', 'INDX_MEMBERS')

# 4. Your analysis here
print(ref_data)
print(hist_data.head())
print(index_members.head())
```

### Additional Resources

- **Bloomberg Terminal**: Type `FLDS<GO>` to search for fields
- **Bloomberg API Help**: `WAPI<GO>` for API documentation
- **XBBG Documentation**: https://github.com/alpha-xone/xbbg
- **Bloomberg Support**: Press F1 twice in Bloomberg Terminal

---

**Note**: This guide assumes you have a valid Bloomberg Terminal license and the terminal is running and logged in. All field names and ticker formats follow Bloomberg conventions and may be updated by Bloomberg over time.

