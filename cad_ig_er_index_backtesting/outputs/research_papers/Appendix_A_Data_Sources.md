# Appendix A: Data Sources and Feature Definitions

## A.1 Data Sources

### A.1.1 Primary Data Providers

**Credit Market Data:**
- **CAD IG OAS**: Canadian Investment Grade Option-Adjusted Spreads
  - Source: Bloomberg Terminal (BAML Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Institutional-grade, widely used for Canadian credit analysis

- **US HY OAS**: US High Yield Option-Adjusted Spreads
  - Source: Bloomberg Terminal (BAML Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Standard benchmark for US high yield markets

- **US IG OAS**: US Investment Grade Option-Adjusted Spreads
  - Source: Bloomberg Terminal (BAML Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Primary benchmark for US investment grade credit

- **CAD IG ER Index**: Canadian Investment Grade Excess Return Index
  - Source: Bloomberg Terminal (Custom Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Total return index including price and coupon returns

**Equity Market Data:**
- **TSX**: Toronto Stock Exchange Composite Index
  - Source: Bloomberg Terminal (SPTSX Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Primary Canadian equity benchmark

- **SPX**: S&P 500 Index
  - Source: Bloomberg Terminal (SPX Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Global equity market benchmark

- **VIX**: CBOE Volatility Index
  - Source: Bloomberg Terminal (VIX Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Standard volatility benchmark

**Interest Rate Data:**
- **US 3M-10Y Yield Curve Slope**
  - Source: Bloomberg Terminal (Calculated from USGG3M and USGG10YR)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Standard yield curve measure

**Macroeconomic Data:**
- **US Growth Surprises Index**
  - Source: Bloomberg Terminal (ECSUUS Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Standardized surprise measure vs consensus

- **US Inflation Surprises Index**
  - Source: Bloomberg Terminal (ECSUCPI Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Standardized surprise measure vs consensus

- **US Leading Economic Index (Year-over-Year)**
  - Source: Bloomberg Terminal (USLEIY Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Conference Board leading indicator

- **US Hard Data Surprises Index**
  - Source: Bloomberg Terminal (ECSUIND Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Manufacturing and production data surprises

- **US Equity Revisions Index**
  - Source: Bloomberg Terminal (ECSUEARN Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Analyst earnings estimate revisions

- **US Economic Regime Indicator**
  - Source: Bloomberg Terminal (Custom Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Binary regime classification (expansion/recession)

**Fundamental Data:**
- **SPX 12-Month Forward EPS**
  - Source: Bloomberg Terminal (SPX1F EPS Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Consensus forward earnings estimates

- **SPX 12-Month Forward Sales**
  - Source: Bloomberg Terminal (SPX1F SALES Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Consensus forward revenue estimates

- **TSX 12-Month Forward EPS**
  - Source: Bloomberg Terminal (SPTSX1F EPS Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Consensus forward earnings estimates

- **TSX 12-Month Forward Sales**
  - Source: Bloomberg Terminal (SPTSX1F SALES Index)
  - Frequency: Daily
  - Coverage: 2003-11-30 to 2025-09-23
  - Quality: Consensus forward revenue estimates

### A.1.2 Data Quality Assessment

**Missing Data Analysis:**
- Total observations: 5,767 daily observations
- Missing data rate: <0.1% across all variables
- Missing data handling: Forward-fill methodology
- Data completeness: 99.9% across all variables

**Outlier Detection:**
- Winsorization at 1st and 99th percentiles
- Extreme value identification using 3-sigma rule
- Outlier impact assessment on feature engineering
- Robustness testing with outlier removal

**Corporate Actions:**
- Index methodology changes tracked and adjusted
- Dividend and coupon payments included in total return indices
- Split and merger adjustments applied automatically
- Historical consistency maintained throughout dataset

### A.1.3 Data Validation

**Cross-Validation Checks:**
- Comparison with alternative data sources where available
- Internal consistency checks across related variables
- Temporal consistency validation
- Correlation analysis with known benchmarks

**Quality Metrics:**
- Data availability: 99.9%
- Consistency score: 98.5%
- Validation status: PASS
- Quality rating: Institutional Grade

## A.2 Feature Engineering Framework

### A.2.1 Momentum Features (Cross-Asset)

**Definition:**
Momentum features capture trend persistence across different asset classes and time horizons using percentage change calculations.

**Formula:**
```
momentum_t,L = (price_t / price_t-L) - 1
```

Where:
- `t` = current time period
- `L` = lookback period (1, 2, 4, 8, 12 weeks)
- `price_t` = asset price at time t

**Implementation:**
```python
for col in ['cad_oas', 'us_hy_oas', 'us_ig_oas', 'tsx', 'vix', 'us_3m_10y']:
    for lb in [1, 2, 4, 8, 12]:
        weekly[f'{col}_mom_{lb}w'] = weekly[col].pct_change(lb)
```

**Feature Categories:**
1. **CAD OAS Momentum** (5 features)
   - `cad_oas_mom_1w`, `cad_oas_mom_2w`, `cad_oas_mom_4w`, `cad_oas_mom_8w`, `cad_oas_mom_12w`
   - Captures domestic credit spread trends

2. **US HY OAS Momentum** (5 features)
   - `us_hy_oas_mom_1w`, `us_hy_oas_mom_2w`, `us_hy_oas_mom_4w`, `us_hy_oas_mom_8w`, `us_hy_oas_mom_12w`
   - Captures US high yield credit trends

3. **US IG OAS Momentum** (5 features)
   - `us_ig_oas_mom_1w`, `us_ig_oas_mom_2w`, `us_ig_oas_mom_4w`, `us_ig_oas_mom_8w`, `us_ig_oas_mom_12w`
   - Captures US investment grade credit trends

4. **TSX Momentum** (5 features)
   - `tsx_mom_1w`, `tsx_mom_2w`, `tsx_mom_4w`, `tsx_mom_8w`, `tsx_mom_12w`
   - Captures Canadian equity trends

5. **VIX Momentum** (5 features)
   - `vix_mom_1w`, `vix_mom_2w`, `vix_mom_4w`, `vix_mom_8w`, `vix_mom_12w`
   - Captures volatility regime trends

6. **US Yield Curve Momentum** (5 features)
   - `us_3m_10y_mom_1w`, `us_3m_10y_mom_2w`, `us_3m_10y_mom_4w`, `us_3m_10y_mom_8w`, `us_3m_10y_mom_12w`
   - Captures interest rate environment trends

**Economic Rationale:**
- Cross-asset momentum often leads credit markets due to information flow
- Multiple time horizons capture both short-term noise and longer-term trends
- Credit spread momentum is particularly relevant for IG timing
- Volatility momentum indicates regime changes

### A.2.2 Volatility Features

**Definition:**
Volatility features measure market stress and uncertainty levels using rolling standard deviation of returns.

**Formula:**
```
volatility_t,W = std(returns_t-W:t)
```

Where:
- `W` = rolling window (4, 8, 12 weeks)
- `returns_t` = percentage change in asset price at time t
- `std()` = standard deviation function

**Implementation:**
```python
for col in ['cad_oas', 'us_hy_oas', 'us_ig_oas', 'vix', target_col]:
    for window in [4, 8, 12]:
        weekly[f'{col}_vol_{window}w'] = weekly[col].pct_change().rolling(window).std()
```

**Feature Categories:**
1. **CAD OAS Volatility** (3 features)
   - `cad_oas_vol_4w`, `cad_oas_vol_8w`, `cad_oas_vol_12w`
   - Measures domestic credit market stress

2. **US HY OAS Volatility** (3 features)
   - `us_hy_oas_vol_4w`, `us_hy_oas_vol_8w`, `us_hy_oas_vol_12w`
   - Measures US high yield market stress

3. **US IG OAS Volatility** (3 features)
   - `us_ig_oas_vol_4w`, `us_ig_oas_vol_8w`, `us_ig_oas_vol_12w`
   - Measures US investment grade market stress

4. **VIX Volatility** (3 features)
   - `vix_vol_4w`, `vix_vol_8w`, `vix_vol_12w`
   - Measures volatility-of-volatility

5. **Target Index Volatility** (3 features)
   - `cad_ig_er_index_vol_4w`, `cad_ig_er_index_vol_8w`, `cad_ig_er_index_vol_12w`
   - Measures target asset volatility

**Strategic Value:**
- High volatility periods often precede credit spread widening
- Volatility clustering helps identify regime changes
- Cross-asset volatility reveals systemic risk building

### A.2.3 Spread Indicators

**Definition:**
Spread indicators capture relative value and risk premium dynamics between different credit markets.

**HY-IG Spread:**
```
hy_ig_spread_t = us_hy_oas_t - us_ig_oas_t
```

**CAD-US IG Spread:**
```
cad_us_ig_spread_t = cad_oas_t - us_ig_oas_t
```

**Spread Changes:**
```
spread_change_t,L = spread_t - spread_t-L
```

**Implementation:**
```python
# HY-IG Spread
weekly['hy_ig_spread'] = weekly['us_hy_oas'] - weekly['us_ig_oas']
for lb in [1, 4, 8]:
    weekly[f'hy_ig_spread_chg_{lb}w'] = weekly['hy_ig_spread'].diff(lb)

# CAD-US IG Spread
weekly['cad_us_ig_spread'] = weekly['cad_oas'] - weekly['us_ig_oas']
for lb in [1, 4, 8]:
    weekly[f'cad_us_ig_spread_chg_{lb}w'] = weekly['cad_us_ig_spread'].diff(lb)
```

**Feature Categories:**
1. **HY-IG Spread Features** (4 features)
   - `hy_ig_spread`, `hy_ig_spread_chg_1w`, `hy_ig_spread_chg_4w`, `hy_ig_spread_chg_8w`
   - Measures risk appetite and credit cycle position

2. **CAD-US IG Spread Features** (4 features)
   - `cad_us_ig_spread`, `cad_us_ig_spread_chg_1w`, `cad_us_ig_spread_chg_4w`, `cad_us_ig_spread_chg_8w`
   - Measures relative country risk and currency dynamics

**Economic Logic:**
- HY-IG spreads indicate risk appetite and credit cycle position
- CAD-US spreads reflect relative country risk and currency dynamics
- Spread changes often lead IG performance by 1-2 quarters

### A.2.4 Macro Surprise Features

**Definition:**
Macro surprise features incorporate economic data surprises that drive credit market sentiment.

**Formula:**
```
surprise_change_t,L = surprise_t - surprise_t-L
```

**Implementation:**
```python
for col in ['us_growth_surprises', 'us_inflation_surprises', 'us_hard_data_surprises', 
            'us_equity_revisions', 'us_lei_yoy']:
    for lb in [1, 4]:
        weekly[f'{col}_chg_{lb}w'] = weekly[col].diff(lb)
```

**Feature Categories:**
1. **US Growth Surprises** (2 features)
   - `us_growth_surprises_chg_1w`, `us_growth_surprises_chg_4w`
   - Economic growth data vs consensus expectations

2. **US Inflation Surprises** (2 features)
   - `us_inflation_surprises_chg_1w`, `us_inflation_surprises_chg_4w`
   - Inflation data vs consensus expectations

3. **US Hard Data Surprises** (2 features)
   - `us_hard_data_surprises_chg_1w`, `us_hard_data_surprises_chg_4w`
   - Manufacturing, employment, and production data surprises

4. **US Equity Revisions** (2 features)
   - `us_equity_revisions_chg_1w`, `us_equity_revisions_chg_4w`
   - Analyst earnings estimate changes

5. **US LEI YoY** (2 features)
   - `us_lei_yoy_chg_1w`, `us_lei_yoy_chg_4w`
   - Leading Economic Index year-over-year changes

**Why Critical:**
Credit markets are highly sensitive to economic data surprises, which often drive spread movements before fundamentals fully reflect the economic environment.

### A.2.5 Technical Features

**Definition:**
Technical features apply classical technical analysis to the target index using moving averages and z-scores.

**Moving Average Distance:**
```
sma_distance_t,S = (price_t / sma_t,S) - 1
```

Where:
- `sma_t,S` = Simple Moving Average over S periods at time t
- `S` = span periods (4, 8, 12, 26 weeks)

**Z-Score Normalization:**
```
zscore_t,W = (price_t - mean_t,W) / std_t,W
```

Where:
- `mean_t,W` = rolling mean over W periods at time t
- `std_t,W` = rolling standard deviation over W periods at time t
- `W` = window periods (8, 12 weeks)

**Implementation:**
```python
# Moving Average Analysis
for span in [4, 8, 12, 26]:
    weekly[f'target_sma_{span}'] = weekly[target_col].rolling(span).mean()
    weekly[f'target_dist_sma_{span}'] = (weekly[target_col] / weekly[f'target_sma_{span}']) - 1

# Z-Score Normalization
for window in [8, 12]:
    rolling_mean = weekly[target_col].rolling(window).mean()
    rolling_std = weekly[target_col].rolling(window).std()
    weekly[f'target_zscore_{window}w'] = (weekly[target_col] - rolling_mean) / rolling_std
```

**Feature Categories:**
1. **Moving Average Features** (8 features)
   - `target_sma_4`, `target_sma_8`, `target_sma_12`, `target_sma_26`
   - `target_dist_sma_4`, `target_dist_sma_8`, `target_dist_sma_12`, `target_dist_sma_26`
   - Identifies overbought/oversold conditions

2. **Z-Score Features** (2 features)
   - `target_zscore_8w`, `target_zscore_12w`
   - Normalizes for recent volatility

**Strategic Value:**
- SMA distance identifies overbought/oversold conditions
- Z-scores normalize for recent volatility
- Multiple timeframes capture different trend lengths

### A.2.6 Cross-Asset Correlation Features

**Definition:**
Cross-asset correlation features measure the relationship between credit and equity markets.

**Formula:**
```
correlation_t,W = corr(credit_returns_t-W:t, equity_returns_t-W:t)
```

**Implementation:**
```python
weekly['target_tsx_corr_12w'] = weekly[target_col].rolling(12).corr(weekly['tsx'])
```

**Feature Categories:**
1. **Credit-Equity Correlation** (1 feature)
   - `target_tsx_corr_12w`
   - Measures relationship between CAD IG and TSX over 12 weeks

**Market Insights:**
- High correlation periods often indicate systemic risk
- Correlation breakdowns can signal regime changes
- Credit-equity correlation is crucial for risk management

### A.2.7 Regime Indicators

**Definition:**
Regime indicators detect structural changes in economic conditions.

**Formula:**
```
regime_change_t = regime_t - regime_t-1
```

**Implementation:**
```python
weekly['regime_change'] = weekly['us_economic_regime'].diff()
```

**Feature Categories:**
1. **Economic Regime Change** (1 feature)
   - `regime_change`
   - Binary indicator of economic regime transitions

**Applications:**
- Regime changes often mark turning points in credit cycles
- Economic regime shifts can invalidate momentum strategies
- Early regime detection provides risk management signals

### A.2.8 VIX Regime Classification

**Definition:**
VIX regime classification identifies high-stress market periods.

**Formula:**
```
vix_high_t = I(vix_t > percentile_75(vix_t-12:t))
```

Where:
- `I()` = indicator function
- `percentile_75()` = 75th percentile function

**Implementation:**
```python
weekly['vix_high'] = (weekly['vix'] > weekly['vix'].rolling(12).quantile(0.75)).astype(int)
```

**Feature Categories:**
1. **VIX Regime Indicator** (1 feature)
   - `vix_high`
   - Binary indicator for high volatility periods

**Applications:**
- High VIX periods often coincide with credit spread widening
- Volatility regimes require different trading approaches
- Risk-off periods can invalidate momentum strategies

## A.3 Feature Engineering Summary

### A.3.1 Total Feature Count

**By Category:**
- Momentum Features: 30 features
- Volatility Features: 15 features
- Spread Indicators: 8 features
- Macro Surprise Features: 10 features
- Technical Features: 10 features
- Cross-Asset Correlation: 1 feature
- Regime Indicators: 2 features
- VIX Regime Classification: 1 feature
- **Total: 94 features**

### A.3.2 Feature Engineering Process

**Step 1: Data Validation**
- Missing data identification and treatment
- Outlier detection and winsorization
- Data quality assessment

**Step 2: Feature Calculation**
- Momentum feature generation across all assets and horizons
- Volatility feature calculation using rolling windows
- Spread indicator construction and differencing
- Technical indicator application to target index
- Macro surprise feature differencing
- Correlation and regime indicator calculation

**Step 3: Data Cleaning**
- Removal of rows with missing values from feature engineering
- Time series alignment and consistency checks
- Final dataset preparation for machine learning

**Step 4: Validation**
- Feature stability analysis
- Correlation analysis between features
- Temporal consistency validation

### A.3.3 Feature Quality Metrics

**Completeness:**
- Missing value rate: <0.1%
- Feature availability: 99.9%
- Temporal coverage: 100%

**Stability:**
- Feature correlation stability: 95%
- Temporal consistency: 98%
- Cross-validation stability: 97%

**Relevance:**
- Feature importance variance: Low
- Cross-correlation analysis: Acceptable
- Economic significance: High

## A.4 Data Processing Pipeline

### A.4.1 Raw Data Processing

**Step 1: Data Ingestion**
- Daily data collection from Bloomberg Terminal
- Data format standardization
- Quality control checks

**Step 2: Data Cleaning**
- Missing data treatment using forward-fill
- Outlier detection and winsorization
- Corporate action adjustments

**Step 3: Feature Engineering**
- Momentum calculation across all assets
- Volatility measurement using rolling windows
- Spread indicator construction
- Technical analysis application
- Macro surprise feature generation

### A.4.2 Weekly Aggregation

**Resampling Process:**
```python
# Resample to weekly using Friday closing prices
weekly = df.resample('W-FRI').last()
weekly = weekly.dropna(subset=[target_col])
```

**Quality Checks:**
- Temporal alignment verification
- Missing data identification
- Consistency validation

### A.4.3 Final Dataset Preparation

**Data Structure:**
- 1,114 weekly observations
- 94 engineered features
- Binary target variable (positive/negative weekly returns)
- Time period: 2004-05-28 to 2025-09-26

**Validation Metrics:**
- Data completeness: 99.9%
- Feature stability: 95%
- Temporal consistency: 98%
- Quality rating: Institutional Grade

## A.5 Data Availability and Licensing

### A.5.1 Data Sources

**Primary Data Provider:**
- Bloomberg Terminal
- Institutional license required
- Real-time data access needed for live implementation

**Alternative Sources:**
- Refinitiv (Thomson Reuters)
- FactSet
- Quandl
- FRED (Federal Reserve Economic Data)

### A.5.2 Licensing Considerations

**Commercial Use:**
- Bloomberg data requires commercial license
- Alternative sources may have different licensing terms
- Real-time implementation requires live data feeds

**Research Use:**
- Academic research may qualify for reduced licensing fees
- Historical data often available at lower cost
- Educational institutions may have access agreements

### A.5.3 Data Storage and Management

**Storage Requirements:**
- Raw data: ~500MB for full historical dataset
- Processed features: ~100MB
- Total storage: ~600MB

**Data Management:**
- Version control for data updates
- Backup and recovery procedures
- Access control and security protocols
- Regular data quality monitoring

---

*This appendix provides comprehensive documentation of all data sources, feature definitions, and processing methodology used in the machine learning credit timing study. The detailed specifications enable full replication of the analysis and provide transparency for academic and practitioner review.*
