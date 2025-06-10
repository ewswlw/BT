# VectorBT & QuantStats Common Issues Guide for AI

**Purpose:** Educational guide for AI assistants to recognize and fix common VectorBT and QuantStats configuration issues  
**Scope:** Frequency handling, performance calculations, and validation patterns  
**Status:** Living Document - Updated with Each Discovery  

---

## 📋 Overview

This guide documents recurring patterns in VectorBT and QuantStats usage that frequently cause incorrect results. These issues typically manifest when AI assistants help users with backtesting code, especially when dealing with non-daily data frequencies.

## 🎯 AI Learning Objectives

After reviewing this guide, AI assistants should be able to:
1. **Recognize frequency mismatch symptoms** in backtesting output
2. **Identify calculation inconsistencies** between different libraries
3. **Apply correct frequency configurations** for various data types
4. **Implement validation checks** to catch issues early
5. **Provide accurate troubleshooting** for performance metric discrepancies

---

## 🚨 Pattern #1: VectorBT Frequency Mismatch

### How to Recognize This Issue

**Key Symptom**: Trade durations that are impossible for the data frequency
- Monthly data showing trade durations of 2-3 days
- Weekly data showing durations of hours
- Any duration shorter than the underlying data frequency

**User Quote Pattern**: *"How can this be possible if frequency of data is [X]?"*

**When This Occurs**: When VectorBT portfolio objects are created without explicit frequency matching the data

### Root Cause Pattern

**Common Mistake**: Using default or incorrect frequency in VectorBT portfolio creation:

```python
# ❌ PROBLEMATIC PATTERNS:
portfolio = vbt.Portfolio.from_orders(
    close=price_data,
    size=strategy_signal,
    freq='D'  # ❌ Daily freq for monthly data
)

portfolio = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits
    # ❌ No freq specified - defaults to 'D'
)
```

### AI Recognition Checklist

**Instantly Suspicious Results**:
- Period showing data point count instead of actual time span
- Trade durations impossible for data frequency
- Sharpe ratios > 3.0 (often inflated due to frequency errors)
- Drawdown durations that seem too short

**Expected vs Actual Comparison**:
```python
# For monthly data spanning 57 years:
# ❌ Wrong: Period: 683 days (just the data point count)
# ✅ Correct: Period: 20,490 days (~57 years)

# For monthly trading:
# ❌ Wrong: Avg trade duration: 2-30 days 
# ✅ Correct: Avg trade duration: 90+ days (3+ months)
```

### AI Solution Pattern

**Step 1: Identify Data Frequency**
```python
# AI should determine data frequency first:
data_freq = 'monthly' if len(data) < 500 and timespan > 10_years else 'daily'
trading_freq = infer_frequency_from_data(data.index)
```

**Step 2: Apply Correct VectorBT Frequency**
```python
# ✅ CORRECT FREQUENCY MAPPING:
freq_mapping = {
    'daily': 'D',
    'weekly': pd.Timedelta(days=7),
    'monthly': pd.Timedelta(days=30),
    'quarterly': pd.Timedelta(days=90),
    'yearly': pd.Timedelta(days=365)
}

# ✅ CORRECTED PORTFOLIO CREATION:
portfolio = vbt.Portfolio.from_orders(
    close=price_data,
    size=signals,
    freq=freq_mapping[data_frequency]  # Match to data
)
```

**Step 3: VectorBT Frequency Limitations**
```python
# ❌ NOT SUPPORTED by VectorBT:
freq='M'    # Monthly - deprecated in pandas
freq='Y'    # Yearly - deprecated  
freq='Q'    # Quarterly - deprecated

# ✅ ALTERNATIVES that work:
freq=pd.Timedelta(days=30)    # Monthly approximation
freq=pd.Timedelta(weeks=4)    # 4-week approximation
freq='30D'                    # 30-day string format
```

### AI Validation Protocol

**Post-Fix Verification Checklist**:
1. **Period Check**: Does total period match actual time span?
2. **Duration Reality**: Are trade durations >= minimum possible for frequency?
3. **Ratio Sanity**: Are Sharpe ratios in reasonable range (0.1-2.0)?
4. **Consistency**: Do all portfolio objects use same frequency?

**Expected Improvement After Fix**:
```python
# Realistic metrics after frequency correction:
# - Period: Actual calendar time, not data point count
# - Trade durations: Multiple periods (e.g., 3+ months for monthly data)
# - Sharpe ratios: Moderate values (0.3-1.5 typically)
# - Performance: May decrease but becomes accurate
```

### AI Implementation Checklist

**When Creating VectorBT Portfolios**:
- [ ] Explicitly set frequency for ALL portfolio objects
- [ ] Use consistent frequency across all portfolios  
- [ ] Validate trade durations make sense for data frequency
- [ ] Cross-check period calculations with expected timespan

---

## 🚨 Pattern #2: Volatility Annualization Factor Mismatch

### How to Recognize This Issue

**Key Symptom**: Volatility calculations using wrong annualization factor for data frequency

**Common Mistake Pattern**:
```python
# ❌ WRONG: Using daily annualization for monthly data
df['Vol_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)  # Daily factor
annual_vol = strategy_returns.std() * np.sqrt(252)             # Daily factor
annual_return = strategy_returns.mean() * 252                  # Daily factor
```

### AI Solution Pattern

**Correct Annualization by Frequency**:
```python
# ✅ CORRECT: Match annualization to data frequency
annualization_factors = {
    'daily': {'vol': np.sqrt(252), 'ret': 252},
    'weekly': {'vol': np.sqrt(52), 'ret': 52},
    'monthly': {'vol': np.sqrt(12), 'ret': 12},
    'quarterly': {'vol': np.sqrt(4), 'ret': 4},
    'yearly': {'vol': np.sqrt(1), 'ret': 1}
}

# Apply correct factor based on detected frequency
freq = detect_data_frequency(data)
vol_factor = annualization_factors[freq]['vol']
ret_factor = annualization_factors[freq]['ret']

annual_vol = returns.std() * vol_factor
annual_return = returns.mean() * ret_factor
```

### AI Recognition Checklist

**When to Suspect Annualization Errors**:
- Monthly data with `sqrt(252)` anywhere in volatility calculations
- Extremely high volatility values (>100% for typical strategies)
- Returns that seem inflated relative to reasonable expectations

---

## 🚨 Pattern #3: VectorBT Returns Accessor Frequency Compatibility

### How to Recognize This Issue

**Key Symptom**: VectorBT returns accessor throwing pandas frequency deprecation warnings

**Error Pattern**:
```
FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
FutureWarning: 'Y' is deprecated and will be removed in a future version, please use 'YE' instead.
```

### AI Solution Pattern

**Set Global VectorBT Frequency at Import**:
```python
# ✅ CORRECT: Set frequency compatibility at import
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
    # Set frequency for monthly data compatibility
    vbt.settings.array_wrapper['freq'] = '30d'  # Monthly approximation
except ImportError:
    VECTORBT_AVAILABLE = False
```

**Use Returns Accessor with Error Handling**:
```python
try:
    # VectorBT returns accessor with global frequency setting
    stats = returns_series.vbt.returns.stats()
    print(stats)
except Exception as e:
    # Fallback to manual calculations
    print(f"VectorBT accessor failed: {e}")
    print(f"Mean: {returns_series.mean():.4f}")
    print(f"Std: {returns_series.std():.4f}")
```

---

## 🛠️ AI Best Practices Framework

### Frequency Detection Function
```python
def detect_data_frequency(data_index):
    """AI helper to detect data frequency from index"""
    median_diff = pd.Series(data_index).diff().median()
    
    if median_diff <= pd.Timedelta(days=1):
        return 'daily'
    elif median_diff <= pd.Timedelta(days=8):
        return 'weekly'  
    elif median_diff <= pd.Timedelta(days=32):
        return 'monthly'
    elif median_diff <= pd.Timedelta(days=95):
        return 'quarterly'
    else:
        return 'yearly'
```

### Universal VectorBT Setup Function
```python
def setup_vectorbt_for_frequency(frequency):
    """AI helper to configure VectorBT for any frequency"""
    freq_settings = {
        'daily': 'D',
        'weekly': '7d', 
        'monthly': '30d',
        'quarterly': '90d',
        'yearly': '365d'
    }
    
    if frequency in freq_settings:
        vbt.settings.array_wrapper['freq'] = freq_settings[frequency]
        return freq_settings[frequency]
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
```

### Validation Template
```python
def validate_portfolio_metrics(portfolio, expected_timespan_years, data_frequency):
    """AI validation template for VectorBT portfolios"""
    
    # Extract key metrics
    period_days = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
    avg_trade_duration = portfolio.stats()['Avg Winning Trade Duration']
    
    # Validation checks
    checks = {
        'period_realistic': abs(period_days / 365.25 - expected_timespan_years) < 1,
        'trade_duration_valid': validate_trade_duration(avg_trade_duration, data_frequency),
        'sharpe_reasonable': 0.1 <= portfolio.stats()['Sharpe Ratio'] <= 3.0
    }
    
    return all(checks.values()), checks

def validate_trade_duration(duration, frequency):
    """Check if trade duration makes sense for data frequency"""
    min_durations = {
        'daily': pd.Timedelta(days=1),
        'weekly': pd.Timedelta(days=7),
        'monthly': pd.Timedelta(days=30),
        'quarterly': pd.Timedelta(days=90)
    }
    
    return duration >= min_durations.get(frequency, pd.Timedelta(days=1))
```

---

## 🚨 Pattern #4: QuantStats CAGR Calculation Error with Monthly Data

### How to Recognize This Issue

**Key Symptom**: Significant discrepancy between QuantStats CAGR and VectorBT/manual calculations

**User Quote Pattern**: *"why is the CAGR% so diff vs Annualized Return [%]"*

**Typical Discrepancy**: QuantStats showing ~30% lower CAGR than correct calculation

### Root Cause Pattern

**Common Mistake**: Using QuantStats `cagr()` function without frequency specification:
```python
# ❌ PROBLEMATIC: QuantStats assumes daily frequency
def calculate_metrics(returns):
    metrics = {}
    metrics['cagr'] = qs.stats.cagr(returns)  # ❌ No frequency specified
    return metrics
```

**Failed Attempts**:
```python
# ❌ These don't work as expected:
qs.stats.cagr(returns, periods=12)        # Often makes results worse
qs.stats.cagr(returns, periods=252)       # Still incorrect for monthly
```

### AI Solution Pattern

**Manual CAGR Calculation for Non-Daily Data**:
```python
def calculate_correct_cagr(returns, data_frequency):
    """AI helper for accurate CAGR calculation"""
    
    # Frequency mappings
    periods_per_year = {
        'daily': 252,
        'weekly': 52, 
        'monthly': 12,
        'quarterly': 4,
        'yearly': 1
    }
    
    # Get total return using QuantStats (this part works correctly)
    total_return = qs.stats.comp(returns)
    
    # Calculate years manually based on data frequency
    years = len(returns) / periods_per_year[data_frequency]
    
    # Manual CAGR calculation
    cagr = (1 + total_return) ** (1/years) - 1
    
    return cagr

# ✅ CORRECT implementation:
def calculate_comprehensive_metrics(returns, data_frequency):
    metrics = {}
    
    # Use manual CAGR for non-daily data
    if data_frequency != 'daily':
        metrics['cagr'] = calculate_correct_cagr(returns, data_frequency)
    else:
        metrics['cagr'] = qs.stats.cagr(returns)  # QuantStats OK for daily
    
    # Other QuantStats functions typically work fine
    metrics['total_return'] = qs.stats.comp(returns)
    metrics['sharpe'] = qs.stats.sharpe(returns)
    metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
    
    return metrics
```

### AI Cross-Validation Template

**Always verify CAGR across multiple methods**:
```python
def validate_cagr_calculation(returns, data_frequency, vbt_result=None):
    """AI validation template for CAGR calculations"""
    
    # Method 1: Manual calculation (most reliable)
    total_ret = qs.stats.comp(returns)
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly': 4}
    years = len(returns) / periods_per_year[data_frequency]
    manual_cagr = (1 + total_ret) ** (1/years) - 1
    
    # Method 2: QuantStats (potentially problematic)
    qs_cagr = qs.stats.cagr(returns)
    
    # Method 3: VectorBT (if available)
    results = {
        'manual_cagr': manual_cagr,
        'quantstats_cagr': qs_cagr,
        'difference_pct': abs(manual_cagr - qs_cagr) / manual_cagr * 100
    }
    
    if vbt_result:
        results['vectorbt_cagr'] = vbt_result
        results['vbt_manual_diff'] = abs(manual_cagr - vbt_result) / manual_cagr * 100
    
    # Flag if QuantStats differs significantly from manual
    results['quantstats_reliable'] = results['difference_pct'] < 5.0
    
    return results
```

### AI Warning Triggers

**When to Suspect QuantStats CAGR Issues**:
- Monthly, weekly, quarterly, or yearly data frequency
- CAGR difference > 10% between QuantStats and VectorBT
- QuantStats CAGR seems "too conservative" relative to total returns
- User questioning discrepancies between different calculation methods

---

## 🎯 AI Implementation Workflow

### Complete AI Setup Template
```python
def setup_backtesting_environment(data, user_frequency_hint=None):
    """Complete AI setup for VectorBT + QuantStats backtesting"""
    
    # Step 1: Auto-detect or confirm data frequency
    detected_freq = detect_data_frequency(data.index)
    if user_frequency_hint and user_frequency_hint != detected_freq:
        print(f"⚠️  User says {user_frequency_hint}, detected {detected_freq}")
        frequency = user_frequency_hint  # Trust user input
    else:
        frequency = detected_freq
    
    # Step 2: Setup VectorBT for detected frequency
    vbt_freq = setup_vectorbt_for_frequency(frequency)
    
    # Step 3: Prepare calculation functions
    def safe_cagr(returns):
        if frequency == 'daily':
            return qs.stats.cagr(returns)  # QuantStats OK for daily
        else:
            return calculate_correct_cagr(returns, frequency)  # Manual for others
    
    def safe_annualization(returns):
        factors = annualization_factors[frequency]
        return {
            'annual_vol': returns.std() * factors['vol'],
            'annual_ret': returns.mean() * factors['ret']
        }
    
    return {
        'frequency': frequency,
        'vbt_freq': vbt_freq,
        'cagr_func': safe_cagr,
        'annualize_func': safe_annualization,
        'validation_func': lambda pf, years: validate_portfolio_metrics(pf, years, frequency)
    }

# Usage in AI code:
setup = setup_backtesting_environment(price_data, user_frequency_hint='monthly')

# Create portfolios with correct frequency
strategy_pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits,
    freq=setup['vbt_freq']  # Automatically correct
)

# Calculate CAGR safely
strategy_cagr = setup['cagr_func'](strategy_returns)

# Validate results
is_valid, checks = setup['validation_func'](strategy_pf, expected_years=57)
if not is_valid:
    print(f"⚠️  Validation failed: {checks}")
```

## 🔍 AI Quick Diagnostic Checklist

**Before delivering any backtesting results, AI should verify:**

### Frequency Consistency ✅
- [ ] VectorBT portfolio frequency matches data frequency
- [ ] All portfolio objects use same frequency setting
- [ ] Annualization factors match data frequency
- [ ] CAGR calculation appropriate for frequency

### Sanity Checks ✅
- [ ] Trade durations >= minimum possible for frequency
- [ ] Total period matches expected timespan
- [ ] Sharpe ratios in reasonable range (0.1-3.0)
- [ ] Performance metrics consistent across libraries

### Cross-Validation ✅
- [ ] CAGR difference < 5% between manual and library calculations
- [ ] VectorBT and manual results align
- [ ] No suspicious outliers in performance metrics
- [ ] User intuition checks pass ("does this make sense?")

### Error Prevention ✅
- [ ] No `freq='M'`, `freq='Y'` in VectorBT (use Timedelta)
- [ ] No `sqrt(252)` for non-daily data
- [ ] No `qs.stats.cagr()` for non-daily without verification
- [ ] Proper error handling for all library calls

## 🚀 Summary for AI Assistants

**Key Takeaway**: When helping users with VectorBT and QuantStats backtesting, always think "frequency first" - most issues stem from frequency mismatches between data, calculations, and library assumptions.

**Golden Rules**:
1. **Detect data frequency explicitly** - never assume
2. **Configure VectorBT frequency to match data** - never use defaults
3. **Use manual CAGR for non-daily data** - don't trust QuantStats
4. **Validate results with sanity checks** - impossible durations are red flags
5. **Cross-check between libraries** - consistency is key

**User Warning Signs to Watch For**:
- *"How can this be possible if frequency is..."*
- *"Why is CAGR so different from..."*
- *"These durations seem too short/long..."*
- *"The numbers don't match between..."*

When you see these patterns, immediately investigate frequency configuration issues using this guide.

---

**END OF GUIDE** 