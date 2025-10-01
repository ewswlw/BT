# Tactical Asset Allocation Framework

A production-grade implementation of 4 institutional TAA strategies from academic research, using VectorBT and Bloomberg data.

## 📊 Strategies Implemented

### 1. **VIX Timing Strategy** (2013-2025)
- **Paper**: "Timing Leveraged Equity Exposure in a TAA Model" (QuantSeeker, 2025)
- **Logic**: VIX-based filter to time SPXL (3x leveraged SPY) vs Cash
- **Assets**: Defensive rotation (TLT, GLD, DBC, UUP, BTAL) + SPXL/Cash fallback
- **Expected**: CAGR >25%, Sharpe 1.41, Max DD 15-16%

### 2. **Defense First Base** (2008-2025)
- **Paper**: "A Simple and Effective Tactical Allocation Strategy" (QuantSeeker, 2025)
- **Logic**: Rank defensives by momentum, allocate to SPY when momentum < T-bill
- **Assets**: TLT, GLD, DBC, UUP, BTAL + SPY fallback
- **Expected**: CAGR 9.5%, Sharpe 0.83, Max DD ~12%

### 3. **Defense First Leveraged** (2008-2025)
- **Paper**: Same as Defense First Base
- **Logic**: Identical to #2 but uses SPXL instead of SPY
- **Assets**: TLT, GLD, DBC, UUP, BTAL + SPXL fallback
- **Expected**: CAGR ~21%, Sharpe 0.99, similar volatility to SPY

### 4. **Sector Rotation** (1999-2025)
- **Paper**: "Replicating an Asset Allocation Model" (QuantSeeker, 2025)
- **Logic**: Multi-signal (momentum, vol, correlation, trend) sector selection
- **Assets**: 9-11 sector ETFs (dynamic universe) + Cash
- **Expected**: Sharpe 0.60, Max DD ~20-25%

---

## 🏗️ Architecture

```
tactical asset allocation/
├── strategies/              # Strategy implementations
│   ├── base_strategy.py
│   ├── vix_timing_strategy.py
│   ├── defense_first_base_strategy.py
│   ├── defense_first_levered_strategy.py
│   └── sector_rotation_strategy.py
│
├── data/                    # Data management
│   ├── etf_inception_dates.py   # Dynamic asset availability
│   ├── data_loader.py           # Bloomberg xbbg integration
│   └── processed/               # Cached data
│
├── backtests/               # VectorBT backtest engine
│   └── backtest_engine.py
│
├── reporting/               # Report generation
│   └── report_generator.py
│
├── tests/                   # Comprehensive test suite
│   ├── test_etf_inception_dates.py
│   ├── test_strategies.py
│   └── test_integration.py
│
├── outputs/                 # Generated results
│   ├── reports/            # HTML tearsheets
│   └── results/            # Text reports + CSVs
│
├── main.py                  # Main runner script
└── README.md
```

---

## 🚀 Quick Start

### Installation

```powershell
# Ensure you're in the Poetry environment
cd "c:\Users\Eddy\YTM Capital Dropbox\...\BT"
poetry shell

# Install dependencies (if not already installed)
poetry add vectorbt xbbg pandas numpy pytest pdfplumber
```

### Running Backtests

```powershell
# Navigate to tactical asset allocation directory
cd "tactical asset allocation"

# Run all 4 strategies
poetry run python main.py
```

### Running Tests

```powershell
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_strategies.py -v

# Run with coverage
poetry run pytest tests/ --cov=. --cov-report=html
```

---

## 📋 Key Features

### ✅ **Dynamic Asset Management**
- Respects ETF inception dates (e.g., BTAL launches May 2011)
- Sector universe evolves: 9 → 10 → 11 sectors over time
- Prevents lookahead bias by only using available assets

### ✅ **T-1 Signal Generation**
- All signals use previous day's close (avoid lookahead)
- Monthly rebalancing at month-end
- Full allocation (100% invested or in cash)

### ✅ **Zero Transaction Costs**
- Per requirements, assumes 0% fees
- VectorBT configured for pure signal testing

### ✅ **Bloomberg Data Integration**
- Uses `xbbg` for reliable Bloomberg data
- Automatic caching to minimize API calls
- Configurable lookback periods

### ✅ **Comprehensive Metrics**
- VectorBT portfolio stats
- Manual calculations (CAGR, Sharpe, Sortino, Max DD)
- QuantStats-style comparison vs SPY
- Skew, Kurtosis, Tail Risk metrics

### ✅ **Professional Reporting**
- HTML tearsheets (interactive charts)
- Text comparison reports
- Weight history CSV exports
- Inspired by `comprehensive_strategy_comparison.txt`

---

## 📊 Data Sources

### Bloomberg Tickers (via xbbg)

| Asset | Ticker | Bloomberg Code | Inception |
|-------|--------|----------------|-----------|
| S&P 500 | SPY | SPY US Equity | 1993-01-29 |
| S&P 500 3x | SPXL | SPXL US Equity | 2008-11-05 |
| 20+ Yr Treasury | TLT | TLT US Equity | 2002-07-26 |
| Gold | GLD | GLD US Equity | 2004-11-18 |
| Commodities | DBC | DBC US Equity | 2006-02-03 |
| US Dollar | UUP | UUP US Equity | 2007-02-26 |
| Anti-Beta | BTAL | BTAL US Equity | 2011-05-11 |
| VIX Index | VIX | VIX Index | - |
| 3M T-Bill | TBILL_3M | USGG3M Index | - |

**Sector ETFs**: XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY

---

## 🧪 Testing Strategy

### Unit Tests
- **ETF Inception Dates**: Validates registry accuracy and dynamic availability
- **Strategy Logic**: Tests signal generation, momentum, filtering
- **Base Methods**: Tests utility functions (returns, volatility, etc.)

### Integration Tests
- **End-to-End Workflow**: Data → Strategy → Backtest → Report
- **Weight Validation**: Ensures implementable allocations
- **Data Consistency**: Checks alignment and quality

### Validation Tests
- **Paper Replication**: Compares results to paper claims (where possible)
- **Date Constraints**: Verifies inception date compliance
- **Performance Metrics**: Validates expected performance ranges

---

## 📈 Expected Outputs

### Generated Files

```
outputs/
├── reports/
│   ├── VIX_TIMING_tearsheet.html
│   ├── DEFENSE_FIRST_BASE_tearsheet.html
│   ├── DEFENSE_FIRST_LEVERED_tearsheet.html
│   └── SECTOR_ROTATION_tearsheet.html
│
└── results/
    ├── VIX_TIMING_report.txt
    ├── DEFENSE_FIRST_BASE_report.txt
    ├── DEFENSE_FIRST_LEVERED_report.txt
    ├── SECTOR_ROTATION_report.txt
    ├── VIX_TIMING_weights.csv
    ├── DEFENSE_FIRST_BASE_weights.csv
    ├── DEFENSE_FIRST_LEVERED_weights.csv
    ├── SECTOR_ROTATION_weights.csv
    └── strategy_comparison.txt
```

### Report Format

Each report includes:
1. **Strategy Information**: Paper, author, dates, rules
2. **VectorBT Portfolio Stats**: Total return, Sharpe, Calmar, etc.
3. **Manual Calculations**: CAGR, volatility, Sortino, skew, kurtosis
4. **QuantStats Comparison**: Side-by-side vs SPY
5. **Recent Allocations**: Last 10 monthly weights
6. **Trading Rules**: Complete strategy logic

---

## ⚠️ Important Notes

### Date Constraints
- **VIX Timing**: Limited to 2013+ (paper start date)
- **Defense First Base**: Starts 2008-02-01 (paper start)
- **Defense First Leveraged**: Starts 2008-11-05 (SPXL inception)
- **Sector Rotation**: Starts 1999-12-01 (sector ETF availability)

### Asset Evolution
- **BTAL**: Only available from May 2011+
- **Sectors**: 9 original (1998) → +XLRE (2015) → +XLC (2018)

### Paper Claims vs Reality
- Some papers use synthetic data before ETF inception
- This implementation uses **real ETF data only**
- Results may differ from papers due to:
  - Real vs synthetic SPXL data
  - Transaction costs (0% here vs reality)
  - Data vendor differences

---

## 🔧 Configuration

### Strategy Parameters

```python
# VIX Timing
VIXTimingStrategy(
    lookbacks=[10, 15, 20],        # Realized vol lookbacks
    use_multi_lookback=True        # Graded allocation
)

# Defense First
DefenseFirstBaseStrategy(
    lookbacks=[21, 63, 126, 252],  # Momentum lookbacks
    fixed_weights=[0.40, 0.30, 0.20, 0.10]  # Rank weights
)

# Sector Rotation
SectorRotationStrategy(
    lookbacks=[21, 63, 126, 252],  # All signals
    top_n_sectors=5                # Number to hold
)
```

---

## 📚 References

1. **QuantSeeker (2025)**: "Timing Leveraged Equity Exposure in a TAA Model"
2. **QuantSeeker (2025)**: "A Simple and Effective Tactical Allocation Strategy"
3. **QuantSeeker (2025)**: "Stress-Testing a Tactical Allocation Model"
4. **QuantSeeker (2025)**: "Replicating an Asset Allocation Model" (based on Giordano 2018, 2019)

---

## 🤝 Contributing

To extend this framework:
1. Add new strategies by inheriting from `BaseTAAStrategy`
2. Implement `generate_signals()` and `get_strategy_info()`
3. Add comprehensive tests in `tests/`
4. Update this README with strategy details

---

## 📞 Support

For questions or issues:
1. Check test suite for usage examples
2. Review strategy docstrings for detailed logic
3. See paper conversion `.txt` files for full details

---

## ⚖️ License

This is a private research implementation. All strategies are based on publicly available academic research.

---

**Last Updated**: 2025-10-01  
**Python**: 3.11+  
**Key Dependencies**: vectorbt, xbbg, pandas, numpy, pytest

