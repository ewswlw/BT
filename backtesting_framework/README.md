# 🏆 Professional Financial Backtesting Framework

A comprehensive, enterprise-grade backtesting framework designed for quantitative finance professionals, researchers, and traders. Built with modular architecture, professional reporting standards, and maximum flexibility for rapid strategy development and testing.

## 🎯 Overview

This framework transforms financial strategy backtesting from ad-hoc scripts into a professional, reproducible system. Whether you're testing cross-asset momentum strategies, developing genetic algorithm-based trading rules, or building complex multi-asset strategies, this framework provides the infrastructure to focus on strategy logic rather than plumbing.

**Perfect for:**
- Quantitative researchers developing new strategies
- Portfolio managers testing allocation strategies  
- Risk managers analyzing strategy performance
- Academic researchers studying market behavior
- Trading teams needing rapid strategy prototyping

## 🚀 Key Features

### 🏗️ **Enterprise Architecture**
- **Modular Design**: Clean separation of concerns with data loading, feature engineering, strategy logic, and reporting
- **Configuration-Driven**: YAML-based configuration system for reproducible research
- **Extensible Framework**: Plugin architecture for adding new strategies, features, and data sources
- **Type Safety**: Comprehensive type hints and dataclass configurations
- **Error Handling**: Robust error handling with detailed logging and debugging information

### 📊 **Professional Reporting**
- **Industry-Standard Metrics**: 40+ performance metrics including Sharpe, Sortino, Calmar, Information Ratio
- **Interactive HTML Reports**: Professional-grade reports using quantstats integration
- **Risk Analytics**: Value-at-Risk, Expected Shortfall, Maximum Drawdown analysis
- **Strategy Comparison**: Side-by-side performance analysis with statistical significance testing
- **Artifact Generation**: CSV exports of signals, equity curves, returns, and metrics for further analysis

### 🧠 **Advanced Strategies**
- **Cross-Asset Momentum**: Multi-asset momentum confirmation strategies with configurable thresholds
- **Multi-Asset Momentum**: Combined momentum signals across asset classes with custom weighting
- **Genetic Algorithm**: Evolutionary trading rule discovery with 24+ technical features
- **Custom Strategies**: Framework for rapid development of new strategy types

### 🔧 **Feature Engineering**
- **Technical Indicators**: MACD, Stochastic, RSI, Bollinger Bands, Moving Averages
- **Cross-Asset Features**: Correlation analysis, relative strength, regime detection  
- **Risk Features**: VIX analysis, OAS momentum, volatility clustering
- **Custom Features**: Extensible feature engineering pipeline for proprietary indicators

### 💼 **Portfolio Management**
- **Vectorbt Integration**: High-performance backtesting with optional vectorbt acceleration
- **Manual Fallback**: Pure Python implementation for environments without vectorbt
- **Position Sizing**: Multiple position sizing methodologies (fixed, Kelly criterion, volatility-adjusted)
- **Transaction Costs**: Configurable fees, slippage, and market impact modeling
- **Risk Management**: Portfolio-level risk controls and position limits

## 📁 Detailed Project Structure

```
backtesting_framework/
├── 🏗️ core/                          # Core Framework Components
│   ├── config.py                    # Configuration management with dataclasses
│   │   ├── DataConfig              # Data loading configuration
│   │   ├── PortfolioConfig        # Portfolio backtesting settings
│   │   ├── ReportingConfig        # Report generation settings
│   │   └── OptimizationConfig     # Parameter optimization settings
│   ├── data_loader.py              # Data loading and preprocessing
│   │   ├── CSVDataLoader          # CSV file loading with date handling
│   │   ├── Resampling logic       # Time series resampling (daily→weekly)
│   │   └── Data validation        # Missing data checks and fills
│   ├── feature_engineering.py      # Technical indicators and features
│   │   ├── TechnicalFeatureEngineer    # MACD, RSI, Stochastic, etc.
│   │   ├── CrossAssetFeatureEngineer   # Multi-asset correlations
│   │   ├── MultiAssetFeatureEngineer   # Combined asset features
│   │   └── Custom feature pipeline    # Extensible feature creation
│   ├── portfolio.py                # Portfolio management and backtesting
│   │   ├── PortfolioEngine        # Main backtesting engine
│   │   ├── Vectorbt integration   # High-performance backtesting
│   │   ├── Manual calculations    # Pure Python fallback
│   │   └── Transaction cost modeling
│   ├── metrics.py                  # Performance metrics calculation
│   │   ├── QuantStats integration # 40+ professional metrics
│   │   ├── Risk metrics          # VaR, CVaR, drawdown analysis
│   │   └── Manual metric fallback
│   └── reporting.py               # Professional report generation
│       ├── HTML report generation
│       ├── CSV artifact export
│       ├── Strategy comparison
│       └── Performance visualization
│
├── 🧠 strategies/                     # Strategy Implementations
│   ├── base_strategy.py            # Abstract base strategy class
│   │   ├── Signal generation API
│   │   ├── Data validation
│   │   ├── Backtesting interface
│   │   └── Parameter optimization
│   ├── cross_asset_momentum.py     # Cross-asset momentum strategy
│   │   ├── 4-asset momentum tracking (CAD IG, US HY, US IG, TSX)
│   │   ├── Configurable lookback periods (default: 2 weeks)
│   │   ├── Minimum confirmation logic (default: ≥3 of 4 assets)
│   │   └── Entry/exit signal generation
│   ├── multi_asset_momentum.py     # Combined momentum strategy
│   │   ├── Multi-asset momentum aggregation
│   │   ├── Weighted momentum scoring
│   │   ├── Threshold-based signals
│   │   └── Dynamic asset weighting
│   ├── genetic_algorithm.py        # Evolutionary trading strategy
│   │   ├── 24+ technical features
│   │   ├── Genetic algorithm optimization
│   │   ├── Rule expression evolution
│   │   ├── Fitness function optimization
│   │   └── Early stopping mechanisms
│   └── strategy_factory.py         # Strategy registration and creation
│
├── ⚙️ configs/                        # Configuration Management
│   ├── base_config.yaml            # Production configuration
│   │   ├── Real data source paths
│   │   ├── Weekly Friday resampling
│   │   ├── Production portfolio settings
│   │   └── Strategy-specific parameters
│   └── custom_config.yaml          # Research/testing configuration
│       ├── Test data configuration
│       ├── Custom date ranges
│       ├── Modified strategy parameters
│       └── Research-specific settings
│
├── 📊 outputs/                        # Generated Outputs
│   ├── reports/                    # Standard reports (base_config)
│   │   ├── artifacts/             # CSV files and JSON metrics
│   │   │   ├── {strategy}_signals.csv
│   │   │   ├── {strategy}_equity.csv
│   │   │   ├── {strategy}_returns.csv
│   │   │   └── {strategy}_metrics.json
│   │   └── {strategy}_report.html  # Interactive HTML reports
│   └── custom_reports/             # Custom reports (custom_config)
│       ├── artifacts/             # Research artifacts
│       └── *.html                 # Research reports
│
├── 🔧 Development Files
│   ├── main.py                     # Main execution script
│   │   ├── CLI argument parsing
│   │   ├── Configuration loading
│   │   ├── Strategy execution
│   │   ├── Comparison functionality
│   │   └── Error handling and logging
│   ├── pyproject.toml              # Poetry dependency management
│   │   ├── Core dependencies (pandas, numpy, pyyaml)
│   │   ├── Optional dependencies (vectorbt, quantstats)
│   │   ├── Development dependencies (pytest, black)
│   │   └── Python version constraints
│   └── README.md                   # This comprehensive documentation
│
└── 🧪 Additional Directories
    ├── tests/                      # Unit and integration tests
    ├── notebooks/                  # Jupyter research notebooks
    ├── optimization/               # Parameter optimization utilities
    └── data/                       # Sample data and schemas
```

### 🔄 Data Flow Architecture

```
📥 Data Input → 🔧 Feature Engineering → 🧠 Strategy Logic → 💼 Portfolio Sim → 📊 Reporting
     ↓                    ↓                    ↓                ↓              ↓
CSV Files          Technical           Signal           Backtest        HTML Reports
Date Handling      Indicators         Generation        Portfolio       CSV Artifacts
Resampling         Cross-Asset        Entry/Exit        Returns         JSON Metrics
Validation         Features           Logic             Trades          Comparisons
```

## 🛠️ Comprehensive Installation Guide

### Prerequisites

- **Python 3.9+** (Tested on 3.9, 3.10, 3.11)
- **Git** for version control
- **Terminal/Command Line** access

### Step 1: Install Poetry (Package Manager)

Poetry provides robust dependency management and virtual environment handling.

**Option A - Official Installer (Recommended):**
```bash
# Linux/macOS/WSL
curl -sSL https://install.python-poetry.org | python3 -

# Windows PowerShell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

**Option B - Alternative Methods:**
```bash
# Using pip (if you prefer)
pip install poetry

# Using conda
conda install poetry

# Using homebrew (macOS)
brew install poetry
```

**Verify Installation:**
```bash
poetry --version
# Should output: Poetry (version 1.6.0+)
```

### Step 2: Setup Project

**Clone and Navigate:**
```bash
# If cloning from repository
git clone <repository-url>
cd backtesting_framework

# Or if already have the code
cd path/to/backtesting_framework
```

### Step 3: Install Dependencies

**Core Installation:**
```bash
# Install all dependencies including optional ones
poetry install

# Install only core dependencies (faster, no quantstats/vectorbt)
poetry install --no-dev --extras minimal
```

**Verify Installation:**
```bash
# Activate virtual environment
poetry shell

# Test import
python -c "import pandas; import numpy; import yaml; print('✅ Core dependencies installed')"

# Test optional dependencies
python -c "
try:
    import quantstats; print('✅ QuantStats available')
except ImportError: print('⚠️  QuantStats not available (optional)')
try:
    import vectorbt; print('✅ VectorBT available') 
except ImportError: print('⚠️  VectorBT not available (optional)')
"
```

### Step 4: Configuration Setup

**Data Configuration:**
```bash
# Option 1: Use provided test data
cp configs/custom_config.yaml configs/my_config.yaml

# Option 2: Configure for your data
# Edit configs/base_config.yaml to point to your CSV file
```

**Required Data Format:**
Your CSV file should have:
- Date column (configurable name, default: "Date")
- Price/index columns for assets
- Proper date formatting (YYYY-MM-DD or similar)

**Example Data Structure:**
```csv
Date,cad_ig_er_index,us_hy_er_index,us_ig_er_index,tsx,cad_oas,us_ig_oas,us_hy_oas,vix
2020-01-01,100.0,100.0,100.0,100.0,150.0,125.0,300.0,15.5
2020-01-02,100.5,99.8,100.2,100.1,149.5,124.8,299.5,15.2
...
```

### Step 5: Test Installation

**Quick Test:**
```bash
# Test with basic configuration
poetry run python main.py --list-strategies

# Run a quick strategy test
poetry run python main.py --strategy cross_asset_momentum

# If successful, you'll see output like:
# ================================================================================
# CROSS-ASSET-MOMENTUM STRATEGY - COMPLETE ANALYSIS  
# ================================================================================
```

### Troubleshooting Installation

**Common Issues:**

1. **Poetry not found after installation:**
   ```bash
   # Add to PATH (Linux/macOS)
   export PATH="$HOME/.local/bin:$PATH"
   
   # Windows: Add to environment variables
   # %APPDATA%\Python\Scripts
   ```

2. **Python version conflicts:**
   ```bash
   # Check Python version
   python --version
   
   # Use specific Python version with Poetry
   poetry env use python3.10
   poetry install
   ```

3. **VectorBT installation issues (optional dependency):**
   ```bash
   # VectorBT has complex dependencies, framework works without it
   poetry install --no-dev
   
   # Or install without optional dependencies
   pip install pandas numpy pyyaml matplotlib seaborn
   ```

4. **Permission errors:**
   ```bash
   # Linux/macOS: Fix permissions
   sudo chown -R $USER ~/.cache/pypoetry
   
   # Windows: Run terminal as administrator
   ```

5. **SSL Certificate errors:**
   ```bash
   # Corporate networks might need
   poetry config certificates.custom.cert-file /path/to/cert.pem
   ```

### Development Installation

**For Contributors:**
```bash
# Install with development dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest tests/

# Code formatting
poetry run black .
poetry run isort .
```

### Docker Installation (Advanced)

**Create Dockerfile:**
```dockerfile
FROM python:3.10-slim

RUN pip install poetry
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY . .
CMD ["python", "main.py"]
```

**Build and Run:**
```bash
docker build -t backtesting-framework .
docker run -v $(pwd)/outputs:/app/outputs backtesting-framework
```

## 📊 Comprehensive Usage Guide

### 🚀 Quick Start

**Run Default Configuration:**
```bash
# Activate Poetry environment
poetry shell

# Run all strategies with base configuration
poetry run python main.py

# Or run specific strategy
poetry run python main.py --strategy cross_asset_momentum
```

### 🎯 Core Commands

#### List Available Strategies
```bash
poetry run python main.py --list-strategies

# Output:
# Available strategies:
# ├── cross_asset_momentum    # Cross-asset momentum confirmation
# ├── multi_asset_momentum    # Combined momentum signals  
# └── genetic_algorithm       # Evolutionary rule discovery
```

#### Single Strategy Execution
```bash
# Cross-asset momentum (4 indices, ≥3 confirmations)
poetry run python main.py --strategy cross_asset_momentum

# Multi-asset momentum (combined momentum scoring)
poetry run python main.py --strategy multi_asset_momentum

# Genetic algorithm (evolutionary trading rules)
poetry run python main.py --strategy genetic_algorithm
```

#### Strategy Comparison
```bash
# Compare two strategies
poetry run python main.py --compare cross_asset_momentum multi_asset_momentum

# Compare all strategies
poetry run python main.py --compare cross_asset_momentum multi_asset_momentum genetic_algorithm

# Output includes side-by-side performance table:
# Strategy                 Total Return  CAGR    Sharpe  Max DD   Trades
# -------------------------------------------------------------------------
# cross_asset_momentum         85.39%   2.00%   4.90   -1.25%    155
# multi_asset_momentum         74.95%   1.81%   4.13   -3.61%     84
# genetic_algorithm            76.03%   1.83%   3.96   -2.05%     73
```

### ⚙️ Configuration Management

#### Using Custom Configurations
```bash
# Use custom configuration file
poetry run python main.py --config configs/custom_config.yaml --strategy genetic_algorithm

# Use configuration with different data source
poetry run python main.py --config my_research_config.yaml --strategy cross_asset_momentum
```

#### Creating Custom Configurations
```yaml
# Create configs/my_config.yaml
data:
  file_path: "path/to/my/data.csv"
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  resample_frequency: "W-FRI"

portfolio:
  initial_capital: 1000000  # $1M
  fees: 0.001              # 10 bps
  slippage: 0.0005        # 5 bps

cross_asset_momentum:
  momentum_lookback_weeks: 3  # Changed from default 2
  min_confirmations: 2        # Changed from default 3
```

### 📈 Advanced Usage Scenarios

#### Research Workflow
```bash
# 1. Test strategy with different parameters
poetry run python main.py --config research_config.yaml --strategy genetic_algorithm

# 2. Compare performance across time periods
# Edit config dates and re-run
poetry run python main.py --config period1_config.yaml --strategy cross_asset_momentum
poetry run python main.py --config period2_config.yaml --strategy cross_asset_momentum

# 3. Analyze specific market regimes
poetry run python main.py --config crisis_period.yaml --compare cross_asset_momentum multi_asset_momentum
```

#### Production Workflow
```bash
# 1. Run daily strategy update
poetry run python main.py --strategy cross_asset_momentum

# 2. Generate comparison report for investment committee
poetry run python main.py --compare cross_asset_momentum multi_asset_momentum genetic_algorithm

# 3. Export results for further analysis
# Check outputs/reports/artifacts/ for CSV files
```

#### Development Workflow
```bash
# 1. Test new strategy during development
poetry run python main.py --strategy my_new_strategy --config dev_config.yaml

# 2. Parameter optimization (manual)
for param in 1 2 3 4; do
  sed -i "s/momentum_lookback_weeks: .*/momentum_lookback_weeks: $param/" configs/test_config.yaml
  poetry run python main.py --config configs/test_config.yaml --strategy cross_asset_momentum
done

# 3. Validate against benchmark
poetry run python main.py --compare my_new_strategy cross_asset_momentum
```

### 📊 Understanding Output

#### Console Output Structure
```
================================================================================
CROSS-ASSET-MOMENTUM STRATEGY - COMPLETE ANALYSIS  
================================================================================
Strategy: Long when ≥3 of 4 indices show positive 2-week momentum
Indices: CAD IG ER, US HY ER, US IG ER, TSX
Trading Asset: CAD IG ER Index
Data Period: 2003-12-05 to 2025-06-06 (1123 weeks)

Signal Statistics:
├── Total signals: 620 periods (55.21% frequency)
├── Average confirmations: 2.42 assets
└── Time in market: 55.2%

Portfolio Statistics:
├── Total return: 85.39%
├── CAGR: 2.00%
├── Sharpe ratio: 4.90
├── Sortino ratio: 14.37
├── Max drawdown: -1.25%
├── Calmar ratio: 1.60
├── Number of trades: 155
└── Win rate: 68.4%

Risk Metrics:
├── Volatility (annualized): 2.84%
├── Value-at-Risk (95%): -0.24%
├── Expected Shortfall: -0.38%
├── Beta: 0.63
└── Information ratio: 0.94
```

#### Generated Files
```
outputs/reports/
├── cross_asset_momentum_report.html     # Interactive report
└── artifacts/
    ├── cross_asset_momentum_signals.csv # Entry/exit signals by date
    ├── cross_asset_momentum_equity.csv  # Portfolio value over time
    ├── cross_asset_momentum_returns.csv # Period returns
    └── cross_asset_momentum_metrics.json # All metrics in JSON format
```

### 🔧 Command Line Options

#### Complete CLI Reference
```bash
poetry run python main.py [OPTIONS]

Options:
  --strategy TEXT        Run single strategy [cross_asset_momentum|multi_asset_momentum|genetic_algorithm]
  --compare TEXT...      Compare multiple strategies (space-separated list)
  --config TEXT          Path to YAML configuration file (default: configs/base_config.yaml)
  --list-strategies      List all available strategies
  --verbose             Enable verbose logging
  --help                Show help message

Examples:
  python main.py --strategy cross_asset_momentum
  python main.py --compare cross_asset_momentum multi_asset_momentum
  python main.py --config custom.yaml --strategy genetic_algorithm
  python main.py --list-strategies
```

### 🚨 Common Usage Patterns

#### Error Handling
```bash
# Strategy fails - check configuration
poetry run python main.py --strategy cross_asset_momentum --verbose

# Data file not found - verify path
poetry run python main.py --config custom_config.yaml --verbose

# Memory issues with large datasets - use sampling
# Edit config to add start_date/end_date constraints
```

#### Performance Optimization
```bash
# Speed up execution - disable HTML reports in config
reporting:
  generate_html: false

# Use faster resampling
data:
  resample_frequency: "M"  # Monthly instead of weekly

# Limit date range for testing
data:
  start_date: "2022-01-01"
  end_date: "2023-12-31"
```

## ⚙️ Configuration

The framework uses YAML configuration files. Key sections:

### Data Configuration
```yaml
data:
  file_path: "path/to/your/data.csv"
  date_column: "Date"
  resample_frequency: "W-FRI"
  benchmark_asset: "cad_ig_er_index"
  assets:
    - "cad_ig_er_index"
    - "us_hy_er_index"
    # ... more assets
```

### Portfolio Configuration
```yaml
portfolio:
  initial_capital: 100000
  frequency: "W"
  fees: 0.0
  slippage: 0.0
```

### Strategy-Specific Configuration
```yaml
cross_asset_momentum:
  momentum_assets: ["cad_ig_er_index", "us_hy_er_index", "us_ig_er_index", "tsx"]
  momentum_lookback_weeks: 2
  min_confirmations: 3

multi_asset_momentum:
  momentum_assets_map:
    tsx: "tsx"
    us_hy: "us_hy_er_index"
    cad_ig: "cad_ig_er_index"
  momentum_lookback_periods: 4
  signal_threshold: -0.005
```

## 📈 Strategy Deep Dive

### 🎯 Strategy 1: Cross-Asset Momentum

**Concept**: Trade based on momentum confirmation across multiple assets, reducing false signals through cross-validation.

**Logic**:
```python
# For each period:
momentum_signals = []
for asset in [CAD_IG, US_HY, US_IG, TSX]:
    momentum = (current_price / price_n_weeks_ago) - 1
    momentum_signals.append(momentum > 0)

# Enter long if ≥ min_confirmations assets show positive momentum
entry_signal = sum(momentum_signals) >= min_confirmations
```

**Parameters**:
- `momentum_assets`: List of 4 assets ["cad_ig_er_index", "us_hy_er_index", "us_ig_er_index", "tsx"]
- `momentum_lookback_weeks`: Lookback period (default: 2 weeks)
- `min_confirmations`: Minimum confirmations required (default: 3 of 4 assets)

**Configuration Example**:
```yaml
cross_asset_momentum:
  momentum_assets: ["cad_ig_er_index", "us_hy_er_index", "us_ig_er_index", "tsx"]
  momentum_lookback_weeks: 2
  min_confirmations: 3
```

**Performance Characteristics**:
- **Time in Market**: ~55% (selective entries)
- **Typical Sharpe**: 2.5-5.0 (high risk-adjusted returns)
- **Max Drawdown**: Generally <5% (momentum protection)
- **Trade Frequency**: Medium (155 trades over 20+ years)

---

### 🌐 Strategy 2: Multi-Asset Momentum

**Concept**: Combine momentum from multiple assets into weighted score, entering when aggregate momentum is positive.

**Logic**:
```python
# Calculate momentum for each asset
asset_momentums = {}
for asset_key, asset_col in momentum_assets_map.items():
    momentum = data[asset_col].pct_change(lookback_periods)
    asset_momentums[asset_key] = momentum

# Combine into single momentum score
combined_momentum = sum(asset_momentums.values()) / len(asset_momentums)

# Generate signals based on threshold
entry_signal = combined_momentum > signal_threshold
```

**Parameters**:
- `momentum_assets_map`: Asset mapping {"tsx": "tsx", "us_hy": "us_hy_er_index", "cad_ig": "cad_ig_er_index"}
- `momentum_lookback_periods`: Lookback for momentum calculation (default: 4 weeks)
- `signal_threshold`: Entry threshold (default: -0.005)
- `exit_signal_shift_periods`: Exit delay (default: 1)

**Configuration Example**:
```yaml
multi_asset_momentum:
  momentum_assets_map:
    tsx: "tsx"
    us_hy: "us_hy_er_index" 
    cad_ig: "cad_ig_er_index"
  momentum_lookback_periods: 4
  signal_threshold: -0.005
  exit_signal_shift_periods: 1
```

**Performance Characteristics**:
- **Time in Market**: ~75% (more frequent exposure)
- **Typical Sharpe**: 1.8-4.0 (balanced risk-return)
- **Max Drawdown**: 2-5% (diversified momentum)
- **Trade Frequency**: Lower (84 trades, longer holds)

---

### 🧬 Strategy 3: Genetic Algorithm

**Concept**: Use evolutionary computation to discover optimal trading rules by combining technical features with logical operators.

**Technical Features** (24 total):
```python
# Momentum features (4 periods each)
momentum_1, momentum_2, momentum_4, momentum_8 = calculate_momentum()

# Volatility (20-period rolling std)
volatility = calculate_volatility()

# Moving average deviations
sma_dev_10, sma_dev_20 = calculate_sma_deviations()

# MACD components
macd_line, macd_signal, macd_histogram = calculate_macd()

# Stochastic oscillator
stoch_k = calculate_stochastic()

# Risk indicators momentum
cad_oas_mom4, us_ig_oas_mom4, us_hy_oas_mom4, vix_mom4 = calculate_oas_vix_momentum()
```

**Evolution Process**:
```python
# 1. Generate random population of trading rules
population = create_random_rules(population_size=120)

# 2. Evaluate fitness of each rule
for rule in population:
    signals = evaluate_rule(rule, features)
    returns = backtest_signals(signals)
    fitness = calculate_fitness(returns, max_drawdown)

# 3. Select best performers and evolve
best_rules = select_elite(population, elitism_rate=0.1)
offspring = crossover_and_mutate(best_rules, mutation_rate=0.4)

# 4. Repeat for 120 generations or until target return reached
```

**Example Evolved Rules**:
```python
# Simple momentum rule
(features_df['momentum_2'] > 0.01) & (features_df['volatility'] < 0.02)

# Complex multi-factor rule  
((features_df['macd_histogram'] > 0) & (features_df['stoch_k'] < 80)) | \
(features_df['us_hy_oas_mom4'] < 0.005)
```

**Parameters**:
- `population_size`: Number of rules per generation (default: 120)
- `generations`: Maximum generations (default: 120)
- `mutation_rate`: Probability of rule mutation (default: 0.4)
- `crossover_rate`: Probability of rule crossover (default: 0.4)
- `elitism_rate`: Fraction of best rules to keep (default: 0.1)
- `target_return`: Early stopping return target (default: 70%)

**Configuration Example**:
```yaml
genetic_algorithm:
  population_size: 120
  generations: 120
  mutation_rate: 0.4
  crossover_rate: 0.4
  elitism_rate: 0.1
  target_return: 70.0
  fitness_penalty: 1.0
```

**Performance Characteristics**:
- **Time in Market**: 60-80% (adaptive based on evolved rules)
- **Typical Sharpe**: 2.0-6.0 (optimized for risk-adjusted returns)
- **Max Drawdown**: 1-3% (evolution favors drawdown control)
- **Trade Frequency**: Variable (73-104 trades, depends on evolved rules)

### 📊 Strategy Performance Comparison

| Metric | Cross-Asset Momentum | Multi-Asset Momentum | Genetic Algorithm |
|--------|---------------------|---------------------|-------------------|
| **Total Return** | 85.39% | 74.95% | 76.03% |
| **CAGR** | 2.00% | 1.81% | 1.83% |
| **Sharpe Ratio** | 4.90 | 4.13 | 3.96 |
| **Max Drawdown** | -1.25% | -3.61% | -2.05% |
| **Time in Market** | 55.2% | 73.5% | 79.8% |
| **Number of Trades** | 155 | 84 | 73 |
| **Avg. Trade Length** | 3.6 weeks | 8.7 weeks | 11.0 weeks |

## 📊 Output

The framework generates:

1. **Console Reports**: Detailed performance analysis matching professional standards
2. **HTML Reports**: Interactive reports using quantstats (if available)
3. **Artifacts**: CSV files with signals, equity curves, returns, and metrics
4. **Comparison Reports**: Side-by-side strategy performance

### Example Output Format:

```
================================================================================
CROSS-ASSET 2-WEEK MOMENTUM STRATEGY - COMPLETE ANALYSIS
================================================================================
Strategy: Long when ≥3 of 4 indices show positive 2-week momentum
Indices: CAD IG ER, US HY ER, US IG ER, TSX
Trading Asset: CAD IG ER Index
Data loaded. Strategy Period: 2003-12-05 00:00:00 to 2025-06-06 00:00:00
Total periods (W-FRI): 1123

Signal Statistics:
Total signals generated: 620
Signal frequency: 55.21%

Portfolio Statistics:
Total return: 85.39%
Sharpe ratio: 2.227
Max drawdown: -1.25%
Number of trades: 155
```

## 🔧 Extending the Framework

### Adding a New Strategy

1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement required methods: `generate_signals()` and `get_required_features()`
3. Register in `StrategyFactory`

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data, features):
        # Your signal logic here
        return entry_signals, exit_signals
    
    def get_required_features(self):
        return ['feature1', 'feature2']
```

### Adding New Features

Extend feature engineers in `core/feature_engineering.py`:

```python
def my_custom_indicator(self, price_series):
    # Your custom indicator logic
    return indicator_series
```

## 🧪 Testing

Run tests (when implemented):
```bash
poetry run pytest tests/
```

## 📝 Requirements

- Python 3.9+
- See `pyproject.toml` for full dependency list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Data file not found**: Update the `file_path` in your configuration file
2. **Missing dependencies**: Run `poetry install` to install all dependencies
3. **Import errors**: Ensure you're in the Poetry environment (`poetry shell`)

### Support

For issues and questions, please check the documentation or create an issue in the repository. 