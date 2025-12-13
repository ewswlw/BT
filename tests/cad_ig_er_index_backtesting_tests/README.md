# Comprehensive Test Suite for cad_ig_er_index_backtesting

This directory contains a comprehensive test suite covering all components and edge cases of the cad_ig_er_index_backtesting module.

## Test Structure

### Core Component Tests

- **test_data_loader.py**: Tests for data loading and preprocessing
  - CSVDataProvider edge cases
  - MultiIndexCSVDataProvider edge cases
  - DataLoader filtering, quality checks, transformations
  - DataValidator utility methods
  - All missing data strategies
  - Outlier detection and handling
  - Gap detection
  - Data validation

- **test_config.py**: Tests for configuration management
  - DataConfig with all fields
  - PortfolioConfig
  - ReportingConfig
  - Config loading from YAML
  - Config creation from dictionaries
  - Default value handling

- **test_feature_engineering.py**: Tests for feature engineering
  - TechnicalFeatureEngineer
  - CrossAssetFeatureEngineer
  - MultiAssetFeatureEngineer
  - All feature types (momentum, volatility, SMA, MACD, RSI, Bollinger Bands, OAS)

- **test_portfolio.py**: Tests for portfolio management
  - PortfolioEngine backtesting
  - BacktestResult creation
  - Fees and slippage handling
  - Leverage handling
  - Edge cases (empty data, no signals, etc.)

- **test_metrics.py**: Tests for metrics calculation
  - Basic metrics
  - Metrics with benchmarks
  - Different frequencies
  - Edge cases (negative returns, zero returns, empty data)

- **test_strategies.py**: Tests for strategy implementations
  - BaseStrategy abstract class
  - StrategyFactory
  - Strategy creation and validation
  - Strategy integration with data

- **test_integration.py**: End-to-end integration tests
  - Complete pipeline workflows
  - Multi-index pipeline
  - Filtering integration
  - Quality checks integration
  - Edge cases and error handling

## Running Tests

### Run all tests:
```bash
pytest tests/cad_ig_er_index_backtesting/ -v
```

### Run specific test file:
```bash
pytest tests/cad_ig_er_index_backtesting/test_data_loader.py -v
```

### Run with coverage:
```bash
pytest tests/cad_ig_er_index_backtesting/ --cov=cad_ig_er_index_backtesting --cov-report=html
```

### Run specific test:
```bash
pytest tests/cad_ig_er_index_backtesting/test_data_loader.py::TestDataLoader::test_filter_by_asset_classes_include -v
```

## Test Coverage

The test suite covers:

1. **Data Loading**
   - Single-index CSV loading
   - Multi-index CSV loading
   - Date filtering
   - Empty file handling
   - Invalid file handling
   - Malformed data handling

2. **Data Filtering**
   - Asset class include/exclude
   - Column filtering within asset classes
   - Edge cases (empty filters, non-existent classes)

3. **Data Quality**
   - Missing data strategies (forward_fill, backward_fill, interpolate, drop, fill_value)
   - Outlier detection (IQR, z-score)
   - Outlier handling (flag, remove, clip)
   - Gap detection
   - Data validation (min/max)

4. **Data Transformation**
   - Resampling
   - Column renaming
   - Type conversion
   - Multi-index flattening

5. **Feature Engineering**
   - All technical indicators
   - Cross-asset features
   - Multi-asset features
   - Missing data handling

6. **Portfolio Management**
   - Basic backtesting
   - Fees and slippage
   - Leverage
   - Edge cases

7. **Metrics**
   - All metric calculations
   - Benchmark comparisons
   - Different frequencies

8. **Strategies**
   - Strategy creation
   - Signal generation
   - Integration with pipeline

9. **Integration**
   - End-to-end workflows
   - Configuration loading
   - Error handling

## Edge Cases Covered

- Empty data
- Missing columns
- Invalid dates
- Malformed CSV files
- Invalid configurations
- Mismatched indices
- Zero returns
- Negative returns
- All NaN columns
- Infinite values
- Outliers
- Data gaps
- Invalid date ranges
- Missing required features
- Empty filters
- Non-existent asset classes

## Fixtures

Common fixtures are provided in `conftest.py`:
- `sample_data`: Sample price data
- `multi_asset_data`: Multi-asset data
- `multi_index_data`: Multi-index DataFrame
- `temp_csv_file`: Temporary CSV file
- `temp_multi_index_csv`: Temporary multi-index CSV
- `sample_signals`: Sample entry/exit signals

## Notes

- All tests use temporary files that are cleaned up automatically
- Tests are designed to be independent and can run in any order
- Edge cases are thoroughly tested to ensure robust error handling
- Integration tests verify the complete pipeline works end-to-end

