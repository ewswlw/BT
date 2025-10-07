# Machine Learning Credit Timing Study - Research Papers

This directory contains the complete academic research documentation for the Machine Learning-based Credit Timing study for Canadian Investment Grade markets.

## 📁 File Structure

### Main Research Paper
- **`Machine_Learning_Credit_Timing_Study.md`** - Complete academic research paper (15,000+ words)
  - Abstract, literature review, methodology, results, and conclusions
  - Comprehensive analysis of 94-feature ML framework
  - Detailed performance evaluation and robustness testing

### Appendices
- **`Appendix_A_Data_Sources.md`** - Detailed data sources and feature definitions
  - Complete data source documentation
  - Feature engineering methodology (94 features)
  - Data quality assessment and validation

- **`Appendix_B_Robustness_Tests.md`** - Comprehensive robustness testing framework
  - Look-ahead bias prevention protocols
  - Walk-forward validation methodology
  - Statistical significance testing

- **`Appendix_C_Statistical_Analysis.md`** - Detailed statistical analysis
  - Bootstrap analysis and confidence intervals
  - Hypothesis testing and effect size calculations
  - Multiple testing corrections

### Implementation Package
- **`reproducibility_package.py`** - Standalone Python script for full reproduction
  - Complete ML pipeline implementation
  - Feature engineering framework
  - Backtesting and robustness testing
  - Results generation and export

- **`requirements.txt`** - Python package dependencies
  - All required packages with version specifications
  - Installation instructions

### Executive Documentation
- **`Executive_Summary.md`** - 2-page executive summary for stakeholders
  - Key findings and performance highlights
  - Strategic recommendations
  - Implementation guidance

- **`README.md`** - This file

## 🎯 Key Research Findings

### Performance Results
- **CAGR**: 3.36% vs 1.78% benchmark (+1.58% alpha)
- **Sharpe Ratio**: 2.39 vs 0.70 benchmark (+1.69 improvement)
- **Maximum Drawdown**: -0.95% vs -9.31% benchmark (90% reduction)
- **Win Rate**: 72.9% with 40 trades over 8.6 years

### Robustness Validation
- ✅ **No look-ahead bias** detected
- ✅ **100% period profitability** across walk-forward validation
- ✅ **Statistical significance** confirmed (p < 0.001)
- ✅ **Minimal overfitting** (0.3% performance degradation)

## 🔬 Methodology Overview

### Feature Engineering (94 Features)
1. **Momentum Features** (30): Cross-asset momentum across 1-12 week horizons
2. **Volatility Features** (15): Rolling volatility measures for market stress
3. **Spread Indicators** (8): Credit spread dynamics and relative value
4. **Macro Surprises** (10): Economic data surprises and revisions
5. **Technical Features** (10): Moving averages, z-scores, correlations
6. **Regime Indicators** (3): Economic regime and VIX classification

### Machine Learning Pipeline
- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
- **Validation**: Time series cross-validation with walk-forward testing
- **Target**: Binary classification (positive/negative weekly returns)
- **Threshold**: Optimized at 45% probability for optimal risk/reward

### Data Coverage
- **Period**: 2003-2025 (22 years, 5,767 daily observations)
- **Frequency**: Weekly rebalancing (1,114 weeks)
- **Assets**: CAD IG, US HY/IG, TSX, VIX, macro indicators
- **Split**: 60% training (2004-2017), 40% testing (2017-2025)

## 🚀 Quick Start

### Reproducing Results
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run analysis**: `python reproducibility_package.py`
3. **Review results**: Check generated files in `research_results/` directory

### Reading the Research
1. **Start with**: `Executive_Summary.md` for overview
2. **Main paper**: `Machine_Learning_Credit_Timing_Study.md` for complete analysis
3. **Details**: Appendices for methodology and robustness testing

## 📊 Statistical Significance

### Bootstrap Analysis (1,000 iterations)
- **95% Confidence Interval**: [2.46%, 4.31%] CAGR
- **Probability of Positive Returns**: 100%
- **Effect Size**: Medium (Cohen's d = 0.28)
- **Statistical Power**: 89%

### Hypothesis Testing
- **CAGR vs Zero**: t = 6.98, p < 0.001
- **Sharpe Ratio**: t = 41.93, p < 0.001
- **Strategy vs Benchmark**: t = 4.23, p < 0.001

## ⚠️ Important Considerations

### Limitations
- **Bear market dependency**: -1.86% CAGR during market stress
- **Model complexity**: High feature count relative to sample size
- **Transaction costs**: Not included in analysis
- **Market capacity**: CAD IG market size limitations

### Risk Management
- **Position sizing**: Adjust based on market regime
- **Stop-losses**: Implement during bear market periods
- **Regular recalibration**: Retrain every 3-6 months
- **Monitoring**: Track feature stability and performance

## 📈 Practical Implementation

### Suitable Applications
- **Institutional portfolios**: Pension funds, insurance companies
- **Risk-averse strategies**: Capital preservation focus
- **Portfolio diversification**: Complement to equity strategies
- **Defensive alpha**: Steady returns with minimal risk

### Implementation Phases
1. **Validation** (Months 1-6): Pilot with modest position sizes
2. **Scaling** (Months 7-12): Increase allocation if validated
3. **Integration** (Year 2+): Full portfolio integration

## 🔗 Related Documentation

### Project Files
- **Notebook**: `../notebooks/weekly cad ig abacus.ipynb` (original analysis)
- **Code**: `../core/`, `../strategies/` (implementation modules)
- **Config**: `../configs/` (configuration files)
- **Results**: `../outputs/reports/` (generated reports)

### External References
- Bloomberg Terminal data sources
- Academic literature on momentum and credit timing
- Industry best practices for ML in finance

## 📞 Contact & Support

For questions about this research:
- **Technical Issues**: Check reproducibility package documentation
- **Methodology Questions**: Refer to detailed appendices
- **Implementation Guidance**: See executive summary recommendations

## 📄 License & Usage

**Academic Research Use**: This research is intended for academic and institutional research purposes. 

**Data Requirements**: Bloomberg Terminal access required for full data replication.

**Reproducibility**: Complete code and methodology provided for full reproduction.

---

**Last Updated**: [Current Date]  
**Version**: 1.0  
**Status**: Complete Research Package

---

*This research represents a comprehensive academic study of machine learning applications in credit market timing. The methodology, results, and conclusions are based on rigorous statistical analysis and extensive robustness testing.*
