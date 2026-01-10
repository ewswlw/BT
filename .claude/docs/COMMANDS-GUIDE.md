# Commands Guide

Slash commands activate specialized Claude behaviors for specific tasks. This guide covers all available commands and how to use them effectively.

## Table of Contents

- [Overview](#overview)
- [Available Commands](#available-commands)
- [Usage Examples](#usage-examples)
- [Creating Custom Commands](#creating-custom-commands)
- [Best Practices](#best-practices)

## Overview

### What are Slash Commands?

Slash commands are special triggers that activate specialized Claude personas and workflows. When you use a slash command, Claude receives additional context and instructions tailored to that specific task.

### How They Work

```
User types: /backtest
         ↓
Claude loads: .claude/commands/backtest.md
         ↓
Claude responds with: Specialized backtesting expertise
```

### Command Format

Commands are markdown files that define:
- Role and responsibilities
- Specialized knowledge
- Workflow procedures
- Output formats
- Guidelines and best practices

## Available Commands

### /backtest

**Purpose**: Run comprehensive backtesting analysis

**Use When**:
- Running strategy backtests
- Validating strategy implementations
- Analyzing historical performance
- Generating performance reports

**Key Features**:
- Pre-backtest validation (data quality, parameters)
- Execution monitoring
- Comprehensive performance metrics
- Risk analysis
- Benchmark comparison
- Optimization suggestions

**Example Usage**:
```
/backtest

Please run a backtest for the momentum strategy with these parameters:
- Lookback: 126 days (6 months)
- Rebalance frequency: Monthly
- Universe: S&P 500
- Start date: 2020-01-01
- End date: 2023-12-31

Include transaction costs of 10 bps per trade.
```

**Expected Output**:
- Performance metrics (Sharpe, Sortino, Max DD)
- Equity curve and drawdown chart
- Monthly/yearly return breakdown
- Risk analysis
- Benchmark comparison
- Recommendations

---

### /analyze-strategy

**Purpose**: Deep dive analysis of trading strategies

**Use When**:
- Evaluating strategy performance
- Comparing multiple strategies
- Identifying strategy weaknesses
- Assessing statistical significance
- Understanding factor exposures

**Key Features**:
- Statistical analysis (significance tests, distribution analysis)
- Risk decomposition
- Factor attribution
- Regime analysis
- Overfitting detection
- Implementation quality review

**Example Usage**:
```
/analyze-strategy

Analyze the momentum strategy results from the last backtest.
Focus on:
1. Statistical significance of returns
2. Performance across different market regimes
3. Factor exposures (momentum, value, size)
4. Concentration risks
5. Potential improvements
```

**Expected Output**:
- Executive summary
- Detailed performance breakdown
- Statistical significance tests
- Factor analysis
- Risk assessment
- Improvement recommendations

---

### /optimize-params

**Purpose**: Parameter optimization with robustness validation

**Use When**:
- Tuning strategy parameters
- Finding optimal parameter ranges
- Testing parameter sensitivity
- Validating robustness

**Key Features**:
- Multiple optimization methods (grid, random, genetic, Bayesian)
- In-sample / out-of-sample splitting
- Walk-forward optimization
- Sensitivity analysis
- Overfitting detection
- Robustness testing

**Example Usage**:
```
/optimize-params

Optimize the moving average crossover strategy:
- Fast MA: Test 10 to 100 days (step 10)
- Slow MA: Test 50 to 300 days (step 25)
- Objective: Maximize Sharpe Ratio
- Method: Grid search with walk-forward validation
- In-sample: 2015-2020
- Out-of-sample: 2021-2023
```

**Expected Output**:
- Optimal parameters found
- In-sample vs out-of-sample performance
- Parameter sensitivity heatmap
- Stability analysis
- Degradation metrics
- Recommendations

---

### /review-code

**Purpose**: Comprehensive code review for quantitative systems

**Use When**:
- Before merging code
- After implementing new features
- When debugging issues
- During refactoring
- For production readiness checks

**Key Features**:
- Correctness verification (no look-ahead bias, proper calculations)
- Financial correctness (returns, corporate actions, costs)
- Data integrity checks
- Performance analysis
- Error handling review
- Security assessment
- Testing coverage

**Example Usage**:
```
/review-code

Please review the strategy implementation in:
cad_ig_er_index_backtesting/strategies/momentum.py

Focus on:
1. Correctness (especially look-ahead bias)
2. Data handling
3. Performance efficiency
4. Code quality and maintainability
```

**Expected Output**:
- Critical issues (must fix)
- High priority suggestions
- Medium/low priority improvements
- Positive observations
- Verdict (approve/request changes)

---

### /risk-analysis

**Purpose**: Comprehensive risk assessment

**Use When**:
- Before deploying strategies
- During portfolio reviews
- After significant market moves
- For regulatory reporting
- When assessing new risks

**Key Features**:
- Market risk metrics (VaR, CVaR, Greeks)
- Strategy-specific risks
- Operational risk assessment
- Liquidity analysis
- Stress testing
- Concentration risk
- Risk limit recommendations

**Example Usage**:
```
/risk-analysis

Perform a comprehensive risk analysis on the momentum portfolio:
1. Calculate VaR (95% and 99%)
2. Stress test against 2008 financial crisis scenario
3. Analyze concentration by sector and factor
4. Assess liquidity for positions
5. Recommend risk limits
```

**Expected Output**:
- Overall risk rating
- Key risk metrics
- Stress test results
- Concentration analysis
- Risk warnings
- Recommended limits
- Monitoring dashboard

---

### /data-pipeline

**Purpose**: Data pipeline management and troubleshooting

**Use When**:
- Building new data pipelines
- Debugging data issues
- Improving data quality
- Optimizing pipeline performance
- Setting up monitoring

**Key Features**:
- Pipeline architecture design
- Data quality frameworks
- ETL best practices
- Error handling strategies
- Performance optimization
- Monitoring and alerting

**Example Usage**:
```
/data-pipeline

Help me build a pipeline to ingest daily price data:
- Sources: Yahoo Finance and Alpha Vantage
- Data: OHLCV for S&P 500 constituents
- Validation: Check for missing data, outliers
- Storage: Parquet format, partitioned by date
- Schedule: Daily at 6 PM after market close
```

**Expected Output**:
- Pipeline architecture diagram
- Implementation code
- Data quality checks
- Error handling logic
- Monitoring setup
- Testing strategy

---

## Usage Examples

### Example 1: Full Strategy Development Workflow

```
# Step 1: Understand codebase
/review-code
Show me an overview of the backtesting framework structure.

# Step 2: Implement strategy
Can you help me implement a simple momentum strategy?
[Work with Claude to write code]

# Step 3: Review implementation
/review-code
Review the momentum.py file I just created.

# Step 4: Run backtest
/backtest
Run a backtest on the momentum strategy with standard parameters.

# Step 5: Analyze results
/analyze-strategy
Analyze the backtest results and identify potential improvements.

# Step 6: Optimize
/optimize-params
Optimize the lookback period for best risk-adjusted returns.

# Step 7: Risk assessment
/risk-analysis
Perform comprehensive risk analysis before deployment.
```

### Example 2: Debugging Data Issues

```
/data-pipeline
I'm seeing missing data for some stocks in my backtest.
Help me debug the data pipeline and add better quality checks.

[Claude provides diagnostic steps and solutions]
```

### Example 3: Quick Code Review

```
/review-code
Quick review of the changes in my last commit.
Focus on correctness and any obvious issues.
```

## Creating Custom Commands

### Step 1: Create Command File

Create a new file in `.claude/commands/`:

```bash
touch .claude/commands/my-command.md
```

### Step 2: Define Command Structure

```markdown
# /my-command - Brief Description

You are a specialized assistant for [specific purpose].

## Your Role
[Define the primary role and mission]

## Context
[Provide relevant context about the project]

## Key Responsibilities

### 1. [First Responsibility]
[Details...]

### 2. [Second Responsibility]
[Details...]

## Important Guidelines
1. [Guideline 1]
2. [Guideline 2]

## Output Format
[Specify expected output format]

## Example Workflow
```
[Step-by-step process]
```

Remember: [Key principle or reminder]
```

### Step 3: Test Your Command

```
/my-command
[Test with a typical use case]
```

### Example: Custom Performance Report Command

```markdown
# /report - Generate Performance Report

You are a performance reporting specialist.

## Your Role
Generate comprehensive, professional performance reports for trading strategies.

## Report Components

### 1. Executive Summary
- 2-3 sentence overview
- Key highlights
- Major concerns

### 2. Performance Metrics
- Returns (total, CAGR, monthly, yearly)
- Risk metrics (volatility, Sharpe, Sortino, Max DD)
- Trading metrics (turnover, win rate)

### 3. Visualizations
- Equity curve
- Drawdown chart
- Monthly returns heatmap
- Rolling Sharpe ratio

### 4. Analysis
- Strengths and weaknesses
- Market regime performance
- Factor attribution

### 5. Recommendations
- Actionable improvements
- Risk management suggestions

## Output Format
Generate a markdown report suitable for stakeholders.

## Guidelines
1. Be concise but comprehensive
2. Use tables and charts
3. Highlight key insights
4. Provide context for numbers
5. Be honest about limitations

Remember: Reports should be professional and actionable.
```

## Best Practices

### 1. Choose the Right Command

| Task | Command |
|------|---------|
| Run backtest | `/backtest` |
| Analyze results | `/analyze-strategy` |
| Tune parameters | `/optimize-params` |
| Review code | `/review-code` |
| Assess risk | `/risk-analysis` |
| Fix data issues | `/data-pipeline` |

### 2. Provide Clear Context

```
# Good
/backtest
Run a backtest on momentum strategy with 126-day lookback,
monthly rebalancing, S&P 500 universe, 2020-2023.

# Less Good
/backtest
Run a backtest
```

### 3. Be Specific About Requirements

```
# Good
/analyze-strategy
Focus on statistical significance and factor exposures.
Compare to SPY benchmark.

# Less Good
/analyze-strategy
Analyze this strategy
```

### 4. Combine Commands in Workflow

Use multiple commands for complex tasks:
```
/review-code → /backtest → /analyze-strategy → /optimize-params → /risk-analysis
```

### 5. Iterate Based on Results

```
/backtest
[Review results]

/analyze-strategy
[Identify issues]

[Make improvements]

/backtest
[Test again]
```

## Command Cheat Sheet

| Command | Use For | Key Output |
|---------|---------|------------|
| `/backtest` | Running strategy tests | Performance metrics, equity curve |
| `/analyze-strategy` | Deep performance analysis | Statistical tests, factor analysis |
| `/optimize-params` | Parameter tuning | Optimal parameters, sensitivity analysis |
| `/review-code` | Code quality checks | Issues found, suggestions |
| `/risk-analysis` | Risk assessment | VaR, stress tests, limits |
| `/data-pipeline` | Data infrastructure | Pipeline design, quality checks |

## Troubleshooting

### Command Not Recognized

**Problem**: `/my-command` not found

**Solutions**:
1. Check file exists: `.claude/commands/my-command.md`
2. Verify filename matches command (no `/` in filename)
3. Check markdown syntax is valid
4. Ensure commands_path is configured
5. Restart Claude session

### Command Doesn't Behave as Expected

**Problem**: Command active but not working correctly

**Solutions**:
1. Review command markdown file
2. Check for markdown syntax errors
3. Verify instructions are clear and specific
4. Test with different phrasings
5. Add more detailed guidelines

### Command is Too Verbose

**Problem**: Command provides too much unnecessary detail

**Solutions**:
1. Add guidelines for conciseness
2. Specify output format more precisely
3. Use bullet points instead of paragraphs
4. Set word limits in command definition

## Summary

Slash commands provide:
- ✅ **Specialized expertise** for specific tasks
- ✅ **Consistent workflows** and outputs
- ✅ **Context-aware assistance**
- ✅ **Faster task completion**
- ✅ **Better quality results**

Master these commands to maximize your productivity with Claude!

---

**Next**: Read [AGENTS-GUIDE.md](./AGENTS-GUIDE.md) to learn about specialized agents.
