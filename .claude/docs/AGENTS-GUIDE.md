# Agents Guide

Agents are specialized AI personas with deep expertise in specific domains. This guide explains how to work effectively with agents in the BT backtesting framework.

## Table of Contents

- [Overview](#overview)
- [Available Agents](#available-agents)
- [Using Agents Effectively](#using-agents-effectively)
- [Agent Comparison](#agent-comparison)
- [Best Practices](#best-practices)
- [Creating Custom Agents](#creating-custom-agents)

## Overview

### What are Agents?

Agents are persistent specialized personas that bring domain expertise to extended conversations. Unlike slash commands (which activate for a single task), agents maintain their specialty throughout the conversation.

### Agents vs Commands

| Feature | Agents | Commands |
|---------|--------|----------|
| **Duration** | Persistent conversation | Single task |
| **Use Case** | Extended work session | Specific workflow |
| **Context** | Maintains specialty | Activates and completes |
| **Best For** | Complex problems | Defined procedures |

### How to Use Agents

Simply reference the agent in your conversation:

```
User: I need help from the quantitative analyst agent.
      Please analyze this strategy implementation.

Claude: [Responds with quantitative analyst expertise throughout the conversation]
```

## Available Agents

### 1. Quantitative Analyst Agent

**File**: `.claude/agents/quantitative-analyst.md`

**Specialization**: Quantitative trading strategy development and analysis

**Core Competencies**:
- Strategy development and design
- Statistical analysis and hypothesis testing
- Performance evaluation and metrics
- Factor models and portfolio construction
- Risk-adjusted optimization

**Knowledge Domains**:
- Modern Portfolio Theory
- Factor investing (Fama-French, momentum, value)
- Time series econometrics
- Market microstructure
- Derivatives pricing
- Python scientific stack (pandas, numpy, scipy, statsmodels)

**Communication Style**:
- Precise and technical
- Quantitative with specific numbers
- Honest about limitations
- Evidence-based conclusions
- Clear about uncertainty

**When to Use**:
- Designing new trading strategies
- Analyzing backtest results
- Evaluating statistical significance
- Understanding factor exposures
- Optimizing portfolio construction

**Example Interaction**:
```
User: @quantitative-analyst

I have a momentum strategy that buys the top 10% of stocks by 12-month return
and shorts the bottom 10%, rebalanced monthly. The backtest Sharpe is 1.8.

Is this statistically significant? What are the main risks?

Agent Response:
Let me analyze the statistical significance and risks...

Statistical Significance:
- Sharpe ratio of 1.8 is strong, but we need more context
- What's the sample period? (Need at least 3-5 years for robustness)
- Number of rebalances: If monthly over 5 years = 60 observations
- T-statistic: 1.8 * sqrt(60) ≈ 13.9 (highly significant if returns are i.i.d.)
- However, momentum returns show autocorrelation, so effective N is lower

Key Risks:
1. Momentum crashes: Historical 20-30% drawdowns in momentum factors
2. Crowding: Momentum is well-known, may be crowded
3. Transaction costs: Monthly rebalance with 20% turnover could be costly
4. Short side: Short availability and costs could impact returns

[Continues with detailed analysis...]
```

**Key Principles**:
- Occam's Razor: Simpler is better
- Out-of-sample validation required
- Economic logic must make sense
- Risk-adjusted returns matter
- Practical implementability essential

---

### 2. Risk Manager Agent

**File**: `.claude/agents/risk-manager.md`

**Specialization**: Risk assessment, monitoring, and mitigation

**Core Competencies**:
- Risk identification and measurement
- VaR and stress testing
- Risk monitoring and alerting
- Risk mitigation strategies
- Limit setting and enforcement

**Risk Categories**:
- Market risk (directional, volatility, interest rate)
- Credit risk
- Liquidity risk
- Operational risk
- Model risk

**Communication Style**:
- Direct and cautious
- Quantitative with probabilities
- Errs on side of safety
- Proactive about risks
- Objective and fact-based

**When to Use**:
- Before deploying new strategies
- Setting risk limits
- Stress testing portfolios
- Evaluating new risks
- Creating monitoring frameworks

**Example Interaction**:
```
User: @risk-manager

We're about to deploy a momentum strategy with these characteristics:
- Max position: 5% per stock
- Target leverage: 1.5x
- Typical holding period: 1 month
- Max drawdown in backtest: -22%

What risk limits should we set?

Agent Response:
RISK ASSESSMENT: MEDIUM-HIGH

Let me recommend a comprehensive risk framework...

Position Limits:
✓ Max 5% per position is reasonable
⚠️ Also set: Max 25% per sector
⚠️ Also set: Max 10% per factor exposure

Portfolio Limits:
⚠️ Target leverage 1.5x: Set hard stop at 1.7x
✓ Consider volatility targeting instead

Risk Metrics:
- Daily VaR (95%): Set limit at 2% of portfolio
- Max Drawdown: Set kill switch at -25% (vs -22% backtest)
- Volatility: Set limit at 20% annualized

Monitoring (Daily):
- Current drawdown from peak
- Position concentration (HHI)
- Leverage level
- VaR estimate

Critical Alerts:
🛑 STOP TRADING if:
- Drawdown exceeds -20%
- Single position > 8%
- Daily loss > 5%
- Data quality issues

[Continues with detailed risk framework...]
```

**Key Principles**:
- Capital preservation first
- Measure what matters
- Set hard limits
- Monitor continuously
- Plan for worst case

---

### 3. Code Reviewer Agent

**File**: `.claude/agents/code-reviewer.md`

**Specialization**: Code quality, correctness, and best practices

**Review Priorities** (in order):
1. Correctness (does it work right?)
2. Safety (can it cause losses?)
3. Performance (is it efficient?)
4. Maintainability (can others understand it?)
5. Style (does it follow conventions?)

**Key Review Areas**:
- Correctness & Logic (especially look-ahead bias)
- Financial correctness (returns, corporate actions, costs)
- Data integrity
- Performance & efficiency
- Error handling
- Testing coverage
- Code quality
- Security

**Communication Style**:
- Constructive and educational
- Explains the "why"
- Provides specific examples
- Balances idealism with pragmatism
- Thorough but not nitpicky

**When to Use**:
- Before merging code
- After implementing features
- When debugging
- For production readiness
- During refactoring

**Example Interaction**:
```
User: @code-reviewer

Please review this momentum strategy code:

```python
def calculate_momentum(prices):
    returns = prices.pct_change(126)
    signals = returns > returns.median()
    return signals
```

Agent Response:
SEVERITY: CRITICAL - Must Fix

**Critical Issues:**

1. ⚠️ LOOK-AHEAD BIAS (Line 3)
   - Problem: `returns.median()` uses future data
   - Impact: Backtest will show inflated returns
   - Fix: Use rolling or shift
   ```python
   signals = returns > returns.shift(1).rolling(252).median()
   ```

2. ⚠️ MISSING DATA HANDLING (Line 2)
   - Problem: First 126 rows will be NaN
   - Impact: Strategy won't trade initially
   - Fix: Handle explicitly
   ```python
   returns = prices.pct_change(126)
   if returns.isna().any():
       returns = returns.fillna(0)  # Or other strategy
   ```

**Suggestions:**

1. Add type hints
   ```python
   def calculate_momentum(prices: pd.Series) -> pd.Series:
   ```

2. Add documentation
   ```python
   """
   Calculate momentum signals based on 126-day returns.

   Args:
       prices: Price series

   Returns:
       Boolean series: True for long positions
   """
   ```

**Verdict:** 🔄 CHANGES REQUESTED - Critical bias issue must be fixed

[Continues with detailed review...]
```

**Key Principles**:
- Correctness first, always
- Be thorough but pragmatic
- Explain reasoning
- Provide examples
- Focus on shipping working code

---

### 4. Data Engineer Agent

**File**: `.claude/agents/data-engineer.md`

**Specialization**: Data pipelines, ETL, and data infrastructure

**Core Competencies**:
- Pipeline architecture and design
- Data quality frameworks
- ETL/ELT processes
- Storage and retrieval optimization
- Monitoring and alerting

**Technical Expertise**:
- Python (pandas, polars, dask, airflow)
- SQL (PostgreSQL, TimescaleDB, ClickHouse)
- Formats (Parquet, Arrow, HDF5)
- Orchestration (Airflow, Prefect, Dagster)

**Communication Style**:
- Practical and solution-oriented
- Focuses on reliability and quality
- Emphasizes monitoring
- Data-driven decisions

**When to Use**:
- Building data pipelines
- Debugging data issues
- Improving data quality
- Optimizing performance
- Setting up monitoring

**Example Interaction**:
```
User: @data-engineer

I'm getting missing data for some stocks in my backtests.
Sometimes there are gaps of 5-10 days.

Agent Response:
Let's diagnose and fix this systematically.

**Step 1: Understand the Issue**

First, let's check the extent:
```python
# Check for missing dates
def find_missing_dates(df, symbol):
    date_range = pd.date_range(df.index.min(), df.index.max(), freq='B')
    missing = date_range.difference(df.index)
    return missing

# Run for each symbol
for symbol in symbols:
    missing = find_missing_dates(data[symbol], symbol)
    if len(missing) > 0:
        print(f"{symbol}: {len(missing)} missing days")
```

**Step 2: Identify Root Cause**

Possible causes:
1. Data source issue (vendor gaps)
2. Download failures (network errors)
3. Halted/suspended trading
4. Delisted stocks
5. Pipeline bugs

**Step 3: Implement Solution**

```python
# Add data quality check to pipeline
def validate_completeness(df, symbol):
    expected_days = get_trading_days(df.index.min(), df.index.max())
    actual_days = len(df)
    completeness = actual_days / len(expected_days)

    if completeness < 0.95:
        logger.warning(f"{symbol}: Only {completeness:.1%} complete")
        # Attempt backfill
        backfill_missing_data(symbol, missing_dates)

    return completeness
```

[Continues with comprehensive solution...]
```

**Key Principles**:
- Data quality is paramount
- Make pipelines idempotent
- Log everything
- Handle errors gracefully
- Monitor continuously

---

## Agent Comparison

### When to Use Which Agent

| Situation | Best Agent |
|-----------|------------|
| Designing a new strategy | Quantitative Analyst |
| Evaluating factor exposures | Quantitative Analyst |
| Setting risk limits | Risk Manager |
| Stress testing portfolio | Risk Manager |
| Reviewing code for bugs | Code Reviewer |
| Checking for look-ahead bias | Code Reviewer |
| Building data pipeline | Data Engineer |
| Fixing data quality issues | Data Engineer |

### Agent Collaboration

Agents can work together:

```
Phase 1: Strategy Design
@quantitative-analyst: Design momentum strategy

Phase 2: Implementation
[Implement with Claude's help]

Phase 3: Code Review
@code-reviewer: Review implementation

Phase 4: Risk Assessment
@risk-manager: Set risk limits

Phase 5: Data Validation
@data-engineer: Ensure data quality
```

## Using Agents Effectively

### 1. Set Clear Context

```
# Good
@quantitative-analyst

I have 5 years of daily data for S&P 500 stocks.
I want to develop a momentum strategy based on 12-month returns.
Target Sharpe ratio > 1.5.

# Less Good
@quantitative-analyst
Help me with a strategy.
```

### 2. Ask Specific Questions

```
# Good
@risk-manager
What VaR limit should I set for a strategy with 15% annualized volatility
and -20% max drawdown in backtest?

# Less Good
@risk-manager
What do you think about risk?
```

### 3. Provide Relevant Data

```
# Good
@code-reviewer
Review this function. Backtest shows Sharpe of 2.5 which seems too high.
[Provide code]

# Less Good
@code-reviewer
Something is wrong with my code.
```

### 4. Iterate and Refine

```
@quantitative-analyst
Initial question...

[Review response]

Follow-up based on what I learned...

[Continue iteration]
```

## Best Practices

### DO:
✅ Reference agents explicitly
✅ Provide full context upfront
✅ Ask specific, focused questions
✅ Share relevant code/data
✅ Iterate based on feedback
✅ Combine agents for complex tasks
✅ Follow agent recommendations

### DON'T:
❌ Switch agents mid-task unnecessarily
❌ Provide insufficient context
❌ Ask vague questions
❌ Ignore agent warnings
❌ Skip agent recommendations
❌ Use wrong agent for task

## Creating Custom Agents

### Template

Create `.claude/agents/my-agent.md`:

```markdown
# My Custom Agent

## Agent Identity
**Name**: [Agent Name]
**Specialization**: [Specific domain]
**Model**: claude-sonnet-4-5
**Priority**: [What matters most]

## Core Mission
[One sentence: What is this agent's primary purpose?]

## Core Competencies

### [Competency 1]
[Description]

### [Competency 2]
[Description]

## Knowledge Domains
- Domain 1
- Domain 2

## Behavioral Guidelines

### Approach
1. [How agent approaches problems]
2. [Decision-making framework]

### Communication Style
- [How agent communicates]
- [Tone and style]

### Quality Standards
- [What standards agent upholds]

## Task Workflows

### [Workflow Name]
```
Steps...
```

## Output Standards
[What outputs look like]

## Key Principles
1. [Principle 1]
2. [Principle 2]

Remember: [Key reminder for this agent]
```

## Summary

Agents provide:
- ✅ **Deep expertise** in specific domains
- ✅ **Persistent specialization** across conversations
- ✅ **Consistent quality standards**
- ✅ **Domain-specific workflows**
- ✅ **Professional guidance**

Use agents to bring specialized expertise to your work!

---

**Return to**: [README.md](./README.md) for overview.
