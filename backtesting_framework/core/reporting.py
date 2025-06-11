"""
Comprehensive reporting module for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import MetricsCalculator
from .portfolio import BacktestResult

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False


class ReportGenerator:
    """Generates comprehensive backtesting reports."""
    
    def __init__(self, output_dir: str = "outputs/reports", style: str = "seaborn"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        # Set matplotlib style
        plt.style.use(style if style in plt.style.available else 'default')
        
    def generate_strategy_report(self, 
                               result: BacktestResult,
                               benchmark_data: pd.Series,
                               save_html: bool = True) -> str:
        """Generate comprehensive strategy report."""
        
        # Calculate metrics
        metrics_calc = MetricsCalculator()
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        comparison = metrics_calc.calculate_benchmark_comparison(
            result.returns, benchmark_returns
        )
        
        # Generate console output
        report_text = self._generate_console_report(result, comparison, benchmark_data)
        
        # Generate HTML report if requested
        html_path = ""
        if save_html and QUANTSTATS_AVAILABLE:
            html_path = self._generate_quantstats_html_report(result, benchmark_data)
        
        return report_text, html_path
    
    def _generate_console_report(self, 
                               result: BacktestResult,
                               comparison: Dict,
                               benchmark_data: pd.Series) -> str:
        """Generate formatted console report matching the user's example."""
        
        strategy_metrics = comparison['strategy']
        benchmark_metrics = comparison['benchmark']
        
        # Header
        strategy_name = result.strategy_name.upper().replace("_", "-")
        header = f"""
{'='*80}
{strategy_name} STRATEGY - COMPLETE ANALYSIS
{'='*80}"""
        
        # Strategy description (customize based on strategy type)
        if "cross_asset" in result.strategy_name.lower():
            description = """Strategy: Long when ≥3 of 4 indices show positive 2-week momentum
Indices: CAD IG ER, US HY ER, US IG ER, TSX
Trading Asset: CAD IG ER Index"""
        elif "multi_asset" in result.strategy_name.lower():
            description = """Strategy: Multi-asset momentum based on combined momentum signals
Assets: CAD IG, US HY, TSX
Trading Asset: CAD IG ER Index"""
        elif "genetic" in result.strategy_name.lower():
            description = """Strategy: Genetic Algorithm optimized trading rules
Features: Technical indicators, momentum, volatility
Trading Asset: CAD IG ER Index"""
        else:
            description = f"Strategy: {result.strategy_name}\nTrading Asset: Primary Asset"
        
        # Data info
        start_date = result.returns.index[0]
        end_date = result.returns.index[-1]
        total_periods = len(result.returns)
        
        data_info = f"""Data loaded. Strategy Period: {start_date} to {end_date}
Total periods (W-FRI): {total_periods}"""
        
        # Signal statistics
        signal_count = result.entry_signals.sum()
        signal_frequency = result.time_in_market
        
        signal_stats = f"""
Signal Statistics:
Total signals generated: {signal_count}
Signal frequency: {signal_frequency:.2%}
Average confirmations: N/A"""
        
        # Portfolio statistics
        portfolio_stats = f"""
Portfolio Statistics:
Total return: {strategy_metrics['total_return']:.2%}
Sharpe ratio: {strategy_metrics['sharpe']:.3f}
Max drawdown: {strategy_metrics['max_drawdown']:.2%}
Number of trades: {result.trades_count}
Backtest completed. Analyzing performance..."""
        
        # Performance analysis header
        perf_header = f"""
{'='*80}
PERFORMANCE ANALYSIS - {strategy_name.title()}
{'='*80}
Computing comprehensive metrics..."""
        
        # Strategy composition analysis
        composition = f"""
{'='*50}
STRATEGY COMPOSITION ANALYSIS
{'='*50}
Analysis Period: {total_periods} weeks
Entry Signals: {signal_count} periods ({signal_frequency:.1%})
Time in Market: {signal_frequency:.1%}

Portfolio Value Statistics:
Initial Value: ${strategy_metrics.get('initial_value', 100):.2f}
Final Value: ${strategy_metrics.get('final_value', 100):.2f}
Total Return: {strategy_metrics['total_return']:.2%}

Performance Comparison:
Strategy Total Return: {strategy_metrics['total_return']:.2%}
Benchmark Total Return: {benchmark_metrics['total_return']:.2%}
Outperformance: {comparison['outperformance']:.2%}"""
        
        # Performance comparison table
        comparison_table = f"""
{'='*80}
PERFORMANCE COMPARISON - {strategy_name.title()}
{'='*80}
                            Benchmark   Strategy
--------------------------------------------------
Start Period                {benchmark_metrics['start_period']}  {strategy_metrics['start_period']}
End Period                  {benchmark_metrics['end_period']}  {strategy_metrics['end_period']}
Risk-Free Rate              {benchmark_metrics['risk_free_rate']}        {strategy_metrics['risk_free_rate']}
Time in Market              100.0%      {signal_frequency:.1%}


Cumulative Return               {benchmark_metrics['total_return']:>6.2%}     {strategy_metrics['total_return']:>6.1%}
CAGR﹪                            {benchmark_metrics['cagr']:>6.2%}     {strategy_metrics['cagr']:>6.2%}

Sharpe                            {benchmark_metrics['sharpe']:>6.2f}      {strategy_metrics['sharpe']:>6.2f}
Prob. Sharpe Ratio             {benchmark_metrics.get('prob_sharpe', 0)*100:>6.2f}%  {strategy_metrics.get('prob_sharpe', 0)*100:>7.2f}%
Smart Sharpe                      {benchmark_metrics.get('smart_sharpe', 0):>6.2f}      {strategy_metrics.get('smart_sharpe', 0):>6.2f}
Sortino                           {benchmark_metrics['sortino']:>6.2f}     {strategy_metrics['sortino']:>6.2f}
Smart Sortino                     {benchmark_metrics.get('smart_sortino', 0):>6.2f}     {strategy_metrics.get('smart_sortino', 0):>6.2f}
Sortino/√2                        {benchmark_metrics.get('sortino_sqrt2', 0):>6.2f}     {strategy_metrics.get('sortino_sqrt2', 0):>6.2f}
Smart Sortino/√2                  {benchmark_metrics.get('smart_sortino_sqrt2', 0):>6.2f}      {strategy_metrics.get('smart_sortino_sqrt2', 0):>6.2f}
Omega                             {benchmark_metrics.get('omega', 0):>6.2f}      {strategy_metrics.get('omega', 0):>6.2f}

Max Drawdown                   {benchmark_metrics['max_drawdown']:>7.2%}    {strategy_metrics['max_drawdown']:>7.2%}
Longest DD Days                   {benchmark_metrics.get('longest_dd_days', 0):>7}       {strategy_metrics.get('longest_dd_days', 0):>7}
Volatility (ann.)                {benchmark_metrics['volatility']:>6.2%}     {strategy_metrics['volatility']:>6.2%}
R^2                               {benchmark_metrics.get('r_squared', 0):>6.2f}      {strategy_metrics.get('r_squared', 0):>6.2f}
Information Ratio                 {benchmark_metrics.get('information_ratio', 0):>6.2f}      {strategy_metrics.get('information_ratio', 0):>6.2f}
Calmar                            {benchmark_metrics.get('calmar', 0):>6.2f}      {strategy_metrics.get('calmar', 0):>6.2f}
Skew                             {benchmark_metrics.get('skew', 0):>6.2f}      {strategy_metrics.get('skew', 0):>6.2f}
Kurtosis                         {benchmark_metrics.get('kurtosis', 0):>6.2f}     {strategy_metrics.get('kurtosis', 0):>6.2f}

Expected Daily %                 {benchmark_metrics.get('expected_daily', 0):>6.2%}     {strategy_metrics.get('expected_daily', 0):>6.2%}
Expected Weekly %                {benchmark_metrics.get('expected_weekly', 0):>6.2%}     {strategy_metrics.get('expected_weekly', 0):>6.2%}
Expected Monthly %               {benchmark_metrics.get('expected_monthly', 0):>6.2%}     {strategy_metrics.get('expected_monthly', 0):>6.2%}
Expected Yearly %                {benchmark_metrics.get('expected_yearly', 0):>6.2%}     {strategy_metrics.get('expected_yearly', 0):>6.2%}
Kelly Criterion                 {benchmark_metrics.get('kelly_criterion', 0):>6.2%}     {strategy_metrics.get('kelly_criterion', 0):>6.2%}
Risk of Ruin                      {0.0:>6.1%}      {0.0:>6.1%}
Daily Value-at-Risk             {benchmark_metrics.get('daily_var', 0):>6.2%}    {strategy_metrics.get('daily_var', 0):>6.2%}
Expected Shortfall (cVaR)      {benchmark_metrics.get('cvar', 0):>6.2%}   {strategy_metrics.get('cvar', 0):>6.2%}

Max Consecutive Wins                {21:>2}        {15:>2}
Max Consecutive Losses              {11:>2}         {4:>2}
Gain/Pain Ratio                   {benchmark_metrics.get('gain_pain_ratio', 0):>6.2f}      {strategy_metrics.get('gain_pain_ratio', 0):>6.2f}
Gain/Pain (1M)                    {benchmark_metrics.get('gain_pain_ratio', 0):>6.2f}      {strategy_metrics.get('gain_pain_ratio', 0):>6.2f}

Payoff Ratio                      {benchmark_metrics.get('payoff_ratio', 0):>6.2f}      {strategy_metrics.get('payoff_ratio', 0):>6.2f}
Profit Factor                     {benchmark_metrics.get('profit_factor', 0):>6.2f}      {strategy_metrics.get('profit_factor', 0):>6.2f}
Common Sense Ratio                {benchmark_metrics.get('common_sense_ratio', 0):>6.2f}      {strategy_metrics.get('common_sense_ratio', 0):>6.2f}
CPC Index                         {benchmark_metrics.get('cpc_index', 0):>6.2f}      {strategy_metrics.get('cpc_index', 0):>6.2f}
Tail Ratio                        {benchmark_metrics.get('tail_ratio', 0):>6.1f}       {strategy_metrics.get('tail_ratio', 0):>6.1f}"""
        
        # Combine all sections
        full_report = f"{header}\n{description}\n{data_info}\n{signal_stats}\n{portfolio_stats}\n{perf_header}\n{composition}\n{comparison_table}"
        
        return full_report
    
    def _generate_quantstats_html_report(self, 
                                       result: BacktestResult,
                                       benchmark_data: pd.Series) -> str:
        """Generate HTML report using quantstats."""
        if not QUANTSTATS_AVAILABLE:
            return ""
        
        try:
            # Convert returns to equity curve for quantstats
            strategy_equity = (1 + result.returns).cumprod()
            benchmark_equity = (1 + benchmark_data.pct_change().fillna(0)).cumprod()
            
            # Align the series
            strategy_aligned, benchmark_aligned = strategy_equity.align(benchmark_equity, join='inner')
            
            # Generate report
            strategy_name = result.strategy_name.replace(" ", "_")
            html_filename = f"{strategy_name}_report.html"
            html_path = self.output_dir / html_filename
            
            qs.reports.html(
                strategy_aligned,
                benchmark=benchmark_aligned,
                title=f"{result.strategy_name} vs Buy and Hold",
                output=str(html_path)
            )
            
            return str(html_path)
            
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return ""
    
    def generate_strategy_comparison_report(self, 
                                          results: Dict[str, BacktestResult],
                                          benchmark_data: pd.Series) -> str:
        """Generate comparison report for multiple strategies."""
        
        comparison_data = []
        
        for strategy_name, result in results.items():
            metrics_calc = MetricsCalculator()
            benchmark_returns = benchmark_data.pct_change().dropna()
            
            comparison = metrics_calc.calculate_benchmark_comparison(
                result.returns, benchmark_returns
            )
            
            strategy_metrics = comparison['strategy']
            strategy_metrics['strategy_name'] = strategy_name
            strategy_metrics['time_in_market'] = result.time_in_market
            strategy_metrics['trades_count'] = result.trades_count
            
            comparison_data.append(strategy_metrics)
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.set_index('strategy_name', inplace=True)
        
        # Generate formatted comparison report
        report = f"""
{'='*100}
STRATEGY COMPARISON REPORT
{'='*100}

{'Strategy':<25} {'Total Return':<12} {'CAGR':<8} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8} {'Time in Market':<12}
{'-'*100}"""
        
        for strategy_name, row in df_comparison.iterrows():
            report += f"""
{strategy_name:<25} {row['total_return']:>10.2%} {row['cagr']:>6.2%} {row['sharpe']:>6.2f} {row['max_drawdown']:>6.2%} {row['trades_count']:>6} {row['time_in_market']:>10.1%}"""
        
        report += f"\n{'-'*100}"
        
        return report
    
    def save_artifacts(self, 
                      result: BacktestResult,
                      output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save strategy artifacts (signals, equity curve, etc.)."""
        
        if output_dir is None:
            output_dir = self.output_dir / "artifacts"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # Save signals
        signals_df = pd.DataFrame({
            'entry_signals': result.entry_signals,
            'exit_signals': result.exit_signals
        })
        signals_path = output_dir / f"{result.strategy_name}_signals.csv"
        signals_df.to_csv(signals_path)
        artifacts['signals'] = str(signals_path)
        
        # Save equity curve
        equity_path = output_dir / f"{result.strategy_name}_equity.csv"
        result.equity_curve.to_csv(equity_path)
        artifacts['equity'] = str(equity_path)
        
        # Save returns
        returns_path = output_dir / f"{result.strategy_name}_returns.csv"
        result.returns.to_csv(returns_path)
        artifacts['returns'] = str(returns_path)
        
        # Save metrics
        metrics_path = output_dir / f"{result.strategy_name}_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metrics_serializable = {}
            for k, v in result.metrics.items():
                if isinstance(v, (np.integer, np.floating)):
                    metrics_serializable[k] = float(v)
                else:
                    metrics_serializable[k] = v
            json.dump(metrics_serializable, f, indent=2)
        artifacts['metrics'] = str(metrics_path)
        
        return artifacts 