"""
Report Generator

Creates comprehensive HTML tearsheets and text comparison reports.
Inspired by comprehensive_strategy_comparison.txt format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import vectorbt as vbt
from datetime import datetime


class ReportGenerator:
    """
    Generates professional reports for TAA strategies.
    
    Outputs:
    - HTML tearsheets (using VectorBT/QuantStats)
    - Text comparison reports (matching comprehensive_strategy_comparison.txt)
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Base directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_html_tearsheet(
        self,
        portfolio: vbt.Portfolio,
        strategy_name: str,
        benchmark_returns: pd.Series
    ):
        """
        Generate HTML tearsheet using VectorBT.
        
        Args:
            portfolio: VectorBT Portfolio object
            strategy_name: Name of strategy
            benchmark_returns: Benchmark return series
        """
        html_path = self.output_dir / 'reports' / f"{strategy_name}_tearsheet.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating HTML tearsheet for {strategy_name}...")
        
        # Create plots
        fig = portfolio.plot(subplots=[
            'cum_returns',
            'drawdowns',
            'underwater'
        ])
        
        # Save to HTML
        fig.write_html(str(html_path))
        
        print(f"✓ HTML tearsheet saved: {html_path}")
        
    def generate_text_report(
        self,
        strategy_name: str,
        metrics: Dict,
        strategy_info: Dict,
        weights: pd.DataFrame
    ) -> str:
        """
        Generate comprehensive text report.
        
        Args:
            strategy_name: Name of strategy
            metrics: Dictionary of calculated metrics
            strategy_info: Strategy metadata
            weights: Weight DataFrame
            
        Returns:
            Formatted text report string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 100)
        report_lines.append(f"TACTICAL ASSET ALLOCATION BACKTEST REPORT: {strategy_name}")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Strategy Information
        report_lines.append("-" * 100)
        report_lines.append("STRATEGY INFORMATION")
        report_lines.append("-" * 100)
        for key, value in strategy_info.items():
            if key != 'rules':
                report_lines.append(f"{key:30s}: {value}")
        report_lines.append("")
        
        # Trading Rules
        if 'rules' in strategy_info:
            report_lines.append("-" * 100)
            report_lines.append("TRADING RULES")
            report_lines.append("-" * 100)
            report_lines.append(strategy_info['rules'])
            report_lines.append("")
        
        # VectorBT Portfolio Stats
        report_lines.append("-" * 100)
        report_lines.append("VECTORBT PORTFOLIO STATISTICS")
        report_lines.append("-" * 100)
        if 'vectorbt_stats' in metrics:
            for key, value in metrics['vectorbt_stats'].items():
                report_lines.append(f"{key:40s}: {value}")
        report_lines.append("")
        
        # Manual Calculations
        report_lines.append("-" * 100)
        report_lines.append("PERFORMANCE METRICS (MANUAL CALCULATIONS)")
        report_lines.append("-" * 100)
        
        if 'manual_calcs' in metrics:
            manual = metrics['manual_calcs']
            
            report_lines.append("\nSTRATEGY METRICS:")
            for key, value in manual['strategy'].items():
                report_lines.append(f"  {key:25s}: {value}")
            
            report_lines.append("\nBENCHMARK METRICS:")
            for key, value in manual['benchmark'].items():
                report_lines.append(f"  {key:25s}: {value}")
        
        report_lines.append("")
        
        # QuantStats-Style Comparison
        report_lines.append("-" * 100)
        report_lines.append("QUANTSTATS-STYLE COMPARISON")
        report_lines.append("-" * 100)
        if 'quantstats_style' in metrics:
            for key, value in metrics['quantstats_style'].items():
                report_lines.append(f"{key:40s}: {value}")
        report_lines.append("")
        
        # Recent Allocations
        report_lines.append("-" * 100)
        report_lines.append("LAST 10 MONTHLY ALLOCATIONS")
        report_lines.append("-" * 100)
        report_lines.append(weights.tail(10).to_string())
        report_lines.append("")
        
        report_lines.append("=" * 100)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)
    
    def save_text_report(self, report_text: str, strategy_name: str):
        """
        Save text report to file.
        
        Args:
            report_text: Formatted report string
            strategy_name: Name of strategy
        """
        report_path = self.output_dir / 'results' / f"{strategy_name}_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Text report saved: {report_path}")
    
    def generate_comparison_report(
        self,
        all_results: Dict[str, Dict]
    ) -> str:
        """
        Generate multi-strategy comparison report.
        
        Args:
            all_results: Dictionary of {strategy_name: metrics}
            
        Returns:
            Formatted comparison report
        """
        report_lines = []
        
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE STRATEGY COMPARISON")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Strategies: {len(all_results)}")
        report_lines.append("")
        
        # Build comparison table
        comparison_data = []
        
        for strat_name, result in all_results.items():
            if 'manual_calcs' in result['metrics']:
                manual = result['metrics']['manual_calcs']['strategy']
                
                row = {
                    'Strategy': strat_name,
                    'CAGR': manual.get('cagr', 'N/A'),
                    'Volatility': f"{manual.get('volatility', 0):.2%}",
                    'Sharpe': f"{manual.get('sharpe', 0):.2f}",
                    'Sortino': f"{manual.get('sortino', 0):.2f}",
                    'Max DD': f"{manual.get('max_drawdown', 0):.2%}",
                    'Skew': f"{manual.get('skewness', 0):.2f}",
                    'Kurt': f"{manual.get('kurtosis', 0):.2f}",
                    'Time in Market': manual.get('time_in_market', 'N/A'),
                }
                
                comparison_data.append(row)
        
        # Create DataFrame and format
        df = pd.DataFrame(comparison_data)
        
        report_lines.append("-" * 100)
        report_lines.append("STRATEGY COMPARISON TABLE")
        report_lines.append("-" * 100)
        report_lines.append(df.to_string(index=False))
        report_lines.append("")
        
        # Add individual strategy details
        for strat_name, result in all_results.items():
            report_lines.append("-" * 100)
            report_lines.append(f"STRATEGY: {strat_name}")
            report_lines.append("-" * 100)
            
            if 'strategy_info' in result:
                info = result['strategy_info']
                report_lines.append(f"Paper: {info.get('paper', 'N/A')}")
                report_lines.append(f"Period: {info.get('start_date', 'N/A')} to {info.get('end_date', 'N/A')}")
                report_lines.append(f"Rebalance: {info.get('rebalance_frequency', 'N/A')}")
            
            report_lines.append("")
        
        report_lines.append("=" * 100)
        report_lines.append("END OF COMPARISON REPORT")
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)
    
    def save_comparison_report(self, report_text: str):
        """
        Save comparison report to file.
        
        Args:
            report_text: Formatted comparison report
        """
        report_path = self.output_dir / 'results' / 'strategy_comparison.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Comparison report saved: {report_path}")

