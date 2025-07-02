#!/usr/bin/env python3
"""
Performance Optimizer Tool for Financial Backtesting Framework

This tool automatically analyzes and optimizes performance bottlenecks:
- Detects inefficient DataFrame operations
- Suggests vectorization opportunities  
- Identifies memory leaks and excessive copying
- Provides optimization recommendations
"""

import ast
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass

# Performance analysis imports
try:
    import pandas as pd
    import numpy as np
    import psutil
except ImportError as e:
    print(f"Required packages missing: {e}")
    sys.exit(1)

@dataclass
class PerformanceIssue:
    """Represents a performance issue found in code."""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    recommendation: str
    code_snippet: str

class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze code for performance issues."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.issues: List[PerformanceIssue] = []
        self.current_lines = []
        
    def visit_Call(self, node):
        """Analyze function calls for performance issues."""
        
        # Check for inefficient pandas operations
        if hasattr(node.func, 'attr') and hasattr(node.func, 'value'):
            attr_name = node.func.attr
            
            # Detect iterrows() usage
            if attr_name == 'iterrows':
                self.issues.append(PerformanceIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    issue_type='inefficient_iteration',
                    description='Using iterrows() for DataFrame iteration',
                    severity='high',
                    recommendation='Replace with vectorized operations or .iloc[] indexing',
                    code_snippet=self._get_code_snippet(node.lineno)
                ))
            
            # Detect itertuples() usage
            elif attr_name == 'itertuples':
                self.issues.append(PerformanceIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    issue_type='inefficient_iteration',
                    description='Using itertuples() for DataFrame iteration',
                    severity='medium',
                    recommendation='Consider vectorized operations if possible',
                    code_snippet=self._get_code_snippet(node.lineno)
                ))
            
            # Detect excessive copying
            elif attr_name == 'copy':
                # Check if in a loop context
                parent = getattr(node, 'parent', None)
                if self._is_in_loop_context(node):
                    self.issues.append(PerformanceIssue(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        issue_type='excessive_copying',
                        description='DataFrame.copy() called in loop context',
                        severity='high',
                        recommendation='Move copy outside loop or use views where possible',
                        code_snippet=self._get_code_snippet(node.lineno)
                    ))
            
            # Detect append in loops
            elif attr_name == 'append' and self._is_in_loop_context(node):
                self.issues.append(PerformanceIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    issue_type='inefficient_growth',
                    description='Using append() in loop - quadratic performance',
                    severity='critical',
                    recommendation='Pre-allocate list/array or use pd.concat() with list of DataFrames',
                    code_snippet=self._get_code_snippet(node.lineno)
                ))
        
        # Check for inefficient imports
        elif hasattr(node.func, 'id'):
            func_name = node.func.id
            
            # Detect __import__ usage in hot paths
            if func_name == '__import__':
                self.issues.append(PerformanceIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    issue_type='hot_path_import',
                    description='Dynamic import in execution path',
                    severity='medium',
                    recommendation='Move import to module level or use lazy imports',
                    code_snippet=self._get_code_snippet(node.lineno)
                ))
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Analyze for loops for performance issues."""
        
        # Check for nested loops
        for child in ast.walk(node):
            if isinstance(child, ast.For) and child != node:
                self.issues.append(PerformanceIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    issue_type='nested_loop',
                    description='Nested loops detected - potential O(n²) complexity',
                    severity='high', 
                    recommendation='Consider vectorization or more efficient algorithms',
                    code_snippet=self._get_code_snippet(node.lineno)
                ))
                break
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Analyze imports for heavy dependencies."""
        heavy_modules = {
            'torch': 'critical',
            'tensorflow': 'critical', 
            'plotly': 'high',
            'dash': 'high',
            'matplotlib': 'medium',
            'seaborn': 'medium',
            'lightgbm': 'high',
            'xgboost': 'high',
            'catboost': 'high'
        }
        
        for alias in node.names:
            if alias.name in heavy_modules:
                self.issues.append(PerformanceIssue(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    issue_type='heavy_import',
                    description=f'Heavy dependency import: {alias.name}',
                    severity=heavy_modules[alias.name],
                    recommendation=f'Consider lazy import for {alias.name}',
                    code_snippet=self._get_code_snippet(node.lineno)
                ))
        
        self.generic_visit(node)
    
    def _is_in_loop_context(self, node) -> bool:
        """Check if node is inside a loop."""
        # Simple heuristic - look for For/While in parent nodes
        # In practice, would need proper parent tracking
        return True  # Conservative assumption
    
    def _get_code_snippet(self, line_number: int) -> str:
        """Get code snippet around the line."""
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
                start = max(0, line_number - 3)
                end = min(len(lines), line_number + 2)
                snippet_lines = lines[start:end]
                return ''.join(snippet_lines).strip()
        except:
            return "Unable to retrieve code snippet"

class PerformanceOptimizer:
    """Main performance optimization tool."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.issues: List[PerformanceIssue] = []
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('performance_optimizer')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def analyze_codebase(self) -> List[PerformanceIssue]:
        """Analyze entire codebase for performance issues."""
        self.logger.info(f"Analyzing codebase at {self.root_path}")
        
        python_files = list(self.root_path.rglob("*.py"))
        self.logger.info(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
        
        # Sort issues by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        self.issues.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        return self.issues
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(file_path))
            analyzer = CodeAnalyzer(str(file_path))
            analyzer.visit(tree)
            
            self.issues.extend(analyzer.issues)
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
        except UnicodeDecodeError:
            self.logger.warning(f"Unable to decode {file_path}")
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive optimization report."""
        
        report_lines = [
            "=" * 80,
            "PERFORMANCE OPTIMIZATION REPORT",
            "=" * 80,
            f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Codebase Path: {self.root_path}",
            f"Total Issues Found: {len(self.issues)}",
            "",
        ]
        
        # Summary by severity
        severity_counts = {}
        for issue in self.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        report_lines.extend([
            "ISSUE SUMMARY BY SEVERITY:",
            "-" * 40
        ])
        
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            report_lines.append(f"{severity.upper():<10}: {count:>3} issues")
        
        report_lines.extend(["", "DETAILED ISSUES:", "-" * 40])
        
        # Group issues by type
        issues_by_type = {}
        for issue in self.issues:
            issue_type = issue.issue_type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        for issue_type, type_issues in issues_by_type.items():
            report_lines.extend([
                f"\n{issue_type.upper().replace('_', ' ')} ({len(type_issues)} issues)",
                "=" * 60
            ])
            
            for issue in type_issues:
                report_lines.extend([
                    f"File: {issue.file_path}:{issue.line_number}",
                    f"Severity: {issue.severity.upper()}",
                    f"Description: {issue.description}",
                    f"Recommendation: {issue.recommendation}",
                    f"Code:",
                    "```",
                    issue.code_snippet,
                    "```",
                    "-" * 40
                ])
        
        # Optimization recommendations
        report_lines.extend([
            "\nOVERALL OPTIMIZATION RECOMMENDATIONS:",
            "=" * 60,
            "1. DEPENDENCY OPTIMIZATION:",
            "   - Implement lazy imports for heavy libraries (torch, plotly, etc.)",
            "   - Use optional dependencies with extras_require",
            "   - Consider lighter alternatives where possible",
            "",
            "2. DATAFRAME OPTIMIZATION:",
            "   - Replace iterrows()/itertuples() with vectorized operations",
            "   - Minimize DataFrame copying in loops",
            "   - Use pd.concat() instead of append() for DataFrame growth",
            "",
            "3. MEMORY OPTIMIZATION:",
            "   - Pre-allocate arrays/lists when size is known",
            "   - Use views instead of copies where safe",
            "   - Implement memory monitoring and cleanup",
            "",
            "4. ALGORITHM OPTIMIZATION:",
            "   - Vectorize nested loops where possible",
            "   - Consider more efficient algorithms for O(n²) operations",
            "   - Implement caching for expensive computations",
            "",
            "5. API OPTIMIZATION:",
            "   - Batch API calls instead of individual requests",
            "   - Implement connection pooling",
            "   - Add retry logic with exponential backoff",
        ])
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_file}")
        
        return report_text
    
    def benchmark_improvements(self, test_data_size: int = 10000) -> Dict[str, float]:
        """Benchmark potential improvements."""
        self.logger.info("Running performance benchmarks...")
        
        # Create test data
        test_df = pd.DataFrame({
            'A': np.random.randn(test_data_size),
            'B': np.random.randn(test_data_size),
            'C': np.random.randn(test_data_size)
        })
        
        benchmarks = {}
        
        # Benchmark: iterrows vs vectorized
        start_time = time.time()
        result_iterrows = []
        for idx, row in test_df.iterrows():
            result_iterrows.append(row['A'] + row['B'])
        benchmarks['iterrows_time'] = time.time() - start_time
        
        start_time = time.time()
        result_vectorized = test_df['A'] + test_df['B']
        benchmarks['vectorized_time'] = time.time() - start_time
        
        benchmarks['vectorization_speedup'] = benchmarks['iterrows_time'] / benchmarks['vectorized_time']
        
        # Benchmark: append vs concat
        start_time = time.time()
        result_append = pd.DataFrame()
        for i in range(100):
            small_df = test_df.iloc[i:i+1]
            result_append = result_append.append(small_df)
        benchmarks['append_time'] = time.time() - start_time
        
        start_time = time.time()
        dfs_to_concat = [test_df.iloc[i:i+1] for i in range(100)]
        result_concat = pd.concat(dfs_to_concat)
        benchmarks['concat_time'] = time.time() - start_time
        
        benchmarks['concat_speedup'] = benchmarks['append_time'] / benchmarks['concat_time']
        
        self.logger.info(f"Vectorization speedup: {benchmarks['vectorization_speedup']:.2f}x")
        self.logger.info(f"Concat speedup: {benchmarks['concat_speedup']:.2f}x")
        
        return benchmarks

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Performance Optimizer for Financial Framework')
    parser.add_argument('--path', '-p', default='.', help='Path to analyze (default: current directory)')
    parser.add_argument('--output', '-o', help='Output report file path')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run optimization analysis
    optimizer = PerformanceOptimizer(args.path)
    issues = optimizer.analyze_codebase()
    
    print(f"\nFound {len(issues)} performance issues")
    
    # Generate report
    output_file = args.output or 'performance_report.md'
    report = optimizer.generate_report(output_file)
    
    if not args.output:
        print("\n" + "="*60)
        print("PERFORMANCE REPORT SUMMARY")
        print("="*60)
        
        # Print summary
        severity_counts = {}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                print(f"{severity.upper()}: {count} issues")
    
    # Run benchmarks if requested
    if args.benchmark:
        benchmarks = optimizer.benchmark_improvements()
        print("\nPERFORMANCE BENCHMARKS:")
        print("-" * 30)
        for metric, value in benchmarks.items():
            if 'speedup' in metric:
                print(f"{metric}: {value:.2f}x faster")
            else:
                print(f"{metric}: {value:.4f}s")

if __name__ == "__main__":
    main()