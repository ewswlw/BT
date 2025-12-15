"""
Comprehensive Validation Report Generator.

Generates detailed text reports with all validation outputs and educational
explanations of what each measure means.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from .validation_results import ValidationResults


class ValidationReportGenerator:
    """Generates comprehensive validation reports with explanations."""
    
    def __init__(self, output_dir: str = "outputs/validation"):
        """
        Initialize ValidationReportGenerator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        strategy_name: str,
        validation_results: ValidationResults,
        config: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            strategy_name: Name of the strategy
            validation_results: ValidationResults object
            config: Optional configuration dictionary
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use fixed filename (no timestamp) so it overwrites on each run
        report_path = self.output_dir / f"{strategy_name}_validation_report.txt"
        
        report_lines = []
        
        # Header (still includes timestamp for reference)
        report_lines.extend(self._generate_header(strategy_name, timestamp))
        
        # Executive Summary
        report_lines.extend(self._generate_executive_summary(validation_results))
        
        # Data Quality
        report_lines.extend(self._generate_data_quality_section(validation_results))
        
        # Cross-Validation
        report_lines.extend(self._generate_cv_section(validation_results))
        
        # Sample Weighting
        report_lines.extend(self._generate_sample_weights_section(validation_results))
        
        # Deflated Metrics
        report_lines.extend(self._generate_deflated_metrics_section(validation_results))
        
        # Walk-Forward Analysis
        report_lines.extend(self._generate_walk_forward_section(validation_results))
        
        # Overfitting Detection
        report_lines.extend(self._generate_overfitting_section(validation_results))
        
        # Minimum Backtest Length
        report_lines.extend(self._generate_min_backtest_length_section(validation_results))
        
        # Metric Explanations
        report_lines.extend(self._generate_metric_explanations())
        
        # Recommendations
        report_lines.extend(self._generate_recommendations(validation_results))
        
        # Footer
        report_lines.extend(self._generate_footer())
        
        # Write report
        report_text = "\n".join(report_lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return str(report_path)
    
    def _generate_header(self, strategy_name: str, timestamp: str) -> List[str]:
        """Generate report header."""
        return [
            "=" * 100,
            f"COMPREHENSIVE VALIDATION REPORT",
            f"Strategy: {strategy_name.upper()}",
            f"Generated: {timestamp}",
            "=" * 100,
            "",
            "This report provides a comprehensive validation analysis based on the methodologies",
            "outlined in 'Advances in Financial Machine Learning' by Marcos López de Prado.",
            "",
            "The validation framework addresses common pitfalls in backtesting including:",
            "  - Data leakage and look-ahead bias",
            "  - Multiple testing and selection bias",
            "  - Overfitting detection",
            "  - Non-IID data handling",
            "  - Robustness assessment",
            "",
            "=" * 100,
            ""
        ]
    
    def _generate_executive_summary(self, results: ValidationResults) -> List[str]:
        """Generate executive summary section."""
        lines = [
            "SECTION 1: EXECUTIVE SUMMARY",
            "-" * 100,
            ""
        ]
        
        # Overall status
        status = self._assess_overall_status(results)
        lines.append(f"Overall Validation Status: {status['status']}")
        lines.append(f"Risk Level: {status['risk_level']}")
        lines.append("")
        
        # Key findings
        lines.append("KEY FINDINGS:")
        lines.append("-" * 50)
        
        if results.cv_scores:
            cv_consistency = "GOOD" if results.cv_std < 0.1 else "MODERATE" if results.cv_std < 0.2 else "POOR"
            lines.append(f"  • Cross-Validation Consistency: {cv_consistency} (std: {results.cv_std:.4f})")
            lines.append(f"  • Average CV Performance: {results.cv_mean:.4f}")
        
        if results.deflated_sharpe is not None:
            dsr_status = "PASS" if results.deflated_sharpe > 0.95 else "MODERATE" if results.deflated_sharpe > 0.80 else "FAIL"
            lines.append(f"  • Deflated Sharpe Ratio: {results.deflated_sharpe:.4f} ({dsr_status})")
        
        if results.pbo is not None:
            pbo_status = "LOW RISK" if results.pbo < 0.3 else "MODERATE RISK" if results.pbo < 0.7 else "HIGH RISK"
            lines.append(f"  • Overfitting Risk (PBO): {results.pbo:.2%} ({pbo_status})")
        
        if results.min_backtest_length and results.current_length:
            adequate = "YES" if results.current_length >= results.min_backtest_length else "NO"
            lines.append(f"  • Minimum Backtest Length Met: {adequate}")
            lines.append(f"    Required: {results.min_backtest_length} periods")
            lines.append(f"    Current: {results.current_length} periods")
        
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 50)
        lines.extend(status['recommendations'])
        lines.append("")
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_cv_section(self, results: ValidationResults) -> List[str]:
        """Generate cross-validation section with explanations."""
        lines = [
            "SECTION 2: CROSS-VALIDATION ANALYSIS",
            "-" * 100,
            ""
        ]
        
        if not results.cv_scores:
            lines.append("No cross-validation results available.")
            lines.append("")
            return lines
        
        # Results
        lines.append("CROSS-VALIDATION RESULTS:")
        lines.append("-" * 50)
        lines.append(f"Method: {results.cv_method}")
        lines.append(f"Number of Folds: {len(results.cv_scores)}")
        lines.append("")
        
        for i, score in enumerate(results.cv_scores, 1):
            lines.append(f"  Fold {i}: {score:.6f}")
        
        lines.append("")
        lines.append(f"Mean CV Score: {results.cv_mean:.6f}")
        lines.append(f"Std CV Score:  {results.cv_std:.6f}")
        cv_ratio = (results.cv_std/results.cv_mean)*100 if results.cv_mean != 0 else float('inf')
        lines.append(f"Coefficient of Variation: {cv_ratio:.2f}%")
        lines.append("")
        
        # Interpretation
        lines.append("WHAT THIS MEANS:")
        lines.append("-" * 50)
        lines.append("Cross-validation is a technique used to assess how well your model generalizes")
        lines.append("to unseen data. In financial ML, we use PURGED cross-validation to prevent")
        lines.append("data leakage from overlapping labels.")
        lines.append("")
        lines.append("KEY CONCEPTS:")
        lines.append("  1. PURGING: Removes training samples that overlap with test samples")
        lines.append("     in time. This prevents look-ahead bias.")
        lines.append("")
        lines.append("  2. EMBARGO: Adds a buffer period after the test set where no training")
        lines.append("     samples are used. This accounts for serial correlation.")
        lines.append("")
        lines.append("  3. CONSISTENCY: Low standard deviation across folds indicates stable")
        lines.append("     performance. High variation suggests the model may be overfitting")
        lines.append("     to specific time periods.")
        lines.append("")
        
        # Interpretation guidelines
        lines.append("HOW TO INTERPRET:")
        lines.append("-" * 50)
        cv_ratio = results.cv_std / results.cv_mean if results.cv_mean != 0 else float('inf')
        
        if cv_ratio < 0.1:
            lines.append("✓ EXCELLENT: Very consistent performance across folds.")
            lines.append("  Your model shows stable generalization.")
        elif cv_ratio < 0.2:
            lines.append("⚠ MODERATE: Some variation across folds.")
            lines.append("  Consider investigating periods with poor performance.")
        else:
            lines.append("✗ POOR: High variation across folds.")
            lines.append("  This suggests overfitting or regime-dependent performance.")
            lines.append("  Consider:")
            lines.append("    - Reducing model complexity")
            lines.append("    - Adding regularization")
            lines.append("    - Using ensemble methods")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_deflated_metrics_section(self, results: ValidationResults) -> List[str]:
        """Generate deflated metrics section with explanations."""
        lines = [
            "SECTION 3: DEFLATED PERFORMANCE METRICS",
            "-" * 100,
            ""
        ]
        
        if results.deflated_sharpe is None:
            lines.append("Deflated Sharpe Ratio not calculated.")
            lines.append("")
            return lines
        
        # Results
        lines.append("DEFLATED SHARPE RATIO:")
        lines.append("-" * 50)
        if results.estimated_sharpe is not None:
            lines.append(f"Estimated Sharpe Ratio: {results.estimated_sharpe:.4f}")
        lines.append(f"Number of Trials: {results.n_trials}")
        lines.append(f"Deflated Sharpe Ratio: {results.deflated_sharpe:.4f}")
        lines.append("")
        
        # Explanation
        lines.append("WHAT IS DEFLATED SHARPE RATIO?")
        lines.append("-" * 50)
        lines.append("The Deflated Sharpe Ratio (DSR) adjusts the estimated Sharpe ratio to account")
        lines.append("for MULTIPLE TESTING BIAS. When you test many strategies, you're likely to")
        lines.append("find one that performs well purely by chance.")
        lines.append("")
        lines.append("FORMULA:")
        lines.append("  DSR = Φ((SR - E[SR_max]) / √Var[SR_max])")
        lines.append("")
        lines.append("Where:")
        lines.append("  • SR = Your estimated Sharpe ratio")
        lines.append("  • E[SR_max] = Expected maximum Sharpe ratio under null hypothesis")
        lines.append("  • Var[SR_max] = Variance of Sharpe ratio")
        lines.append("  • Φ = Standard normal CDF")
        lines.append("")
        lines.append("The DSR tells you the PROBABILITY that your Sharpe ratio is genuine,")
        lines.append("not just the result of multiple testing.")
        lines.append("")
        
        # Interpretation
        lines.append("HOW TO INTERPRET:")
        lines.append("-" * 50)
        if results.deflated_sharpe > 0.95:
            lines.append(f"✓ EXCELLENT: {results.deflated_sharpe:.1%} confidence this is not a false positive.")
            lines.append("  Your strategy likely has genuine edge.")
        elif results.deflated_sharpe > 0.80:
            lines.append(f"⚠ MODERATE: {results.deflated_sharpe:.1%} confidence this is not a false positive.")
            lines.append("  Proceed with caution. Consider more testing.")
        else:
            lines.append(f"✗ POOR: {results.deflated_sharpe:.1%} confidence this is not a false positive.")
            lines.append("  High risk of false discovery. Do not trade this strategy.")
            lines.append("  Consider:")
            lines.append("    - Testing on more data")
            lines.append("    - Reducing number of parameter combinations tested")
            lines.append("    - Using more conservative significance levels")
        
        lines.append("")
        
        # Probabilistic Sharpe Ratio
        if results.probabilistic_sharpe is not None:
            lines.append("PROBABILISTIC SHARPE RATIO:")
            lines.append("-" * 50)
            lines.append(f"PSR Value: {results.probabilistic_sharpe:.4f}")
            lines.append("")
            lines.append("WHAT IS PROBABILISTIC SHARPE RATIO?")
            lines.append("-" * 50)
            lines.append("PSR computes the probability that your estimated Sharpe ratio exceeds")
            lines.append("a benchmark (typically 0 or 1.0). Unlike standard Sharpe, PSR accounts")
            lines.append("for the distribution's skewness and kurtosis.")
            lines.append("")
            lines.append("HOW TO INTERPRET:")
            lines.append("-" * 50)
            if results.probabilistic_sharpe > 0.95:
                lines.append("✓ EXCELLENT: Very high probability (>95%) that SR exceeds benchmark.")
            elif results.probabilistic_sharpe > 0.80:
                lines.append("⚠ MODERATE: Good probability (80-95%) that SR exceeds benchmark.")
            else:
                lines.append("✗ POOR: Low probability (<80%) that SR exceeds benchmark.")
            lines.append("")
        
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_overfitting_section(self, results: ValidationResults) -> List[str]:
        """Generate overfitting detection section."""
        lines = [
            "SECTION 4: OVERFITTING DETECTION",
            "-" * 100,
            ""
        ]
        
        if results.pbo is None:
            lines.append("Probability of Backtest Overfitting (PBO) not calculated.")
            lines.append("PBO requires testing multiple strategy configurations.")
            lines.append("")
            return lines
        
        # Results
        lines.append("PROBABILITY OF BACKTEST OVERFITTING (PBO):")
        lines.append("-" * 50)
        lines.append(f"PBO: {results.pbo:.2%}")
        if results.is_sharpe is not None:
            lines.append(f"In-Sample Sharpe: {results.is_sharpe:.4f}")
        if results.oos_sharpe is not None:
            lines.append(f"Out-of-Sample Sharpe: {results.oos_sharpe:.4f}")
        lines.append("")
        
        # Explanation
        lines.append("WHAT IS PBO?")
        lines.append("-" * 50)
        lines.append("PBO measures the probability that your backtest performance is due to")
        lines.append("overfitting rather than genuine predictive power. It's calculated by:")
        lines.append("")
        lines.append("  1. Splitting data into In-Sample (IS) and Out-of-Sample (OOS) sets")
        lines.append("  2. Testing multiple strategy configurations on IS data")
        lines.append("  3. Selecting the best IS configuration")
        lines.append("  4. Evaluating that configuration on OOS data")
        lines.append("  5. Calculating how many configurations perform worse than median OOS")
        lines.append("")
        lines.append("If the best IS configuration performs poorly on OOS, you have overfitting.")
        lines.append("")
        
        # Interpretation
        lines.append("HOW TO INTERPRET:")
        lines.append("-" * 50)
        if results.pbo < 0.3:
            lines.append("✓ LOW RISK: PBO < 30%")
            lines.append("  Low probability of overfitting. Strategy may be genuine.")
        elif results.pbo < 0.7:
            lines.append("⚠ MODERATE RISK: PBO 30-70%")
            lines.append("  Moderate overfitting risk. Proceed with caution.")
            lines.append("  Consider:")
            lines.append("    - More out-of-sample testing")
            lines.append("    - Reducing model complexity")
            lines.append("    - Using regularization")
        else:
            lines.append("✗ HIGH RISK: PBO > 70%")
            lines.append("  High probability of overfitting. DO NOT TRADE THIS STRATEGY.")
            lines.append("  The backtest results are likely spurious.")
            lines.append("")
            lines.append("  Actions required:")
            lines.append("    - Start over with a new hypothesis")
            lines.append("    - Reduce number of parameters tested")
            lines.append("    - Use more conservative validation")
            lines.append("    - Consider ensemble methods")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_sample_weights_section(self, results: ValidationResults) -> List[str]:
        """Generate sample weights section."""
        lines = [
            "SECTION 5: SAMPLE WEIGHTING ANALYSIS",
            "-" * 100,
            ""
        ]
        
        if results.sample_weights is None:
            lines.append("Sample weights not calculated.")
            lines.append("")
            return lines
        
        # Statistics
        lines.append("SAMPLE WEIGHT STATISTICS:")
        lines.append("-" * 50)
        lines.append(f"Mean Weight: {results.sample_weights.mean():.6f}")
        lines.append(f"Std Weight:  {results.sample_weights.std():.6f}")
        lines.append(f"Min Weight:  {results.sample_weights.min():.6f}")
        lines.append(f"Max Weight:  {results.sample_weights.max():.6f}")
        lines.append("")
        
        if results.label_uniqueness is not None:
            lines.append("LABEL UNIQUENESS:")
            lines.append("-" * 50)
            lines.append(f"Mean Uniqueness: {results.label_uniqueness.mean():.4f}")
            lines.append(f"Min Uniqueness:  {results.label_uniqueness.min():.4f}")
            lines.append("")
            lines.append("Uniqueness measures how much each label overlaps with others.")
            lines.append("Values closer to 1.0 indicate less overlap (more unique labels).")
            lines.append("")
        
        if results.overlapping_labels_count:
            lines.append(f"Overlapping Labels Detected: {results.overlapping_labels_count}")
            lines.append("")
        
        # Explanation
        lines.append("WHY SAMPLE WEIGHTS MATTER:")
        lines.append("-" * 50)
        lines.append("Financial data violates the IID (Independent and Identically Distributed)")
        lines.append("assumption. Labels often overlap in time, creating dependencies.")
        lines.append("")
        lines.append("PROBLEM:")
        lines.append("  • Standard ML assumes independent samples")
        lines.append("  • Overlapping labels violate this assumption")
        lines.append("  • This leads to overfitting and poor generalization")
        lines.append("")
        lines.append("SOLUTION:")
        lines.append("  • Weight samples by their uniqueness")
        lines.append("  • More unique labels get higher weights")
        lines.append("  • This accounts for the non-IID nature of financial data")
        lines.append("")
        lines.append("HOW TO INTERPRET:")
        lines.append("-" * 50)
        weight_cv = results.sample_weights.std() / results.sample_weights.mean() if results.sample_weights.mean() != 0 else 0
        if weight_cv > 0.5:
            lines.append("⚠ HIGH VARIATION: Significant differences in sample weights.")
            lines.append("  This indicates many overlapping labels. Sample weighting is CRITICAL.")
        elif weight_cv > 0.2:
            lines.append("⚠ MODERATE VARIATION: Some variation in sample weights.")
            lines.append("  Sample weighting recommended.")
        else:
            lines.append("✓ LOW VARIATION: Relatively uniform weights.")
            lines.append("  Few overlapping labels detected.")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_walk_forward_section(self, results: ValidationResults) -> List[str]:
        """Generate walk-forward analysis section."""
        lines = [
            "SECTION 6: WALK-FORWARD ANALYSIS",
            "-" * 100,
            ""
        ]
        
        if not results.walk_forward_results:
            lines.append("Walk-forward analysis not performed.")
            lines.append("")
            return lines
        
        # Results summary
        lines.append("WALK-FORWARD RESULTS SUMMARY:")
        lines.append("-" * 50)
        
        sharpe_scores = [r.get('sharpe', 0) for r in results.walk_forward_results if r.get('sharpe') is not None]
        if sharpe_scores:
            lines.append(f"Number of Periods: {len(sharpe_scores)}")
            lines.append(f"Mean Sharpe: {np.mean(sharpe_scores):.4f}")
            lines.append(f"Std Sharpe:  {np.std(sharpe_scores):.4f}")
            lines.append(f"Min Sharpe:  {np.min(sharpe_scores):.4f}")
            lines.append(f"Max Sharpe:  {np.max(sharpe_scores):.4f}")
            lines.append("")
        
        # Detailed results
        lines.append("PERIOD-BY-PERIOD RESULTS:")
        lines.append("-" * 50)
        for i, result in enumerate(results.walk_forward_results[:10], 1):  # Show first 10
            lines.append(f"\nPeriod {i}:")
            lines.append(f"  Train Period: {result.get('train_start', 'N/A')} to {result.get('train_end', 'N/A')}")
            lines.append(f"  Test Period:  {result.get('test_start', 'N/A')} to {result.get('test_end', 'N/A')}")
            if result.get('sharpe') is not None:
                lines.append(f"  Sharpe Ratio: {result['sharpe']:.4f}")
            if result.get('return') is not None:
                lines.append(f"  Return: {result['return']:.2%}")
        
        if len(results.walk_forward_results) > 10:
            lines.append(f"\n... ({len(results.walk_forward_results) - 10} more periods)")
        
        lines.append("")
        
        # Explanation
        lines.append("WHAT IS WALK-FORWARD ANALYSIS?")
        lines.append("-" * 50)
        lines.append("Walk-forward analysis simulates real-world trading by:")
        lines.append("")
        lines.append("  1. Training on historical data (expanding or rolling window)")
        lines.append("  2. Testing on subsequent unseen data")
        lines.append("  3. Moving forward in time and repeating")
        lines.append("")
        lines.append("This mimics how you would actually trade: you only know past data")
        lines.append("when making decisions.")
        lines.append("")
        lines.append("TYPES:")
        lines.append("  • EXPANDING WINDOW: Training set grows over time")
        lines.append("  • ROLLING WINDOW: Fixed-size training window that moves forward")
        lines.append("")
        
        # Interpretation
        lines.append("HOW TO INTERPRET:")
        lines.append("-" * 50)
        if sharpe_scores:
            sharpe_cv = np.std(sharpe_scores) / np.mean(sharpe_scores) if np.mean(sharpe_scores) != 0 else float('inf')
            if sharpe_cv < 0.3:
                lines.append("✓ STABLE: Consistent performance across periods.")
                lines.append("  Strategy shows robustness over time.")
            elif sharpe_cv < 0.6:
                lines.append("⚠ MODERATE: Some variation across periods.")
                lines.append("  Strategy may be regime-dependent.")
            else:
                lines.append("✗ UNSTABLE: High variation across periods.")
                lines.append("  Strategy performance is inconsistent.")
                lines.append("  Consider regime detection and adaptation.")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_min_backtest_length_section(self, results: ValidationResults) -> List[str]:
        """Generate minimum backtest length section."""
        lines = [
            "SECTION 7: MINIMUM BACKTEST LENGTH ANALYSIS",
            "-" * 100,
            ""
        ]
        
        if results.min_backtest_length is None:
            lines.append("Minimum backtest length not calculated.")
            lines.append("")
            return lines
        
        # Results
        lines.append("MINIMUM BACKTEST LENGTH REQUIREMENT:")
        lines.append("-" * 50)
        lines.append(f"Required Length: {results.min_backtest_length} periods")
        lines.append(f"Current Length:  {results.current_length} periods")
        
        if results.current_length:
            adequacy = results.current_length >= results.min_backtest_length
            lines.append(f"Adequacy Status: {'✓ ADEQUATE' if adequacy else '✗ INADEQUATE'}")
            lines.append("")
        
        # Explanation
        lines.append("WHAT IS MINIMUM BACKTEST LENGTH?")
        lines.append("-" * 50)
        lines.append("López de Prado provides a formula for the minimum backtest length needed")
        lines.append("to have confidence in your results:")
        lines.append("")
        lines.append("  MinBTL = ((SR* / SR)^2 - 1) × N")
        lines.append("")
        lines.append("Where:")
        lines.append("  • SR* = Target Sharpe ratio")
        lines.append("  • SR = Expected Sharpe ratio")
        lines.append("  • N = Number of observations in original backtest")
        lines.append("")
        lines.append("This ensures you have enough data to distinguish genuine edge from noise.")
        lines.append("")
        
        # Interpretation
        lines.append("HOW TO INTERPRET:")
        lines.append("-" * 50)
        if results.current_length:
            if adequacy:
                lines.append("✓ ADEQUATE: You have sufficient data for reliable backtesting.")
            else:
                lines.append("✗ INADEQUATE: You need more data for reliable backtesting.")
                lines.append(f"  Shortfall: {results.min_backtest_length - results.current_length} periods")
                lines.append("  Recommendations:")
                lines.append("    - Wait for more data")
                lines.append("    - Use lower frequency data (e.g., weekly instead of daily)")
                lines.append("    - Reduce target Sharpe ratio expectations")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_data_quality_section(self, results: ValidationResults) -> List[str]:
        """Generate data quality section."""
        lines = [
            "SECTION 8: DATA QUALITY VALIDATION",
            "-" * 100,
            ""
        ]
        
        if not results.data_quality_metrics:
            lines.append("Data quality metrics not available.")
            lines.append("")
            return lines
        
        dq = results.data_quality_metrics
        
        lines.append("DATA QUALITY METRICS:")
        lines.append("-" * 50)
        if 'completeness' in dq:
            lines.append(f"Completeness: {dq['completeness']:.2%}")
        if 'n_samples' in dq:
            lines.append(f"Number of Samples: {dq['n_samples']}")
        if 'n_features' in dq:
            lines.append(f"Number of Features: {dq['n_features']}")
        if 'missing_values' in dq:
            lines.append(f"Missing Values: {dq['missing_values']}")
        lines.append("")
        
        lines.append("=" * 100)
        lines.append("")
        
        return lines
    
    def _generate_metric_explanations(self) -> List[str]:
        """Generate comprehensive metric explanations section."""
        return [
            "SECTION 9: DETAILED METRIC EXPLANATIONS",
            "-" * 100,
            "",
            "This section provides detailed explanations of all validation metrics used in this report.",
            "",
            "=" * 100,
            "",
            "1. DEFLATED SHARPE RATIO (DSR)",
            "-" * 100,
            "",
            "WHAT IT IS:",
            "  The Deflated Sharpe Ratio adjusts your estimated Sharpe ratio to account for",
            "  multiple testing bias. When testing many strategies, you're likely to find",
            "  one that performs well by chance alone.",
            "",
            "WHY IT MATTERS:",
            "  Without adjustment, you may think you found a great strategy when you actually",
            "  just got lucky after testing many variations.",
            "",
            "HOW TO READ IT:",
            "  • DSR > 0.95: Excellent - Very high confidence this is genuine",
            "  • DSR 0.80-0.95: Moderate - Good confidence, proceed with caution",
            "  • DSR < 0.80: Poor - High risk of false discovery, do not trade",
            "",
            "INDUSTRY BENCHMARK:",
            "  Institutional strategies typically require DSR > 0.95 before deployment.",
            "",
            "=" * 100,
            "",
            "2. PROBABILITY OF BACKTEST OVERFITTING (PBO)",
            "-" * 100,
            "",
            "WHAT IT IS:",
            "  PBO measures the probability that your backtest results are due to overfitting",
            "  rather than genuine predictive power. It compares in-sample vs out-of-sample",
            "  performance.",
            "",
            "WHY IT MATTERS:",
            "  A strategy that works in-sample but fails out-of-sample is overfit and will",
            "  fail in live trading.",
            "",
            "HOW TO READ IT:",
            "  • PBO < 0.30: Low risk - Strategy likely genuine",
            "  • PBO 0.30-0.70: Moderate risk - Proceed with caution",
            "  • PBO > 0.70: High risk - Do not trade, likely overfit",
            "",
            "RED FLAGS:",
            "  • Large gap between IS and OOS performance",
            "  • Best IS configuration performs poorly on OOS",
            "  • PBO > 0.70",
            "",
            "=" * 100,
            "",
            "3. PURGED CROSS-VALIDATION",
            "-" * 100,
            "",
            "WHAT IT IS:",
            "  Standard cross-validation fails in finance because labels overlap in time.",
            "  Purged CV removes training samples that overlap with test samples, preventing",
            "  data leakage.",
            "",
            "WHY IT MATTERS:",
            "  Without purging, you're using future information to predict the past, which",
            "  creates look-ahead bias.",
            "",
            "KEY COMPONENTS:",
            "  • PURGING: Remove overlapping samples",
            "  • EMBARGO: Add buffer period after test set",
            "  • TIME-AWARE: Respects temporal order",
            "",
            "HOW TO READ IT:",
            "  • Low CV std: Consistent performance across folds (good)",
            "  • High CV std: High variation, possible overfitting (bad)",
            "",
            "=" * 100,
            "",
            "4. SAMPLE WEIGHTS",
            "-" * 100,
            "",
            "WHAT IT IS:",
            "  Financial labels often overlap in time, violating the IID assumption.",
            "  Sample weights adjust for this by giving more weight to unique labels.",
            "",
            "WHY IT MATTERS:",
            "  Ignoring overlapping labels leads to overfitting and poor generalization.",
            "",
            "TYPES:",
            "  • UNIQUENESS WEIGHTS: Based on label overlap",
            "  • TIME DECAY WEIGHTS: More recent samples weighted higher",
            "  • SEQUENTIAL BOOTSTRAP: Accounts for dependencies",
            "",
            "HOW TO READ IT:",
            "  • High weight variation: Many overlapping labels, weighting critical",
            "  • Low weight variation: Few overlaps, standard ML may work",
            "",
            "=" * 100,
            "",
            "5. WALK-FORWARD ANALYSIS",
            "-" * 100,
            "",
            "WHAT IT IS:",
            "  Simulates real trading by training on past data and testing on future data,",
            "  then moving forward in time.",
            "",
            "WHY IT MATTERS:",
            "  This is how you actually trade - you only know past data when making decisions.",
            "",
            "TYPES:",
            "  • EXPANDING: Training set grows over time",
            "  • ROLLING: Fixed-size window that moves forward",
            "",
            "HOW TO READ IT:",
            "  • Stable performance: Strategy robust over time",
            "  • Variable performance: Strategy may be regime-dependent",
            "",
            "=" * 100,
            "",
            "6. MINIMUM BACKTEST LENGTH",
            "-" * 100,
            "",
            "WHAT IT IS:",
            "  The minimum amount of data needed to have confidence in backtest results.",
            "",
            "WHY IT MATTERS:",
            "  Too little data means you can't distinguish signal from noise.",
            "",
            "HOW TO READ IT:",
            "  • Adequate: You have enough data",
            "  • Inadequate: Need more data or lower expectations",
            "",
            "=" * 100,
            "",
            "7. PROBABILISTIC SHARPE RATIO (PSR)",
            "-" * 100,
            "",
            "WHAT IT IS:",
            "  PSR computes the probability that your Sharpe ratio exceeds a benchmark,",
            "  accounting for skewness and kurtosis.",
            "",
            "WHY IT MATTERS:",
            "  Standard Sharpe assumes normal returns, which is rarely true in finance.",
            "",
            "HOW TO READ IT:",
            "  • PSR > 0.95: Very high probability SR exceeds benchmark",
            "  • PSR 0.80-0.95: Good probability",
            "  • PSR < 0.80: Low probability",
            "",
            "=" * 100,
            "",
            "GENERAL VALIDATION BEST PRACTICES:",
            "-" * 100,
            "",
            "1. ALWAYS use purged cross-validation for financial data",
            "2. ALWAYS calculate deflated Sharpe ratio when testing multiple strategies",
            "3. ALWAYS check PBO before deploying a strategy",
            "4. ALWAYS use sample weights for overlapping labels",
            "5. ALWAYS perform walk-forward analysis",
            "6. ALWAYS verify minimum backtest length",
            "7. NEVER optimize on test set",
            "8. NEVER ignore overlapping labels",
            "9. NEVER trust a single backtest result",
            "10. ALWAYS validate on actual strategy data (not synthetic)",
            "",
            "=" * 100,
            ""
        ]
    
    def _generate_recommendations(self, results: ValidationResults) -> List[str]:
        """Generate recommendations section."""
        recommendations = []
        
        recommendations.append("SECTION 10: RECOMMENDATIONS")
        recommendations.append("-" * 100)
        recommendations.append("")
        
        # CV recommendations
        if results.cv_scores:
            cv_ratio = results.cv_std / results.cv_mean if results.cv_mean != 0 else float('inf')
            if cv_ratio > 0.2:
                recommendations.append("1. CROSS-VALIDATION CONSISTENCY:")
                recommendations.append("   • High variation detected across CV folds")
                recommendations.append("   • Consider reducing model complexity")
                recommendations.append("   • Add regularization or ensemble methods")
                recommendations.append("")
        
        # PBO recommendations
        if results.pbo is not None and results.pbo > 0.7:
            recommendations.append("2. OVERFITTING RISK:")
            recommendations.append("   • HIGH PBO detected - DO NOT TRADE")
            recommendations.append("   • Strategy likely overfit to historical data")
            recommendations.append("   • Start over with new hypothesis")
            recommendations.append("   • Reduce number of parameters tested")
            recommendations.append("")
        
        # Deflated Sharpe recommendations
        if results.deflated_sharpe is not None and results.deflated_sharpe < 0.80:
            recommendations.append("3. MULTIPLE TESTING BIAS:")
            recommendations.append("   • Low deflated Sharpe ratio")
            recommendations.append("   • High risk of false discovery")
            recommendations.append("   • Test on more data or reduce trials")
            recommendations.append("")
        
        # Sample weights recommendations
        if results.sample_weights is not None:
            weight_cv = results.sample_weights.std() / results.sample_weights.mean() if results.sample_weights.mean() != 0 else 0
            if weight_cv > 0.3:
                recommendations.append("4. SAMPLE WEIGHTING:")
                recommendations.append("   • Significant overlapping labels detected")
                recommendations.append("   • Ensure sample weights are used in training")
                recommendations.append("")
        
        # Minimum backtest length
        if results.min_backtest_length and results.current_length:
            if results.current_length < results.min_backtest_length:
                recommendations.append("5. DATA ADEQUACY:")
                recommendations.append(f"   • Need {results.min_backtest_length - results.current_length} more periods")
                recommendations.append("   • Consider using lower frequency data")
                recommendations.append("")
        
        if len(recommendations) == 2:  # Only header
            recommendations.append("No specific recommendations at this time.")
            recommendations.append("Continue monitoring and validation.")
            recommendations.append("")
        
        recommendations.append("=" * 100)
        recommendations.append("")
        
        return recommendations
    
    def _generate_footer(self) -> List[str]:
        """Generate report footer."""
        return [
            "=" * 100,
            "END OF VALIDATION REPORT",
            "=" * 100,
            "",
            "For questions or clarifications, refer to:",
            "  'Advances in Financial Machine Learning' by Marcos López de Prado",
            "",
            "Generated by: Comprehensive Validation Framework",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
    
    def _assess_overall_status(self, results: ValidationResults) -> Dict:
        """Assess overall validation status."""
        status = "PASS"
        risk_level = "LOW"
        recommendations = []
        
        # Check PBO
        if results.pbo is not None:
            if results.pbo > 0.7:
                status = "FAIL"
                risk_level = "HIGH"
                recommendations.append("  ⚠ CRITICAL: High overfitting risk (PBO > 70%)")
            elif results.pbo > 0.3:
                status = "WARNING"
                risk_level = "MODERATE"
                recommendations.append("  ⚠ WARNING: Moderate overfitting risk (PBO 30-70%)")
        
        # Check Deflated Sharpe
        if results.deflated_sharpe is not None:
            if results.deflated_sharpe < 0.80:
                status = "FAIL" if status == "PASS" else status
                risk_level = "HIGH" if risk_level == "LOW" else risk_level
                recommendations.append("  ⚠ CRITICAL: Low deflated Sharpe ratio (< 0.80)")
            elif results.deflated_sharpe < 0.95:
                if status == "PASS":
                    status = "WARNING"
                    risk_level = "MODERATE"
                recommendations.append("  ⚠ WARNING: Moderate deflated Sharpe ratio (0.80-0.95)")
        
        # Check CV consistency
        if results.cv_scores:
            cv_ratio = results.cv_std / results.cv_mean if results.cv_mean != 0 else float('inf')
            if cv_ratio > 0.2:
                if status == "PASS":
                    status = "WARNING"
                    risk_level = "MODERATE"
                recommendations.append("  ⚠ WARNING: High CV variation detected")
        
        if not recommendations:
            recommendations.append("  ✓ All validation checks passed")
            recommendations.append("  ✓ Strategy shows promise for further testing")
        
        return {
            'status': status,
            'risk_level': risk_level,
            'recommendations': recommendations
        }

