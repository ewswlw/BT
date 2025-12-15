"""
Validation framework for financial machine learning strategies.

This module implements validation methodologies from "Advances in Financial 
Machine Learning" by Marcos López de Prado, including:
- Purged Cross-Validation
- Combinatorial Purged Cross-Validation (CPCV)
- Sample Weighting for Overlapping Labels
- Deflated Sharpe Ratio
- Probability of Backtest Overfitting (PBO)
- Walk-Forward Analysis
- Synthetic Data Testing
"""

from .validation_config import ValidationConfig
from .validation_results import ValidationResults
from .purged_cv import PurgedKFold
from .cpcv import CombinatorialPurgedCV
from .sample_weights import (
    LabelUniquenessCalculator,
    compute_label_uniqueness,
    sequential_bootstrap,
    compute_sample_weights,
    compute_time_decay_weights
)
from .deflated_metrics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio
)
from .pbo import ProbabilityBacktestOverfitting, calculate_pbo
from .walk_forward import WalkForwardAnalyzer
from .synthetic_data import SyntheticDataGenerator
from .validation_framework import ValidationFramework
from .validation_report import ValidationReportGenerator

__version__ = "1.0.0"

__all__ = [
    "ValidationConfig",
    "ValidationResults",
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "LabelUniquenessCalculator",
    "compute_label_uniqueness",
    "sequential_bootstrap",
    "compute_sample_weights",
    "compute_time_decay_weights",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "ProbabilityBacktestOverfitting",
    "calculate_pbo",
    "WalkForwardAnalyzer",
    "SyntheticDataGenerator",
    "ValidationFramework",
    "ValidationReportGenerator",
]

