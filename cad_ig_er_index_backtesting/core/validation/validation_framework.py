"""
Validation Framework Orchestrator.

Coordinates all validation components and provides unified interface
for comprehensive strategy validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import warnings

from .validation_config import ValidationConfig
from .validation_results import ValidationResults
from .purged_cv import PurgedKFold
from .cpcv import CombinatorialPurgedCV
from .sample_weights import (
    compute_label_uniqueness,
    compute_sample_weights,
    compute_time_decay_weights,
    count_overlapping_labels
)
from .deflated_metrics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    min_backtest_length
)
from .pbo import ProbabilityBacktestOverfitting
from .walk_forward import WalkForwardAnalyzer
from .synthetic_data import SyntheticDataGenerator


class ValidationFramework:
    """
    Main validation framework orchestrator.
    
    Coordinates all validation methods and provides unified interface.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize ValidationFramework.
        
        Args:
            config: ValidationConfig object
        """
        self.config = config
        self.results = ValidationResults()
        
        # Initialize components
        self.pbo_calculator = ProbabilityBacktestOverfitting()
        self.walk_forward_analyzer = WalkForwardAnalyzer(
            train_period=config.walk_forward_train_period,
            test_period=config.walk_forward_test_period,
            step=config.walk_forward_step
        )
        self.synthetic_data_generator = SyntheticDataGenerator()
    
    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: pd.Series,
        samples_info_sets: Optional[List[Tuple]] = None,
        model_fn: Optional[Callable] = None
    ) -> ValidationResults:
        """
        Run comprehensive validation.
        
        Args:
            X: Features DataFrame
            y: Target Series
            returns: Returns Series (for performance metrics)
            samples_info_sets: List of (start_time, end_time) tuples for labels
            model_fn: Optional function to train/evaluate model
                    Signature: (X_train, y_train, X_test, y_test, sample_weights) -> score
                    
        Returns:
            ValidationResults object with all validation metrics
        """
        self.results = ValidationResults()
        self.results.current_length = len(X)
        self.results.strategy_name = ""  # Will be set by caller
        
        # 1. Cross-Validation
        if self.config.cv_method in ["purged_kfold", "cpcv"]:
            self._run_cross_validation(X, y, samples_info_sets, model_fn)
        
        # 2. Sample Weights
        if self.config.use_sample_weights and samples_info_sets:
            self._calculate_sample_weights(X, samples_info_sets)
        
        # 3. Deflated Metrics
        if len(returns) > 0:
            self._calculate_deflated_metrics(returns)
        
        # 4. Walk-Forward Analysis
        if self.config.cv_method == "walk_forward":
            self._run_walk_forward(X, returns, model_fn)
        
        # 5. PBO (if multiple configurations available)
        # Note: PBO requires multiple strategy configurations
        # This would be called separately with returns_matrix
        
        # 6. Minimum Backtest Length
        if self.config.min_backtest_length_enabled and len(returns) > 0:
            self._calculate_min_backtest_length(returns)
        
        # 7. Data Quality
        self._assess_data_quality(X, y)
        
        return self.results
    
    def _run_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        samples_info_sets: Optional[List[Tuple]],
        model_fn: Optional[Callable]
    ) -> None:
        """Run cross-validation."""
        try:
            if self.config.cv_method == "purged_kfold":
                cv = PurgedKFold(
                    n_splits=self.config.n_splits,
                    samples_info_sets=samples_info_sets,
                    pct_embargo=self.config.embargo_pct
                )
            elif self.config.cv_method == "cpcv":
                cv = CombinatorialPurgedCV(
                    n_splits=self.config.n_splits,
                    n_test_groups=self.config.n_test_groups,
                    samples_info_sets=samples_info_sets,
                    pct_embargo=self.config.embargo_pct
                )
            else:
                return
            
            scores = []
            sample_weights_list = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Get sample weights if available
                weights = None
                if self.config.use_sample_weights and self.results.sample_weights is not None:
                    weights = self.results.sample_weights.iloc[train_idx].values
                
                # Evaluate model
                if model_fn is not None:
                    try:
                        score = model_fn(X_train, y_train, X_test, y_test, weights)
                        scores.append(score)
                    except Exception as e:
                        warnings.warn(f"Model evaluation failed in CV fold: {e}")
                else:
                    # Default: use simple accuracy if binary classification
                    # Skip scoring if no model function provided (will be done by strategy)
                    pass
            
            if scores:
                self.results.cv_scores = scores
                self.results.cv_mean = float(np.mean(scores))
                self.results.cv_std = float(np.std(scores))
                self.results.cv_method = self.config.cv_method
        except Exception as e:
            warnings.warn(f"Cross-validation failed: {e}")
    
    def _calculate_sample_weights(
        self,
        X: pd.DataFrame,
        samples_info_sets: List[Tuple]
    ) -> None:
        """Calculate sample weights."""
        try:
            # Count overlapping labels
            self.results.overlapping_labels_count = count_overlapping_labels(samples_info_sets)
            
            # Compute uniqueness
            price_index = X.index
            uniqueness = compute_label_uniqueness(samples_info_sets, price_index)
            self.results.label_uniqueness = uniqueness
            
            # Compute sample weights based on method
            if self.config.weight_method == "uniqueness":
                weights = compute_sample_weights(uniqueness)
            elif self.config.weight_method == "time_decay":
                weights = compute_time_decay_weights(
                    samples_info_sets,
                    uniqueness,
                    self.config.time_decay_factor
                )
            else:  # sequential_bootstrap
                # For sequential bootstrap, we'd need to sample
                # For now, use uniqueness weights
                weights = compute_sample_weights(uniqueness)
            
            self.results.sample_weights = weights
        except Exception as e:
            warnings.warn(f"Sample weight calculation failed: {e}")
    
    def _calculate_deflated_metrics(self, returns: pd.Series) -> None:
        """Calculate deflated metrics."""
        try:
            returns_clean = returns.dropna()
            if len(returns_clean) < 3:
                return
            
            # Calculate Sharpe ratio
            sr = (returns_clean.mean() * 252) / (returns_clean.std() * np.sqrt(252))
            self.results.estimated_sharpe = float(sr)
            
            # Deflated Sharpe Ratio
            skew = returns_clean.skew()
            kurt = returns_clean.kurtosis() + 3  # scipy returns excess kurtosis
            
            dsr = deflated_sharpe_ratio(
                sr,
                self.config.n_trials,
                len(returns_clean),
                skew,
                kurt
            )
            self.results.deflated_sharpe = float(dsr)
            self.results.n_trials = self.config.n_trials
            
            # Probabilistic Sharpe Ratio
            psr = probabilistic_sharpe_ratio(returns_clean)
            self.results.probabilistic_sharpe = float(psr)
        except Exception as e:
            warnings.warn(f"Deflated metrics calculation failed: {e}")
    
    def _run_walk_forward(
        self,
        X: pd.DataFrame,
        returns: pd.Series,
        model_fn: Optional[Callable]
    ) -> None:
        """Run walk-forward analysis."""
        try:
            # Create model function wrapper
            def wf_model_fn(train_data, train_returns, test_data):
                # This would need to be customized based on actual model
                # For now, return basic metrics
                return {}
            
            results = self.walk_forward_analyzer.analyze(X, returns, wf_model_fn)
            
            # Convert to dict format
            wf_results = []
            for r in results:
                wf_results.append({
                    'train_start': r.train_start,
                    'train_end': r.train_end,
                    'test_start': r.test_start,
                    'test_end': r.test_end,
                    'sharpe': r.sharpe,
                    'return': r.return_,
                    'max_drawdown': r.max_drawdown
                })
            
            self.results.walk_forward_results = wf_results
        except Exception as e:
            warnings.warn(f"Walk-forward analysis failed: {e}")
    
    def _calculate_min_backtest_length(self, returns: pd.Series) -> None:
        """Calculate minimum backtest length."""
        try:
            returns_clean = returns.dropna()
            if len(returns_clean) < 3:
                return
            
            sr = (returns_clean.mean() * 252) / (returns_clean.std() * np.sqrt(252))
            
            min_len = min_backtest_length(
                self.config.target_sharpe,
                sr,
                len(returns_clean)
            )
            
            self.results.min_backtest_length = min_len
        except Exception as e:
            warnings.warn(f"Minimum backtest length calculation failed: {e}")
    
    def _assess_data_quality(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """Assess data quality."""
        try:
            completeness = 1.0 - (X.isnull().sum().sum() / (len(X) * len(X.columns)))
            
            self.results.data_quality_metrics = {
                'completeness': float(completeness),
                'n_samples': len(X),
                'n_features': len(X.columns),
                'missing_values': int(X.isnull().sum().sum())
            }
        except Exception as e:
            warnings.warn(f"Data quality assessment failed: {e}")
    
    def calculate_pbo(
        self,
        returns_matrix: pd.DataFrame
    ) -> None:
        """
        Calculate Probability of Backtest Overfitting.
        
        Args:
            returns_matrix: DataFrame with returns for multiple configurations
                          Rows = time periods, Columns = configurations
        """
        try:
            pbo_results = self.pbo_calculator.calculate_pbo(returns_matrix)
            
            self.results.pbo = float(pbo_results.pbo)
            self.results.is_sharpe = float(pbo_results.is_sharpe)
            self.results.oos_sharpe = float(pbo_results.oos_sharpe)
        except Exception as e:
            warnings.warn(f"PBO calculation failed: {e}")
    
    def test_robustness(
        self,
        real_returns: pd.Series,
        strategy_fn: Callable
    ) -> None:
        """
        Test strategy robustness on synthetic data.
        
        Args:
            real_returns: Historical returns
            strategy_fn: Function that takes prices and returns metrics
        """
        try:
            robustness_results = self.synthetic_data_generator.test_robustness(
                real_returns,
                strategy_fn,
                self.config.n_simulations
            )
            
            self.results.synthetic_data_results = robustness_results
        except Exception as e:
            warnings.warn(f"Robustness testing failed: {e}")

