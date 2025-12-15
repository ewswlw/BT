"""
Unit tests for validation framework components.

Tests all validation methods including purged CV, sample weights,
deflated metrics, PBO, walk-forward analysis, and report generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import validation components
try:
    from cad_ig_er_index_backtesting.core.validation import (
        ValidationConfig,
        ValidationResults,
        PurgedKFold,
        CombinatorialPurgedCV,
        compute_label_uniqueness,
        compute_sample_weights,
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
        ProbabilityBacktestOverfitting,
        WalkForwardAnalyzer,
        ValidationFramework,
        ValidationReportGenerator
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    pytest.skip("Validation framework not available", allow_module_level=True)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    prices = 100 * (1 + returns).cumprod()
    
    # Create features
    features = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    }, index=dates)
    
    # Create binary target
    target = pd.Series((returns.shift(-1) > 0).astype(int), index=dates)
    
    return {
        'returns': returns,
        'prices': prices,
        'features': features,
        'target': target,
        'dates': dates
    }


@pytest.fixture
def validation_config():
    """Create validation config for testing."""
    return ValidationConfig(
        cv_method="purged_kfold",
        n_splits=3,
        embargo_pct=0.01,
        use_sample_weights=True,
        n_trials=10
    )


class TestPurgedKFold:
    """Test PurgedKFold cross-validation."""
    
    def test_basic_split(self, sample_data):
        """Test basic purged CV split."""
        cv = PurgedKFold(n_splits=3, pct_embargo=0.01)
        X = sample_data['features']
        y = sample_data['target']
        
        splits = list(cv.split(X, y))
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_with_samples_info_sets(self, sample_data):
        """Test purged CV with samples_info_sets."""
        dates = sample_data['dates']
        samples_info_sets = [(d, d + timedelta(days=7)) for d in dates]
        
        cv = PurgedKFold(
            n_splits=3,
            samples_info_sets=samples_info_sets,
            pct_embargo=0.01
        )
        
        X = sample_data['features']
        y = sample_data['target']
        
        splits = list(cv.split(X, y))
        assert len(splits) == 3


class TestSampleWeights:
    """Test sample weight calculations."""
    
    def test_compute_label_uniqueness(self, sample_data):
        """Test label uniqueness calculation."""
        dates = sample_data['dates']
        label_times = [(d, d + timedelta(days=7)) for d in dates[:20]]
        price_index = dates
        
        uniqueness = compute_label_uniqueness(label_times, price_index)
        
        assert len(uniqueness) == 20
        assert (uniqueness >= 0).all()
        assert (uniqueness <= 1).all()
    
    def test_compute_sample_weights(self, sample_data):
        """Test sample weight computation."""
        uniqueness = pd.Series([0.5, 0.7, 0.9, 1.0, 0.8])
        weights = compute_sample_weights(uniqueness)
        
        assert len(weights) == len(uniqueness)
        assert weights.sum() == pytest.approx(len(weights), rel=1e-6)


class TestDeflatedMetrics:
    """Test deflated metrics calculations."""
    
    def test_deflated_sharpe_ratio(self):
        """Test deflated Sharpe ratio calculation."""
        dsr = deflated_sharpe_ratio(
            sharpe=2.0,
            n_trials=100,
            n_observations=1000
        )
        
        assert 0 <= dsr <= 1
    
    def test_probabilistic_sharpe_ratio(self, sample_data):
        """Test probabilistic Sharpe ratio."""
        returns = sample_data['returns']
        psr = probabilistic_sharpe_ratio(returns)
        
        assert 0 <= psr <= 1


class TestPBO:
    """Test Probability of Backtest Overfitting."""
    
    def test_pbo_calculation(self):
        """Test PBO calculation."""
        # Create returns matrix with multiple configurations
        np.random.seed(42)
        returns_matrix = pd.DataFrame({
            'config1': np.random.randn(100) * 0.01,
            'config2': np.random.randn(100) * 0.01,
            'config3': np.random.randn(100) * 0.01,
            'config4': np.random.randn(100) * 0.01,
            'config5': np.random.randn(100) * 0.01,
        })
        
        calculator = ProbabilityBacktestOverfitting()
        results = calculator.calculate_pbo(returns_matrix)
        
        assert 0 <= results.pbo <= 1
        assert results.is_sharpe is not None
        assert results.oos_sharpe is not None


class TestWalkForward:
    """Test walk-forward analysis."""
    
    def test_walk_forward_analyzer(self, sample_data):
        """Test walk-forward analyzer."""
        analyzer = WalkForwardAnalyzer(
            train_period=20,
            test_period=10,
            step=5
        )
        
        features = sample_data['features']
        returns = sample_data['returns']
        
        results = analyzer.analyze(features, returns)
        
        assert len(results) > 0
        for result in results:
            assert result.train_start < result.test_start
            assert result.test_start < result.test_end


class TestValidationFramework:
    """Test ValidationFramework orchestrator."""
    
    def test_validation_framework(self, sample_data, validation_config):
        """Test full validation framework."""
        framework = ValidationFramework(validation_config)
        
        X = sample_data['features']
        y = sample_data['target']
        returns = sample_data['returns']
        
        # Prepare samples_info_sets
        dates = sample_data['dates']
        samples_info_sets = [(d, d + timedelta(days=7)) for d in dates]
        
        results = framework.validate(X, y, returns, samples_info_sets)
        
        assert results is not None
        assert isinstance(results, ValidationResults)


class TestValidationReport:
    """Test validation report generation."""
    
    def test_report_generation(self, sample_data, validation_config):
        """Test report generation."""
        generator = ValidationReportGenerator()
        
        results = ValidationResults()
        results.cv_scores = [0.6, 0.65, 0.7]
        results.cv_mean = 0.65
        results.cv_std = 0.05
        results.cv_method = "purged_kfold"
        results.deflated_sharpe = 0.85
        results.probabilistic_sharpe = 0.90
        
        report_path = generator.generate_report(
            "test_strategy",
            results,
            {}
        )
        
        assert Path(report_path).exists()
        
        # Read and verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert "VALIDATION REPORT" in content
            assert "test_strategy" in content.upper()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

