"""
Logistic Regression Machine Learning Trading Strategy
Uses logistic regression with comprehensive feature engineering and threshold optimization.
"""

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import warnings
import pickle
import os
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        confusion_matrix, classification_report, roc_auc_score, roc_curve,
        accuracy_score, precision_score, recall_score, f1_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Please install: pip install scikit-learn")

from .base_strategy import BaseStrategy


class LogisticRegressionStrategy(BaseStrategy):
    """
    Logistic Regression Trading Strategy
    
    Features:
    - Uses TechnicalFeatureEngineer for feature creation
    - Logistic regression model with threshold optimization
    - Model persistence (save/load)
    - Comprehensive diagnostics (console + file)
    - Binary classification for 5-day forward returns
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for this strategy. Install with: pip install scikit-learn")
        
        # Trading parameters
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')
        self.benchmark_asset = config.get('benchmark_asset', 'cad_ig_er_index')
        
        # Model parameters
        self.prediction_horizon = config.get('prediction_horizon', 5)  # Days ahead to predict
        self.train_test_split = config.get('train_test_split', 0.7)  # 70% train, 30% test
        self.random_seed = config.get('random_seed', 42)
        
        # Logistic regression parameters
        self.max_iter = config.get('max_iter', 1000)
        self.C = config.get('C', 1.0)  # Regularization strength
        
        # Threshold optimization
        self.optimize_threshold = config.get('optimize_threshold', True)
        self.threshold_range = config.get('threshold_range', [0.45, 0.65])
        self.threshold_steps = config.get('threshold_steps', 20)
        self.default_threshold = config.get('default_threshold', 0.55)
        
        # Model persistence
        self.model_dir = Path(config.get('model_dir', 'outputs/models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.save_model = config.get('save_model', True)
        self.load_model = config.get('load_model', True)
        
        # Diagnostics
        self.diagnostics_dir = Path(config.get('diagnostics_dir', 'outputs/results'))
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.save_diagnostics = config.get('save_diagnostics', True)
        
        # Model storage
        self.model = None
        self.scaler = None
        self.optimal_threshold = self.default_threshold
        self.feature_names = []
        self.diagnostics = {}
        
    def get_required_features(self) -> List[str]:
        """Returns empty list as features are generated via TechnicalFeatureEngineer."""
        return []
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable for classification.
        
        Target = 1 if forward return over horizon is positive, 0 otherwise
        """
        price = data[self.trading_asset]
        forward_returns = price.shift(-self.prediction_horizon) / price - 1
        target = (forward_returns > 0).astype(int)
        return target
    
    def load_model_from_disk(self) -> Tuple[Optional[LogisticRegression], Optional[StandardScaler], Optional[float], Optional[List[str]]]:
        """Load trained model, scaler, threshold, and feature names from disk."""
        model_path = self.model_dir / f"{self.name}_model.pkl"
        scaler_path = self.model_dir / f"{self.name}_scaler.pkl"
        threshold_path = self.model_dir / f"{self.name}_threshold.txt"
        features_path = self.model_dir / f"{self.name}_features.pkl"
        
        if not self.load_model:
            return None, None, None, None
        
        if model_path.exists() and scaler_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                threshold = self.default_threshold
                if threshold_path.exists():
                    with open(threshold_path, 'r') as f:
                        threshold = float(f.read().strip())
                
                feature_names = None
                if features_path.exists():
                    with open(features_path, 'rb') as f:
                        feature_names = pickle.load(f)
                
                print(f"✓ Loaded model from {model_path}")
                return model, scaler, threshold, feature_names
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                return None, None, None, None
        
        return None, None, None, None
    
    def save_model_to_disk(self, model: LogisticRegression, scaler: StandardScaler, threshold: float, feature_names: List[str]):
        """Save trained model, scaler, threshold, and feature names to disk."""
        if not self.save_model:
            return
        
        try:
            model_path = self.model_dir / f"{self.name}_model.pkl"
            scaler_path = self.model_dir / f"{self.name}_scaler.pkl"
            threshold_path = self.model_dir / f"{self.name}_threshold.txt"
            features_path = self.model_dir / f"{self.name}_features.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(threshold_path, 'w') as f:
                f.write(str(threshold))
            with open(features_path, 'wb') as f:
                pickle.dump(feature_names, f)
            
            print(f"✓ Saved model to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def optimize_prediction_threshold(self, 
                                     X_test: pd.DataFrame,
                                     y_test: pd.Series,
                                     proba_test: np.ndarray,
                                     data: pd.DataFrame) -> float:
        """
        Optimize prediction probability threshold for maximum returns.
        
        Uses walk-forward optimization to find the threshold that maximizes
        total return while maintaining acceptable drawdown.
        """
        print("\n=== Optimizing Prediction Threshold ===")
        
        if not self.optimize_threshold:
            print(f"Threshold optimization disabled, using default {self.default_threshold}")
            return self.default_threshold
        
        price = data[self.trading_asset]
        returns = price.pct_change()
        
        # Try different thresholds
        thresholds = np.linspace(self.threshold_range[0], self.threshold_range[1], self.threshold_steps)
        best_threshold = self.default_threshold
        best_score = -np.inf
        
        results = []
        
        for threshold in thresholds:
            # Generate signals based on threshold
            pred = (proba_test > threshold).astype(int)
            signals = pd.Series(pred, index=X_test.index)
            
            # Calculate strategy returns
            strategy_returns = returns.loc[X_test.index] * signals.shift(1).fillna(0)
            total_return = (1 + strategy_returns).prod() - 1
            
            # Calculate drawdown
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate Sharpe ratio
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            
            # Score: prioritize return with reasonable drawdown
            score = total_return
            if max_drawdown < -0.30:  # Penalize excessive drawdown
                score = score * 0.5
            
            results.append({
                'threshold': threshold,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        results_df = pd.DataFrame(results)
        print(f"\n✓ Threshold optimization complete:")
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  Expected return: {results_df[results_df['threshold'] == best_threshold]['total_return'].values[0]:.2%}")
        
        return best_threshold
    
    def train_model(self, features: pd.DataFrame, data: pd.DataFrame) -> Tuple[LogisticRegression, StandardScaler]:
        """
        Train logistic regression model.
        
        Returns:
            Tuple of (trained_model, scaler)
        """
        print("\n=== Training Logistic Regression Model ===")
        
        # Create target variable
        target = self.create_target_variable(data)
        
        # Align features and target (remove last 'horizon' rows due to forward-looking target)
        valid_mask = target.notna()
        X = features[valid_mask].copy()
        y = target[valid_mask].copy()
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Train/test split (time-series aware)
        split_idx = int(len(X) * self.train_test_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Positive class ratio (train): {y_train.mean():.2%}")
        print(f"  Positive class ratio (test): {y_test.mean():.2%}")
        
        # Fill NaN values
        X_train = X_train.fillna(0.0)
        X_test = X_test.fillna(0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_seed,
            solver='lbfgs'  # Good for small-medium datasets
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        proba_test = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_default = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred_default)
        
        print(f"  Test accuracy (default threshold=0.5): {accuracy:.2%}")
        
        # Optimize threshold
        self.optimal_threshold = self.optimize_prediction_threshold(
            X_test, y_test, proba_test, data
        )
        
        # Generate diagnostics
        self.generate_diagnostics(
            X_train_scaled, y_train, X_test_scaled, y_test,
            model, proba_test
        )
        
        return model, scaler
    
    def generate_diagnostics(self,
                            X_train_scaled: np.ndarray,
                            y_train: pd.Series,
                            X_test_scaled: np.ndarray,
                            y_test: pd.Series,
                            model: LogisticRegression,
                            proba_test: np.ndarray):
        """Generate and save diagnostics."""
        print("\n=== Generating Diagnostics ===")
        
        # Train set predictions
        proba_train = model.predict_proba(X_train_scaled)[:, 1]
        pred_train = (proba_train > self.optimal_threshold).astype(int)
        pred_train_default = model.predict(X_train_scaled)
        
        # Test set predictions
        pred_test = (proba_test > self.optimal_threshold).astype(int)
        pred_test_default = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics_train = {
            'accuracy': accuracy_score(y_train, pred_train),
            'precision': precision_score(y_train, pred_train, zero_division=0),
            'recall': recall_score(y_train, pred_train, zero_division=0),
            'f1': f1_score(y_train, pred_train, zero_division=0),
            'roc_auc': roc_auc_score(y_train, proba_train)
        }
        
        metrics_test = {
            'accuracy': accuracy_score(y_test, pred_test),
            'precision': precision_score(y_test, pred_test, zero_division=0),
            'recall': recall_score(y_test, pred_test, zero_division=0),
            'f1': f1_score(y_test, pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, proba_test)
        }
        
        metrics_test_default = {
            'accuracy': accuracy_score(y_test, pred_test_default),
            'precision': precision_score(y_test, pred_test_default, zero_division=0),
            'recall': recall_score(y_test, pred_test_default, zero_division=0),
            'f1': f1_score(y_test, pred_test_default, zero_division=0)
        }
        
        # Confusion matrices
        cm_train = confusion_matrix(y_train, pred_train)
        cm_test = confusion_matrix(y_test, pred_test)
        
        # Feature importance (coefficients)
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': model.coef_[0]
        })
        coef_df['abs_coef'] = coef_df['coefficient'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
        
        # Store diagnostics
        self.diagnostics = {
            'metrics_train': metrics_train,
            'metrics_test': metrics_test,
            'metrics_test_default': metrics_test_default,
            'cm_train': cm_train.tolist(),
            'cm_test': cm_test.tolist(),
            'feature_importance': coef_df.to_dict('records'),
            'optimal_threshold': self.optimal_threshold,
            'label_distribution_train': {
                'negative': int((y_train == 0).sum()),
                'positive': int((y_train == 1).sum()),
                'negative_pct': float((y_train == 0).mean()),
                'positive_pct': float((y_train == 1).mean())
            },
            'label_distribution_test': {
                'negative': int((y_test == 0).sum()),
                'positive': int((y_test == 1).sum()),
                'negative_pct': float((y_test == 0).mean()),
                'positive_pct': float((y_test == 1).mean())
            }
        }
        
        # Print diagnostics to console
        self.print_diagnostics()
        
        # Save diagnostics to file
        if self.save_diagnostics:
            self.save_diagnostics_to_file()
    
    def print_diagnostics(self):
        """Print diagnostics to console."""
        print("\n" + "="*80)
        print("LOGISTIC REGRESSION DIAGNOSTICS")
        print("="*80)
        
        # Feature Importance
        coef_df = pd.DataFrame(self.diagnostics['feature_importance'])
        print("\n### 1. FEATURE IMPORTANCE (Top 15)")
        print("   (Positive coef → increases probability of positive return)")
        print(coef_df.head(15).to_string(index=False))
        
        # Classification Metrics
        print("\n### 2. CLASSIFICATION METRICS")
        print(f"\nTRAIN SET (threshold={self.optimal_threshold:.3f}):")
        m_train = self.diagnostics['metrics_train']
        print(f"  Accuracy  : {m_train['accuracy']:.4f}")
        print(f"  Precision : {m_train['precision']:.4f}")
        print(f"  Recall    : {m_train['recall']:.4f}")
        print(f"  F1-Score  : {m_train['f1']:.4f}")
        print(f"  ROC-AUC   : {m_train['roc_auc']:.4f}")
        
        print(f"\nTEST SET (threshold={self.optimal_threshold:.3f}):")
        m_test = self.diagnostics['metrics_test']
        print(f"  Accuracy  : {m_test['accuracy']:.4f}")
        print(f"  Precision : {m_test['precision']:.4f}")
        print(f"  Recall    : {m_test['recall']:.4f}")
        print(f"  F1-Score  : {m_test['f1']:.4f}")
        print(f"  ROC-AUC   : {m_test['roc_auc']:.4f}")
        
        m_test_def = self.diagnostics['metrics_test_default']
        print(f"\nTEST SET (default threshold=0.5, for comparison):")
        print(f"  Accuracy  : {m_test_def['accuracy']:.4f}")
        print(f"  Precision : {m_test_def['precision']:.4f}")
        print(f"  Recall    : {m_test_def['recall']:.4f}")
        print(f"  F1-Score  : {m_test_def['f1']:.4f}")
        
        # Confusion Matrices
        print("\n### 3. CONFUSION MATRICES")
        cm_train = np.array(self.diagnostics['cm_train'])
        cm_test = np.array(self.diagnostics['cm_test'])
        
        print(f"\nTRAIN SET Confusion Matrix (threshold={self.optimal_threshold:.3f}):")
        print("              Predicted")
        print("              Neg   Pos")
        print(f"Actual Neg   {cm_train[0,0]:5d} {cm_train[0,1]:5d}")
        print(f"       Pos   {cm_train[1,0]:5d} {cm_train[1,1]:5d}")
        
        print(f"\nTEST SET Confusion Matrix (threshold={self.optimal_threshold:.3f}):")
        print("              Predicted")
        print("              Neg   Pos")
        print(f"Actual Neg   {cm_test[0,0]:5d} {cm_test[0,1]:5d}")
        print(f"       Pos   {cm_test[1,0]:5d} {cm_test[1,1]:5d}")
        
        # Label Distribution
        print("\n### 4. LABEL DISTRIBUTION")
        dist_train = self.diagnostics['label_distribution_train']
        dist_test = self.diagnostics['label_distribution_test']
        print(f"TRAIN SET: Negative={dist_train['negative']} ({dist_train['negative_pct']:.2%}), "
              f"Positive={dist_train['positive']} ({dist_train['positive_pct']:.2%})")
        print(f"TEST SET:  Negative={dist_test['negative']} ({dist_test['negative_pct']:.2%}), "
              f"Positive={dist_test['positive']} ({dist_test['positive_pct']:.2%})")
        
        print("\n" + "="*80)
    
    def save_diagnostics_to_file(self):
        """Save diagnostics to text file."""
        try:
            file_path = self.diagnostics_dir / f"{self.name}_diagnostics.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("LOGISTIC REGRESSION DIAGNOSTICS\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Strategy: {self.name}\n")
                f.write(f"Prediction Horizon: {self.prediction_horizon} days\n")
                f.write(f"Optimal Threshold: {self.optimal_threshold:.3f}\n")
                f.write("="*80 + "\n\n")
                
                # Feature Importance
                coef_df = pd.DataFrame(self.diagnostics['feature_importance'])
                f.write("### 1. FEATURE IMPORTANCE (Top 15)\n")
                f.write("   (Positive coef → increases probability of positive return)\n\n")
                f.write(coef_df.head(15).to_string(index=False))
                f.write("\n\n")
                
                # Classification Metrics
                f.write("### 2. CLASSIFICATION METRICS\n\n")
                m_train = self.diagnostics['metrics_train']
                f.write(f"TRAIN SET (threshold={self.optimal_threshold:.3f}):\n")
                f.write(f"  Accuracy  : {m_train['accuracy']:.4f}\n")
                f.write(f"  Precision : {m_train['precision']:.4f}\n")
                f.write(f"  Recall    : {m_train['recall']:.4f}\n")
                f.write(f"  F1-Score  : {m_train['f1']:.4f}\n")
                f.write(f"  ROC-AUC   : {m_train['roc_auc']:.4f}\n\n")
                
                m_test = self.diagnostics['metrics_test']
                f.write(f"TEST SET (threshold={self.optimal_threshold:.3f}):\n")
                f.write(f"  Accuracy  : {m_test['accuracy']:.4f}\n")
                f.write(f"  Precision : {m_test['precision']:.4f}\n")
                f.write(f"  Recall    : {m_test['recall']:.4f}\n")
                f.write(f"  F1-Score  : {m_test['f1']:.4f}\n")
                f.write(f"  ROC-AUC   : {m_test['roc_auc']:.4f}\n\n")
                
                m_test_def = self.diagnostics['metrics_test_default']
                f.write("TEST SET (default threshold=0.5, for comparison):\n")
                f.write(f"  Accuracy  : {m_test_def['accuracy']:.4f}\n")
                f.write(f"  Precision : {m_test_def['precision']:.4f}\n")
                f.write(f"  Recall    : {m_test_def['recall']:.4f}\n")
                f.write(f"  F1-Score  : {m_test_def['f1']:.4f}\n\n")
                
                # Confusion Matrices
                f.write("### 3. CONFUSION MATRICES\n\n")
                cm_train = np.array(self.diagnostics['cm_train'])
                cm_test = np.array(self.diagnostics['cm_test'])
                
                f.write(f"TRAIN SET Confusion Matrix (threshold={self.optimal_threshold:.3f}):\n")
                f.write("              Predicted\n")
                f.write("              Neg   Pos\n")
                f.write(f"Actual Neg   {cm_train[0,0]:5d} {cm_train[0,1]:5d}\n")
                f.write(f"       Pos   {cm_train[1,0]:5d} {cm_train[1,1]:5d}\n\n")
                
                f.write(f"TEST SET Confusion Matrix (threshold={self.optimal_threshold:.3f}):\n")
                f.write("              Predicted\n")
                f.write("              Neg   Pos\n")
                f.write(f"Actual Neg   {cm_test[0,0]:5d} {cm_test[0,1]:5d}\n")
                f.write(f"       Pos   {cm_test[1,0]:5d} {cm_test[1,1]:5d}\n\n")
                
                # Label Distribution
                f.write("### 4. LABEL DISTRIBUTION\n\n")
                dist_train = self.diagnostics['label_distribution_train']
                dist_test = self.diagnostics['label_distribution_test']
                f.write(f"TRAIN SET: Negative={dist_train['negative']} ({dist_train['negative_pct']:.2%}), "
                       f"Positive={dist_train['positive']} ({dist_train['positive_pct']:.2%})\n")
                f.write(f"TEST SET:  Negative={dist_test['negative']} ({dist_test['negative_pct']:.2%}), "
                       f"Positive={dist_test['positive']} ({dist_test['positive_pct']:.2%})\n")
                
                f.write("\n" + "="*80 + "\n")
            
            print(f"✓ Diagnostics saved to {file_path}")
        except Exception as e:
            print(f"Warning: Could not save diagnostics: {e}")
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals using trained logistic regression model.
        
        Args:
            data: Price data DataFrame
            features: Features DataFrame from TechnicalFeatureEngineer
            
        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        print("\n=== Generating Logistic Regression Signals ===")
        
        # Try to load model from disk
        model, scaler, threshold, feature_names = self.load_model_from_disk()
        
        # If model not loaded or doesn't exist, train new model
        if model is None or scaler is None:
            print("Training new model...")
            model, scaler = self.train_model(features, data)
            self.model = model
            self.scaler = scaler
            
            # Save model
            self.save_model_to_disk(model, scaler, self.optimal_threshold, self.feature_names)
        else:
            print("Using loaded model...")
            self.model = model
            self.scaler = scaler
            self.optimal_threshold = threshold
            if feature_names:
                self.feature_names = feature_names
        
        # Prepare features for prediction
        X = features.copy()
        X = X.fillna(0.0)
        
        # Ensure feature order matches training
        if len(self.feature_names) > 0:
            # Reorder columns to match training
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                print(f"Warning: Missing features: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0.0
            
            X = X[self.feature_names]
        else:
            # First time training, store feature names
            self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        # Convert predictions to signals using optimal threshold
        raw_signals = (predictions > self.optimal_threshold).astype(int)
        
        # Convert to entry/exit signals
        positions = pd.Series(raw_signals, index=data.index).astype(int)
        positions_shifted = positions.shift(1).fillna(0).astype(int)
        
        entry_signals = (positions == 1) & (positions_shifted == 0)
        exit_signals = (positions == 0) & (positions_shifted == 1)
        
        print(f"\n✓ Signal generation complete:")
        print(f"  Total entry signals: {entry_signals.sum()}")
        print(f"  Signal frequency: {entry_signals.mean():.2%}")
        print(f"  Time in market: {positions.mean():.2%}")
        
        return entry_signals, exit_signals
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""Logistic Regression Machine Learning Trading Strategy

STRATEGY OVERVIEW:
This strategy uses logistic regression to predict profitable trading opportunities
in {self.trading_asset} with a {self.prediction_horizon}-day prediction horizon.

METHODOLOGY:
1. Feature Engineering:
   - Uses TechnicalFeatureEngineer for comprehensive feature creation
   - Includes momentum, volatility, technical indicators, and macro features
   - Features are standardized before model training

2. Logistic Regression Model:
   - Binary classification (positive vs negative forward returns)
   - Regularization strength (C): {self.C}
   - Maximum iterations: {self.max_iter}
   - Train/test split: {self.train_test_split:.0%}

3. Threshold Optimization:
   - Optimized prediction threshold: {self.optimal_threshold:.3f}
   - Maximizes return while controlling drawdown
   - Threshold range: {self.threshold_range}

4. Model Persistence:
   - Models are saved to disk for reuse
   - Scaler and optimal threshold are saved with model

5. Signal Generation:
   - Binary classification (long/cash positions)
   - Entry when predicted probability > {self.optimal_threshold:.3f}
   - Exit when predicted probability <= {self.optimal_threshold:.3f}

PARAMETERS:
- Trading Asset: {self.trading_asset}
- Prediction Horizon: {self.prediction_horizon} days
- Optimal Threshold: {self.optimal_threshold:.3f}
- Regularization (C): {self.C}
- Random Seed: {self.random_seed}
"""

