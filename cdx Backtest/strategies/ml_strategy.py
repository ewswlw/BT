"""
ML ensemble strategy with Random Forest, LightGBM, XGBoost, and CatBoost.
"""

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_strategy import BaseStrategy
from core.feature_selection import FeatureSelector

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: catboost not available")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available")


class MLStrategy(BaseStrategy):
    """
    ML ensemble strategy with multiple models.
    
    Uses Random Forest, LightGBM, XGBoost, and CatBoost in an ensemble.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for this strategy")
        
        # Trading parameters
        self.trading_asset = config.get('trading_asset', 'us_ig_cdx_er_index')
        self.forward_return_periods = config.get('forward_return_periods', 7)  # 7-day forward return
        
        # Model configurations
        self.use_random_forest = config.get('use_random_forest', True)
        self.use_lightgbm = config.get('use_lightgbm', True)
        self.use_xgboost = config.get('use_xgboost', True)
        self.use_catboost = config.get('use_catboost', False)
        
        # Ensemble parameters
        self.ensemble_weights = config.get('ensemble_weights', [0.3, 0.3, 0.3, 0.1])  # RF, LGB, XGB, CB
        self.probability_threshold = config.get('probability_threshold', 0.55)
        
        # Model storage
        self.trained_models = []
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Hyperparameter optimization
        self.optimize_hyperparameters = config.get('optimize_hyperparameters', False)
        self.n_trials = config.get('n_trials', 50)
        
        # Feature selection
        self.use_feature_selection = config.get('use_feature_selection', True)
        self.feature_selection_method = config.get('feature_selection_method', 'combined')
        self.top_k_features = config.get('top_k_features', 150)
        self.feature_selector = FeatureSelector(method=self.feature_selection_method, 
                                               top_k=self.top_k_features) if self.use_feature_selection else None
    
    def get_required_features(self) -> List[str]:
        """Returns empty list as features are generated internally."""
        return []
    
    def create_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable with very selective threshold.
        
        Uses 80th percentile OR absolute return threshold (whichever is higher).
        Targets only the very best opportunities.
        """
        if self.trading_asset not in data.columns:
            raise ValueError(f"Trading asset {self.trading_asset} not found in data")
        
        # Calculate forward return
        forward_return = data[self.trading_asset].pct_change(self.forward_return_periods).shift(-self.forward_return_periods)
        
        # Use 80th percentile as base threshold (very selective - targets top 20% of returns)
        percentile_threshold = forward_return.quantile(0.80)
        
        # Also use absolute return threshold (aim for at least 0.2% per period - aggressive)
        absolute_threshold = 0.002  # 0.2% per period
        
        # Use the higher of the two thresholds
        threshold = max(percentile_threshold, absolute_threshold)
        
        # Target is 1 if return is above threshold
        target = (forward_return > threshold).astype(int)
        
        return target
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> List[Dict]:
        """Train ensemble of models."""
        models = []
        
        # Prepare data - replace infinity and NaN
        X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Check for any remaining issues
        if X_train_clean.isnull().any().any():
            X_train_clean = X_train_clean.fillna(0)
        
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        self.feature_names = X_train.columns.tolist()
        
        model_idx = 0
        
        # Random Forest
        if self.use_random_forest and SKLEARN_AVAILABLE:
            print("  Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_leaf=5,
                max_features=0.4,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            models.append({
                'model': rf,
                'type': 'random_forest',
                'weight': self.ensemble_weights[model_idx] if model_idx < len(self.ensemble_weights) else 0.25
            })
            model_idx += 1
        
        # LightGBM
        if self.use_lightgbm and LIGHTGBM_AVAILABLE:
            print("  Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            models.append({
                'model': lgb_model,
                'type': 'lightgbm',
                'weight': self.ensemble_weights[model_idx] if model_idx < len(self.ensemble_weights) else 0.25
            })
            model_idx += 1
        
        # XGBoost
        if self.use_xgboost and XGBOOST_AVAILABLE:
            print("  Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            models.append({
                'model': xgb_model,
                'type': 'xgboost',
                'weight': self.ensemble_weights[model_idx] if model_idx < len(self.ensemble_weights) else 0.25
            })
            model_idx += 1
        
        # CatBoost
        if self.use_catboost and CATBOOST_AVAILABLE:
            print("  Training CatBoost...")
            cb_model = cb.CatBoostClassifier(
                iterations=500,
                depth=10,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            cb_model.fit(X_train, y_train)
            models.append({
                'model': cb_model,
                'type': 'catboost',
                'weight': self.ensemble_weights[model_idx] if model_idx < len(self.ensemble_weights) else 0.25
            })
            model_idx += 1
        
        # Normalize weights
        total_weight = sum(m['weight'] for m in models)
        if total_weight > 0:
            for m in models:
                m['weight'] = m['weight'] / total_weight
        
        self.trained_models = models
        return models
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions."""
        if not self.trained_models:
            raise ValueError("Models not trained yet")
        
        predictions = []
        weights = []
        
        for model_dict in self.trained_models:
            model = model_dict['model']
            model_type = model_dict['type']
            weight = model_dict['weight']
            
            if model_type == 'random_forest':
                X_scaled = self.scaler.transform(X)
                proba = model.predict_proba(X_scaled)[:, 1]
            else:
                # LightGBM, XGBoost, CatBoost use original features
                proba = model.predict_proba(X.values)[:, 1]
            
            predictions.append(proba)
            weights.append(weight)
        
        # Weighted ensemble
        ensemble_proba = np.zeros(len(X))
        for proba, weight in zip(predictions, weights):
            ensemble_proba += proba * weight
        
        return ensemble_proba
    
    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                        train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals using trained ML models.
        
        Args:
            train_data: Training price data
            test_data: Test price data
            train_features: Training features
            test_features: Test features
            
        Returns:
            Tuple of (entry_signals, exit_signals) for test_data
        """
        print(f"\n=== ML Strategy: {self.name} ===")
        
        # Create target for training
        print("Creating target variable...")
        y_train = self.create_target(train_data)
        
        # Remove duplicate indices
        train_data = train_data[~train_data.index.duplicated(keep='first')]
        test_data = test_data[~test_data.index.duplicated(keep='first')]
        train_features = train_features[~train_features.index.duplicated(keep='first')]
        test_features = test_features[~test_features.index.duplicated(keep='first')]
        
        # Align features and target
        train_features_aligned, y_train_aligned = train_features.align(y_train, join='inner', axis=0)
        
        # Remove NaN rows
        valid_mask = ~(train_features_aligned.isnull().any(axis=1) | y_train_aligned.isnull())
        train_features_clean = train_features_aligned[valid_mask]
        y_train_clean = y_train_aligned[valid_mask]
        
        if len(train_features_clean) < 100:
            raise ValueError(f"Insufficient training data: {len(train_features_clean)} samples")
        
        # Feature selection
        if self.use_feature_selection and self.feature_selector:
            print(f"Selecting top {self.top_k_features} features using {self.feature_selection_method}...")
            train_features_clean = self.feature_selector.select_features(
                train_features_clean, y_train_clean, method=self.feature_selection_method
            )
            selected_feature_names = self.feature_selector.get_selected_features()
            print(f"Selected {len(selected_feature_names)} features")
        
        # Train models
        print("Training ensemble models...")
        self.train_models(train_features_clean, y_train_clean)
        
        # Generate predictions for test data
        print("Generating predictions for test data...")
        test_features_aligned = test_features.reindex(test_data.index).ffill()
        test_features_clean = test_features_aligned.fillna(0)
        
        # Apply feature selection to test features if used
        if self.use_feature_selection and self.feature_selector and hasattr(self.feature_selector, 'selected_features'):
            selected_features = self.feature_selector.get_selected_features()
            if selected_features:
                # Only keep selected features
                available_selected = [f for f in selected_features if f in test_features_clean.columns]
                test_features_clean = test_features_clean[available_selected]
        
        # Ensure same features as training
        missing_features = set(train_features_clean.columns) - set(test_features_clean.columns)
        if missing_features:
            for feat in missing_features:
                test_features_clean[feat] = 0
        
        extra_features = set(test_features_clean.columns) - set(train_features_clean.columns)
        if extra_features:
            test_features_clean = test_features_clean[train_features_clean.columns]
        
        # Clean test features - replace infinity and NaN
        test_features_clean = test_features_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Get probabilities
        probabilities = self.predict_proba(test_features_clean)
        
        # Convert to signals
        raw_signals = (probabilities > self.probability_threshold).astype(int)
        raw_signals_series = pd.Series(raw_signals, index=test_data.index)
        
        # Apply holding period
        print(f"Applying {self.holding_period_days}-day holding period...")
        adjusted_signals = self.apply_holding_period(raw_signals_series, self.holding_period_days)
        
        # Convert to entry/exit signals
        positions = adjusted_signals.astype(int)
        positions_shifted = positions.shift(1).fillna(0).astype(int)
        
        entry_signals = (positions == 1) & (positions_shifted == 0)
        exit_signals = (positions == 0) & (positions_shifted == 1)
        
        print(f"  Entry signals: {entry_signals.sum()}")
        print(f"  Exit signals: {exit_signals.sum()}")
        print(f"  Time in market: {positions.mean():.2%}")
        
        return entry_signals, exit_signals

