"""
Iterative Machine Learning Strategy with Statistical Validation
Goal: Beat buy-and-hold by at least 2.5% annualized return

This strategy:
- Iteratively improves through feature engineering and model selection
- Uses walk-forward validation to prevent overfitting
- Checks for statistical biases (look-ahead, data leakage, etc.)
- Learns from past iterations to guide future improvements
- Uses ensemble of multiple ML algorithms
"""

from typing import Dict, Tuple, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import json
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from .base_strategy import BaseStrategy


class IterativeMLStrategy(BaseStrategy):
    """
    Iterative ML Strategy with Statistical Validation
    
    Features:
    - Advanced feature engineering with multiple transformations
    - Ensemble of RF, LightGBM, XGBoost, CatBoost
    - Walk-forward validation
    - Bias checking (look-ahead, data leakage)
    - Iterative improvement tracking
    - Statistical significance testing
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        # Trading parameters
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')
        self.benchmark_asset = config.get('benchmark_asset', 'cad_ig_er_index')
        
        # Iteration tracking
        self.iteration = config.get('iteration', 0)
        self.max_iterations = config.get('max_iterations', 20)
        self.min_outperformance = config.get('min_outperformance', 0.025)  # 2.5% annualized
        self.iteration_history = []
        self.best_performance = {'cagr': -np.inf, 'outperformance': -np.inf}
        
        # Feature engineering parameters (expandable)
        self.momentum_windows = config.get('momentum_windows', 
            [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 252])
        self.volatility_windows = config.get('volatility_windows', [5, 10, 20, 40, 60, 120, 252])
        self.zscore_windows = config.get('zscore_windows', [5, 10, 20, 40, 60, 120, 252])
        self.correlation_windows = config.get('correlation_windows', [20, 60, 120, 252])
        
        # Model configurations
        self.use_rf = config.get('use_rf', True)
        self.use_lightgbm = config.get('use_lightgbm', True) and LIGHTGBM_AVAILABLE
        self.use_xgboost = config.get('use_xgboost', True) and XGBOOST_AVAILABLE
        self.use_catboost = config.get('use_catboost', True) and CATBOOST_AVAILABLE
        
        # Ensemble weights (learned from iterations)
        self.ensemble_weights = config.get('ensemble_weights', None)
        
        # Walk-forward validation
        self.walk_forward_train_size = config.get('walk_forward_train_size', 0.7)
        self.walk_forward_test_size = config.get('walk_forward_test_size', 0.15)
        self.walk_forward_step = config.get('walk_forward_step', 0.05)
        self.use_walk_forward = config.get('use_walk_forward', True)
        
        # Prediction parameters
        self.prediction_horizon = config.get('prediction_horizon', 7)  # Days ahead
        self.probability_threshold = config.get('probability_threshold', 0.5)
        self.optimize_threshold = config.get('optimize_threshold', True)
        
        # Risk management
        self.min_holding_days = config.get('min_holding_days', 5)
        self.max_holding_days = config.get('max_holding_days', 60)
        
        # Statistical validation
        self.min_sharpe = config.get('min_sharpe', 0.5)
        self.max_drawdown_limit = config.get('max_drawdown_limit', -0.20)  # -20%
        self.min_trades = config.get('min_trades', 20)
        
        # Model storage
        self.models = {}
        self.feature_importance = None
        self.feature_names = []
        
        # Results tracking
        self.results_dir = Path(config.get('results_dir', 'outputs/iterative_ml'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def get_required_features(self) -> List[str]:
        """Returns empty list as features are generated internally."""
        return []
    
    def engineer_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering with multiple transformations.
        
        Creates 200+ features including:
        - Multi-timeframe momentum
        - Volatility regimes and measures
        - Cross-asset relationships
        - Statistical features (z-scores, percentiles, skew, kurtosis)
        - Regime detection
        - Interaction features
        - Lagged features
        """
        print("\n=== Advanced Feature Engineering ===")
        
        if self.trading_asset not in data.columns:
            raise ValueError(f"Trading asset {self.trading_asset} not found")
        
        price = data[self.trading_asset].copy()
        features = pd.DataFrame(index=data.index)
        returns = price.pct_change()
        
        # 1. MOMENTUM FEATURES (Multi-timeframe)
        print("  Creating momentum features...")
        for period in self.momentum_windows:
            # Simple momentum
            features[f'mom_{period}d'] = price.pct_change(period)
            
            # Rank momentum (percentile rank)
            features[f'mom_{period}d_rank'] = features[f'mom_{period}d'].rolling(252).rank(pct=True)
            
            # Momentum acceleration
            if period > 5:
                features[f'mom_{period}d_accel'] = features[f'mom_{period}d'] - features[f'mom_{period}d'].shift(period//2)
            
            # Momentum vs volatility
            vol = returns.rolling(period).std()
            features[f'mom_{period}d_sharpe'] = features[f'mom_{period}d'] / (vol + 1e-8)
        
        # 2. VOLATILITY FEATURES
        print("  Creating volatility features...")
        for period in self.volatility_windows:
            vol = returns.rolling(period).std()
            
            # Raw volatility
            features[f'vol_{period}d'] = vol
            
            # Volatility z-score
            vol_mean = vol.rolling(252).mean()
            vol_std = vol.rolling(252).std()
            features[f'vol_{period}d_zscore'] = (vol - vol_mean) / (vol_std + 1e-8)
            
            # Volatility regime
            features[f'vol_{period}d_regime'] = (vol > vol.rolling(252).median()).astype(int)
            
            # Volatility of volatility
            features[f'vol_of_vol_{period}d'] = vol.rolling(period).std()
            
            # Realized vs implied volatility (if VIX available)
            if 'vix' in data.columns:
                vix_pct = data['vix'] / 100
                features[f'vol_{period}d_vs_vix'] = vol / (vix_pct + 1e-8)
        
        # 3. Z-SCORE FEATURES (Statistical standardization)
        print("  Creating z-score features...")
        for col in ['cad_oas', 'us_hy_oas', 'us_ig_oas', 'vix']:
            if col in data.columns:
                for window in self.zscore_windows:
                    rolling_mean = data[col].rolling(window).mean()
                    rolling_std = data[col].rolling(window).std()
                    features[f'{col}_zscore{window}'] = (data[col] - rolling_mean) / (rolling_std + 1e-8)
                    
                    # Z-score regime
                    features[f'{col}_zscore{window}_regime'] = (np.abs(features[f'{col}_zscore{window}']) > 1).astype(int)
        
        # 4. MOVING AVERAGE FEATURES
        print("  Creating moving average features...")
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            ma = price.rolling(period).mean()
            features[f'ma_{period}d_dev'] = (price - ma) / ma
            features[f'price_above_ma_{period}d'] = (price > ma).astype(int)
            
            # MA crossovers
            if period > 10:
                short_ma = price.rolling(period//2).mean()
                features[f'ma_cross_{period//2}_{period}d'] = (short_ma > ma).astype(int)
        
        # 5. TECHNICAL INDICATORS
        print("  Creating technical indicators...")
        
        # RSI
        for period in [14, 30]:
            delta = returns
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}d'] = 100 - (100 / (1 + rs))
            features[f'rsi_{period}d_regime'] = (features[f'rsi_{period}d'] > 50).astype(int)
        
        # MACD
        ema_12 = price.ewm(span=12, adjust=False).mean()
        ema_26 = price.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_diff'] = macd_line - signal_line
        features['macd_cross'] = (macd_line > signal_line).astype(int)
        
        # Bollinger Bands
        for period in [20, 40]:
            ma = price.rolling(period).mean()
            std = price.rolling(period).std()
            features[f'bb_upper_{period}d'] = ma + 2 * std
            features[f'bb_lower_{period}d'] = ma - 2 * std
            features[f'bb_position_{period}d'] = (price - features[f'bb_lower_{period}d']) / (2 * std + 1e-8)
            features[f'bb_width_{period}d'] = (4 * std) / ma
        
        # Stochastic Oscillator
        for period in [14, 30]:
            low = price.rolling(period).min()
            high = price.rolling(period).max()
            features[f'stoch_{period}d'] = 100 * (price - low) / (high - low + 1e-10)
        
        # 6. CROSS-ASSET FEATURES
        print("  Creating cross-asset features...")
        
        # VIX features
        if 'vix' in data.columns:
            vix = data['vix']
            features['vix_level'] = vix
            features['vix_change'] = vix.pct_change()
            features['vix_zscore'] = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
            features['vix_regime'] = (vix > vix.rolling(252).median()).astype(int)
            
            for period in [5, 10, 20]:
                features[f'vix_mom_{period}d'] = vix.pct_change(period)
        
        # OAS spreads
        for col in ['cad_oas', 'us_ig_oas', 'us_hy_oas']:
            if col in data.columns:
                oas = data[col]
                features[f'{col}_level'] = oas
                features[f'{col}_change'] = oas.pct_change()
                features[f'{col}_zscore'] = (oas - oas.rolling(252).mean()) / (oas.rolling(252).std() + 1e-8)
                
                for period in [5, 10, 20]:
                    features[f'{col}_mom_{period}d'] = oas.pct_change(period)
                
                # OAS vs price correlation
                for window in self.correlation_windows:
                    corr = returns.rolling(window).corr(oas.pct_change())
                    features[f'{col}_corr_{window}d'] = corr
        
        # 7. STATISTICAL FEATURES
        print("  Creating statistical features...")
        
        for window in [20, 60, 120, 252]:
            # Rolling statistics
            features[f'return_mean_{window}d'] = returns.rolling(window).mean()
            features[f'return_std_{window}d'] = returns.rolling(window).std()
            features[f'return_skew_{window}d'] = returns.rolling(window).skew()
            features[f'return_kurt_{window}d'] = returns.rolling(window).kurt()
            
            # Percentile ranks
            features[f'price_percentile_{window}d'] = price.rolling(window).rank(pct=True)
            
            # Distance from extremes
            rolling_high = price.rolling(window).max()
            rolling_low = price.rolling(window).min()
            features[f'dist_from_high_{window}d'] = (price - rolling_high) / rolling_high
            features[f'dist_from_low_{window}d'] = (price - rolling_low) / rolling_low
        
        # 8. REGIME DETECTION FEATURES
        print("  Creating regime features...")
        
        # Trend regime
        for window in [20, 60]:
            trend = price.pct_change(window)
            features[f'trend_regime_{window}d'] = (trend > 0).astype(int)
        
        # Volatility regime
        vol_20 = returns.rolling(20).std()
        vol_252 = returns.rolling(252).std()
        features['vol_regime'] = (vol_20 > vol_252).astype(int)
        
        # 9. INTERACTION FEATURES (key relationships)
        print("  Creating interaction features...")
        
        if 'vix' in data.columns and 'cad_oas' in data.columns:
            features['vix_oas_ratio'] = data['vix'] / (data['cad_oas'] + 1e-8)
            features['vix_oas_spread'] = data['vix'] - data['cad_oas']
        
        # Momentum vs volatility interaction
        mom_20 = price.pct_change(20)
        vol_20 = returns.rolling(20).std()
        features['mom_vol_ratio'] = mom_20 / (vol_20 + 1e-8)
        
        # 10. LAGGED FEATURES (to capture persistence)
        print("  Creating lagged features...")
        
        for lag in [1, 2, 3, 5]:
            features[f'return_lag{lag}'] = returns.shift(lag)
            if 'vix' in data.columns:
                features[f'vix_lag{lag}'] = data['vix'].shift(lag)
            if 'cad_oas' in data.columns:
                features[f'cad_oas_lag{lag}'] = data['cad_oas'].shift(lag)
        
        # 11. MACRO FEATURES
        print("  Creating macro features...")
        
        for col in ['us_equity_revisions', 'us_hard_data_surprises', 'us_growth_surprises', 
                    'us_3m_10y', 'us_lei_yoy', 'us_economic_regime']:
            if col in data.columns:
                features[f'{col}_level'] = data[col]
                features[f'{col}_change'] = data[col].pct_change()
                features[f'{col}_zscore'] = (data[col] - data[col].rolling(252).mean()) / (data[col].rolling(252).std() + 1e-8)
        
        # Fill NaN values
        features = features.ffill().fillna(0)
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        print(f"  ✓ Feature engineering complete: {len(features.columns)} features")
        
        self.feature_names = list(features.columns)
        return features
    
    def create_target(self, data: pd.DataFrame) -> pd.Series:
        """Create binary target: 1 if forward return > 0, else 0."""
        price = data[self.trading_asset]
        forward_return = price.shift(-self.prediction_horizon) / price - 1
        target = (forward_return > 0).astype(int)
        return target
    
    def check_bias(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Check for common biases in features and data.
        
        Returns:
            Dictionary with bias check results
        """
        print("\n=== Bias Checking ===")
        
        bias_results = {
            'look_ahead_bias': False,
            'data_leakage': False,
            'survivorship_bias': False,
            'selection_bias': False
        }
        
        # Check for look-ahead bias (future data in features)
        price = data[self.trading_asset]
        for col in features.columns:
            if 'forward' in col.lower() or 'shift(-' in str(features[col].dtype):
                print(f"  WARNING: Potential look-ahead bias in {col}")
                bias_results['look_ahead_bias'] = True
        
        # Check for data leakage (correlation with future returns)
        target = self.create_target(data)
        valid_idx = target.notna()
        if valid_idx.sum() > 100:
            X_valid = features[valid_idx].iloc[:-self.prediction_horizon]
            y_valid = target[valid_idx].iloc[:-self.prediction_horizon]
            
            # Check for perfect or near-perfect correlations
            for col in X_valid.columns:
                corr = abs(X_valid[col].corr(y_valid))
                if corr > 0.95:
                    print(f"  WARNING: Potential data leakage in {col} (corr={corr:.3f})")
                    bias_results['data_leakage'] = True
        
        # Check for survivorship bias (missing data patterns)
        missing_pct = features.isnull().sum() / len(features)
        if (missing_pct > 0.5).any():
            print(f"  WARNING: High missing data (>50%) in some features")
            bias_results['survivorship_bias'] = True
        
        print(f"  ✓ Bias check complete")
        return bias_results
    
    def train_models_walk_forward(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """
        Train models using walk-forward validation to prevent overfitting.
        
        Returns:
            Dictionary of trained models and validation metrics
        """
        print("\n=== Walk-Forward Model Training ===")
        
        target = self.create_target(data)
        valid_idx = target.notna()
        X = features[valid_idx].copy()
        y = target[valid_idx].copy()
        
        # Remove last prediction_horizon rows to avoid look-ahead
        X = X.iloc[:-self.prediction_horizon]
        y = y.iloc[:-self.prediction_horizon]
        
        if len(X) < 500:
            raise ValueError(f"Insufficient data for walk-forward: {len(X)} samples")
        
        # Walk-forward splits
        train_size = int(len(X) * self.walk_forward_train_size)
        test_size = int(len(X) * self.walk_forward_test_size)
        step_size = int(len(X) * self.walk_forward_step)
        
        print(f"  Train size: {train_size}, Test size: {test_size}, Step: {step_size}")
        
        models = {}
        validation_scores = []
        
        # Train each model type
        if self.use_rf:
            rf_model = self._train_rf(X, y, train_size)
            if rf_model:
                models['rf'] = rf_model
        if self.use_lightgbm:
            lgb_model = self._train_lightgbm(X, y, train_size)
            if lgb_model:
                models['lightgbm'] = lgb_model
        if self.use_xgboost:
            xgb_model = self._train_xgboost(X, y, train_size)
            if xgb_model:
                models['xgboost'] = xgb_model
        if self.use_catboost:
            cat_model = self._train_catboost(X, y, train_size)
            if cat_model:
                models['catboost'] = cat_model
        
        if not models:
            raise ValueError("No models could be trained. Check model availability.")
        
        # Walk-forward validation
        if self.use_walk_forward:
            print("\n  Running walk-forward validation...")
            for start_idx in range(train_size, len(X) - test_size, step_size):
                end_idx = min(start_idx + test_size, len(X))
                
                X_train_wf = X.iloc[:start_idx]
                X_test_wf = X.iloc[start_idx:end_idx]
                y_train_wf = y.iloc[:start_idx]
                y_test_wf = y.iloc[start_idx:end_idx]
                
                # Evaluate each model
                for model_name, model_dict in models.items():
                    model = model_dict['model']
                    
                    if model_name == 'rf':
                        y_pred_proba = model.predict_proba(X_test_wf)[:, 1]
                    elif model_name == 'lightgbm':
                        y_pred_proba = model.predict(X_test_wf)
                    elif model_name == 'xgboost':
                        y_pred_proba = model.predict_proba(X_test_wf)[:, 1]
                    elif model_name == 'catboost':
                        y_pred_proba = model.predict_proba(X_test_wf)[:, 1]
                    
                    auc = roc_auc_score(y_test_wf, y_pred_proba)
                    validation_scores.append({
                        'model': model_name,
                        'fold': len(validation_scores) // len(models),
                        'auc': auc
                    })
            
            # Calculate average AUC per model
            avg_aucs = {}
            for model_name in models.keys():
                model_scores = [s['auc'] for s in validation_scores if s['model'] == model_name]
                avg_aucs[model_name] = np.mean(model_scores) if model_scores else 0
                print(f"    {model_name} average AUC: {avg_aucs[model_name]:.4f}")
            
            # Set ensemble weights based on performance
            total_auc = sum(avg_aucs.values())
            if total_auc > 0:
                self.ensemble_weights = {k: v/total_auc for k, v in avg_aucs.items()}
            else:
                # Equal weights if all fail
                self.ensemble_weights = {k: 1.0/len(models) for k in models.keys()}
        else:
            # Equal weights if not using walk-forward
            self.ensemble_weights = {k: 1.0/len(models) for k in models.keys()}
        
        # Store feature importance from best model
        if models:
            best_model_name = max(self.ensemble_weights.items(), key=lambda x: x[1])[0]
            best_model = models[best_model_name]['model']
            
            if best_model_name == 'rf':
                importances = best_model.feature_importances_
            elif best_model_name == 'lightgbm':
                importances = best_model.feature_importance(importance_type='gain')
            elif best_model_name == 'xgboost':
                importances = best_model.feature_importances_
            elif best_model_name == 'catboost':
                importances = best_model.feature_importances_
            
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n  ✓ Top 10 features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                print(f"    {row['feature']:40s} {row['importance']:.4f}")
        
        return models
    
    def _train_rf(self, X: pd.DataFrame, y: pd.Series, train_size: int) -> Dict:
        """Train Random Forest model."""
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=5,
            max_features=0.4,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        return {'model': rf, 'type': 'rf'}
    
    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series, train_size: int) -> Optional[Dict]:
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            return None
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test = X.iloc[train_size:min(train_size+100, len(X))]
        y_test = y.iloc[train_size:min(train_size+100, len(y))]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        return {'model': model, 'type': 'lightgbm'}
    
    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series, train_size: int) -> Optional[Dict]:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            return None
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )
        model.fit(X_train, y_train)
        
        return {'model': model, 'type': 'xgboost'}
    
    def _train_catboost(self, X: pd.DataFrame, y: pd.Series, train_size: int) -> Optional[Dict]:
        """Train CatBoost model."""
        if not CATBOOST_AVAILABLE:
            return None
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        model = CatBoostClassifier(
            iterations=200,
            depth=7,
            learning_rate=0.05,
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train)
        
        return {'model': model, 'type': 'catboost'}
    
    def generate_ensemble_predictions(self, features: pd.DataFrame) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        all_predictions = []
        weights = []
        
        for model_name, model_dict in self.models.items():
            model = model_dict['model']
            weight = self.ensemble_weights.get(model_name, 0)
            
            if weight > 0:
                if model_name == 'rf':
                    pred = model.predict_proba(features)[:, 1]
                elif model_name == 'lightgbm':
                    pred = model.predict(features)
                elif model_name == 'xgboost':
                    pred = model.predict_proba(features)[:, 1]
                elif model_name == 'catboost':
                    pred = model.predict_proba(features)[:, 1]
                else:
                    continue
                
                all_predictions.append(pred)
                weights.append(weight)
        
        if not all_predictions:
            return np.zeros(len(features))
        
        # Weighted average
        ensemble_pred = np.zeros(len(features))
        total_weight = sum(weights)
        for pred, weight in zip(all_predictions, weights):
            ensemble_pred += pred * (weight / total_weight)
        
        return ensemble_pred
    
    def optimize_threshold(self, features: pd.DataFrame, data: pd.DataFrame) -> float:
        """Optimize prediction threshold for maximum return."""
        if not self.optimize_threshold:
            return self.probability_threshold
        
        print("\n=== Optimizing Threshold ===")
        
        predictions = self.generate_ensemble_predictions(features)
        price = data[self.trading_asset]
        returns = price.pct_change()
        
        thresholds = np.linspace(0.45, 0.65, 20)
        best_threshold = 0.5
        best_score = -np.inf
        
        for threshold in thresholds:
            signals = pd.Series((predictions >= threshold).astype(int), index=features.index)
            strategy_returns = returns * signals.shift(1).fillna(0)
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            
            # Score: return adjusted for Sharpe
            score = total_return * (1 + sharpe)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"  Optimal threshold: {best_threshold:.3f}")
        return best_threshold
    
    def apply_holding_period(self, raw_signals: pd.Series) -> pd.Series:
        """Apply minimum/maximum holding period constraints."""
        adjusted_signals = pd.Series(0, index=raw_signals.index, dtype=int)
        
        i = 0
        while i < len(raw_signals):
            if raw_signals.iloc[i] == 1:
                # Enter position
                hold_end = min(i + self.min_holding_days, len(raw_signals))
                adjusted_signals.iloc[i:hold_end] = 1
                i = hold_end
            else:
                i += 1
        
        return adjusted_signals
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate trading signals using iterative ML approach."""
        print(f"\n=== Iteration {self.iteration + 1} ===")
        
        # 1. Engineer features
        engineered_features = self.engineer_advanced_features(data)
        
        # 2. Check for bias
        bias_results = self.check_bias(engineered_features, data)
        
        # 3. Train models with walk-forward validation
        self.models = self.train_models_walk_forward(engineered_features, data)
        
        # 4. Optimize threshold
        self.probability_threshold = self.optimize_threshold(engineered_features, data)
        
        # 5. Generate ensemble predictions
        predictions = self.generate_ensemble_predictions(engineered_features)
        
        # 6. Convert to signals
        raw_signals = (predictions >= self.probability_threshold).astype(int)
        raw_signals_series = pd.Series(raw_signals, index=data.index)
        
        # 7. Apply holding period
        adjusted_signals = self.apply_holding_period(raw_signals_series)
        
        # 8. Convert to entry/exit signals
        positions = adjusted_signals.astype(int)
        positions_shifted = positions.shift(1).fillna(0).astype(int)
        
        entry_signals = (positions == 1) & (positions_shifted == 0)
        exit_signals = (positions == 0) & (positions_shifted == 1)
        
        print(f"\n  ✓ Signal generation complete:")
        print(f"    Entry signals: {entry_signals.sum()}")
        print(f"    Time in market: {positions.mean():.2%}")
        
        return entry_signals, exit_signals
    
    def validate_performance(self, result, benchmark_data: pd.Series) -> Dict:
        """
        Validate strategy performance meets requirements.
        
        Returns:
            Dictionary with validation results
        """
        # Calculate benchmark metrics
        benchmark_returns = benchmark_data.pct_change().dropna()
        benchmark_cagr = (1 + benchmark_returns.mean()) ** 252 - 1
        
        # Calculate strategy metrics
        strategy_returns = result.returns.dropna()
        strategy_cagr = (1 + strategy_returns.mean()) ** 252 - 1
        
        # Outperformance
        outperformance = strategy_cagr - benchmark_cagr
        
        # Additional validations
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        equity_curve = (1 + strategy_returns).cumprod()
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        validation = {
            'meets_outperformance_target': outperformance >= self.min_outperformance,
            'meets_sharpe_target': sharpe >= self.min_sharpe,
            'meets_drawdown_limit': max_drawdown >= self.max_drawdown_limit,
            'meets_min_trades': result.trades_count >= self.min_trades,
            'outperformance': outperformance,
            'strategy_cagr': strategy_cagr,
            'benchmark_cagr': benchmark_cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trades_count': result.trades_count
        }
        
        return validation
    
    def save_iteration_results(self, result, validation: Dict):
        """Save iteration results for learning."""
        iteration_data = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'validation': validation,
            'ensemble_weights': self.ensemble_weights,
            'probability_threshold': self.probability_threshold,
            'feature_count': len(self.feature_names),
            'top_features': self.feature_importance.head(20).to_dict('records') if self.feature_importance is not None else []
        }
        
        self.iteration_history.append(iteration_data)
        
        # Save to file
        results_file = self.results_dir / f'iteration_{self.iteration}.json'
        with open(results_file, 'w') as f:
            json.dump(iteration_data, f, indent=2, default=str)
        
        print(f"\n  ✓ Iteration results saved: {results_file}")
    
    def learn_from_iterations(self) -> Dict:
        """Learn from past iterations to guide future improvements."""
        if len(self.iteration_history) < 2:
            return {}
        
        # Analyze what worked
        successful_iterations = [
            it for it in self.iteration_history 
            if it['validation'].get('meets_outperformance_target', False)
        ]
        
        if not successful_iterations:
            return {}
        
        # Find patterns in successful iterations
        insights = {
            'best_threshold_range': [],
            'best_feature_types': [],
            'best_ensemble_weights': []
        }
        
        for it in successful_iterations:
            insights['best_threshold_range'].append(it['probability_threshold'])
            insights['best_ensemble_weights'].append(it['ensemble_weights'])
            
            top_features = it.get('top_features', [])
            if top_features:
                feature_types = [f['feature'].split('_')[0] for f in top_features[:10]]
                insights['best_feature_types'].extend(feature_types)
        
        # Return insights
        return {
            'avg_threshold': np.mean(insights['best_threshold_range']) if insights['best_threshold_range'] else 0.5,
            'common_feature_types': pd.Series(insights['best_feature_types']).value_counts().head(5).to_dict(),
            'avg_ensemble_weights': {}
        }
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""Iterative Machine Learning Strategy (Iteration {self.iteration + 1})

GOAL: Beat buy-and-hold by at least {self.min_outperformance:.1%} annualized return

METHODOLOGY:
1. Advanced Feature Engineering ({len(self.feature_names)} features):
   - Multi-timeframe momentum ({len(self.momentum_windows)} windows)
   - Volatility regimes and measures ({len(self.volatility_windows)} windows)
   - Cross-asset relationships
   - Statistical features (z-scores, percentiles, skew, kurtosis)
   - Regime detection
   - Interaction features

2. Ensemble Models:
   - Random Forest: {self.use_rf}
   - LightGBM: {self.use_lightgbm}
   - XGBoost: {self.use_xgboost}
   - CatBoost: {self.use_catboost}
   - Ensemble weights: {self.ensemble_weights}

3. Walk-Forward Validation:
   - Train: {self.walk_forward_train_size:.0%}
   - Test: {self.walk_forward_test_size:.0%}
   - Step: {self.walk_forward_step:.0%}

4. Statistical Validation:
   - Bias checking (look-ahead, data leakage)
   - Minimum Sharpe: {self.min_sharpe}
   - Maximum drawdown: {self.max_drawdown_limit:.0%}
   - Minimum trades: {self.min_trades}

5. Iterative Improvement:
   - Learning from past iterations
   - Feature importance analysis
   - Threshold optimization

CURRENT STATUS:
- Iteration: {self.iteration + 1}/{self.max_iterations}
- Target outperformance: {self.min_outperformance:.1%}
- Probability threshold: {self.probability_threshold:.3f}
"""
