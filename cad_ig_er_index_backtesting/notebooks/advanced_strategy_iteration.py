#!/usr/bin/env python3
"""
Advanced Strategy Iteration Framework for CAD-IG-ER Index
Goal: Beat buy-and-hold by 2.5% annualized through creative feature engineering and rigorous validation

Author: Data Science & ML Trading Expert
Date: 2025-12-14
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import sys
import os

# Add parent directory to path
script_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import xgboost as xgb
    import lightgbm as lgb
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ML libraries not available: {e}")
    SKLEARN_AVAILABLE = False

# Statistical libraries
try:
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.stats import spearmanr, pearsonr, skew, kurtosis
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except:
    VECTORBT_AVAILABLE = False

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

print("="*100)
print(" ADVANCED STRATEGY ITERATION FRAMEWORK")
print("="*100)
print(f"Timestamp: {datetime.now()}")
print(f"Python ML Stack: {'✓' if SKLEARN_AVAILABLE else '✗'}")
print(f"SciPy Stats: {'✓' if SCIPY_AVAILABLE else '✗'}")
print(f"VectorBT: {'✓' if VECTORBT_AVAILABLE else '✗'}")
print("="*100)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

class DataAnalyzer:
    """Deep data analysis and pattern discovery."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.target_col = 'cad_ig_er_index'

    def load_data(self):
        """Load and perform initial data analysis."""
        print("\n" + "="*100)
        print("STEP 1: LOADING AND ANALYZING DATA")
        print("="*100)

        self.data = pd.read_csv(self.data_path, parse_dates=['Date'])
        self.data.set_index('Date', inplace=True)

        print(f"\n📊 Data Shape: {self.data.shape}")
        print(f"📅 Date Range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"📈 Total Days: {len(self.data)}")
        print(f"📉 Years of Data: {(self.data.index.max() - self.data.index.min()).days / 365.25:.1f}")

        print("\n📋 Available Features:")
        for i, col in enumerate(self.data.columns, 1):
            missing = self.data[col].isna().sum()
            print(f"  {i:2d}. {col:30s} | Missing: {missing:4d} ({missing/len(self.data)*100:5.2f}%)")

        return self.data

    def analyze_target_characteristics(self):
        """Analyze target index characteristics."""
        print("\n" + "="*100)
        print("STEP 2: TARGET INDEX ANALYSIS")
        print("="*100)

        # Calculate returns
        self.data['returns'] = self.data[self.target_col].pct_change()
        self.data['log_returns'] = np.log(self.data[self.target_col] / self.data[self.target_col].shift(1))

        # Statistics
        returns = self.data['returns'].dropna()

        print("\n📊 RETURN STATISTICS:")
        print(f"  Mean Daily Return:     {returns.mean()*100:8.4f}%")
        print(f"  Median Daily Return:   {returns.median()*100:8.4f}%")
        print(f"  Std Dev (Daily):       {returns.std()*100:8.4f}%")
        print(f"  Annualized Vol:        {returns.std()*np.sqrt(252)*100:8.2f}%")
        print(f"  Skewness:              {skew(returns):8.4f}")
        print(f"  Kurtosis:              {kurtosis(returns):8.4f}")

        # Calculate buy-and-hold performance
        total_return = (self.data[self.target_col].iloc[-1] / self.data[self.target_col].iloc[0]) - 1
        years = (self.data.index[-1] - self.data.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1/years) - 1

        print("\n📈 BUY-AND-HOLD PERFORMANCE:")
        print(f"  Total Return:          {total_return*100:8.2f}%")
        print(f"  CAGR:                  {cagr*100:8.2f}%")
        print(f"  🎯 Target CAGR:         {(cagr + 0.025)*100:8.2f}%")

        # Analyze drawdowns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        print("\n📉 RISK METRICS:")
        print(f"  Max Drawdown:          {max_dd*100:8.2f}%")
        print(f"  Sharpe Ratio (rf=0):   {returns.mean() / returns.std() * np.sqrt(252):8.2f}")

        # Autocorrelation analysis
        autocorr_1d = returns.autocorr(lag=1)
        autocorr_5d = returns.autocorr(lag=5)
        autocorr_20d = returns.autocorr(lag=20)

        print("\n🔄 AUTOCORRELATION (Momentum/Mean Reversion):")
        print(f"  Lag 1 day:             {autocorr_1d:8.4f}")
        print(f"  Lag 5 days:            {autocorr_5d:8.4f}")
        print(f"  Lag 20 days:           {autocorr_20d:8.4f}")

        return {
            'cagr': cagr,
            'target_cagr': cagr + 0.025,
            'total_return': total_return,
            'max_dd': max_dd,
            'volatility': returns.std() * np.sqrt(252)
        }

    def discover_regime_patterns(self):
        """Discover market regime patterns."""
        print("\n" + "="*100)
        print("STEP 3: REGIME PATTERN DISCOVERY")
        print("="*100)

        # VIX regimes
        vix = self.data['vix']
        vix_percentiles = vix.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))

        print("\n📊 VIX REGIME ANALYSIS:")
        print(f"  Current VIX:           {vix.iloc[-1]:8.2f}")
        print(f"  1Y Percentile:         {vix_percentiles.iloc[-1]:8.1f}%")
        print(f"  Mean VIX:              {vix.mean():8.2f}")
        print(f"  Median VIX:            {vix.median():8.2f}")

        # OAS spread regimes
        oas = self.data['cad_oas']
        oas_percentiles = oas.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))

        print("\n📊 OAS SPREAD ANALYSIS:")
        print(f"  Current OAS:           {oas.iloc[-1]:8.2f}")
        print(f"  1Y Percentile:         {oas_percentiles.iloc[-1]:8.1f}%")
        print(f"  Mean OAS:              {oas.mean():8.2f}")
        print(f"  Median OAS:            {oas.median():8.2f}")

        # Economic regime analysis
        if 'us_economic_regime' in self.data.columns:
            regime = self.data['us_economic_regime']
            regime_counts = regime.value_counts()
            print("\n📊 ECONOMIC REGIME DISTRIBUTION:")
            for r, count in regime_counts.items():
                pct = count / len(regime) * 100
                print(f"  Regime {r:3.0f}:            {count:6d} days ({pct:5.1f}%)")

        return {
            'vix_percentiles': vix_percentiles,
            'oas_percentiles': oas_percentiles
        }


# ============================================================================
# 2. CREATIVE FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """Generate creative features based on statistical patterns."""

    def __init__(self, data):
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)

    def create_all_features(self):
        """Create comprehensive feature set."""
        print("\n" + "="*100)
        print("STEP 4: CREATIVE FEATURE ENGINEERING")
        print("="*100)

        # Track feature counts
        feature_counts = {}

        # 1. Multi-timeframe momentum
        print("\n🔧 Creating momentum features...")
        feature_counts['momentum'] = self._create_momentum_features()

        # 2. Volatility features
        print("🔧 Creating volatility features...")
        feature_counts['volatility'] = self._create_volatility_features()

        # 3. Statistical features
        print("🔧 Creating statistical features...")
        feature_counts['statistical'] = self._create_statistical_features()

        # 4. Cross-asset features
        print("🔧 Creating cross-asset features...")
        feature_counts['cross_asset'] = self._create_cross_asset_features()

        # 5. Macro economic features
        print("🔧 Creating macro features...")
        feature_counts['macro'] = self._create_macro_features()

        # 6. Technical indicators
        print("🔧 Creating technical indicators...")
        feature_counts['technical'] = self._create_technical_features()

        # 7. Regime features
        print("🔧 Creating regime features...")
        feature_counts['regime'] = self._create_regime_features()

        # 8. CREATIVE: Pattern recognition features
        print("🔧 Creating pattern recognition features...")
        feature_counts['patterns'] = self._create_pattern_features()

        # 9. CREATIVE: Higher-order features
        print("🔧 Creating higher-order features...")
        feature_counts['higher_order'] = self._create_higher_order_features()

        print("\n📊 FEATURE ENGINEERING SUMMARY:")
        total = 0
        for category, count in feature_counts.items():
            print(f"  {category:20s}: {count:4d} features")
            total += count
        print(f"  {'TOTAL':20s}: {total:4d} features")

        return self.features

    def _create_momentum_features(self):
        """Multi-timeframe momentum."""
        count = 0
        price = self.data['cad_ig_er_index']

        # Standard momentum windows
        for window in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 252]:
            self.features[f'mom_{window}d'] = price.pct_change(window)
            count += 1

        # Log returns momentum
        for window in [5, 10, 20, 60]:
            self.features[f'log_mom_{window}d'] = np.log(price / price.shift(window))
            count += 1

        # Acceleration (momentum of momentum)
        for window in [5, 10, 20, 60]:
            mom = price.pct_change(window)
            self.features[f'mom_accel_{window}d'] = mom - mom.shift(window)
            count += 1

        return count

    def _create_volatility_features(self):
        """Volatility-based features."""
        count = 0
        returns = self.data['cad_ig_er_index'].pct_change()

        # Realized volatility
        for window in [5, 10, 20, 40, 60, 120, 252]:
            self.features[f'vol_{window}d'] = returns.rolling(window).std()
            count += 1

        # Volatility percentile
        for window in [20, 60, 252]:
            vol = returns.rolling(window).std()
            self.features[f'vol_pctile_{window}d'] = vol.rolling(252).apply(
                lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50
            )
            count += 1

        # VIX features
        if 'vix' in self.data.columns:
            vix = self.data['vix']
            for window in [5, 10, 20, 60, 252]:
                self.features[f'vix_zscore_{window}d'] = (vix - vix.rolling(window).mean()) / vix.rolling(window).std()
                count += 1

            # VIX term structure
            self.features['vix_mom_5d'] = vix.pct_change(5)
            self.features['vix_mom_20d'] = vix.pct_change(20)
            count += 2

        return count

    def _create_statistical_features(self):
        """Statistical pattern features."""
        count = 0
        returns = self.data['cad_ig_er_index'].pct_change()

        # Rolling statistics
        for window in [20, 60, 120]:
            self.features[f'skew_{window}d'] = returns.rolling(window).skew()
            self.features[f'kurt_{window}d'] = returns.rolling(window).kurt()
            count += 2

        # Percentile ranks
        price = self.data['cad_ig_er_index']
        for window in [20, 60, 120, 252]:
            self.features[f'price_pctile_{window}d'] = price.rolling(window).apply(
                lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50
            )
            count += 1

        return count

    def _create_cross_asset_features(self):
        """Cross-asset relationships."""
        count = 0

        # OAS spread features
        if 'cad_oas' in self.data.columns:
            oas = self.data['cad_oas']

            # OAS momentum
            for window in [5, 10, 20, 60]:
                self.features[f'oas_mom_{window}d'] = oas.pct_change(window)
                count += 1

            # OAS z-score
            for window in [20, 60, 252]:
                self.features[f'oas_zscore_{window}d'] = (oas - oas.rolling(window).mean()) / oas.rolling(window).std()
                count += 1

        # Equity market features
        for asset in ['tsx', 's&p_500']:
            if asset in self.data.columns:
                price = self.data[asset]
                for window in [5, 20, 60]:
                    self.features[f'{asset}_mom_{window}d'] = price.pct_change(window)
                    count += 1

        # Yield curve
        if 'us_3m_10y' in self.data.columns:
            curve = self.data['us_3m_10y']
            self.features['yield_curve'] = curve
            self.features['yield_curve_mom_20d'] = curve - curve.shift(20)
            self.features['yield_curve_mom_60d'] = curve - curve.shift(60)
            count += 3

        return count

    def _create_macro_features(self):
        """Macroeconomic indicators."""
        count = 0

        macro_cols = [
            'us_growth_surprises', 'us_inflation_surprises',
            'us_lei_yoy', 'us_hard_data_surprises', 'us_equity_revisions'
        ]

        for col in macro_cols:
            if col in self.data.columns:
                val = self.data[col]

                # Raw value
                self.features[col] = val
                count += 1

                # Changes
                for window in [5, 20, 60]:
                    self.features[f'{col}_chg_{window}d'] = val - val.shift(window)
                    count += 1

        return count

    def _create_technical_features(self):
        """Technical indicators."""
        count = 0
        price = self.data['cad_ig_er_index']

        # Moving averages
        for window in [10, 20, 50, 100, 200]:
            ma = price.rolling(window).mean()
            self.features[f'ma_{window}d'] = price / ma - 1  # Distance from MA
            count += 1

        # Bollinger Bands
        for window in [20, 60]:
            ma = price.rolling(window).mean()
            std = price.rolling(window).std()
            self.features[f'bb_upper_{window}d'] = (price - (ma + 2*std)) / std
            self.features[f'bb_lower_{window}d'] = (price - (ma - 2*std)) / std
            count += 2

        # RSI
        for window in [14, 30]:
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss
            self.features[f'rsi_{window}d'] = 100 - (100 / (1 + rs))
            count += 1

        return count

    def _create_regime_features(self):
        """Market regime detection."""
        count = 0

        # Volatility regime
        returns = self.data['cad_ig_er_index'].pct_change()
        vol_20d = returns.rolling(20).std()
        vol_median = vol_20d.rolling(252).median()
        self.features['high_vol_regime'] = (vol_20d > vol_median).astype(int)
        count += 1

        # VIX regime
        if 'vix' in self.data.columns:
            vix = self.data['vix']
            vix_median = vix.rolling(252).median()
            self.features['high_vix_regime'] = (vix > vix_median).astype(int)
            count += 1

        # OAS regime
        if 'cad_oas' in self.data.columns:
            oas = self.data['cad_oas']
            oas_median = oas.rolling(252).median()
            self.features['high_oas_regime'] = (oas > oas_median).astype(int)
            count += 1

        # Economic regime
        if 'us_economic_regime' in self.data.columns:
            self.features['econ_regime'] = self.data['us_economic_regime']
            count += 1

        return count

    def _create_pattern_features(self):
        """CREATIVE: Pattern recognition features."""
        count = 0
        price = self.data['cad_ig_er_index']
        returns = price.pct_change()

        # Consecutive up/down days
        self.features['consec_up'] = (returns > 0).astype(int).groupby(
            (returns <= 0).cumsum()
        ).cumsum()
        self.features['consec_down'] = (returns < 0).astype(int).groupby(
            (returns >= 0).cumsum()
        ).cumsum()
        count += 2

        # New highs/lows
        for window in [20, 60, 252]:
            rolling_max = price.rolling(window).max()
            rolling_min = price.rolling(window).min()
            self.features[f'new_high_{window}d'] = (price == rolling_max).astype(int)
            self.features[f'new_low_{window}d'] = (price == rolling_min).astype(int)
            count += 2

        # Distance from highs/lows
        for window in [60, 252]:
            rolling_max = price.rolling(window).max()
            rolling_min = price.rolling(window).min()
            self.features[f'pct_from_high_{window}d'] = (price - rolling_max) / rolling_max
            self.features[f'pct_from_low_{window}d'] = (price - rolling_min) / rolling_min
            count += 2

        # Reversal patterns (after big moves)
        for window in [5, 20]:
            big_down = returns.rolling(window).sum() < -0.02  # 2% down
            big_up = returns.rolling(window).sum() > 0.02      # 2% up
            self.features[f'reversal_from_down_{window}d'] = big_down.astype(int)
            self.features[f'reversal_from_up_{window}d'] = big_up.astype(int)
            count += 2

        return count

    def _create_higher_order_features(self):
        """CREATIVE: Higher-order interactions and transformations."""
        count = 0

        # Interaction features
        if 'vix' in self.data.columns and 'cad_oas' in self.data.columns:
            # Spread x VIX interaction
            vix_z = (self.data['vix'] - self.data['vix'].rolling(60).mean()) / self.data['vix'].rolling(60).std()
            oas_z = (self.data['cad_oas'] - self.data['cad_oas'].rolling(60).mean()) / self.data['cad_oas'].rolling(60).std()
            self.features['vix_oas_interaction'] = vix_z * oas_z
            count += 1

        # Momentum x Volatility
        mom_20 = self.data['cad_ig_er_index'].pct_change(20)
        vol_20 = self.data['cad_ig_er_index'].pct_change().rolling(20).std()
        self.features['mom_vol_ratio'] = mom_20 / (vol_20 + 1e-6)
        count += 1

        # Risk-adjusted momentum
        for window in [20, 60]:
            mom = self.data['cad_ig_er_index'].pct_change(window)
            vol = self.data['cad_ig_er_index'].pct_change().rolling(window).std()
            self.features[f'risk_adj_mom_{window}d'] = mom / (vol + 1e-6)
            count += 1

        return count


# ============================================================================
# 3. STRATEGY ITERATION ENGINE
# ============================================================================

class StrategyIterator:
    """Systematic strategy iteration with validation."""

    def __init__(self, data, features, target_col='cad_ig_er_index'):
        self.data = data
        self.features = features
        self.target_col = target_col
        self.results = []

    def create_labels(self, forward_periods=5, threshold=0.0):
        """Create forward-looking labels."""
        price = self.data[self.target_col]
        forward_returns = price.shift(-forward_periods) / price - 1
        labels = (forward_returns > threshold).astype(int)
        return labels

    def walk_forward_validation(self, strategy_name, model, forward_periods=5,
                                 n_splits=10, verbose=True):
        """
        Walk-forward cross-validation with proper time-series splits.
        Returns out-of-sample predictions and performance metrics.
        """
        print(f"\n{'='*100}")
        print(f"TESTING STRATEGY: {strategy_name}")
        print(f"{'='*100}")

        # Prepare labels
        labels = self.create_labels(forward_periods=forward_periods)

        # Align features and labels
        valid_idx = ~(self.features.isna().any(axis=1) | labels.isna())
        X = self.features[valid_idx].copy()
        y = labels[valid_idx].copy()

        print(f"\n📊 Data Summary:")
        print(f"  Total samples:         {len(X)}")
        print(f"  Features:              {X.shape[1]}")
        print(f"  Positive labels:       {y.sum()} ({y.mean()*100:.1f}%)")

        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)

        predictions = pd.Series(index=X.index, dtype=float)
        probabilities = pd.Series(index=X.index, dtype=float)

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Store predictions
            predictions.iloc[test_idx] = y_pred
            probabilities.iloc[test_idx] = y_prob

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

            fold_metrics.append({
                'fold': fold,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': acc,
                'auc': auc
            })

            if verbose:
                print(f"  Fold {fold:2d} | Train: {len(X_train):5d} | Test: {len(X_test):5d} | "
                      f"Acc: {acc:.3f} | AUC: {auc:.3f}")

        # Overall metrics
        valid_preds = ~predictions.isna()
        overall_acc = accuracy_score(y[valid_preds], predictions[valid_preds])
        overall_auc = roc_auc_score(y[valid_preds], probabilities[valid_preds])

        print(f"\n📊 Cross-Validation Results:")
        print(f"  Mean Accuracy:         {np.mean([m['accuracy'] for m in fold_metrics]):.4f}")
        print(f"  Mean AUC:              {np.mean([m['auc'] for m in fold_metrics]):.4f}")
        print(f"  Overall Accuracy:      {overall_acc:.4f}")
        print(f"  Overall AUC:           {overall_auc:.4f}")

        return {
            'strategy_name': strategy_name,
            'predictions': predictions,
            'probabilities': probabilities,
            'labels': y,
            'fold_metrics': fold_metrics,
            'overall_acc': overall_acc,
            'overall_auc': overall_auc,
            'model': model,
            'feature_importance': getattr(model, 'feature_importances_', None)
        }

    def backtest_signals(self, signals, strategy_name):
        """Backtest binary signals (1=long, 0=cash)."""
        print(f"\n{'='*100}")
        print(f"BACKTESTING: {strategy_name}")
        print(f"{'='*100}")

        # Align signals with price data
        price = self.data[self.target_col].reindex(signals.index)

        # Calculate returns
        returns = price.pct_change()
        strategy_returns = returns * signals.shift(1)  # Enter next day

        # Remove NaN
        valid_idx = ~(strategy_returns.isna() | returns.isna())
        strategy_returns = strategy_returns[valid_idx]
        returns = returns[valid_idx]

        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        bh_return = (1 + returns).prod() - 1

        years = (strategy_returns.index[-1] - strategy_returns.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1/years) - 1
        bh_cagr = (1 + bh_return) ** (1/years) - 1

        # Risk metrics
        vol = strategy_returns.std() * np.sqrt(252)
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

        # Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Time in market
        time_in_market = signals.mean()

        # Number of trades
        trades = signals.diff().abs().sum() / 2

        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"  Strategy CAGR:         {cagr*100:8.2f}%")
        print(f"  Buy-Hold CAGR:         {bh_cagr*100:8.2f}%")
        print(f"  Outperformance:        {(cagr - bh_cagr)*100:8.2f}%")
        print(f"  Total Return:          {total_return*100:8.2f}%")
        print(f"  Sharpe Ratio:          {sharpe:8.2f}")
        print(f"  Volatility (ann.):     {vol*100:8.2f}%")
        print(f"  Max Drawdown:          {max_dd*100:8.2f}%")
        print(f"  Time in Market:        {time_in_market*100:8.1f}%")
        print(f"  Total Trades:          {trades:8.0f}")

        # Check if target achieved
        target_outperformance = 0.025
        target_achieved = (cagr - bh_cagr) >= target_outperformance

        if target_achieved:
            print(f"\n  🎯 ✅ TARGET ACHIEVED! Outperformance: {(cagr - bh_cagr)*100:.2f}% >= 2.50%")
        else:
            print(f"\n  🎯 ❌ Target not met. Need {(target_outperformance - (cagr - bh_cagr))*100:.2f}% more")

        return {
            'strategy_name': strategy_name,
            'cagr': cagr,
            'bh_cagr': bh_cagr,
            'outperformance': cagr - bh_cagr,
            'total_return': total_return,
            'sharpe': sharpe,
            'volatility': vol,
            'max_dd': max_dd,
            'time_in_market': time_in_market,
            'trades': trades,
            'target_achieved': target_achieved,
            'returns': strategy_returns
        }


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""

    # Data path
    data_path = Path(__file__).parent.parent.parent / "market data pipeline" / "processed market data" / "cad_ig_er_index_data_for_backtest.csv"

    # Step 1: Load and analyze data
    analyzer = DataAnalyzer(data_path)
    data = analyzer.load_data()
    target_metrics = analyzer.analyze_target_characteristics()
    regime_info = analyzer.discover_regime_patterns()

    # Step 2: Feature engineering
    engineer = AdvancedFeatureEngineer(data)
    features = engineer.create_all_features()

    print(f"\n{'='*100}")
    print("STEP 5: PREPARING FOR STRATEGY ITERATION")
    print(f"{'='*100}")
    print(f"\n✅ Data loaded: {len(data)} days")
    print(f"✅ Features created: {features.shape[1]} features")
    print(f"✅ Target CAGR: {target_metrics['target_cagr']*100:.2f}%")
    print(f"\n🚀 Ready to iterate on strategies!")

    # Save features for later use
    output_dir = Path(__file__).parent.parent / "outputs" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([data, features], axis=1)
    combined.to_csv(output_dir / "engineered_features.csv")
    print(f"\n💾 Features saved to: {output_dir / 'engineered_features.csv'}")

    # Step 3: Strategy iteration
    iterator = StrategyIterator(data, features)

    # We'll test multiple strategies
    strategies_to_test = []

    # Strategy 1: Random Forest with optimized parameters
    print(f"\n{'='*100}")
    print("ITERATION 1: RANDOM FOREST CLASSIFIER")
    print(f"{'='*100}")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=10,
        max_features=0.3,
        random_state=42,
        n_jobs=-1
    )

    rf_result = iterator.walk_forward_validation(
        strategy_name="RF_Advanced_v1",
        model=rf_model,
        forward_periods=5,
        n_splits=10
    )

    # Convert probabilities to signals (optimize threshold)
    best_threshold = 0.55
    rf_signals = (rf_result['probabilities'] > best_threshold).astype(int)
    rf_backtest = iterator.backtest_signals(rf_signals, "RF_Advanced_v1")

    strategies_to_test.append({
        'name': 'RF_Advanced_v1',
        'cv_result': rf_result,
        'backtest': rf_backtest
    })

    # Strategy 2: Gradient Boosting
    print(f"\n{'='*100}")
    print("ITERATION 2: GRADIENT BOOSTING CLASSIFIER")
    print(f"{'='*100}")

    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=15,
        max_features=0.4,
        random_state=42
    )

    gb_result = iterator.walk_forward_validation(
        strategy_name="GB_Advanced_v1",
        model=gb_model,
        forward_periods=5,
        n_splits=10
    )

    gb_signals = (gb_result['probabilities'] > 0.55).astype(int)
    gb_backtest = iterator.backtest_signals(gb_signals, "GB_Advanced_v1")

    strategies_to_test.append({
        'name': 'GB_Advanced_v1',
        'cv_result': gb_result,
        'backtest': gb_backtest
    })

    # Strategy 3: XGBoost (if available)
    try:
        print(f"\n{'='*100}")
        print("ITERATION 3: XGBOOST CLASSIFIER")
        print(f"{'='*100}")

        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        xgb_result = iterator.walk_forward_validation(
            strategy_name="XGB_Advanced_v1",
            model=xgb_model,
            forward_periods=5,
            n_splits=10
        )

        xgb_signals = (xgb_result['probabilities'] > 0.55).astype(int)
        xgb_backtest = iterator.backtest_signals(xgb_signals, "XGB_Advanced_v1")

        strategies_to_test.append({
            'name': 'XGB_Advanced_v1',
            'cv_result': xgb_result,
            'backtest': xgb_backtest
        })
    except Exception as e:
        print(f"XGBoost not available or error: {e}")

    # Final summary
    print(f"\n{'='*100}")
    print("FINAL STRATEGY COMPARISON")
    print(f"{'='*100}")

    print(f"\n{'Strategy':<25} {'CAGR':>10} {'Outperf':>10} {'Sharpe':>10} {'MaxDD':>10} {'Target':>10}")
    print("-" * 100)

    # Benchmark
    print(f"{'Buy-and-Hold (Benchmark)':<25} {target_metrics['cagr']*100:>9.2f}% {0.0:>9.2f}% {'-':>10} {target_metrics['max_dd']*100:>9.2f}% {'':>10}")
    print("-" * 100)

    for strat in strategies_to_test:
        bt = strat['backtest']
        target_str = '✅' if bt['target_achieved'] else '❌'
        print(f"{strat['name']:<25} {bt['cagr']*100:>9.2f}% {bt['outperformance']*100:>9.2f}% "
              f"{bt['sharpe']:>10.2f} {bt['max_dd']*100:>9.2f}% {target_str:>10}")

    # Find best strategy
    best_strat = max(strategies_to_test, key=lambda x: x['backtest']['outperformance'])

    print(f"\n{'='*100}")
    print(f"🏆 BEST STRATEGY: {best_strat['name']}")
    print(f"{'='*100}")
    print(f"  CAGR:                  {best_strat['backtest']['cagr']*100:.2f}%")
    print(f"  Outperformance:        {best_strat['backtest']['outperformance']*100:.2f}%")
    print(f"  Target Achievement:    {'✅ YES' if best_strat['backtest']['target_achieved'] else '❌ NO'}")

    # Save results
    results_dir = Path(__file__).parent.parent / "outputs" / "strategy_iterations"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for strat in strategies_to_test:
        # Save signals
        signals_path = results_dir / f"{strat['name']}_signals_{timestamp}.csv"
        strat['cv_result']['probabilities'].to_csv(signals_path)
        print(f"\n💾 Saved {strat['name']} signals to: {signals_path}")

    return strategies_to_test, best_strat


if __name__ == "__main__":
    results = main()

    print(f"\n{'='*100}")
    print("✅ ITERATION FRAMEWORK COMPLETE!")
    print(f"{'='*100}")
