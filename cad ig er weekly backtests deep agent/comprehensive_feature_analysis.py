#!/usr/bin/env python3
"""
Comprehensive Feature Importance Analysis for Trading Strategy
Recreating the model and analyzing feature importance in detail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeatureImportanceAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.weekly_df = None
        self.feature_cols = None
        self.best_model = None
        self.scaler = None
        self.X = None
        self.y = None
        
    def load_and_prepare_data(self):
        """Load and prepare the data with all features"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Convert to weekly data (Friday close)
        self.df.set_index('Date', inplace=True)
        weekly_df = self.df.resample('W-FRI').last().dropna()
        
        # Use TSX as the main asset (closest to SPY in this dataset)
        price_col = 'tsx'
        # Available columns: vix, us_3m_10y (similar to TNX), us_economic_regime
        weekly_df = weekly_df[[price_col, 'vix', 'us_3m_10y', 'us_economic_regime', 
                              'cad_oas', 'us_hy_oas', 'us_ig_oas']].dropna()
        
        print(f"Weekly data shape: {weekly_df.shape}")
        
        # Create comprehensive features
        self.create_all_features(weekly_df, price_col)
        
    def create_all_features(self, df, price_col):
        """Create all 69 features as in the original strategy"""
        print("Creating comprehensive feature set...")
        
        # 1. BASIC RETURNS
        df['weekly_ret'] = df[price_col].pct_change()
        df['weekly_ret_fwd'] = df['weekly_ret'].shift(-1)  # Target
        
        # 2. MOMENTUM FEATURES (12 features)
        for period in [4, 8, 12, 26]:
            df[f'momentum_{period}w'] = df[price_col].pct_change(period)
            df[f'momentum_vol_{period}w'] = df[f'momentum_{period}w'] / df['weekly_ret'].rolling(period).std()
        
        # RSI-style momentum
        for period in [8, 12]:
            delta = df['weekly_ret']
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}w'] = 100 - (100 / (1 + rs))
        
        # Price relative to moving averages
        for period in [8, 12, 26]:
            df[f'price_ma_ratio_{period}w'] = df[price_col] / df[price_col].rolling(period).mean()
        
        # 3. VOLATILITY FEATURES (10 features)
        for period in [4, 8, 12, 26]:
            df[f'volatility_{period}w'] = df['weekly_ret'].rolling(period).std()
        
        # VIX features
        df['vix_level'] = df['vix']
        df['vix_change'] = df['vix'].pct_change()
        df['vix_momentum_4w'] = df['vix'].pct_change(4)
        df['vix_ma_ratio_8w'] = df['vix'] / df['vix'].rolling(8).mean()
        
        # Volatility regime
        df['vol_regime'] = (df['volatility_8w'] > df['volatility_8w'].rolling(52).quantile(0.7)).astype(int)
        df['vix_regime'] = (df['vix'] > df['vix'].rolling(52).quantile(0.7)).astype(int)
        
        # 4. TREND FEATURES (8 features)
        for period in [8, 12, 26]:
            df[f'trend_strength_{period}w'] = df[price_col].rolling(period).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == period else np.nan
            )
        
        # Moving average trends
        df['ma_8w'] = df[price_col].rolling(8).mean()
        df['ma_26w'] = df[price_col].rolling(26).mean()
        df['ma_trend_8_26'] = (df['ma_8w'] > df['ma_26w']).astype(int)
        
        # Price position in range
        for period in [12, 26]:
            df[f'price_percentile_{period}w'] = df[price_col].rolling(period).rank(pct=True)
        
        # Trend consistency
        df['trend_consistency_8w'] = (df['weekly_ret'] > 0).rolling(8).sum()
        
        # 5. HIGHER-ORDER FEATURES (8 features)
        # Skewness and kurtosis
        for period in [8, 12]:
            df[f'skewness_{period}w'] = df['weekly_ret'].rolling(period).skew()
            df[f'kurtosis_{period}w'] = df['weekly_ret'].rolling(period).kurt()
        
        # Higher moments of volatility
        df['vol_skewness_8w'] = df['volatility_4w'].rolling(8).skew()
        df['vol_kurtosis_8w'] = df['volatility_4w'].rolling(8).kurt()
        
        # Return autocorrelation
        df['return_autocorr_4w'] = df['weekly_ret'].rolling(8).apply(
            lambda x: x.autocorr(lag=1) if len(x) >= 4 else np.nan
        )
        df['return_autocorr_8w'] = df['weekly_ret'].rolling(12).apply(
            lambda x: x.autocorr(lag=2) if len(x) >= 8 else np.nan
        )
        
        # 6. REGIME DETECTION (8 features)
        # Economic regime proxies
        df['tnx_level'] = df['us_3m_10y']
        df['tnx_change'] = df['us_3m_10y'].pct_change()
        df['tnx_momentum_4w'] = df['us_3m_10y'].pct_change(4)
        
        # Credit spread features
        df['credit_spread_momentum_4w'] = df['us_hy_oas'].pct_change(4)
        
        # Use existing economic regime or create new one
        if 'us_economic_regime' in df.columns:
            df['econ_regime_existing'] = df['us_economic_regime']
        
        # Multi-asset regime
        df['us_economic_regime_new'] = (
            (df['tnx_momentum_4w'] > 0).astype(int) + 
            (df['credit_spread_momentum_4w'] < 0).astype(int) + 
            (df['vix_change'] < 0).astype(int)
        ) / 3
        
        # Risk-on/Risk-off
        df['risk_on_regime'] = (
            (df['momentum_4w'] > 0).astype(int) + 
            (df['vix'] < df['vix'].rolling(26).median()).astype(int)
        ) / 2
        
        # Volatility clustering
        df['vol_clustering'] = (df['volatility_4w'] > df['volatility_4w'].shift(1)).astype(int)
        
        # Market stress indicator
        df['market_stress'] = (
            (df['vix'] > df['vix'].rolling(52).quantile(0.8)).astype(int) +
            (df['volatility_8w'] > df['volatility_8w'].rolling(52).quantile(0.8)).astype(int)
        ) / 2
        
        # 7. CROSS-ASSET SIGNALS (12 features)
        # Credit spread features
        df['cad_oas_level'] = df['cad_oas']
        df['cad_oas_change'] = df['cad_oas'].pct_change()
        df['cad_oas_ma_ratio_8w'] = df['cad_oas'] / df['cad_oas'].rolling(8).mean()
        
        # Bond momentum
        df['tnx_ma_ratio_8w'] = df['us_3m_10y'] / df['us_3m_10y'].rolling(8).mean()
        
        # Cross-asset correlations (rolling)
        df['tsx_vix_corr_8w'] = df['weekly_ret'].rolling(8).corr(df['vix_change'])
        df['tsx_tnx_corr_8w'] = df['weekly_ret'].rolling(8).corr(df['tnx_change'])
        df['tsx_credit_corr_8w'] = df['weekly_ret'].rolling(8).corr(df['cad_oas_change'])
        
        # Multi-asset momentum
        df['multi_asset_momentum'] = (
            df['momentum_4w'] + 
            df['credit_spread_momentum_4w'] + 
            df['tnx_momentum_4w']
        ) / 3
        
        # Risk parity signal
        df['risk_parity_signal'] = (
            (df['momentum_8w'] > 0).astype(int) + 
            (df['tnx_momentum_4w'] < 0).astype(int) + 
            (df['credit_spread_momentum_4w'] < 0).astype(int)
        ) / 3
        
        # Flight to quality
        df['flight_to_quality'] = (
            (df['vix'] > df['vix'].rolling(26).quantile(0.7)).astype(int) *
            (df['tnx_momentum_4w'] < 0).astype(int)
        )
        
        # Carry trade proxy
        df['carry_trade_proxy'] = df['tnx_level'] - df['vix'] / 10
        
        # Credit spread impact
        df['credit_spread_impact'] = df['credit_spread_momentum_4w'] * df['momentum_4w']
        
        # 8. COMPOSITE INDICATORS (9 features)
        # Technical score
        df['technical_score'] = (
            (df['momentum_8w'] > 0).astype(int) +
            (df['price_ma_ratio_8w'] > 1).astype(int) +
            (df['rsi_8w'] < 70).astype(int) +
            (df['rsi_8w'] > 30).astype(int) +
            (df['trend_strength_8w'] > 0).astype(int)
        ) / 5
        
        # Macro score
        df['macro_score'] = (
            (df['us_economic_regime'] > 0.5).astype(int) +
            (df['risk_on_regime'] > 0.5).astype(int) +
            (df['market_stress'] < 0.5).astype(int)
        ) / 3
        
        # Momentum quality
        df['momentum_quality'] = df['momentum_8w'] / df['volatility_8w']
        
        # Risk-adjusted momentum
        df['risk_adj_momentum'] = df['momentum_8w'] / (df['vix'] / 100)
        
        # Trend quality
        df['trend_quality'] = df['trend_strength_8w'] / df['volatility_8w']
        
        # Multi-timeframe momentum
        df['multi_tf_momentum'] = (
            df['momentum_4w'] * 0.4 + 
            df['momentum_8w'] * 0.4 + 
            df['momentum_12w'] * 0.2
        )
        
        # Volatility-adjusted returns
        df['vol_adj_returns_4w'] = df['momentum_4w'] / df['volatility_4w']
        df['vol_adj_returns_8w'] = df['momentum_8w'] / df['volatility_8w']
        
        # Regime-adjusted momentum
        df['regime_adj_momentum'] = df['momentum_8w'] * df['us_economic_regime_new']
        
        # 9. INTERACTION FEATURES (2 features)
        # Momentum * Volatility
        df['momentum_vol_8w'] = df['momentum_8w'] * df['volatility_8w']
        
        # Regime * Momentum
        df['regime_momentum'] = df['us_economic_regime_new'] * df['momentum_8w']
        
        # Store processed data
        self.weekly_df = df
        
        # Create feature list (exclude target and non-predictive columns)
        exclude_cols = ['weekly_ret_fwd', price_col, 'ma_8w', 'ma_26w']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('Date')]
        
        print(f"Total features created: {len(self.feature_cols)}")
        
        # Categorize features
        self.categorize_features()
        
    def categorize_features(self):
        """Categorize features into different types"""
        self.feature_categories = {
            'momentum': [f for f in self.feature_cols if 'momentum' in f or 'rsi' in f or 'price_ma_ratio' in f],
            'volatility': [f for f in self.feature_cols if 'volatility' in f or 'vix' in f or 'vol_' in f],
            'trend': [f for f in self.feature_cols if 'trend' in f or 'ma_trend' in f or 'percentile' in f or 'consistency' in f],
            'higher_order': [f for f in self.feature_cols if 'skewness' in f or 'kurtosis' in f or 'autocorr' in f],
            'regime': [f for f in self.feature_cols if 'regime' in f or 'stress' in f or 'clustering' in f],
            'cross_asset': [f for f in self.feature_cols if any(x in f for x in ['tnx', 'dxy', 'corr', 'multi_asset', 'flight', 'carry', 'dollar'])],
            'composite': [f for f in self.feature_cols if any(x in f for x in ['score', 'quality', 'adj'])],
            'interaction': [f for f in self.feature_cols if any(x in f for x in ['momentum_vol_8w', 'regime_momentum'])]
        }
        
        print("Feature categories:")
        for category, features in self.feature_categories.items():
            print(f"  {category.title()}: {len(features)}")
            
    def train_best_model(self):
        """Train the best model (SVM) as identified in original analysis"""
        print("\nTraining SVM model...")
        
        # Prepare data
        df = self.weekly_df.dropna()
        
        # Create binary target (1 if positive return, 0 otherwise)
        self.y = (df['weekly_ret_fwd'] > 0).astype(int)
        self.X = df[self.feature_cols]
        
        print(f"Training data shape: {self.X.shape}")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")
        
        # Train SVM with scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        self.best_model = SVC(probability=True, random_state=42)
        self.best_model.fit(X_scaled, self.y)
        
        # Calculate accuracy
        y_pred = self.best_model.predict(X_scaled)
        accuracy = accuracy_score(self.y, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        return accuracy

# Initialize and run analysis
analyzer = FeatureImportanceAnalyzer('~/Uploads/with_er_daily.csv')
analyzer.load_and_prepare_data()
accuracy = analyzer.train_best_model()

print(f"\nModel training completed with {accuracy:.4f} accuracy")
print(f"Ready for feature importance analysis...")
