#!/usr/bin/env python3
"""
Comprehensive Trading Strategy Development
Following 9-step workflow with statistical validation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy import stats
import xgboost as xgb
import joblib
import time
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)

class TradingStrategy:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.weekly_df = None
        self.features = None
        self.target = None
        self.models = {}
        self.best_model = None
        self.best_config = None
        self.results = {}
        
    def step1_load_inspect(self):
        """Step 1: Load & Inspect Data"""
        print("=" * 60)
        print("STEP 1: LOAD & INSPECT DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"Total Trading Days: {len(self.df)}")
        
        # Identify best price column (prefer cad_ig_er_index)
        price_cols = ['cad_ig_er_index', 'us_ig_er_index', 'us_hy_er_index', 'tsx']
        self.price_col = 'cad_ig_er_index'  # As specified
        
        print(f"\nSelected Price Column: {self.price_col}")
        print(f"Price Range: {self.df[self.price_col].min():.3f} to {self.df[self.price_col].max():.3f}")
        
        # Calculate basic statistics
        returns = self.df[self.price_col].pct_change().dropna()
        print(f"Daily Return Stats:")
        print(f"  Mean: {returns.mean():.4f}")
        print(f"  Std: {returns.std():.4f}")
        print(f"  Sharpe (annualized): {returns.mean() / returns.std() * np.sqrt(252):.3f}")
        
    def step2_preprocess(self):
        """Step 2: Pre-process to Weekly Frequency"""
        print("\n" + "=" * 60)
        print("STEP 2: PRE-PROCESS TO WEEKLY FREQUENCY")
        print("=" * 60)
        
        # Set Date as index
        df_temp = self.df.set_index('Date')
        
        # Resample to weekly (Friday close)
        self.weekly_df = df_temp.resample('W-FRI').last()
        
        # Forward fill NaNs
        self.weekly_df = self.weekly_df.fillna(method='ffill')
        
        # Create forward weekly returns (target)
        self.weekly_df['weekly_ret_fwd'] = self.weekly_df[self.price_col].pct_change().shift(-1)
        
        # Remove last row (no forward return)
        self.weekly_df = self.weekly_df[:-1]
        
        print(f"Weekly Dataset Shape: {self.weekly_df.shape}")
        print(f"Weekly Date Range: {self.weekly_df.index.min()} to {self.weekly_df.index.max()}")
        print(f"Total Weekly Periods: {len(self.weekly_df)}")
        
        # Basic weekly return stats
        weekly_rets = self.weekly_df['weekly_ret_fwd'].dropna()
        print(f"\nWeekly Return Stats:")
        print(f"  Mean: {weekly_rets.mean():.4f}")
        print(f"  Std: {weekly_rets.std():.4f}")
        print(f"  Positive weeks: {(weekly_rets > 0).sum()}/{len(weekly_rets)} ({(weekly_rets > 0).mean():.1%})")
        
    def step3_feature_engineering(self):
        """Step 3: Comprehensive Feature Engineering"""
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 60)
        
        df = self.weekly_df.copy()
        
        # Price-based features
        price = df[self.price_col]
        
        # 1. MOMENTUM FEATURES
        print("Creating momentum features...")
        for period in [2, 4, 8, 12, 26, 52]:
            df[f'momentum_{period}w'] = price.pct_change(period)
            df[f'price_ratio_{period}w'] = price / price.shift(period)
        
        # 2. VOLATILITY FEATURES  
        print("Creating volatility features...")
        returns = price.pct_change()
        for period in [4, 8, 12, 26]:
            df[f'volatility_{period}w'] = returns.rolling(period).std()
            df[f'volatility_ratio_{period}w'] = (returns.rolling(period).std() / 
                                                returns.rolling(period*2).std())
        
        # 3. TREND FEATURES
        print("Creating trend features...")
        for period in [4, 8, 12, 26]:
            # Moving averages
            df[f'ma_{period}w'] = price.rolling(period).mean()
            df[f'price_ma_ratio_{period}w'] = price / df[f'ma_{period}w']
            
            # Trend strength
            df[f'trend_{period}w'] = (price - price.shift(period)) / price.shift(period)
            
        # 4. TECHNICAL INDICATORS
        print("Creating technical indicators...")
        # RSI-like momentum
        for period in [8, 14, 26]:
            delta = returns
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}w'] = 100 - (100 / (1 + rs))
        
        # Bollinger Band position
        for period in [12, 26]:
            ma = price.rolling(period).mean()
            std = price.rolling(period).std()
            df[f'bb_position_{period}w'] = (price - ma) / (2 * std)
        
        # 5. REGIME FEATURES
        print("Creating regime features...")
        # Economic regime (already in data)
        df['regime_change'] = df['us_economic_regime'].diff()
        
        # Volatility regime
        vol_12w = returns.rolling(12).std()
        df['vol_regime'] = (vol_12w > vol_12w.rolling(52).quantile(0.7)).astype(int)
        
        # Market regime based on trend
        trend_12w = df['momentum_12w']
        df['trend_regime'] = (trend_12w > 0).astype(int)
        
        # 6. CROSS-ASSET FEATURES
        print("Creating cross-asset features...")
        # VIX features
        df['vix_ma_ratio_4w'] = df['vix'] / df['vix'].rolling(4).mean()
        df['vix_momentum_4w'] = df['vix'].pct_change(4)
        
        # Spread features
        df['oas_spread'] = df['cad_oas'] - df['us_ig_oas']
        df['hy_ig_spread'] = df['us_hy_oas'] - df['us_ig_oas']
        
        # Yield curve
        df['yield_curve_momentum'] = df['us_3m_10y'].pct_change(4)
        
        # 7. HIGHER ORDER FEATURES
        print("Creating higher-order features...")
        # Momentum of momentum
        df['momentum_momentum_8w'] = df['momentum_8w'].pct_change(4)
        
        # Volatility of volatility
        df['vol_of_vol_12w'] = df['volatility_12w'].rolling(8).std()
        
        # Cross-correlations (simplified)
        df['price_vix_corr'] = price.rolling(26).corr(df['vix'])
        
        # 8. COMPOSITE FEATURES
        print("Creating composite features...")
        # Risk-adjusted momentum
        df['risk_adj_momentum_8w'] = df['momentum_8w'] / df['volatility_8w']
        df['risk_adj_momentum_12w'] = df['momentum_12w'] / df['volatility_12w']
        
        # Multi-timeframe momentum
        df['momentum_composite'] = (df['momentum_4w'] + df['momentum_8w'] + df['momentum_12w']) / 3
        
        # Trend strength composite
        df['trend_strength'] = (df['trend_4w'] + df['trend_8w'] + df['trend_12w']) / 3
        
        # 9. INTERACTION FEATURES
        print("Creating interaction features...")
        # Momentum * Volatility
        df['momentum_vol_8w'] = df['momentum_8w'] * df['volatility_8w']
        
        # Regime * Momentum
        df['regime_momentum'] = df['us_economic_regime'] * df['momentum_8w']
        
        # VIX * Momentum
        df['vix_momentum_interaction'] = df['vix_momentum_4w'] * df['momentum_4w']
        
        # Store processed data
        self.weekly_df = df
        
        # Create feature list (exclude target and non-predictive columns)
        exclude_cols = ['weekly_ret_fwd', self.price_col] + [col for col in df.columns if 'ma_' in col and 'ratio' not in col]
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Total features created: {len(self.feature_cols)}")
        print("Feature categories:")
        momentum_feats = [f for f in self.feature_cols if 'momentum' in f]
        vol_feats = [f for f in self.feature_cols if 'volatility' in f or 'vol_' in f]
        trend_feats = [f for f in self.feature_cols if 'trend' in f or 'ma_ratio' in f]
        regime_feats = [f for f in self.feature_cols if 'regime' in f]
        print(f"  Momentum: {len(momentum_feats)}")
        print(f"  Volatility: {len(vol_feats)}")
        print(f"  Trend: {len(trend_feats)}")
        print(f"  Regime: {len(regime_feats)}")
        print(f"  Other: {len(self.feature_cols) - len(momentum_feats) - len(vol_feats) - len(trend_feats) - len(regime_feats)}")
        
    def step4_modeling(self):
        """Step 4: Model Training and Testing"""
        print("\n" + "=" * 60)
        print("STEP 4: MODEL TRAINING AND TESTING")
        print("=" * 60)
        
        # Prepare data
        df = self.weekly_df.dropna()
        
        # Create binary target (1 if positive return, 0 otherwise)
        y = (df['weekly_ret_fwd'] > 0).astype(int)
        X = df[self.feature_cols]
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Model configurations
        models_config = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
        }
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features for some models
                if name in ['LogisticRegression', 'SVM', 'MLP']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                
                scores.append(accuracy_score(y_val, y_pred))
            
            avg_score = np.mean(scores)
            model_scores[name] = avg_score
            print(f"  CV Accuracy: {avg_score:.4f} (+/- {np.std(scores):.4f})")
        
        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = models_config[best_model_name]
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {model_scores[best_model_name]:.4f})")
        
        # Train final model on all data
        if best_model_name in ['LogisticRegression', 'SVM', 'MLP']:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.best_model.fit(X_scaled, y)
        else:
            self.best_model.fit(X, y)
            self.scaler = None
        
        # Store model info
        self.best_config = {
            'model_name': best_model_name,
            'cv_accuracy': model_scores[best_model_name],
            'features': self.feature_cols,
            'target_positive_rate': y.mean()
        }
        
        self.X = X
        self.y = y
        
    def step5_signals_portfolio(self):
        """Step 5: Generate Trading Signals and Portfolio Logic"""
        print("\n" + "=" * 60)
        print("STEP 5: TRADING SIGNALS AND PORTFOLIO")
        print("=" * 60)
        
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        
        # Generate predictions
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            predictions = self.best_model.predict(X_scaled)
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = self.best_model.predict(X)
            probabilities = self.best_model.predict_proba(X)[:, 1]
        
        # Create signals dataframe
        signals_df = pd.DataFrame({
            'date': df.index,
            'price': df[self.price_col],
            'actual_return': df['weekly_ret_fwd'],
            'prediction': predictions,
            'probability': probabilities,
            'signal': predictions  # Binary: 1 = invest, 0 = cash
        })
        
        # Calculate portfolio returns
        signals_df['strategy_return'] = signals_df['signal'] * signals_df['actual_return']
        signals_df['benchmark_return'] = signals_df['actual_return']
        
        # Calculate cumulative returns
        signals_df['strategy_cumret'] = (1 + signals_df['strategy_return']).cumprod()
        signals_df['benchmark_cumret'] = (1 + signals_df['benchmark_return']).cumprod()
        
        self.signals_df = signals_df
        
        print(f"Signal Statistics:")
        print(f"  Total periods: {len(signals_df)}")
        print(f"  Invest signals: {signals_df['signal'].sum()} ({signals_df['signal'].mean():.1%})")
        print(f"  Cash periods: {(1-signals_df['signal']).sum()} ({(1-signals_df['signal']).mean():.1%})")
        
    def step6_backtest_metrics(self):
        """Step 6: Detailed Backtesting and Performance Metrics"""
        print("\n" + "=" * 60)
        print("STEP 6: BACKTEST AND PERFORMANCE METRICS")
        print("=" * 60)
        
        df = self.signals_df.dropna()
        
        # Calculate performance metrics
        strategy_returns = df['strategy_return']
        benchmark_returns = df['benchmark_return']
        
        # Total returns
        strategy_total_return = df['strategy_cumret'].iloc[-1] - 1
        benchmark_total_return = df['benchmark_cumret'].iloc[-1] - 1
        
        # Annualized metrics
        n_years = len(df) / 52
        strategy_annual_return = (1 + strategy_total_return) ** (1/n_years) - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (1/n_years) - 1
        
        strategy_annual_vol = strategy_returns.std() * np.sqrt(52)
        benchmark_annual_vol = benchmark_returns.std() * np.sqrt(52)
        
        strategy_sharpe = strategy_annual_return / strategy_annual_vol if strategy_annual_vol > 0 else 0
        benchmark_sharpe = benchmark_annual_return / benchmark_annual_vol if benchmark_annual_vol > 0 else 0
        
        # Drawdown analysis
        strategy_cumret = df['strategy_cumret']
        strategy_peak = strategy_cumret.expanding().max()
        strategy_drawdown = (strategy_cumret - strategy_peak) / strategy_peak
        max_drawdown = strategy_drawdown.min()
        
        benchmark_cumret = df['benchmark_cumret']
        benchmark_peak = benchmark_cumret.expanding().max()
        benchmark_drawdown = (benchmark_cumret - benchmark_peak) / benchmark_peak
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        # Win rate
        win_rate = (strategy_returns > 0).mean()
        
        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(52) if excess_returns.std() > 0 else 0
        
        # Store results
        self.results = {
            'strategy_total_return': strategy_total_return,
            'benchmark_total_return': benchmark_total_return,
            'outperformance_ratio': strategy_total_return / benchmark_total_return if benchmark_total_return > 0 else np.inf,
            'strategy_annual_return': strategy_annual_return,
            'benchmark_annual_return': benchmark_annual_return,
            'strategy_annual_vol': strategy_annual_vol,
            'benchmark_annual_vol': benchmark_annual_vol,
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'win_rate': win_rate,
            'information_ratio': information_ratio,
            'n_years': n_years,
            'n_periods': len(df)
        }
        
        # Print results
        print("PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"Total Return:")
        print(f"  Strategy: {strategy_total_return:.1%}")
        print(f"  Benchmark: {benchmark_total_return:.1%}")
        print(f"  Outperformance Ratio: {self.results['outperformance_ratio']:.2f}x")
        print(f"  Target Achievement: {'✓ PASS' if self.results['outperformance_ratio'] > 2.0 else '✗ FAIL'}")
        
        print(f"\nAnnualized Metrics:")
        print(f"  Strategy Return: {strategy_annual_return:.1%}")
        print(f"  Strategy Volatility: {strategy_annual_vol:.1%}")
        print(f"  Strategy Sharpe: {strategy_sharpe:.3f}")
        print(f"  Benchmark Sharpe: {benchmark_sharpe:.3f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {max_drawdown:.1%}")
        print(f"  Benchmark Max DD: {benchmark_max_drawdown:.1%}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Information Ratio: {information_ratio:.3f}")
        
    def step7_robustness_tests(self):
        """Step 7: Robustness Testing"""
        print("\n" + "=" * 60)
        print("STEP 7: ROBUSTNESS TESTING")
        print("=" * 60)
        
        # 1. Walk-Forward Validation
        print("1. Walk-Forward Validation...")
        self._walk_forward_validation()
        
        # 2. Feature Perturbation Test
        print("\n2. Feature Perturbation Test...")
        self._feature_perturbation_test()
        
        # 3. Target Noise Test
        print("\n3. Target Noise Test...")
        self._target_noise_test()
        
        # 4. Statistical Significance Test
        print("\n4. Statistical Significance Test...")
        self._statistical_significance_test()
        
    def _walk_forward_validation(self):
        """Walk-forward validation test"""
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        y = (df['weekly_ret_fwd'] > 0).astype(int)
        
        # Split into 5 periods for walk-forward
        n_periods = len(df)
        period_size = n_periods // 5
        
        wf_results = []
        
        for i in range(2, 5):  # Start from period 2 to have training data
            train_end = i * period_size
            test_start = train_end
            test_end = (i + 1) * period_size if i < 4 else n_periods
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train model
            if self.scaler:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = type(self.best_model)(**self.best_model.get_params())
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            else:
                model = type(self.best_model)(**self.best_model.get_params())
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            
            # Calculate performance
            actual_returns = df['weekly_ret_fwd'].iloc[test_start:test_end]
            strategy_returns = predictions * actual_returns
            benchmark_returns = actual_returns
            
            strategy_cumret = (1 + strategy_returns).prod() - 1
            benchmark_cumret = (1 + benchmark_returns).prod() - 1
            
            wf_results.append({
                'period': i,
                'strategy_return': strategy_cumret,
                'benchmark_return': benchmark_cumret,
                'outperformance': strategy_cumret / benchmark_cumret if benchmark_cumret > 0 else np.inf
            })
        
        # Analyze results
        avg_outperformance = np.mean([r['outperformance'] for r in wf_results if np.isfinite(r['outperformance'])])
        consistent_periods = sum(1 for r in wf_results if r['outperformance'] > 1.0)
        
        print(f"  Walk-Forward Results:")
        print(f"    Average Outperformance: {avg_outperformance:.2f}x")
        print(f"    Consistent Periods: {consistent_periods}/{len(wf_results)}")
        print(f"    Robustness: {'✓ PASS' if avg_outperformance > 1.5 else '✗ FAIL'}")
        
        self.results['wf_avg_outperformance'] = avg_outperformance
        self.results['wf_consistency'] = consistent_periods / len(wf_results)
        
    def _feature_perturbation_test(self):
        """Test robustness to feature perturbation"""
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        
        # Add noise to features and test performance
        noise_levels = [0.05, 0.10, 0.15]
        perturbation_results = []
        
        for noise_level in noise_levels:
            # Add random noise
            X_noisy = X + np.random.normal(0, noise_level * X.std(), X.shape)
            
            # Generate predictions
            if self.scaler:
                X_noisy_scaled = self.scaler.transform(X_noisy)
                predictions = self.best_model.predict(X_noisy_scaled)
            else:
                predictions = self.best_model.predict(X_noisy)
            
            # Calculate performance
            actual_returns = df['weekly_ret_fwd']
            strategy_returns = predictions * actual_returns
            benchmark_returns = actual_returns
            
            strategy_cumret = (1 + strategy_returns).prod() - 1
            benchmark_cumret = (1 + benchmark_returns).prod() - 1
            
            outperformance = strategy_cumret / benchmark_cumret if benchmark_cumret > 0 else np.inf
            perturbation_results.append(outperformance)
        
        avg_perturbed_performance = np.mean([r for r in perturbation_results if np.isfinite(r)])
        performance_degradation = self.results['outperformance_ratio'] - avg_perturbed_performance
        
        print(f"  Feature Perturbation Results:")
        print(f"    Original Performance: {self.results['outperformance_ratio']:.2f}x")
        print(f"    Average Perturbed Performance: {avg_perturbed_performance:.2f}x")
        print(f"    Performance Degradation: {performance_degradation:.2f}x")
        print(f"    Robustness: {'✓ PASS' if performance_degradation < 0.5 else '✗ FAIL'}")
        
        self.results['perturbation_degradation'] = performance_degradation
        
    def _target_noise_test(self):
        """Test robustness to target noise"""
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        y_original = (df['weekly_ret_fwd'] > 0).astype(int)
        
        # Add noise to target and retrain
        noise_results = []
        
        for noise_level in [0.05, 0.10, 0.15]:
            # Flip some labels randomly
            n_flip = int(noise_level * len(y_original))
            flip_indices = np.random.choice(len(y_original), n_flip, replace=False)
            y_noisy = y_original.copy()
            y_noisy.iloc[flip_indices] = 1 - y_noisy.iloc[flip_indices]
            
            # Retrain model
            if self.scaler:
                X_scaled = self.scaler.transform(X)
                model = type(self.best_model)(**self.best_model.get_params())
                model.fit(X_scaled, y_noisy)
                predictions = model.predict(X_scaled)
            else:
                model = type(self.best_model)(**self.best_model.get_params())
                model.fit(X, y_noisy)
                predictions = model.predict(X)
            
            # Calculate performance
            actual_returns = df['weekly_ret_fwd']
            strategy_returns = predictions * actual_returns
            benchmark_returns = actual_returns
            
            strategy_cumret = (1 + strategy_returns).prod() - 1
            benchmark_cumret = (1 + benchmark_returns).prod() - 1
            
            outperformance = strategy_cumret / benchmark_cumret if benchmark_cumret > 0 else np.inf
            noise_results.append(outperformance)
        
        avg_noise_performance = np.mean([r for r in noise_results if np.isfinite(r)])
        noise_degradation = self.results['outperformance_ratio'] - avg_noise_performance
        
        print(f"  Target Noise Results:")
        print(f"    Original Performance: {self.results['outperformance_ratio']:.2f}x")
        print(f"    Average Noise Performance: {avg_noise_performance:.2f}x")
        print(f"    Noise Degradation: {noise_degradation:.2f}x")
        print(f"    Robustness: {'✓ PASS' if noise_degradation < 0.7 else '✗ FAIL'}")
        
        self.results['noise_degradation'] = noise_degradation
        
    def _statistical_significance_test(self):
        """Test statistical significance of outperformance"""
        df = self.signals_df.dropna()
        
        strategy_returns = df['strategy_return']
        benchmark_returns = df['benchmark_return']
        excess_returns = strategy_returns - benchmark_returns
        
        # T-test for mean excess return
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # Bootstrap test for total return difference
        n_bootstrap = 1000
        bootstrap_outperformance = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_indices = np.random.choice(len(df), len(df), replace=True)
            sample_strategy = strategy_returns.iloc[sample_indices]
            sample_benchmark = benchmark_returns.iloc[sample_indices]
            
            sample_strategy_total = (1 + sample_strategy).prod() - 1
            sample_benchmark_total = (1 + sample_benchmark).prod() - 1
            
            if sample_benchmark_total > 0:
                bootstrap_outperformance.append(sample_strategy_total / sample_benchmark_total)
        
        # Calculate confidence interval
        bootstrap_outperformance = np.array(bootstrap_outperformance)
        ci_lower = np.percentile(bootstrap_outperformance, 2.5)
        ci_upper = np.percentile(bootstrap_outperformance, 97.5)
        
        # Check if 2x is within confidence interval
        target_in_ci = ci_lower <= 2.0 <= ci_upper
        
        print(f"  Statistical Significance Results:")
        print(f"    T-test p-value: {p_value:.4f}")
        print(f"    Significance (p < 0.05): {'✓ PASS' if p_value < 0.05 else '✗ FAIL'}")
        print(f"    Bootstrap 95% CI: [{ci_lower:.2f}x, {ci_upper:.2f}x]")
        print(f"    2x Target in CI: {'✓ PASS' if target_in_ci else '✗ FAIL'}")
        
        self.results['p_value'] = p_value
        self.results['ci_lower'] = ci_lower
        self.results['ci_upper'] = ci_upper
        self.results['statistical_significance'] = p_value < 0.05
        
    def step8_iteration_optimization(self):
        """Step 8: Iterative Optimization"""
        print("\n" + "=" * 60)
        print("STEP 8: ITERATIVE OPTIMIZATION")
        print("=" * 60)
        
        # For time constraints, we'll do a simplified optimization
        # In practice, this would involve more extensive hyperparameter tuning
        
        print("Performing simplified optimization...")
        
        # Test different feature subsets
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        y = (df['weekly_ret_fwd'] > 0).astype(int)
        
        # Feature importance from best model
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Test with top features only
            top_features = feature_importance.head(20)['feature'].tolist()
            X_top = X[top_features]
            
            # Quick validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_top):
                X_train, X_val = X_top.iloc[train_idx], X_top.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = type(self.best_model)(**self.best_model.get_params())
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
            
            top_features_score = np.mean(scores)
            
            print(f"\nFeature Selection Results:")
            print(f"  All features accuracy: {self.best_config['cv_accuracy']:.4f}")
            print(f"  Top 20 features accuracy: {top_features_score:.4f}")
            
            if top_features_score > self.best_config['cv_accuracy']:
                print("  ✓ Top features perform better - updating model")
                self.feature_cols = top_features
                self.best_model.fit(X_top, y)
            else:
                print("  → Keeping all features")
        
        print("Optimization complete.")
        
    def step9_reproducibility(self):
        """Step 9: Ensure Reproducibility"""
        print("\n" + "=" * 60)
        print("STEP 9: REPRODUCIBILITY")
        print("=" * 60)
        
        # Save model and configuration
        model_info = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'config': self.best_config,
            'results': self.results,
            'random_seed': 42
        }
        
        joblib.dump(model_info, '/home/ubuntu/best_trading_model.pkl')
        
        # Save detailed results
        results_df = pd.DataFrame([self.results])
        results_df.to_csv('/home/ubuntu/trading_strategy_results.csv', index=False)
        
        # Save signals for analysis
        self.signals_df.to_csv('/home/ubuntu/trading_signals.csv', index=False)
        
        print("Model and results saved:")
        print("  - best_trading_model.pkl")
        print("  - trading_strategy_results.csv") 
        print("  - trading_signals.csv")
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL STRATEGY SUMMARY")
        print("=" * 60)
        
        print(f"Model: {self.best_config['model_name']}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Training Period: {self.signals_df['date'].min()} to {self.signals_df['date'].max()}")
        
        print(f"\nPERFORMANCE:")
        print(f"  Total Return: {self.results['strategy_total_return']:.1%}")
        print(f"  Benchmark Return: {self.results['benchmark_total_return']:.1%}")
        print(f"  Outperformance: {self.results['outperformance_ratio']:.2f}x")
        print(f"  Target Achievement (>2x): {'✓ PASS' if self.results['outperformance_ratio'] > 2.0 else '✗ FAIL'}")
        
        print(f"\nSTATISTICAL VALIDATION:")
        print(f"  P-value: {self.results['p_value']:.4f}")
        print(f"  Significance (p < 0.05): {'✓ PASS' if self.results['statistical_significance'] else '✗ FAIL'}")
        
        print(f"\nROBUSTNESS:")
        print(f"  Walk-Forward Consistency: {self.results['wf_consistency']:.1%}")
        print(f"  Feature Perturbation Robust: {'✓ PASS' if self.results['perturbation_degradation'] < 0.5 else '✗ FAIL'}")
        print(f"  Target Noise Robust: {'✓ PASS' if self.results['noise_degradation'] < 0.7 else '✗ FAIL'}")
        
        # Overall assessment
        target_met = self.results['outperformance_ratio'] > 2.0
        statistically_significant = self.results['statistical_significance']
        robust = (self.results['wf_consistency'] > 0.6 and 
                 self.results['perturbation_degradation'] < 0.5 and 
                 self.results['noise_degradation'] < 0.7)
        
        overall_pass = target_met and statistically_significant and robust
        
        print(f"\nOVERALL ASSESSMENT: {'✓ STRATEGY PASSES ALL REQUIREMENTS' if overall_pass else '✗ STRATEGY NEEDS IMPROVEMENT'}")
        
        return overall_pass

def main():
    """Main execution function"""
    print("COMPREHENSIVE TRADING STRATEGY DEVELOPMENT")
    print("=" * 60)
    print(f"Start Time: {datetime.now()}")
    
    start_time = time.time()
    
    # Initialize strategy
    strategy = TradingStrategy('/home/ubuntu/Uploads/with_er_daily.csv')
    
    try:
        # Execute 9-step workflow
        strategy.step1_load_inspect()
        strategy.step2_preprocess()
        strategy.step3_feature_engineering()
        strategy.step4_modeling()
        strategy.step5_signals_portfolio()
        strategy.step6_backtest_metrics()
        strategy.step7_robustness_tests()
        strategy.step8_iteration_optimization()
        success = strategy.step9_reproducibility()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nExecution Time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        print(f"End Time: {datetime.now()}")
        
        return success
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
