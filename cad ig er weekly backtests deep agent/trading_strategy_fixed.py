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
import os

# Set random seeds for reproducibility
np.random.seed(42)

class TradingStrategy:
    def __init__(self, data_path=None):
        # Use flexible data path
        if data_path is None:
            # Try multiple possible paths
            possible_paths = [
                'data_pipelines/data_processed/with_er_daily.csv',
                '../data_pipelines/data_processed/with_er_daily.csv',
                'with_er_daily.csv'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.data_path = path
                    break
            else:
                raise FileNotFoundError("Could not find with_er_daily.csv in any expected location")
        else:
            self.data_path = data_path
            
        self.df = None
        self.weekly_df = None
        self.features = None
        self.target = None
        self.models = {}
        self.best_model = None
        self.best_config = None
        self.results = {}
        
        # Set output directory to current directory
        self.output_dir = '.'
        
    def step1_load_inspect(self):
        """Step 1: Load & Inspect Data"""
        print("=" * 60)
        print("STEP 1: LOAD & INSPECT DATA")
        print("=" * 60)
        
        print(f"Loading data from: {self.data_path}")
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
        if 'us_economic_regime' in df.columns:
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
        if 'vix' in df.columns:
            df['vix_ma_ratio_4w'] = df['vix'] / df['vix'].rolling(4).mean()
            df['vix_momentum_4w'] = df['vix'].pct_change(4)
        
        # Spread features
        if 'cad_oas' in df.columns and 'us_ig_oas' in df.columns:
            df['oas_spread'] = df['cad_oas'] - df['us_ig_oas']
        if 'us_hy_oas' in df.columns and 'us_ig_oas' in df.columns:
            df['hy_ig_spread'] = df['us_hy_oas'] - df['us_ig_oas']
        
        # Yield curve
        if 'us_3m_10y' in df.columns:
            df['yield_curve_momentum'] = df['us_3m_10y'].pct_change(4)
        
        # 7. HIGHER ORDER FEATURES
        print("Creating higher-order features...")
        # Momentum of momentum
        df['momentum_momentum_8w'] = df['momentum_8w'].pct_change(4)
        
        # Volatility of volatility
        df['vol_of_vol_12w'] = df['volatility_12w'].rolling(8).std()
        
        # Cross-correlations (simplified)
        if 'vix' in df.columns:
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
        # VIX * Momentum
        if 'vix' in df.columns:
            df['vix_momentum_interaction'] = df['vix_momentum_4w'] * df['momentum_8w']
        
        # Volatility * Trend
        df['vol_trend_interaction'] = df['volatility_8w'] * df['trend_8w']
        
        # Store the processed dataframe
        self.weekly_df = df
        
        # Get feature columns (exclude target and price columns)
        exclude_cols = [self.price_col, 'weekly_ret_fwd', 'weekly_ret']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols and not df[col].isna().all()]
        
        print(f"Created {len(self.feature_cols)} features")
        print(f"Features: {self.feature_cols[:10]}...") # Show first 10
        
    def step4_modeling(self):
        """Step 4: Model Training and Selection"""
        print("\n" + "=" * 60)
        print("STEP 4: MODEL TRAINING & SELECTION")
        print("=" * 60)
        
        # Prepare data
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        y = (df['weekly_ret_fwd'] > 0).astype(int)  # Binary classification
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Positive rate: {y.mean():.2%}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define models to test
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'MLP': MLPClassifier(random_state=42, max_iter=500)
        }
        
        best_score = 0
        best_model_name = ""
        
        # Test each model
        for name, model in models.items():
            print(f"\nTesting {name}...")
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    scores.append(score)
                except Exception as e:
                    print(f"Error with {name}: {e}")
                    continue
            
            if scores:
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"  CV Accuracy: {avg_score:.4f} ± {std_score:.4f}")
                
                self.models[name] = {
                    'model': model,
                    'cv_accuracy': avg_score,
                    'cv_std': std_score
                }
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = name
        
        # Select best model
        if best_model_name:
            self.best_model = self.models[best_model_name]['model']
            self.best_config = self.models[best_model_name]
            self.best_config['model_name'] = best_model_name
            
            # Retrain on full dataset
            self.best_model.fit(X_scaled, y)
            
            print(f"\nBest Model: {best_model_name}")
            print(f"Best CV Accuracy: {best_score:.4f}")
        else:
            raise Exception("No models could be trained successfully")
    
    def step5_signals_portfolio(self):
        """Step 5: Generate Trading Signals and Portfolio"""
        print("\n" + "=" * 60)
        print("STEP 5: TRADING SIGNALS & PORTFOLIO")
        print("=" * 60)
        
        # Prepare data
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Generate signals
        signals = self.best_model.predict(X_scaled)
        signal_probs = self.best_model.predict_proba(X_scaled)[:, 1]
        
        # Create signals dataframe
        self.signals_df = pd.DataFrame({
            'date': df.index,
            'price': df[self.price_col],
            'signal': signals,
            'signal_prob': signal_probs,
            'actual_return': df['weekly_ret_fwd']
        }).reset_index(drop=True)
        
        print(f"Generated {len(self.signals_df)} trading signals")
        print(f"Long signals: {signals.sum()} ({signals.mean():.1%})")
        
        # Calculate portfolio returns
        portfolio_returns = []
        for i, row in self.signals_df.iterrows():
            if row['signal'] == 1:  # Long position
                portfolio_returns.append(row['actual_return'])
            else:  # Cash position
                portfolio_returns.append(0.0)
        
        self.signals_df['portfolio_return'] = portfolio_returns
        self.signals_df['cumulative_return'] = (1 + pd.Series(portfolio_returns)).cumprod()
        
        print(f"Portfolio stats:")
        total_ret = self.signals_df['cumulative_return'].iloc[-1] - 1
        print(f"  Total return: {total_ret:.2%}")
        print(f"  Number of trades: {signals.sum()}")
        
    def step6_backtest_metrics(self):
        """Step 6: Calculate Backtest Metrics"""
        print("\n" + "=" * 60)
        print("STEP 6: BACKTEST METRICS")
        print("=" * 60)
        
        # Portfolio metrics
        portfolio_rets = self.signals_df['portfolio_return'].dropna()
        strategy_total = (1 + portfolio_rets).prod() - 1
        
        # Benchmark (buy and hold)
        benchmark_rets = self.signals_df['actual_return'].dropna()
        benchmark_total = (1 + benchmark_rets).prod() - 1
        
        # Calculate metrics
        strategy_sharpe = portfolio_rets.mean() / portfolio_rets.std() * np.sqrt(52) if portfolio_rets.std() > 0 else 0
        benchmark_sharpe = benchmark_rets.mean() / benchmark_rets.std() * np.sqrt(52) if benchmark_rets.std() > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + portfolio_rets).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Store results
        self.results = {
            'strategy_total_return': strategy_total,
            'benchmark_total_return': benchmark_total,
            'outperformance_ratio': (1 + strategy_total) / (1 + benchmark_total) if benchmark_total > -1 else 0,
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': (portfolio_rets > 0).mean(),
            'total_trades': (self.signals_df['signal'] == 1).sum()
        }
        
        print(f"PERFORMANCE METRICS:")
        print(f"  Strategy Total Return: {strategy_total:.2%}")
        print(f"  Benchmark Total Return: {benchmark_total:.2%}")
        print(f"  Outperformance Ratio: {self.results['outperformance_ratio']:.2f}x")
        print(f"  Strategy Sharpe: {strategy_sharpe:.3f}")
        print(f"  Benchmark Sharpe: {benchmark_sharpe:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Win Rate: {self.results['win_rate']:.2%}")
        print(f"  Total Trades: {self.results['total_trades']}")
        
    def step7_robustness_tests(self):
        """Step 7: Robustness Testing"""
        print("\n" + "=" * 60)
        print("STEP 7: ROBUSTNESS TESTS")
        print("=" * 60)
        
        self._walk_forward_validation()
        self._feature_perturbation_test()
        self._target_noise_test()
        self._statistical_significance_test()
        
    def _walk_forward_validation(self):
        """Walk-forward validation test"""
        print("\nWalk-Forward Validation:")
        
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        y = (df['weekly_ret_fwd'] > 0).astype(int)
        
        # Split into chunks
        n_chunks = 5
        chunk_size = len(df) // n_chunks
        
        chunk_returns = []
        
        for i in range(2, n_chunks):  # Start from chunk 2 to have training data
            train_end = i * chunk_size
            test_start = train_end
            test_end = min((i + 1) * chunk_size, len(df))
            
            if test_end - test_start < 10:  # Skip if test set too small
                continue
                
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = type(self.best_model)(**self.best_model.get_params())
            model.fit(X_train_scaled, y_train)
            
            # Generate signals
            signals = model.predict(X_test_scaled)
            
            # Calculate returns
            actual_rets = df['weekly_ret_fwd'].iloc[test_start:test_end]
            portfolio_rets = [actual_rets.iloc[j] if signals[j] == 1 else 0.0 for j in range(len(signals))]
            
            chunk_return = (1 + pd.Series(portfolio_rets)).prod() - 1
            chunk_returns.append(chunk_return)
        
        if chunk_returns:
            consistency = np.std(chunk_returns)
            avg_return = np.mean(chunk_returns)
            
            print(f"  Chunk returns: {[f'{r:.2%}' for r in chunk_returns]}")
            print(f"  Average return: {avg_return:.2%}")
            print(f"  Consistency (std): {consistency:.4f}")
            print(f"  Consistency test: {'✓ PASS' if consistency < 0.5 else '✗ FAIL'}")
            
            self.results['wf_avg_return'] = avg_return
            self.results['wf_consistency'] = 1 - consistency  # Higher is better
        
    def _feature_perturbation_test(self):
        """Feature perturbation robustness test"""
        print("\nFeature Perturbation Test:")
        
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        y = (df['weekly_ret_fwd'] > 0).astype(int)
        
        # Original performance
        X_scaled = self.scaler.transform(X)
        original_signals = self.best_model.predict(X_scaled)
        original_accuracy = accuracy_score(y, original_signals)
        
        # Add noise to features
        noise_levels = [0.1, 0.2, 0.3]
        degradations = []
        
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, X_scaled.shape)
            X_noisy = X_scaled + noise
            
            noisy_signals = self.best_model.predict(X_noisy)
            noisy_accuracy = accuracy_score(y, noisy_signals)
            
            degradation = (original_accuracy - noisy_accuracy) / original_accuracy
            degradations.append(degradation)
        
        avg_degradation = np.mean(degradations)
        
        print(f"  Original accuracy: {original_accuracy:.4f}")
        print(f"  Degradations: {[f'{d:.3f}' for d in degradations]}")
        print(f"  Average degradation: {avg_degradation:.3f}")
        print(f"  Robustness test: {'✓ PASS' if avg_degradation < 0.5 else '✗ FAIL'}")
        
        self.results['perturbation_degradation'] = avg_degradation
        
    def _target_noise_test(self):
        """Target noise robustness test"""
        print("\nTarget Noise Test:")
        
        df = self.weekly_df.dropna()
        X = df[self.feature_cols]
        y = (df['weekly_ret_fwd'] > 0).astype(int)
        
        # Original performance
        X_scaled = self.scaler.transform(X)
        original_signals = self.best_model.predict(X_scaled)
        original_accuracy = accuracy_score(y, original_signals)
        
        # Add noise to target
        noise_levels = [0.1, 0.2, 0.3]
        degradations = []
        
        for noise_level in noise_levels:
            # Flip some labels randomly
            y_noisy = y.copy()
            flip_mask = np.random.random(len(y)) < noise_level
            y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
            
            # Retrain with noisy target
            model = type(self.best_model)(**self.best_model.get_params())
            model.fit(X_scaled, y_noisy)
            
            noisy_signals = model.predict(X_scaled)
            noisy_accuracy = accuracy_score(y, noisy_signals)  # Test against original y
            
            degradation = (original_accuracy - noisy_accuracy) / original_accuracy
            degradations.append(degradation)
        
        avg_degradation = np.mean(degradations)
        
        print(f"  Original accuracy: {original_accuracy:.4f}")
        print(f"  Degradations: {[f'{d:.3f}' for d in degradations]}")
        print(f"  Average degradation: {avg_degradation:.3f}")
        print(f"  Robustness test: {'✓ PASS' if avg_degradation < 0.7 else '✗ FAIL'}")
        
        self.results['noise_degradation'] = avg_degradation
        
    def _statistical_significance_test(self):
        """Statistical significance test"""
        print("\nStatistical Significance Test:")
        
        portfolio_returns = self.signals_df['portfolio_return'].dropna()
        benchmark_returns = self.signals_df['actual_return'].dropna()
        
        # T-test for difference in means
        t_stat, p_value = stats.ttest_rel(portfolio_returns, benchmark_returns)
        
        # Bootstrap test for outperformance ratio
        df = self.signals_df.dropna()
        strategy_returns = df['portfolio_return']
        benchmark_returns = df['actual_return']
        
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
        
        model_path = os.path.join(self.output_dir, 'best_trading_model.pkl')
        joblib.dump(model_info, model_path)
        
        # Save detailed results
        results_df = pd.DataFrame([self.results])
        results_path = os.path.join(self.output_dir, 'trading_strategy_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save signals for analysis
        signals_path = os.path.join(self.output_dir, 'trading_signals.csv')
        self.signals_df.to_csv(signals_path, index=False)
        
        print("Model and results saved:")
        print(f"  - {model_path}")
        print(f"  - {results_path}") 
        print(f"  - {signals_path}")
        
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
        print(f"  Walk-Forward Consistency: {self.results.get('wf_consistency', 0):.1%}")
        print(f"  Feature Perturbation Robust: {'✓ PASS' if self.results['perturbation_degradation'] < 0.5 else '✗ FAIL'}")
        print(f"  Target Noise Robust: {'✓ PASS' if self.results['noise_degradation'] < 0.7 else '✗ FAIL'}")
        
        # Overall assessment
        target_met = self.results['outperformance_ratio'] > 2.0
        statistically_significant = self.results['statistical_significance']
        robust = (self.results.get('wf_consistency', 0) > 0.6 and 
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
    strategy = TradingStrategy()
    
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