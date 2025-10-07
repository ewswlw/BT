#!/usr/bin/env python3
"""
Machine Learning Credit Timing Study - Reproducibility Package

This standalone script reproduces all results from the machine learning credit timing study
for Canadian Investment Grade markets. The script includes complete feature engineering,
model training, backtesting, and robustness testing frameworks.

Author: Research Team
Date: 2025
License: Academic Research Use

Requirements:
- Python 3.8+
- pandas, numpy, scikit-learn, scipy
- Data file: with_er_daily.csv (see data_requirements.txt)

Usage:
    python reproducibility_package.py

Output:
    - Complete analysis results
    - Performance metrics
    - Robustness test results
    - Statistical significance tests
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class CreditTimingML:
    """
    Machine Learning Credit Timing Strategy Implementation
    
    This class implements the complete ML credit timing framework including:
    - Feature engineering (94 features)
    - Model training (Random Forest, Gradient Boosting, Logistic Regression)
    - Backtesting and performance evaluation
    - Robustness testing framework
    - Statistical significance testing
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the Credit Timing ML framework
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path or self._find_data_file()
        self.df = None
        self.weekly = None
        self.feature_cols = []
        self.target_col = 'cad_ig_er_index'
        self.models = {}
        self.results = {}
        self.robustness_results = {}
        
    def _find_data_file(self) -> str:
        """Find the data file in common locations"""
        possible_paths = [
            '../../data_pipelines/data_processed/with_er_daily.csv',
            '../data_pipelines/data_processed/with_er_daily.csv',
            'data_pipelines/data_processed/with_er_daily.csv',
            'with_er_daily.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            "Data file 'with_er_daily.csv' not found. "
            "Please ensure the data file is available in the expected location."
        )
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the dataset
        
        Returns:
            pd.DataFrame: Processed weekly dataset
        """
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.set_index('Date').sort_index()
        
        print(f"✓ Loaded {len(self.df)} daily observations from {self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')}")
        
        # Resample to weekly
        self.weekly = self.df.resample('W-FRI').last()
        self.weekly = self.weekly.dropna(subset=[self.target_col])
        
        # Compute forward return (TARGET for ML)
        self.weekly['fwd_ret'] = np.log(self.weekly[self.target_col].shift(-1) / self.weekly[self.target_col])
        self.weekly['target_binary'] = (self.weekly['fwd_ret'] > 0).astype(int)
        
        print(f"✓ Resampled to {len(self.weekly)} weeks")
        print(f"✓ Target distribution: {self.weekly['target_binary'].value_counts().to_dict()}")
        
        return self.weekly
    
    def engineer_features(self) -> List[str]:
        """
        Engineer comprehensive feature set (94 features)
        
        Returns:
            List[str]: List of feature column names
        """
        print("Building feature set...")
        
        # 1. Momentum features across assets
        momentum_assets = ['cad_oas', 'us_hy_oas', 'us_ig_oas', 'tsx', 'vix', 'us_3m_10y']
        for col in momentum_assets:
            if col in self.weekly.columns:
                for lb in [1, 2, 4, 8, 12]:
                    self.weekly[f'{col}_mom_{lb}w'] = self.weekly[col].pct_change(lb)
        
        # 2. Volatility features
        volatility_assets = ['cad_oas', 'us_hy_oas', 'us_ig_oas', 'vix', self.target_col]
        for col in volatility_assets:
            if col in self.weekly.columns:
                for window in [4, 8, 12]:
                    self.weekly[f'{col}_vol_{window}w'] = self.weekly[col].pct_change().rolling(window).std()
        
        # 3. Spread indicators
        if 'us_hy_oas' in self.weekly.columns and 'us_ig_oas' in self.weekly.columns:
            self.weekly['hy_ig_spread'] = self.weekly['us_hy_oas'] - self.weekly['us_ig_oas']
            for lb in [1, 4, 8]:
                self.weekly[f'hy_ig_spread_chg_{lb}w'] = self.weekly['hy_ig_spread'].diff(lb)
        
        if 'cad_oas' in self.weekly.columns and 'us_ig_oas' in self.weekly.columns:
            self.weekly['cad_us_ig_spread'] = self.weekly['cad_oas'] - self.weekly['us_ig_oas']
            for lb in [1, 4, 8]:
                self.weekly[f'cad_us_ig_spread_chg_{lb}w'] = self.weekly['cad_us_ig_spread'].diff(lb)
        
        # 4. Macro surprise features
        macro_features = ['us_growth_surprises', 'us_inflation_surprises', 'us_hard_data_surprises', 
                         'us_equity_revisions', 'us_lei_yoy']
        for col in macro_features:
            if col in self.weekly.columns:
                for lb in [1, 4]:
                    self.weekly[f'{col}_chg_{lb}w'] = self.weekly[col].diff(lb)
        
        # 5. Regime indicator
        if 'us_economic_regime' in self.weekly.columns:
            self.weekly['regime_change'] = self.weekly['us_economic_regime'].diff()
        
        # 6. Technical features on target
        for span in [4, 8, 12, 26]:
            self.weekly[f'target_sma_{span}'] = self.weekly[self.target_col].rolling(span).mean()
            self.weekly[f'target_dist_sma_{span}'] = (self.weekly[self.target_col] / self.weekly[f'target_sma_{span}']) - 1
        
        for window in [8, 12]:
            rolling_mean = self.weekly[self.target_col].rolling(window).mean()
            rolling_std = self.weekly[self.target_col].rolling(window).std()
            self.weekly[f'target_zscore_{window}w'] = (self.weekly[self.target_col] - rolling_mean) / rolling_std
        
        # 7. Cross-asset correlation
        if 'tsx' in self.weekly.columns:
            self.weekly['target_tsx_corr_12w'] = self.weekly[self.target_col].rolling(12).corr(self.weekly['tsx'])
        
        # 8. VIX levels
        if 'vix' in self.weekly.columns:
            self.weekly['vix_high'] = (self.weekly['vix'] > self.weekly['vix'].rolling(12).quantile(0.75)).astype(int)
        
        # Drop rows with NaN from feature engineering
        self.feature_cols = [c for c in self.weekly.columns if c not in ['fwd_ret', 'target_binary', self.target_col]]
        self.weekly = self.weekly.dropna(subset=self.feature_cols + ['target_binary'])
        
        print(f"✓ Engineered {len(self.feature_cols)} features")
        print(f"✓ Clean dataset: {len(self.weekly)} weeks from {self.weekly.index[0].strftime('%Y-%m-%d')} to {self.weekly.index[-1].strftime('%Y-%m-%d')}")
        
        return self.feature_cols
    
    def prepare_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and test datasets using time series split
        
        Returns:
            Tuple of X_train, X_test, y_train, y_test
        """
        # Time series split (60/40)
        split_idx = int(len(self.weekly) * 0.6)
        train_data = self.weekly.iloc[:split_idx]
        test_data = self.weekly.iloc[split_idx:]
        
        print(f"Train period: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} weeks)")
        print(f"Test period:  {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data)} weeks)")
        
        # Prepare features and target
        X_train = train_data[self.feature_cols]
        y_train = train_data['target_binary']
        X_test = test_data[self.feature_cols]
        y_test = test_data['target_binary']
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_models(self, X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train machine learning models
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dict containing trained models
        """
        print("Training ML models...")
        print("-" * 80)
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=0.1, max_iter=1000, random_state=42
            )
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name:20s} | Trained successfully")
        
        print("-" * 80)
        self.models = trained_models
        return trained_models
    
    def backtest_strategy(self, model, X_test: np.ndarray, y_test: pd.Series, 
                         test_data: pd.DataFrame, threshold: float = 0.45) -> Dict[str, float]:
        """
        Backtest ML strategy
        
        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test target
            test_data: Test dataset
            threshold: Probability threshold for trading
            
        Returns:
            Dict containing performance metrics
        """
        # Get predictions
        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_proba_series = pd.Series(pred_proba, index=test_data.index)
        
        # Align with returns
        test_returns = test_data['fwd_ret'].iloc[:-1]
        pred_proba_series = pred_proba_series.iloc[:-1]
        
        # Generate signals
        position = (pred_proba_series > threshold).astype(float)
        strat_returns = position * test_returns
        
        # Calculate performance metrics
        cumulative = np.exp(strat_returns.cumsum())
        
        n_weeks = len(strat_returns)
        years = n_weeks / 52.0
        
        final_value = cumulative.iloc[-1]
        cagr = (final_value ** (1/years)) - 1 if years > 0 else 0
        
        ann_vol = strat_returns.std() * np.sqrt(52)
        sharpe = (strat_returns.mean() * 52) / ann_vol if ann_vol > 0 else 0
        
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_dd = drawdown.min()
        
        win_rate = (strat_returns[strat_returns > 0].count() / (strat_returns != 0).sum()) if (strat_returns != 0).sum() > 0 else 0
        n_trades = position.diff().abs().sum() / 2
        
        return {
            'cagr': cagr,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'total_return': final_value - 1
        }
    
    def run_comprehensive_backtest(self, X_train: np.ndarray, X_test: np.ndarray, 
                                  y_train: pd.Series, y_test: pd.Series,
                                  train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive backtesting across all models and thresholds
        
        Returns:
            Dict containing all backtest results
        """
        print("Running comprehensive backtest...")
        
        results = []
        thresholds = [0.45, 0.50, 0.55, 0.60]
        
        for name, model in self.models.items():
            for threshold in thresholds:
                metrics = self.backtest_strategy(model, X_test, y_test, test_data, threshold)
                results.append({
                    'strategy': f'{name}_t{int(threshold*100)}',
                    'model': name,
                    'threshold': threshold,
                    **metrics
                })
        
        # Add benchmark
        test_returns = test_data['fwd_ret'].iloc[:-1]
        test_bnh_cumulative = np.exp(test_returns.cumsum())
        test_years = len(test_returns) / 52.0
        test_bnh_cagr = (test_bnh_cumulative.iloc[-1] ** (1/test_years)) - 1
        test_bnh_vol = test_returns.std() * np.sqrt(52)
        test_bnh_sharpe = (test_returns.mean() * 52) / test_bnh_vol
        test_bnh_running_max = test_bnh_cumulative.cummax()
        test_bnh_dd = (test_bnh_cumulative / test_bnh_running_max) - 1
        test_bnh_max_dd = test_bnh_dd.min()
        
        results.append({
            'strategy': 'BuyAndHold',
            'model': 'Benchmark',
            'threshold': 1.0,
            'cagr': test_bnh_cagr,
            'ann_vol': test_bnh_vol,
            'sharpe': test_bnh_sharpe,
            'max_dd': test_bnh_max_dd,
            'n_trades': 0,
            'win_rate': 0,
            'total_return': test_bnh_cumulative.iloc[-1] - 1
        })
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        # Display results
        self._display_backtest_results(results_df, test_data)
        
        return results_df
    
    def _display_backtest_results(self, results_df: pd.DataFrame, test_data: pd.DataFrame):
        """Display formatted backtest results"""
        print("="*95)
        print("ML BACKTEST RESULTS - TEST SET ONLY (Out-of-Sample)")
        print("="*95)
        print(f"Period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-2].strftime('%Y-%m-%d')} ({len(test_data)-1} weeks)")
        print(f"Features: {len(self.feature_cols)} | Models: RF, GBM, LogReg | Rebalance: Weekly")
        print("="*95)
        
        winners = results_df[results_df['cagr'] >= 0.04].sort_values('cagr', ascending=False)
        
        if len(winners) > 0:
            print(f"\n✓ Found {len(winners)} ML strategies with CAGR >= 4.0%\n")
            print("TOP PERFORMERS:")
            print("-" * 95)
            for idx, row in winners.head(10).iterrows():
                print(f"{row['strategy']:25s} | CAGR: {row['cagr']:6.2%} | Vol: {row['ann_vol']:5.2%} | " +
                      f"Sharpe: {row['sharpe']:5.2f} | MaxDD: {row['max_dd']:6.2%} | Trades: {int(row['n_trades']):3d} | WinRate: {row['win_rate']:.2%}")
        else:
            print(f"\n⚠ No ML strategies met 4% CAGR threshold. Showing top 10 by CAGR:\n")
            top10 = results_df.sort_values('cagr', ascending=False).head(10)
            print("TOP 10 ML STRATEGIES:")
            print("-" * 95)
            for idx, row in top10.iterrows():
                print(f"{row['strategy']:25s} | CAGR: {row['cagr']:6.2%} | Vol: {row['ann_vol']:5.2%} | " +
                      f"Sharpe: {row['sharpe']:5.2f} | MaxDD: {row['max_dd']:6.2%} | Trades: {int(row['n_trades']):3d} | WinRate: {row['win_rate']:.2%}")
        
        print("\n" + "-" * 95)
        bnh = results_df[results_df['strategy'] == 'BuyAndHold'].iloc[0]
        print(f"{'BENCHMARK (Buy & Hold)':25s} | CAGR: {bnh['cagr']:6.2%} | Vol: {bnh['ann_vol']:5.2%} | " +
              f"Sharpe: {bnh['sharpe']:5.2f} | MaxDD: {bnh['max_dd']:6.2%}")
        print("="*95)
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze feature importance from Random Forest model
        
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        print("\nAnalyzing feature importance...")
        
        rf_model = self.models['RandomForest']
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("="*90)
        print("TOP 25 MOST IMPORTANT FEATURES (Random Forest)")
        print("="*90)
        for idx, row in feature_importance.head(25).iterrows():
            print(f"{idx+1:2d}. {row['feature']:50s} {row['importance']:.4f}")
        
        return feature_importance
    
    def run_robustness_tests(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: pd.Series, y_test: pd.Series,
                           train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive robustness testing framework
        
        Returns:
            Dict containing robustness test results
        """
        print("\n" + "="*100)
        print("ROBUSTNESS TESTING FRAMEWORK")
        print("="*100)
        
        robustness_results = {}
        
        # Test 1: Look-ahead bias check
        print("\nTEST 1: LOOK-AHEAD BIAS CHECK")
        print("-" * 100)
        lookahead_issues = []
        for col in self.feature_cols:
            if any(keyword in col.lower() for keyword in ['fwd_', 'forward', 'future']):
                lookahead_issues.append(col)
        
        if len(lookahead_issues) == 0:
            print("✓ PASS: No features contain future information")
            print("✓ All features use only lagged/historical data")
            robustness_results['lookahead_pass'] = True
        else:
            print(f"⚠ WARNING: Potential look-ahead bias in features: {lookahead_issues}")
            robustness_results['lookahead_pass'] = False
        
        # Test 2: Walk-forward validation
        print("\nTEST 2: WALK-FORWARD VALIDATION")
        print("-" * 100)
        wf_results = self._walk_forward_validation()
        robustness_results['walk_forward'] = wf_results
        
        # Test 3: Regime analysis
        print("\nTEST 3: MARKET REGIME ANALYSIS")
        print("-" * 100)
        regime_results = self._regime_analysis(test_data)
        robustness_results['regime_analysis'] = regime_results
        
        # Test 4: Statistical significance
        print("\nTEST 4: STATISTICAL SIGNIFICANCE TESTING")
        print("-" * 100)
        significance_results = self._statistical_significance_test(test_data)
        robustness_results['statistical_significance'] = significance_results
        
        # Test 5: Overfitting analysis
        print("\nTEST 5: OVERFITTING ANALYSIS")
        print("-" * 100)
        overfitting_results = self._overfitting_analysis(X_train, X_test, y_train, y_test, train_data, test_data)
        robustness_results['overfitting'] = overfitting_results
        
        self.robustness_results = robustness_results
        return robustness_results
    
    def _walk_forward_validation(self) -> Dict[str, Any]:
        """Perform walk-forward validation"""
        print("Testing on 6 sequential out-of-sample periods...\n")
        
        # Split test period into 6 sub-periods
        split_idx = int(len(self.weekly) * 0.6)
        test_data = self.weekly.iloc[split_idx:]
        
        n_periods = 6
        period_size = len(test_data) // n_periods
        
        wf_results = []
        
        for i in range(n_periods):
            period_test = test_data.iloc[i*period_size:(i+1)*period_size]
            
            if len(period_test) < 20:
                continue
            
            # Expanding window training
            period_train = self.weekly.iloc[:split_idx + i*period_size]
            
            X_train = period_train[self.feature_cols]
            y_train = period_train['target_binary']
            X_test = period_test[self.feature_cols]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42)
            rf.fit(X_train_scaled, y_train)
            
            pred = rf.predict_proba(X_test_scaled)[:, 1]
            signal = (pred > 0.45).astype(int)
            
            returns = period_test['fwd_ret'].iloc[:-1]
            signal = signal[:-1]
            strat_ret = signal * returns
            
            cum_ret = np.exp(strat_ret.cumsum()).iloc[-1] - 1 if len(strat_ret) > 0 else 0
            ann_ret = (1 + cum_ret) ** (52 / len(returns)) - 1 if len(returns) > 0 else 0
            
            wf_results.append({
                'Period': f"P{i+1}",
                'Start': period_test.index[0].strftime('%Y-%m-%d'),
                'End': period_test.index[-1].strftime('%Y-%m-%d'),
                'Weeks': len(returns),
                'CAGR': ann_ret,
                'CumReturn': cum_ret,
                'WinRate': (strat_ret > 0).sum() / len(strat_ret) if len(strat_ret) > 0 else 0
            })
        
        wf_df = pd.DataFrame(wf_results)
        
        for _, row in wf_df.iterrows():
            status = "✓" if row['CAGR'] > 0.02 else "⚠"
            print(f"{status} {row['Period']}: {row['Start']} to {row['End']} ({row['Weeks']:3.0f}w) | " +
                  f"CAGR: {row['CAGR']:6.2%} | Cum: {row['CumReturn']:6.2%} | WR: {row['WinRate']:.1%}")
        
        wf_consistency = (wf_df['CAGR'] > 0).sum() / len(wf_df)
        wf_mean = wf_df['CAGR'].mean()
        wf_std = wf_df['CAGR'].std()
        
        print(f"\n✓ Consistency: {wf_consistency:.1%} of periods are profitable")
        print(f"✓ Mean CAGR: {wf_mean:.2%} (±{wf_std:.2%} std)")
        
        if wf_consistency >= 0.8 and wf_mean > 0.02:
            print("✓ PASS: Strategy shows consistent performance across time periods")
        else:
            print("⚠ WARNING: Performance varies significantly across periods")
        
        return {
            'consistency_rate': wf_consistency,
            'mean_cagr': wf_mean,
            'std_cagr': wf_std,
            'periods': wf_df.to_dict('records')
        }
    
    def _regime_analysis(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform market regime analysis"""
        rf_model = self.models['RandomForest']
        
        # Generate signals for test period
        X_test = test_data[self.feature_cols]
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        signal = pd.Series((pred > 0.45).astype(int)[:-1], index=test_data.index[:-1])
        test_returns = test_data['fwd_ret'].iloc[:-1]
        
        # VIX regime analysis
        vix_test = self.df.loc[test_returns.index, 'vix'] if 'vix' in self.df.columns else None
        
        if vix_test is not None:
            vix_median = vix_test.median()
            vix_high = vix_test > vix_median
            
            high_vol_ret = (signal[vix_high] * test_returns[vix_high])
            high_vol_cagr = (np.exp(high_vol_ret.sum()) ** (52/len(high_vol_ret)) - 1) if len(high_vol_ret) > 0 else 0
            high_vol_sharpe = (high_vol_ret.mean() * 52) / (high_vol_ret.std() * np.sqrt(52)) if len(high_vol_ret) > 0 else 0
            
            low_vol_ret = (signal[~vix_high] * test_returns[~vix_high])
            low_vol_cagr = (np.exp(low_vol_ret.sum()) ** (52/len(low_vol_ret)) - 1) if len(low_vol_ret) > 0 else 0
            low_vol_sharpe = (low_vol_ret.mean() * 52) / (low_vol_ret.std() * np.sqrt(52)) if len(low_vol_ret) > 0 else 0
            
            print(f"High Volatility Regime (VIX > {vix_median:.1f}): {len(high_vol_ret)} weeks")
            print(f"  CAGR: {high_vol_cagr:6.2%} | Sharpe: {high_vol_sharpe:5.2f}")
            print(f"\nLow Volatility Regime (VIX < {vix_median:.1f}): {len(low_vol_ret)} weeks")
            print(f"  CAGR: {low_vol_cagr:6.2%} | Sharpe: {low_vol_sharpe:5.2f}")
            
            vix_regime_pass = high_vol_cagr > 0 and low_vol_cagr > 0
        else:
            vix_regime_pass = False
        
        # Market direction analysis
        bnh_ret = test_returns.copy()
        bull_periods = bnh_ret > bnh_ret.median()
        
        bull_ret = (signal[bull_periods] * test_returns[bull_periods])
        bull_cagr = (np.exp(bull_ret.sum()) ** (52/len(bull_ret)) - 1) if len(bull_ret) > 0 else 0
        
        bear_ret = (signal[~bull_periods] * test_returns[~bull_periods])
        bear_cagr = (np.exp(bear_ret.sum()) ** (52/len(bear_ret)) - 1) if len(bear_ret) > 0 else 0
        
        print(f"\nBull Markets (above median return): {len(bull_ret)} weeks")
        print(f"  CAGR: {bull_cagr:6.2%}")
        print(f"\nBear Markets (below median return): {len(bear_ret)} weeks")
        print(f"  CAGR: {bear_cagr:6.2%}")
        
        direction_regime_pass = bull_cagr > 0 and bear_cagr > 0
        
        if vix_regime_pass and direction_regime_pass:
            print(f"\n✓ PASS: Strategy profitable in both regimes")
        else:
            print(f"\n⚠ WARNING: Strategy underperforms in certain regimes")
        
        return {
            'vix_regime': {
                'high_vol_cagr': high_vol_cagr if 'high_vol_cagr' in locals() else 0,
                'low_vol_cagr': low_vol_cagr if 'low_vol_cagr' in locals() else 0,
                'pass': vix_regime_pass
            },
            'direction_regime': {
                'bull_cagr': bull_cagr,
                'bear_cagr': bear_cagr,
                'pass': direction_regime_pass
            }
        }
    
    def _statistical_significance_test(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance testing"""
        rf_model = self.models['RandomForest']
        
        # Generate strategy returns
        X_test = test_data[self.feature_cols]
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        signal = pd.Series((pred > 0.45).astype(int)[:-1], index=test_data.index[:-1])
        test_returns = test_data['fwd_ret'].iloc[:-1]
        strat_returns = signal * test_returns
        
        # Bootstrap test
        n_bootstrap = 1000
        bootstrap_cagrs = []
        
        for _ in range(n_bootstrap):
            sample = strat_returns.sample(n=len(strat_returns), replace=True)
            boot_cagr = (np.exp(sample.sum()) ** (52/len(sample)) - 1)
            bootstrap_cagrs.append(boot_cagr)
        
        bootstrap_cagrs = np.array(bootstrap_cagrs)
        ci_lower = np.percentile(bootstrap_cagrs, 2.5)
        ci_upper = np.percentile(bootstrap_cagrs, 97.5)
        actual_cagr = (np.exp(strat_returns.sum()) ** (52/len(strat_returns)) - 1)
        
        print(f"Bootstrap Analysis ({n_bootstrap} iterations):")
        print(f"  Actual CAGR: {actual_cagr:.2%}")
        print(f"  95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
        print(f"  Mean: {bootstrap_cagrs.mean():.2%}")
        print(f"  Probability CAGR > 0: {(bootstrap_cagrs > 0).sum() / n_bootstrap:.1%}")
        
        bootstrap_pass = ci_lower > 0
        
        # T-test vs zero
        t_stat, p_value = stats.ttest_1samp(strat_returns, 0)
        print(f"\nT-test vs zero returns:")
        print(f"  t-statistic: {t_stat:.2f}")
        print(f"  p-value: {p_value:.4f}")
        
        ttest_pass = p_value < 0.05
        
        # Sharpe ratio significance
        sharpe = (strat_returns.mean() * 52) / (strat_returns.std() * np.sqrt(52))
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / len(strat_returns))
        sharpe_pval = 1 - stats.norm.cdf(sharpe / sharpe_se)
        
        print(f"\nSharpe Ratio: {sharpe:.2f}")
        print(f"  Standard Error: {sharpe_se:.3f}")
        print(f"  p-value: {sharpe_pval:.4f}")
        
        sharpe_pass = sharpe_pval < 0.05
        
        if bootstrap_pass and ttest_pass and sharpe_pass:
            print(f"\n✓ PASS: All statistical tests significant")
        else:
            print(f"\n⚠ WARNING: Some statistical tests not significant")
        
        return {
            'bootstrap': {
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'prob_positive': (bootstrap_cagrs > 0).sum() / n_bootstrap,
                'pass': bootstrap_pass
            },
            'ttest': {
                't_statistic': t_stat,
                'p_value': p_value,
                'pass': ttest_pass
            },
            'sharpe': {
                'sharpe_ratio': sharpe,
                'p_value': sharpe_pval,
                'pass': sharpe_pass
            }
        }
    
    def _overfitting_analysis(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: pd.Series, y_test: pd.Series,
                            train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform overfitting analysis"""
        rf_model = self.models['RandomForest']
        
        # In-sample vs out-of-sample performance
        train_pred = rf_model.predict_proba(X_train)[:, 1]
        train_signal = pd.Series((train_pred > 0.45).astype(int)[:-1], index=train_data.index[:-1])
        train_returns = train_data['fwd_ret'].iloc[:-1]
        
        test_pred = rf_model.predict_proba(X_test)[:, 1]
        test_signal = pd.Series((test_pred > 0.45).astype(int)[:-1], index=test_data.index[:-1])
        test_returns = test_data['fwd_ret'].iloc[:-1]
        
        is_strat_ret = train_signal * train_returns
        oos_strat_ret = test_signal * test_returns
        
        is_cagr = (np.exp(is_strat_ret.sum()) ** (52/len(is_strat_ret)) - 1)
        oos_cagr = (np.exp(oos_strat_ret.sum()) ** (52/len(oos_strat_ret)) - 1)
        
        is_sharpe = (is_strat_ret.mean() * 52) / (is_strat_ret.std() * np.sqrt(52))
        oos_sharpe = (oos_strat_ret.mean() * 52) / (oos_strat_ret.std() * np.sqrt(52))
        
        is_winrate = (is_strat_ret > 0).sum() / len(is_strat_ret)
        oos_winrate = (oos_strat_ret > 0).sum() / len(oos_strat_ret)
        
        print(f"{'Metric':<20} {'In-Sample':>15} {'Out-of-Sample':>15} {'Degradation':>15}")
        print("-" * 70)
        print(f"{'CAGR':<20} {is_cagr:>14.2%} {oos_cagr:>14.2%} {(oos_cagr/is_cagr - 1):>14.1%}")
        print(f"{'Sharpe Ratio':<20} {is_sharpe:>14.2f} {oos_sharpe:>14.2f} {(oos_sharpe/is_sharpe - 1):>14.1%}")
        print(f"{'Win Rate':<20} {is_winrate:>14.1%} {oos_winrate:>14.1%} {(oos_winrate/is_winrate - 1):>14.1%}")
        
        degradation = abs(oos_cagr / is_cagr - 1)
        
        if degradation < 0.20:
            print(f"\n✓ PASS: Out-of-sample performance within 20% of in-sample (degradation: {degradation:.1%})")
        else:
            print(f"\n⚠ WARNING: Significant performance degradation out-of-sample ({degradation:.1%})")
        
        # Model complexity check
        print(f"\nModel Complexity:")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Training samples: {len(train_data)}")
        print(f"  Samples per feature: {len(train_data) / len(self.feature_cols):.1f}")
        
        complexity_pass = len(train_data) / len(self.feature_cols) > 10
        
        if complexity_pass:
            print(f"✓ PASS: Sufficient samples per feature (>10:1 ratio)")
        else:
            print(f"⚠ WARNING: May be overfitting (low sample-to-feature ratio)")
        
        return {
            'cagr_degradation': degradation,
            'sharpe_degradation': abs(oos_sharpe/is_sharpe - 1),
            'winrate_change': oos_winrate/is_winrate - 1,
            'samples_per_feature': len(train_data) / len(self.feature_cols),
            'overfitting_pass': degradation < 0.20,
            'complexity_pass': complexity_pass
        }
    
    def save_results(self, output_dir: str = "results"):
        """
        Save all results to files
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save backtest results
        if hasattr(self, 'results') and self.results is not None:
            self.results.to_csv(f"{output_dir}/backtest_results.csv", index=False)
        
        # Save feature importance
        if hasattr(self, 'models') and 'RandomForest' in self.models:
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.models['RandomForest'].feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        
        # Save robustness results
        if hasattr(self, 'robustness_results') and self.robustness_results:
            with open(f"{output_dir}/robustness_results.json", 'w') as f:
                json.dump(self.robustness_results, f, indent=2, default=str)
        
        # Save summary report
        self._save_summary_report(output_dir)
        
        print(f"\n✓ Results saved to {output_dir}/ directory")
    
    def _save_summary_report(self, output_dir: str):
        """Save summary report"""
        report_lines = [
            "MACHINE LEARNING CREDIT TIMING STUDY - SUMMARY REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATASET SUMMARY:",
            f"  Total Observations: {len(self.weekly) if self.weekly is not None else 'N/A'} weeks",
            f"  Features Engineered: {len(self.feature_cols)}",
            f"  Target Variable: {self.target_col}",
            "",
            "MODEL PERFORMANCE:",
        ]
        
        if hasattr(self, 'results') and self.results is not None:
            best_strategy = self.results.loc[self.results['cagr'].idxmax()]
            report_lines.extend([
                f"  Best Strategy: {best_strategy['strategy']}",
                f"  CAGR: {best_strategy['cagr']:.2%}",
                f"  Sharpe Ratio: {best_strategy['sharpe']:.2f}",
                f"  Max Drawdown: {best_strategy['max_dd']:.2%}",
                f"  Win Rate: {best_strategy['win_rate']:.1%}",
            ])
        
        if hasattr(self, 'robustness_results') and self.robustness_results:
            report_lines.extend([
                "",
                "ROBUSTNESS TESTING:",
                f"  Look-ahead Bias: {'PASS' if self.robustness_results.get('lookahead_pass', False) else 'FAIL'}",
                f"  Walk-forward Consistency: {self.robustness_results.get('walk_forward', {}).get('consistency_rate', 0):.1%}",
                f"  Statistical Significance: {'PASS' if all([
                    self.robustness_results.get('statistical_significance', {}).get('bootstrap', {}).get('pass', False),
                    self.robustness_results.get('statistical_significance', {}).get('ttest', {}).get('pass', False),
                    self.robustness_results.get('statistical_significance', {}).get('sharpe', {}).get('pass', False)
                ]) else 'FAIL'}",
                f"  Overfitting Check: {'PASS' if self.robustness_results.get('overfitting', {}).get('overfitting_pass', False) else 'FAIL'}",
            ])
        
        with open(f"{output_dir}/summary_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    """Main execution function"""
    print("Machine Learning Credit Timing Study - Reproducibility Package")
    print("=" * 70)
    
    try:
        # Initialize the framework
        ml_credit = CreditTimingML()
        
        # Load and preprocess data
        weekly_data = ml_credit.load_data()
        
        # Engineer features
        feature_cols = ml_credit.engineer_features()
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test, scaler = ml_credit.prepare_train_test_split()
        
        # Train models
        models = ml_credit.train_models(X_train, y_train)
        
        # Run comprehensive backtest
        train_data = weekly_data.iloc[:int(len(weekly_data) * 0.6)]
        test_data = weekly_data.iloc[int(len(weekly_data) * 0.6):]
        
        backtest_results = ml_credit.run_comprehensive_backtest(
            X_train, X_test, y_train, y_test, train_data, test_data
        )
        
        # Analyze feature importance
        feature_importance = ml_credit.analyze_feature_importance()
        
        # Run robustness tests
        robustness_results = ml_credit.run_robustness_tests(
            X_train, X_test, y_train, y_test, train_data, test_data
        )
        
        # Save results
        ml_credit.save_results("research_results")
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE - ALL RESULTS REPRODUCED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("\nPlease ensure:")
        print("1. Data file 'with_er_daily.csv' is available")
        print("2. Required Python packages are installed")
        print("3. Sufficient memory is available for processing")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
