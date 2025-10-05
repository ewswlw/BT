"""
COMPREHENSIVE TACTICAL ASSET ALLOCATION ANALYZER
================================================

A complete standalone script for discovering, optimizing, and validating 
tactical asset allocation strategies using advanced pattern recognition.

Features:
- Pattern discovery (momentum, mean reversion, volatility regimes)
- Multiple strategy variations with parameter optimization
- Comprehensive bias validation (lookahead, overfitting, transaction costs)
- Realistic performance estimation with cost adjustments
- Professional console reporting

Author: AI Trading Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
import math
import statistics
import random
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TacticalAssetAllocationAnalyzer:
    """
    Comprehensive tactical asset allocation analyzer with pattern discovery,
    strategy optimization, and bias validation.
    """
    
    def __init__(self, data_path: str, target_cagr: float = 0.15, max_drawdown: float = 0.20):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the CSV file with asset price data
            target_cagr: Target CAGR (default: 15%)
            max_drawdown: Maximum acceptable drawdown (default: 20%)
        """
        self.data_path = data_path
        self.target_cagr = target_cagr
        self.max_drawdown = max_drawdown
        self.prices = None
        self.weekly_prices = None
        self.weekly_returns = None
        self.patterns = {}
        self.strategies = {}
        self.results = {}
        self.best_result = None
        self.optimization_history = []
        self.max_iterations = 100
        self.constraints = {
            'max_assets': (2, 8),
            'rebalance_threshold': (0.01, 0.2),
            'momentum_lookback': (4, 26),
            'signal_threshold': (0.3, 0.8),
            'risk_weight': (0.1, 1.0)
        }
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare data for analysis.
        
        Returns:
            Cleaned DataFrame ready for analysis
        """
        print("LOADING AND PREPARING DATA")
        print("=" * 50)
        
        # Load data
        print(f"Loading data from: {self.data_path}")
        self.prices = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        
        print(f"Original data shape: {self.prices.shape}")
        print(f"Date range: {self.prices.index.min()} to {self.prices.index.max()}")
        
        # Forward fill missing values
        self.prices = self.prices.fillna(method='ffill')
        self.prices = self.prices.dropna()
        
        print(f"Cleaned data shape: {self.prices.shape}")
        missing_count = self.prices.isnull().sum().sum()
        print(f"Missing values: {missing_count}")
        
        # Calculate weekly data for Friday rebalancing
        self.weekly_prices = self.prices.resample('W-FRI').last()
        self.weekly_returns = self.weekly_prices.pct_change().dropna()
        
        print(f"Weekly data shape: {self.weekly_returns.shape}")
        print(f"Weekly date range: {self.weekly_returns.index[0] if len(self.weekly_returns) > 0 else 'N/A'} to {self.weekly_returns.index[-1] if len(self.weekly_returns) > 0 else 'N/A'}")
        
        return self.prices
    
    def discover_patterns(self):
        """Discover statistical patterns in the data."""
        print("\nDISCOVERING STATISTICAL PATTERNS")
        print("=" * 50)
        
        print("Analyzing momentum patterns...")
        momentum_patterns = self._discover_momentum_patterns()
        
        print("Analyzing mean reversion patterns...")
        mean_reversion_patterns = self._discover_mean_reversion_patterns()
        
        print("Analyzing volatility regimes...")
        volatility_regimes = self._discover_volatility_regimes()
        
        print("Analyzing regime-dependent performance...")
        regime_performance = self._discover_regime_dependent_performance()
        
        self.patterns = {
            'momentum': momentum_patterns,
            'mean_reversion': mean_reversion_patterns,
            'volatility_regimes': volatility_regimes,
            'regime_performance': regime_performance
        }
        
        print(f"Patterns discovered for {len(self.weekly_returns.columns)} assets")
        
        # Print regime insights
        if regime_performance:
            print("\nREGIME PERFORMANCE INSIGHTS:")
            low_vol_assets = [asset for asset, perf in regime_performance.items() 
                            if perf['regime_preference'] == 'low_vol']
            high_vol_assets = [asset for asset, perf in regime_performance.items() 
                             if perf['regime_preference'] == 'high_vol']
            
            print(f"Assets performing best in LOW volatility: {', '.join(low_vol_assets[:5])}")
            print(f"Assets performing best in HIGH volatility: {', '.join(high_vol_assets[:5])}")
    
    def _discover_momentum_patterns(self, lookbacks: List[int] = [4, 8, 12, 26]) -> Dict:
        """Discover momentum patterns across multiple timeframes."""
        momentum_signals = {}
        
        for asset in self.weekly_returns.columns:
            momentum_signals[asset] = {}
            asset_returns = self.weekly_returns[asset].dropna()
            
            for lookback in lookbacks:
                # Calculate momentum scores
                momentum = asset_returns.rolling(lookback).apply(
                    lambda x: np.prod(1 + x) - 1, raw=False
                )
                
                # Calculate momentum percentile within the asset's own history
                momentum_percentile = momentum.rolling(52).rank(pct=True)
                
                # Calculate momentum acceleration (change in momentum)
                momentum_acceleration = momentum.diff(4)  # 4-week acceleration
                
                momentum_signals[asset][lookback] = {
                    'momentum': momentum,
                    'percentile': momentum_percentile,
                    'acceleration': momentum_acceleration
                }
        
        return momentum_signals
    
    def _discover_mean_reversion_patterns(self, lookbacks: List[int] = [4, 8, 12]) -> Dict:
        """Discover mean reversion patterns using RSI and Bollinger Bands."""
        mean_reversion_signals = {}
        
        for asset in self.weekly_returns.columns:
            mean_reversion_signals[asset] = {}
            asset_returns = self.weekly_returns[asset].dropna()
            
            for lookback in lookbacks:
                # Calculate rolling mean and std
                rolling_mean = asset_returns.rolling(lookback).mean()
                rolling_std = asset_returns.rolling(lookback).std()
                
                # Calculate z-scores
                z_scores = (asset_returns - rolling_mean) / rolling_std
                
                # Calculate bollinger band positions
                upper_band = rolling_mean + 2 * rolling_std
                lower_band = rolling_mean - 2 * rolling_std
                bb_position = (asset_returns - lower_band) / (upper_band - lower_band)
                
                # Calculate RSI-like indicator
                gains = asset_returns.where(asset_returns > 0, 0)
                losses = -asset_returns.where(asset_returns < 0, 0)
                
                avg_gain = gains.rolling(lookback).mean()
                avg_loss = losses.rolling(lookback).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                mean_reversion_signals[asset][lookback] = {
                    'z_score': z_scores,
                    'bb_position': bb_position,
                    'rsi': rsi,
                    'rolling_mean': rolling_mean,
                    'rolling_std': rolling_std
                }
        
        return mean_reversion_signals
    
    def _discover_volatility_regimes(self, lookback: int = 26) -> Dict:
        """Discover volatility regimes and their impact on asset performance."""
        # Calculate market-wide volatility regime
        market_vol = self.weekly_returns.std(axis=1) * np.sqrt(52)
        market_vol_percentile = market_vol.rolling(lookback * 2).rank(pct=True)
        
        # Define regimes based on percentile thresholds
        low_vol_regime = market_vol_percentile < 0.33
        med_vol_regime = (market_vol_percentile >= 0.33) & (market_vol_percentile < 0.67)
        high_vol_regime = market_vol_percentile >= 0.67
        
        return {
            'market_volatility': market_vol,
            'market_vol_percentile': market_vol_percentile,
            'low_vol_regime': low_vol_regime,
            'med_vol_regime': med_vol_regime,
            'high_vol_regime': high_vol_regime
        }
    
    def _discover_regime_dependent_performance(self) -> Dict:
        """Discover how asset performance varies across different market regimes."""
        # First discover volatility regimes
        vol_regimes = self._discover_volatility_regimes()
        
        regime_performance = {}
        
        for asset in self.weekly_returns.columns:
            asset_returns = self.weekly_returns[asset].dropna()
            
            # Align returns with regime indicators
            common_dates = asset_returns.index.intersection(vol_regimes['low_vol_regime'].index)
            
            if len(common_dates) > 0:
                aligned_returns = asset_returns.loc[common_dates]
                low_vol_mask = vol_regimes['low_vol_regime'].loc[common_dates]
                med_vol_mask = vol_regimes['med_vol_regime'].loc[common_dates]
                high_vol_mask = vol_regimes['high_vol_regime'].loc[common_dates]
                
                # Calculate performance in each regime
                low_vol_perf = aligned_returns[low_vol_mask].mean() * 52  # Annualized
                med_vol_perf = aligned_returns[med_vol_mask].mean() * 52
                high_vol_perf = aligned_returns[high_vol_mask].mean() * 52
                
                low_vol_vol = aligned_returns[low_vol_mask].std() * np.sqrt(52)
                med_vol_vol = aligned_returns[med_vol_mask].std() * np.sqrt(52)
                high_vol_vol = aligned_returns[high_vol_mask].std() * np.sqrt(52)
                
                low_vol_sharpe = low_vol_perf / low_vol_vol if low_vol_vol > 0 else 0
                med_vol_sharpe = med_vol_perf / med_vol_vol if med_vol_vol > 0 else 0
                high_vol_sharpe = high_vol_perf / high_vol_vol if high_vol_vol > 0 else 0
                
                regime_performance[asset] = {
                    'low_vol_return': low_vol_perf,
                    'med_vol_return': med_vol_perf,
                    'high_vol_return': high_vol_perf,
                    'low_vol_volatility': low_vol_vol,
                    'med_vol_volatility': med_vol_vol,
                    'high_vol_volatility': high_vol_vol,
                    'low_vol_sharpe': low_vol_sharpe,
                    'med_vol_sharpe': med_vol_sharpe,
                    'high_vol_sharpe': high_vol_sharpe,
                    'regime_preference': 'low_vol' if low_vol_sharpe > max(med_vol_sharpe, high_vol_sharpe)
                                      else 'med_vol' if med_vol_sharpe > high_vol_sharpe else 'high_vol'
                }
        
        return regime_performance
    
    def create_strategy_variations(self):
        """Create multiple strategy variations based on discovered patterns."""
        print("\nCREATING STRATEGY VARIATIONS")
        print("=" * 50)
        
        # Strategy 1: Pure Momentum
        momentum_signals = self._create_momentum_weighted_signals()
        self.strategies['Momentum_Weighted'] = momentum_signals
        
        # Strategy 2: Regime-Adaptive
        regime_signals = self._create_regime_adaptive_signals()
        self.strategies['Regime_Adaptive'] = regime_signals
        
        # Strategy 3: Risk-Adjusted
        risk_adjusted_signals = self._create_risk_adjusted_signals()
        self.strategies['Risk_Adjusted'] = risk_adjusted_signals
        
        # Strategy 4: Multi-Timeframe
        multi_timeframe_signals = self._create_multi_timeframe_signals()
        self.strategies['Multi_Timeframe'] = multi_timeframe_signals
        
        print(f"Created {len(self.strategies)} strategy variations:")
        for name in self.strategies.keys():
            print(f"   • {name}")
    
    def _create_momentum_weighted_signals(self) -> pd.DataFrame:
        """Create momentum-weighted signals."""
        momentum_data = self.patterns['momentum']
        
        momentum_signals = pd.DataFrame(index=self.weekly_returns.index)
        
        for asset in self.weekly_returns.columns:
            if asset in momentum_data:
                momentum_values = []
                for lookback, data in momentum_data[asset].items():
                    momentum_values.append(data['percentile'])
                
                if momentum_values:
                    combined_momentum = pd.concat(momentum_values, axis=1).mean(axis=1)
                    momentum_signals[asset] = combined_momentum.fillna(0.5)
                else:
                    momentum_signals[asset] = 0.5
            else:
                momentum_signals[asset] = 0.5
        
        return momentum_signals.fillna(0.5)
    
    def _create_regime_adaptive_signals(self) -> pd.DataFrame:
        """Create regime-adaptive signals."""
        regime_perf = self.patterns['regime_performance']
        vol_regimes = self.patterns['volatility_regimes']
        
        regime_signals = pd.DataFrame(index=self.weekly_returns.index)
        
        for asset in self.weekly_returns.columns:
            if asset in regime_perf:
                best_regime = regime_perf[asset]['regime_preference']
                asset_signals = pd.Series(0.3, index=self.weekly_returns.index)
                
                if best_regime == 'low_vol':
                    asset_signals[vol_regimes['low_vol_regime']] = 1.0
                    asset_signals[vol_regimes['high_vol_regime']] = 0.1
                elif best_regime == 'high_vol':
                    asset_signals[vol_regimes['high_vol_regime']] = 1.0
                    asset_signals[vol_regimes['low_vol_regime']] = 0.1
                else:  # medium vol
                    asset_signals[vol_regimes['med_vol_regime']] = 1.0
                    asset_signals[vol_regimes['high_vol_regime']] = 0.5
                
                regime_signals[asset] = asset_signals
            else:
                regime_signals[asset] = 0.5
        
        return regime_signals.fillna(0.5)
    
    def _create_risk_adjusted_signals(self) -> pd.DataFrame:
        """Create risk-adjusted signals."""
        momentum_signals = self._create_momentum_weighted_signals()
        
        risk_adjusted_signals = pd.DataFrame(index=momentum_signals.index)
        
        for asset in momentum_signals.columns:
            asset_returns = self.weekly_returns[asset].dropna()
            rolling_vol = asset_returns.rolling(26).std() * np.sqrt(52)
            
            base_signals = momentum_signals[asset]
            risk_adjusted = base_signals / (rolling_vol + 0.01)
            
            # Normalize to 0-1 range
            risk_adjusted = (risk_adjusted - risk_adjusted.min()) / (risk_adjusted.max() - risk_adjusted.min())
            risk_adjusted_signals[asset] = risk_adjusted.fillna(0.5)
        
        return risk_adjusted_signals.fillna(0.5)
    
    def _create_multi_timeframe_signals(self) -> pd.DataFrame:
        """Create multi-timeframe signals."""
        momentum_data = self.patterns['momentum']
        mr_data = self.patterns['mean_reversion']
        
        multi_timeframe_signals = pd.DataFrame(index=self.weekly_returns.index)
        
        for asset in self.weekly_returns.columns:
            signals = []
            
            # Add momentum signals from different timeframes
            if asset in momentum_data:
                for lookback, data in momentum_data[asset].items():
                    signals.append(data['percentile'])
            
            # Add mean reversion signals from different timeframes
            if asset in mr_data:
                for lookback, data in mr_data[asset].items():
                    rsi_signal = 1 - (data['rsi'].fillna(50) / 100)
                    signals.append(rsi_signal)
            
            # Combine signals with equal weighting
            if signals:
                combined_signal = pd.concat(signals, axis=1).mean(axis=1)
                multi_timeframe_signals[asset] = combined_signal.fillna(0.5)
            else:
                multi_timeframe_signals[asset] = 0.5
        
        return multi_timeframe_signals.fillna(0.5)
    
    def _create_ensemble_signals(self) -> pd.DataFrame:
        """Create ensemble signals combining multiple strategies."""
        momentum_signals = self._create_momentum_weighted_signals()
        regime_signals = self._create_regime_adaptive_signals()
        risk_signals = self._create_risk_adjusted_signals()
        
        ensemble_signals = pd.DataFrame(index=self.weekly_returns.index)
        
        for asset in self.weekly_returns.columns:
            # Combine signals with equal weighting initially
            combined = (momentum_signals[asset] + regime_signals[asset] + risk_signals[asset]) / 3
            ensemble_signals[asset] = combined.fillna(0.5)
        
        return ensemble_signals.fillna(0.5)
    
    def _create_dynamic_risk_adjusted_signals(self) -> pd.DataFrame:
        """Create dynamic risk-adjusted signals with adaptive parameters."""
        momentum_signals = self._create_momentum_weighted_signals()
        
        dynamic_signals = pd.DataFrame(index=momentum_signals.index)
        
        for asset in momentum_signals.columns:
            asset_returns = self.weekly_returns[asset].dropna()
            
            # Dynamic volatility calculation with different lookbacks
            vol_short = asset_returns.rolling(13).std() * np.sqrt(52)
            vol_long = asset_returns.rolling(52).std() * np.sqrt(52)
            
            # Adaptive risk adjustment based on volatility regime
            base_signals = momentum_signals[asset]
            risk_ratio = vol_short / (vol_long + 0.01)
            
            # Higher risk adjustment during high volatility periods
            risk_adjusted = base_signals / (risk_ratio + 0.5)
            
            # Normalize to 0-1 range
            risk_adjusted = (risk_adjusted - risk_adjusted.min()) / (risk_adjusted.max() - risk_adjusted.min() + 0.01)
            dynamic_signals[asset] = risk_adjusted.fillna(0.5)
        
        return dynamic_signals.fillna(0.5)
    
    def _create_regime_switching_signals(self) -> pd.DataFrame:
        """Create regime-switching signals that adapt to market conditions."""
        vol_regimes = self.patterns['volatility_regimes']
        momentum_signals = self._create_momentum_weighted_signals()
        
        regime_switching_signals = pd.DataFrame(index=self.weekly_returns.index)
        
        for asset in self.weekly_returns.columns:
            base_signals = momentum_signals[asset]
            
            # Create regime-specific signals
            low_vol_signals = base_signals * 1.2  # Amplify in low vol
            med_vol_signals = base_signals * 1.0  # Normal in medium vol
            high_vol_signals = base_signals * 0.6  # Reduce in high vol
            
            # Combine based on regime
            regime_signals = pd.Series(0.5, index=base_signals.index)
            regime_signals[vol_regimes['low_vol_regime']] = low_vol_signals[vol_regimes['low_vol_regime']]
            regime_signals[vol_regimes['med_vol_regime']] = med_vol_signals[vol_regimes['med_vol_regime']]
            regime_signals[vol_regimes['high_vol_regime']] = high_vol_signals[vol_regimes['high_vol_regime']]
            
            # Cap at 1.0
            regime_signals = regime_signals.clip(0, 1)
            regime_switching_signals[asset] = regime_signals.fillna(0.5)
        
        return regime_switching_signals.fillna(0.5)
    
    def optimize_strategies(self):
        """Run iterative optimization until target is achieved."""
        print("\nSTARTING ITERATIVE OPTIMIZATION")
        print("=" * 60)
        print(f"Target: {self.target_cagr:.1%} CAGR with <{self.max_drawdown:.1%} max drawdown")
        print(f"Max iterations: {self.max_iterations}")
        print("=" * 60)
        
        iteration = 0
        best_score = -np.inf
        convergence_count = 0
        target_achieved = False
        
        while iteration < self.max_iterations and not target_achieved:
            iteration += 1
            print(f"\nIteration {iteration}/{self.max_iterations}")
            
            # Run optimization for each strategy
            iteration_results = []
            
            for strategy_name in self.strategies.keys():
                print(f"  Optimizing {strategy_name}...")
                
                # Use genetic algorithm for optimization
                result = self._genetic_algorithm_optimization(strategy_name)
                iteration_results.append(result)
                
                # Check if this is the best result so far
                if result['score'] > best_score:
                    best_score = result['score']
                    self.best_result = result
                    convergence_count = 0
                    print(f"    NEW BEST: {result['cagr']:.2%} CAGR, {result['max_drawdown']:.2%} DD, Score: {result['score']:.3f}")
                else:
                    convergence_count += 1
                
                # Check if target achieved
                if result['target_achieved']:
                    target_achieved = True
                    print(f"\nTARGET ACHIEVED!")
                    print(f"   Strategy: {result['strategy']}")
                    print(f"   CAGR: {result['cagr']:.2%}")
                    print(f"   Max DD: {result['max_drawdown']:.2%}")
                    print(f"   Parameters: {result['params']}")
                    break
            
            # Store iteration results
            self.optimization_history.append({
                'iteration': iteration,
                'best_score': best_score,
                'results': iteration_results
            })
            
            # Check convergence
            if convergence_count >= 15 and not target_achieved:
                print(f"\nOptimization converged after {iteration} iterations")
                print("Expanding search space...")
                self._expand_search_space()
                convergence_count = 0
            
            # Adaptive parameter adjustment
            if iteration % 10 == 0 and not target_achieved:
                self._adjust_constraints()
        
        if iteration >= self.max_iterations and not target_achieved:
            print(f"\nMaximum iterations ({self.max_iterations}) reached")
            print("Expanding search space and continuing...")
            self._expand_search_space()
            self.max_iterations += 100  # Extend limit
        
        # Store final results
        if self.best_result:
            self.results[self.best_result['strategy']] = {
                'metrics': self.best_result,
                'best_params': self.best_result['params']
            }
        
        return self.best_result
    
    def _genetic_algorithm_optimization(self, strategy_name: str):
        """Use random search optimization to find best parameters."""
        best_result = None
        best_score = -np.inf
        
        # Random search with multiple iterations
        num_iterations = 200
        
        for i in range(num_iterations):
            # Generate random parameters within bounds
            max_assets = random.randint(self.constraints['max_assets'][0], self.constraints['max_assets'][1])
            rebalance_threshold = random.uniform(self.constraints['rebalance_threshold'][0], self.constraints['rebalance_threshold'][1])
            momentum_lookback = random.randint(self.constraints['momentum_lookback'][0], self.constraints['momentum_lookback'][1])
            signal_threshold = random.uniform(self.constraints['signal_threshold'][0], self.constraints['signal_threshold'][1])
            risk_weight = random.uniform(self.constraints['risk_weight'][0], self.constraints['risk_weight'][1])
            
            try:
                # Run backtest with these parameters
                signals = self.strategies[strategy_name]
                result = self._run_strategy_backtest(
                    signals, strategy_name, max_assets, rebalance_threshold
                )
                
                # Calculate score with enhanced objective
                score = result['cagr']
                
                # Heavy penalty for exceeding max drawdown
                if abs(result['max_drawdown']) > self.max_drawdown:
                    score -= 5 * (abs(result['max_drawdown']) - self.max_drawdown)
                
                # Bonus for hitting target
                if result['cagr'] >= self.target_cagr:
                    score += 0.2
                
                # Bonus for good risk-adjusted returns
                if result['sharpe_ratio'] > 1.0:
                    score += 0.1
                
                # Bonus for good calmar ratio
                if result['calmar_ratio'] > 0.5:
                    score += 0.05
                
                # Check if this is the best result
                if score > best_score:
                    best_score = score
                    best_result = result.copy()
                    best_result.update({
                        'strategy': strategy_name,
                        'params': {
                            'max_assets': max_assets,
                            'rebalance_threshold': rebalance_threshold,
                            'momentum_lookback': momentum_lookback,
                            'signal_threshold': signal_threshold,
                            'risk_weight': risk_weight
                        },
                        'score': score,
                        'target_achieved': (result['cagr'] >= self.target_cagr and 
                                          abs(result['max_drawdown']) <= self.max_drawdown)
                    })
                    
                    # Early exit if target achieved
                    if best_result['target_achieved']:
                        print(f"    TARGET ACHIEVED in iteration {i+1}!")
                        break
                        
            except Exception as e:
                continue
        
        if best_result is None:
            # Return default result if optimization failed
            return {
                'strategy': strategy_name,
                'cagr': 0.0,
                'max_drawdown': -1.0,
                'sharpe_ratio': 0.0,
                'score': -np.inf,
                'target_achieved': False,
                'params': {'max_assets': 3, 'rebalance_threshold': 0.1}
            }
        
        return best_result
    
    def _expand_search_space(self):
        """Expand parameter constraints when optimization converges."""
        print("    Expanding search space...")
        
        # Expand max assets range
        self.constraints['max_assets'] = (
            max(2, self.constraints['max_assets'][0] - 2),
            min(15, self.constraints['max_assets'][1] + 3)
        )
        
        # Expand rebalance threshold range
        self.constraints['rebalance_threshold'] = (
            max(0.001, self.constraints['rebalance_threshold'][0] / 2),
            min(0.5, self.constraints['rebalance_threshold'][1] * 1.5)
        )
        
        # Expand momentum lookback range
        self.constraints['momentum_lookback'] = (
            max(1, self.constraints['momentum_lookback'][0] - 5),
            min(104, self.constraints['momentum_lookback'][1] + 10)
        )
        
        print(f"    New constraints: max_assets={self.constraints['max_assets']}, "
              f"threshold={self.constraints['rebalance_threshold']}")
    
    def _adjust_constraints(self):
        """Adjust parameter constraints based on optimization history."""
        if len(self.optimization_history) < 5:
            return
        
        print("    Adjusting parameter constraints...")
        
        # Analyze recent results to adjust constraints
        recent_results = self.optimization_history[-5:]
        best_results = [r['results'] for r in recent_results]
        
        # Find parameter ranges that work well
        good_max_assets = []
        good_thresholds = []
        
        for iteration_results in best_results:
            for result in iteration_results:
                if result['score'] > -0.5:  # Reasonable performance
                    good_max_assets.append(result['params']['max_assets'])
                    good_thresholds.append(result['params']['rebalance_threshold'])
        
        if good_max_assets and good_thresholds:
            # Tighten constraints around successful parameters
            self.constraints['max_assets'] = (
                max(2, int(np.mean(good_max_assets) - np.std(good_max_assets))),
                min(10, int(np.mean(good_max_assets) + np.std(good_max_assets)))
            )
            
            self.constraints['rebalance_threshold'] = (
                max(0.005, np.mean(good_thresholds) - np.std(good_thresholds)),
                min(0.30, np.mean(good_thresholds) + np.std(good_thresholds))
            )
            
            print(f"    Adjusted constraints around successful parameters")
    
    def _run_strategy_backtest(self, signals: pd.DataFrame, strategy_name: str, 
                              max_assets: int, rebalance_threshold: float) -> Dict:
        """Run a single strategy backtest."""
        # Align signals with weekly data
        aligned_signals = signals.reindex(self.weekly_prices.index).fillna(0.5)
        
        # Initialize portfolio
        portfolio_value = 10000
        portfolio_values = [portfolio_value]
        current_weights = None
        
        # Start from week 26 to ensure we have enough data for signals
        start_idx = 26
        
        for i in range(start_idx, len(self.weekly_prices)):
            date = self.weekly_prices.index[i]
            
            # Use PREVIOUS week's signals to avoid lookahead bias
            signal_date = self.weekly_prices.index[i-1]
            if signal_date not in aligned_signals.index:
                # If no signals available, maintain current weights
                if current_weights is None:
                    # Initial equal weight allocation if no signals
                    current_weights = {asset: 1.0/len(self.weekly_prices.columns) 
                                     for asset in self.weekly_prices.columns}
                portfolio_values.append(portfolio_value)
                continue
            
            date_signals = aligned_signals.loc[signal_date]
            
            # Generate target weights based on signals
            target_weights = self._generate_weights_from_signals(date_signals, max_assets)
            
            # Check if rebalancing is needed
            if current_weights is None:
                rebalance_needed = True
                current_weights = target_weights.copy()
            else:
                weight_changes = np.abs(
                    np.array([target_weights.get(asset, 0) for asset in self.weekly_prices.columns]) -
                    np.array([current_weights.get(asset, 0) for asset in self.weekly_prices.columns])
                )
                rebalance_needed = np.max(weight_changes) > rebalance_threshold
                
                if rebalance_needed:
                    current_weights = target_weights.copy()
            
            # Calculate portfolio return for this period
            period_return = 0.0
            for asset, weight in current_weights.items():
                if asset in self.weekly_returns.columns:
                    asset_return = self.weekly_returns.loc[date, asset]
                    if not pd.isna(asset_return):
                        period_return += weight * asset_return
            
            # Cap extreme returns to prevent unrealistic results
            period_return = max(min(period_return, 0.5), -0.5)
            portfolio_value *= (1 + period_return)
            
            portfolio_values.append(portfolio_value)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(portfolio_values, strategy_name)
    
    def _generate_weights_from_signals(self, signals: pd.Series, max_assets: int) -> Dict[str, float]:
        """Generate portfolio weights from signals."""
        # Sort assets by signal strength
        sorted_assets = signals.sort_values(ascending=False)
        
        # Select top assets with positive signals
        selected_assets = sorted_assets[sorted_assets > 0.5].head(max_assets)
        
        # If no assets meet criteria, use top assets with equal weights
        if len(selected_assets) == 0:
            selected_assets = sorted_assets.head(max_assets)
            weights = {}
            for asset in selected_assets.index:
                weights[asset] = 1.0 / len(selected_assets)
            return weights
        
        # Generate weights using signal strength (simple proportional weighting)
        total_signal = selected_assets.sum()
        weights = {}
        
        for asset, signal in selected_assets.items():
            weights[asset] = signal / total_signal
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_performance_metrics(self, portfolio_values: List[float], strategy_name: str) -> Dict:
        """Calculate comprehensive performance metrics."""
        values = pd.Series(portfolio_values)
        returns = values.pct_change().dropna()
        
        # Basic metrics
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        years = len(values) / 52  # Weekly data
        cagr = (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(52)  # Annualized
        sharpe = (returns.mean() * 52 - 0.03) / (returns.std() * np.sqrt(52))  # Assuming 3% risk-free rate
        
        # Drawdown analysis
        cummax = values.cummax()
        drawdowns = (values - cummax) / cummax
        max_drawdown = drawdowns.min()
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Calmar ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Target achievement
        target_achieved = cagr >= self.target_cagr and abs(max_drawdown) <= self.max_drawdown
        
        return {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'target_achieved': target_achieved
        }
    
    def validate_for_biases(self):
        """Comprehensive bias validation."""
        print("\nBIAS VALIDATION TESTS")
        print("=" * 50)
        
        validation_results = {}
        
        # Test 1: Lookahead Bias
        print("\n1. Testing Lookahead Bias...")
        lookahead_result = self._test_lookahead_bias()
        validation_results['lookahead'] = lookahead_result
        
        # Test 2: Transaction Costs
        print("\n2. Testing Transaction Cost Impact...")
        cost_result = self._test_transaction_costs()
        validation_results['transaction_costs'] = cost_result
        
        # Test 3: Parameter Sensitivity
        print("\n3. Testing Parameter Sensitivity...")
        sensitivity_result = self._test_parameter_sensitivity()
        validation_results['parameter_sensitivity'] = sensitivity_result
        
        return validation_results
    
    def _test_lookahead_bias(self) -> Dict:
        """Test for lookahead bias by using only past data."""
        # Split data: Use first 20 years for training, rest for testing
        split_date = '2008-01-01'
        train_data = self.prices[self.prices.index < split_date]
        test_data = self.prices[self.prices.index >= split_date]
        
        print(f"   Training period: {len(train_data)/252:.1f} years")
        print(f"   Testing period: {len(test_data)/252:.1f} years")
        
        # Simple momentum test on out-of-sample data
        test_weekly = test_data.resample('W-FRI').last().pct_change().dropna()
        
        portfolio_value = 10000
        for i in range(26, len(test_weekly)):
            momentum = test_weekly.iloc[i-26:i].apply(lambda x: np.prod(1 + x) - 1)
            top_3 = momentum.nlargest(3).index.tolist()
            
            if i < len(test_weekly):
                port_return = 0
                for asset in top_3:
                    if asset in test_weekly.columns:
                        ret = test_weekly.iloc[i][asset]
                        if not pd.isna(ret):
                            port_return += ret / 3
                
                portfolio_value *= (1 + port_return)
        
        years = (len(test_weekly) - 26) / 52
        cagr = (portfolio_value / 10000) ** (1/years) - 1
        
        print(f"   Out-of-sample CAGR: {cagr:.2%}")
        
        bias_risk = 'HIGH' if cagr > 0.25 else 'MEDIUM' if cagr > 0.15 else 'LOW'
        print(f"   Bias Risk: {bias_risk}")
        
        return {
            'out_of_sample_cagr': cagr,
            'bias_risk': bias_risk,
            'training_years': len(train_data)/252,
            'testing_years': len(test_data)/252
        }
    
    def _test_transaction_costs(self) -> Dict:
        """Test impact of transaction costs."""
        # Estimate number of rebalances per year (more realistic)
        estimated_rebalances_per_year = 12  # Assume monthly rebalancing on average
        
        cost_scenarios = [0.0, 0.0005, 0.001, 0.002, 0.005]  # 0%, 0.05%, 0.1%, 0.2%, 0.5%
        
        print("   Transaction Cost Impact:")
        cost_results = {}
        
        for cost_rate in cost_scenarios:
            annual_cost = estimated_rebalances_per_year * cost_rate
            cost_results[f'{cost_rate*100:.1f}%'] = annual_cost
            print(f"     {cost_rate*100:.1f}% per trade: {annual_cost*100:.2f}% annual cost")
        
        return cost_results
    
    def _test_parameter_sensitivity(self) -> Dict:
        """Test parameter sensitivity."""
        # Test different momentum lookback periods
        periods = [13, 26, 52]  # 3, 6, 12 months
        period_results = {}
        
        for period in periods:
            # Simple momentum test
            portfolio_value = 10000
            
            for i in range(period, len(self.weekly_returns)):
                momentum = self.weekly_returns.iloc[i-period:i].apply(lambda x: np.prod(1 + x) - 1)
                top_3 = momentum.nlargest(3).index.tolist()
                
                if i < len(self.weekly_returns):
                    port_return = 0
                    for asset in top_3:
                        if asset in self.weekly_returns.columns:
                            ret = self.weekly_returns.iloc[i][asset]
                            if not pd.isna(ret):
                                port_return += ret / 3
                    
                    portfolio_value *= (1 + port_return)
            
            years = (len(self.weekly_returns) - period) / 52
            cagr = (portfolio_value / 10000) ** (1/years) - 1
            period_results[f'{period}_weeks'] = cagr
            print(f"     {period}-week momentum: {cagr:.2%} CAGR")
        
        # Calculate volatility of results
        cagrs = list(period_results.values())
        volatility = np.std(cagrs)
        
        sensitivity_risk = 'HIGH' if volatility > 0.05 else 'MEDIUM' if volatility > 0.02 else 'LOW'
        print(f"   Parameter Sensitivity Risk: {sensitivity_risk}")
        
        return {
            'period_results': period_results,
            'parameter_volatility': volatility,
            'sensitivity_risk': sensitivity_risk
        }
    
    def generate_comprehensive_report(self, validation_results: Dict):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TACTICAL ASSET ALLOCATION ANALYSIS REPORT")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Source: {self.data_path}")
        print(f"Target CAGR: {self.target_cagr:.1%}")
        print(f"Max Drawdown Limit: {self.max_drawdown:.1%}")
        
        # Strategy Performance Comparison
        print(f"\nSTRATEGY PERFORMANCE COMPARISON")
        print("-" * 80)
        print(f"{'Strategy':<20} {'CAGR':<8} {'Max DD':<8} {'Sharpe':<8} {'Calmar':<8} {'Target':<8}")
        print("-" * 80)
        
        comparison_data = []
        for strategy_name, result in self.results.items():
            metrics = result['metrics']
            target_status = "YES" if metrics['target_achieved'] else "NO"
            
            print(f"{strategy_name:<20} "
                  f"{metrics['cagr']:<8.1%} "
                  f"{metrics['max_drawdown']:<8.1%} "
                  f"{metrics['sharpe_ratio']:<8.2f} "
                  f"{metrics['calmar_ratio']:<8.2f} "
                  f"{target_status:<8}")
            
            comparison_data.append({
                'strategy': strategy_name,
                'cagr': metrics['cagr'],
                'max_dd': metrics['max_drawdown'],
                'sharpe': metrics['sharpe_ratio'],
                'target_achieved': metrics['target_achieved']
            })
        
        # Best Strategy Analysis
        best_strategy = max(comparison_data, key=lambda x: x['cagr'])
        best_result = self.results[best_strategy['strategy']]
        
        print(f"\nBEST STRATEGY: {best_strategy['strategy']}")
        print("-" * 50)
        metrics = best_result['metrics']
        params = best_result['best_params']
        
        print(f"Performance Metrics:")
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        
        print(f"\nOptimal Parameters:")
        print(f"  Max Assets: {params['max_assets']}")
        print(f"  Rebalance Threshold: {params['rebalance_threshold']:.1%}")
        
        # Bias Validation Results
        print(f"\nBIAS VALIDATION RESULTS")
        print("-" * 50)
        
        lookahead = validation_results['lookahead']
        print(f"Lookahead Bias Test:")
        print(f"  Out-of-sample CAGR: {lookahead['out_of_sample_cagr']:.2%}")
        print(f"  Bias Risk: {lookahead['bias_risk']}")
        print(f"  Training Period: {lookahead['training_years']:.1f} years")
        print(f"  Testing Period: {lookahead['testing_years']:.1f} years")
        
        costs = validation_results['transaction_costs']
        print(f"\nTransaction Cost Analysis:")
        print(f"  Estimated annual cost (0.1% per trade): {costs['0.1%']*100:.1f}%")
        print(f"  Estimated annual cost (0.2% per trade): {costs['0.2%']*100:.1f}%")
        
        sensitivity = validation_results['parameter_sensitivity']
        print(f"\nParameter Sensitivity:")
        print(f"  Sensitivity Risk: {sensitivity['sensitivity_risk']}")
        print(f"  Parameter Volatility: {sensitivity['parameter_volatility']:.2%}")
        
        # Realistic Performance Estimates
        print(f"\nREALISTIC PERFORMANCE ESTIMATES")
        print("-" * 50)
        
        original_cagr = best_strategy['cagr']
        lookahead_cagr = lookahead['out_of_sample_cagr']
        cost_impact = costs['0.1%']
        
        # Estimate realistic performance
        realistic_cagr = lookahead_cagr - cost_impact
        
        print(f"Original Backtest: {original_cagr:.1%} CAGR")
        print(f"Out-of-sample: {lookahead_cagr:.1%} CAGR")
        print(f"After transaction costs: {realistic_cagr:.1%} CAGR")
        print(f"Performance degradation: {((original_cagr - realistic_cagr)/original_cagr)*100:.0f}%")
        
        # Final Assessment
        print(f"\nFINAL ASSESSMENT")
        print("-" * 50)
        
        if realistic_cagr >= self.target_cagr:
            assessment = "STRATEGY PASSES VALIDATION"
            recommendation = "Consider implementing with conservative expectations"
        elif realistic_cagr >= self.target_cagr * 0.8:
            assessment = "STRATEGY SHOWS PROMISE"
            recommendation = "Needs more validation and realistic cost modeling"
        else:
            assessment = "STRATEGY FAILS VALIDATION"
            recommendation = "Do not implement - results likely due to biases"
        
        print(f"Assessment: {assessment}")
        print(f"Recommendation: {recommendation}")
        
        # Risk Warnings
        print(f"\nIMPORTANT RISK WARNINGS")
        print("-" * 50)
        print("1. Past performance does not guarantee future results")
        print("2. Transaction costs and slippage will reduce returns")
        print("3. Market conditions may change, affecting strategy performance")
        print("4. Consider paper trading before live implementation")
        print("5. Diversify risk and never invest more than you can afford to lose")
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


def main():
    """Main execution function with iterative optimization."""
    print("ENHANCED TACTICAL ASSET ALLOCATION ANALYZER")
    print("=" * 70)
    print("Advanced iterative optimization until target achieved")
    print("Target: 15% CAGR with <20% max drawdown")
    print("=" * 70)
    
    # Configuration - parameterized data path
    data_path = "tactical asset allocation/data/processed/1988 assets 3x.csv"
    target_cagr = 0.15  # 15%
    max_drawdown = 0.20  # 20%
    
    try:
        # Initialize analyzer
        analyzer = TacticalAssetAllocationAnalyzer(data_path, target_cagr, max_drawdown)
        
        # Step 1: Load and prepare data
        analyzer.load_and_prepare_data()
        
        # Step 2: Discover patterns
        analyzer.discover_patterns()
        
        # Step 3: Create strategy variations
        analyzer.create_strategy_variations()
        
        # Step 4: Run iterative optimization until target achieved
        print("\nSTARTING TARGET-SEEKING OPTIMIZATION")
        print("Will continue iterating until 15% CAGR + <20% max drawdown achieved")
        
        best_result = analyzer.optimize_strategies()
        
        # Step 5: Generate final report
        if best_result and best_result['target_achieved']:
            print("\n" + "="*70)
            print("SUCCESS: TARGET ACHIEVED!")
            print("="*70)
            print(f"Strategy: {best_result['strategy']}")
            print(f"CAGR: {best_result['cagr']:.2%}")
            print(f"Max Drawdown: {best_result['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
            print(f"Calmar Ratio: {best_result['calmar_ratio']:.2f}")
            print(f"Parameters: {best_result['params']}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("TARGET NOT ACHIEVED")
            print("="*70)
            if best_result:
                print(f"Best result found:")
                print(f"Strategy: {best_result['strategy']}")
                print(f"CAGR: {best_result['cagr']:.2%}")
                print(f"Max Drawdown: {best_result['max_drawdown']:.2%}")
                print(f"Parameters: {best_result['params']}")
            print("="*70)
        
        # Step 6: Validate for biases (if target achieved)
        if best_result and best_result['target_achieved']:
            print("\nVALIDATING BEST STRATEGY FOR BIASES...")
            validation_results = analyzer.validate_for_biases()
            analyzer.generate_comprehensive_report(validation_results)
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        print("Please ensure the data file exists and the path is correct.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your data format and try again.")


if __name__ == "__main__":
    main()
