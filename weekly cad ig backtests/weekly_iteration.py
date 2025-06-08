"""
Weekly CAD IG Backtesting - Iterative Strategy Development using VectorBT
========================================================================

This script implements an iterative backtesting framework for CAD IG (Investment Grade) 
credit strategies using vectorbt for backtesting and quantstats for performance analysis.

Key Features:
- Uses vectorbt for efficient vectorized backtesting
- Uses quantstats for comprehensive performance analysis
- Multiple strategy implementations with iterative optimization
- Automated parameter optimization through grid search
- Detailed logging and error handling

Target: Develop an optimal weekly rebalancing strategy for CAD IG exposure
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import required libraries with error handling
try:
    import vectorbt as vbt
    logger = logging.getLogger(__name__)
    logger.info("VectorBT imported successfully")
except ImportError as e:
    print(f"Error importing vectorbt: {e}")
    print("Installing vectorbt...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vectorbt"])
    import vectorbt as vbt

try:
    import quantstats as qs
    logger = logging.getLogger(__name__)
    logger.info("QuantStats imported successfully")
except ImportError as e:
    print(f"Error importing quantstats: {e}")
    print("Installing quantstats...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "quantstats"])
    import quantstats as qs

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weekly_cad_ig_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyCADIGBacktester:
    """
    VectorBT-based backtesting framework for CAD IG strategies
    
    This class implements multiple strategy variants with iterative optimization
    using vectorbt for efficient backtesting and quantstats for analysis.
    """
    
    def __init__(self, data_path='../data_pipelines/data_processed/with_er_daily.csv'):
        """
        Initialize the backtester with data loading and validation
        
        Args:
            data_path (str): Path to the CSV data file
        """
        logger.info("Initializing VectorBT-based Weekly CAD IG Backtester")
        self.data_path = data_path
        self.data = None
        self.weekly_data = None
        self.cad_ig_returns = None
        self.results = {}
        self.best_strategy = None
        self.best_sharpe = -np.inf
        
        # Load and prepare data
        self._load_data()
        self._prepare_weekly_data()
        self._calculate_features()
        
    def _load_data(self):
        """
        Load and validate the market data with comprehensive error handling
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Convert date column and set as index
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
            
            # Log data overview
            logger.info(f"Loaded data: {len(self.data)} rows, {len(self.data.columns)} columns")
            logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            # Check for missing values in key columns
            key_columns = ['cad_oas', 'cad_ig_er_index', 'us_ig_oas', 'tsx', 'vix']
            missing_data = self.data[key_columns].isnull().sum()
            if missing_data.sum() > 0:
                logger.warning(f"Missing data detected:\n{missing_data}")
                
            # Forward fill missing values for continuity
            self.data.fillna(method='ffill', inplace=True)
            
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _prepare_weekly_data(self):
        """
        Convert daily data to weekly frequency for strategy implementation
        Uses Friday closes or last available day of week for consistency
        """
        try:
            logger.info("Converting daily data to weekly frequency")
            
            # Resample to weekly frequency (Friday closes)
            self.weekly_data = self.data.resample('W-FRI').last()
            
            # Calculate weekly returns for the CAD IG excess return index
            self.cad_ig_returns = self.weekly_data['cad_ig_er_index'].pct_change().dropna()
            
            # Calculate weekly changes in key variables
            self.weekly_data['cad_oas_change'] = self.weekly_data['cad_oas'].pct_change()
            self.weekly_data['us_ig_oas_change'] = self.weekly_data['us_ig_oas'].pct_change()
            self.weekly_data['tsx_weekly_return'] = self.weekly_data['tsx'].pct_change()
            self.weekly_data['vix_change'] = self.weekly_data['vix'].pct_change()
            
            # Remove first row with NaN values from pct_change calculations
            self.weekly_data = self.weekly_data.iloc[1:]
            
            logger.info(f"Weekly data prepared: {len(self.weekly_data)} weeks")
            logger.info(f"Weekly date range: {self.weekly_data.index.min()} to {self.weekly_data.index.max()}")
            
        except Exception as e:
            logger.error(f"Error preparing weekly data: {str(e)}")
            raise
            
    def _calculate_features(self):
        """
        Calculate technical indicators and feature engineering for strategy signals
        Creates moving averages, momentum indicators, volatility measures, and regime features
        """
        try:
            logger.info("Calculating technical features and indicators")
            
            # Moving averages for trend identification
            self.weekly_data['cad_oas_sma_4'] = self.weekly_data['cad_oas'].rolling(4).mean()
            self.weekly_data['cad_oas_sma_12'] = self.weekly_data['cad_oas'].rolling(12).mean()
            self.weekly_data['cad_oas_sma_26'] = self.weekly_data['cad_oas'].rolling(26).mean()
            
            # Momentum indicators
            self.weekly_data['cad_oas_momentum_4'] = (
                self.weekly_data['cad_oas'] / self.weekly_data['cad_oas'].shift(4) - 1
            )
            self.weekly_data['cad_oas_momentum_12'] = (
                self.weekly_data['cad_oas'] / self.weekly_data['cad_oas'].shift(12) - 1
            )
            
            # Volatility measures
            self.weekly_data['cad_ig_volatility_4'] = (
                self.cad_ig_returns.rolling(4).std() * np.sqrt(52)
            )
            self.weekly_data['cad_ig_volatility_12'] = (
                self.cad_ig_returns.rolling(12).std() * np.sqrt(52)
            )
            
            # Relative value indicators
            self.weekly_data['cad_us_spread_ratio'] = (
                self.weekly_data['cad_oas'] / self.weekly_data['us_ig_oas']
            )
            
            # Economic regime features
            self.weekly_data['growth_momentum'] = (
                self.weekly_data['us_growth_surprises'].rolling(4).mean()
            )
            self.weekly_data['inflation_momentum'] = (
                self.weekly_data['us_inflation_surprises'].rolling(4).mean()
            )
            
            # Market stress indicator combining VIX and spreads
            self.weekly_data['market_stress'] = (
                (self.weekly_data['vix'] - self.weekly_data['vix'].rolling(26).mean()) / 
                self.weekly_data['vix'].rolling(26).std() +
                (self.weekly_data['cad_oas'] - self.weekly_data['cad_oas'].rolling(26).mean()) / 
                self.weekly_data['cad_oas'].rolling(26).std()
            ) / 2
            
            # NEW: Extreme positioning features
            self.weekly_data['extreme_spread_percentile'] = (
                self.weekly_data['cad_oas'].rolling(52).rank(pct=True)
            )
            self.weekly_data['extreme_vix_percentile'] = (
                self.weekly_data['vix'].rolling(52).rank(pct=True)
            )
            self.weekly_data['extreme_momentum_signal'] = (
                self.weekly_data['cad_oas_momentum_4'] / self.weekly_data['cad_oas_momentum_4'].rolling(26).std()
            )
            
            logger.info("Feature calculation completed")
            
        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise

    def generate_mean_reversion_signals(self, lookback_window=12, threshold_std=1.0):
        """
        Generate mean reversion signals for VectorBT backtesting
        
        When CAD spreads are wide (above mean + threshold), signal to buy (go long credit)
        When CAD spreads are tight (below mean - threshold), signal to sell (go short credit)
        
        Args:
            lookback_window (int): Rolling window for mean calculation
            threshold_std (float): Standard deviation threshold for signals
            
        Returns:
            pd.Series: Position weights (0 to 1)
        """
        try:
            # Calculate rolling statistics
            rolling_mean = self.weekly_data['cad_oas'].rolling(lookback_window).mean()
            rolling_std = self.weekly_data['cad_oas'].rolling(lookback_window).std()
            
            # Calculate z-score
            z_score = (self.weekly_data['cad_oas'] - rolling_mean) / rolling_std
            
            # Generate position weights using sigmoid function for smooth transitions
            # Positive z-score (wide spreads) = higher allocation
            positions = 1 / (1 + np.exp(-threshold_std * z_score))
            
            # Align with returns index and handle NaN values
            positions = positions.reindex(self.cad_ig_returns.index).fillna(0.5)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {str(e)}")
            return pd.Series(0.5, index=self.cad_ig_returns.index)

    def generate_momentum_signals(self, momentum_window=8, smoothing_window=4):
        """
        Generate momentum signals for VectorBT backtesting
        
        Follow momentum in spread movements - tightening spreads = positive for credit
        
        Args:
            momentum_window (int): Window for momentum calculation
            smoothing_window (int): Window for signal smoothing
            
        Returns:
            pd.Series: Position weights (0 to 1)
        """
        try:
            # Calculate spread momentum (negative because tightening = positive)
            spread_momentum = -self.weekly_data['cad_oas'].pct_change(momentum_window)
            
            # Normalize momentum using rolling z-score
            momentum_mean = spread_momentum.rolling(26).mean()
            momentum_std = spread_momentum.rolling(26).std()
            momentum_zscore = (spread_momentum - momentum_mean) / momentum_std
            
            # Convert to positions using sigmoid
            raw_positions = 1 / (1 + np.exp(-momentum_zscore))
            
            # Smooth positions
            positions = raw_positions.rolling(smoothing_window).mean()
            
            # Align with returns index and handle NaN values
            positions = positions.reindex(self.cad_ig_returns.index).fillna(0.5)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {str(e)}")
            return pd.Series(0.5, index=self.cad_ig_returns.index)

    def generate_regime_signals(self, regime_weight=0.4, vol_adjustment=True):
        """
        Generate regime-based signals for VectorBT backtesting
        
        Use economic regime indicators and volatility conditions
        
        Args:
            regime_weight (float): Weight for regime component
            vol_adjustment (bool): Whether to adjust for volatility regime
            
        Returns:
            pd.Series: Position weights (0 to 1)
        """
        try:
            # Create composite regime score
            regime_score = (
                self.weekly_data['us_economic_regime'] * regime_weight +
                np.clip(self.weekly_data['growth_momentum'], -2, 2) / 4 * 0.3 +
                np.clip(self.weekly_data['inflation_momentum'], -2, 2) / 4 * 0.2 +
                np.clip(-self.weekly_data['market_stress'], -2, 2) / 4 * 0.1
            )
            
            # Base position from regime score
            base_positions = np.clip(regime_score, 0, 1)
            
            if vol_adjustment:
                # Adjust for volatility regime
                vol_median = self.weekly_data['cad_ig_volatility_12'].rolling(52).median()
                vol_adjustment_factor = np.where(
                    self.weekly_data['cad_ig_volatility_12'] > vol_median,
                    0.8, 1.2  # Reduce in high vol, increase in low vol
                )
                positions = base_positions * vol_adjustment_factor
                positions = np.clip(positions, 0, 1.5)
            else:
                positions = base_positions
            
            # Align with returns index and handle NaN values
            positions = positions.reindex(self.cad_ig_returns.index).fillna(0.5)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error generating regime signals: {str(e)}")
            return pd.Series(0.5, index=self.cad_ig_returns.index)

    def generate_relative_value_signals(self, ratio_window=26, sensitivity=2.0):
        """
        Generate relative value signals comparing CAD IG to US IG spreads
        
        Args:
            ratio_window (int): Window for ratio calculations
            sensitivity (float): Sensitivity of allocation to ratio changes
            
        Returns:
            pd.Series: Position weights (0 to 1)
        """
        try:
            # Calculate spread ratio z-score
            ratio_mean = self.weekly_data['cad_us_spread_ratio'].rolling(ratio_window).mean()
            ratio_std = self.weekly_data['cad_us_spread_ratio'].rolling(ratio_window).std()
            ratio_zscore = (self.weekly_data['cad_us_spread_ratio'] - ratio_mean) / ratio_std
            
            # Convert to position (high ratio = expensive CAD = lower allocation)
            positions = 1 / (1 + np.exp(sensitivity * ratio_zscore))
            
            # Add momentum component
            ratio_momentum = self.weekly_data['cad_us_spread_ratio'].pct_change(4)
            momentum_adjustment = np.tanh(ratio_momentum * 10) * 0.2
            
            # Final positions
            positions = np.clip(positions + momentum_adjustment, 0, 1)
            
            # Align with returns index and handle NaN values
            positions = positions.reindex(self.cad_ig_returns.index).fillna(0.5)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error generating relative value signals: {str(e)}")
            return pd.Series(0.5, index=self.cad_ig_returns.index)

    def run_vectorbt_backtest(self, positions, strategy_name):
        """
        Run backtest using VectorBT with position weights
        
        Args:
            positions (pd.Series): Position weights over time
            strategy_name (str): Name of the strategy
            
        Returns:
            vbt.Portfolio: VectorBT portfolio object
        """
        try:
            logger.info(f"Running VectorBT backtest for {strategy_name}")
            
            # Ensure positions and returns are aligned
            common_index = positions.index.intersection(self.cad_ig_returns.index)
            aligned_positions = positions.loc[common_index]
            aligned_returns = self.cad_ig_returns.loc[common_index]
            
            if len(common_index) == 0:
                logger.error(f"No common dates between positions and returns for {strategy_name}")
                return None
            
            # Create VectorBT portfolio
            # Using simple approach: position weights directly determine allocation
            portfolio = vbt.Portfolio.from_signals(
                close=aligned_returns.cumsum().apply(np.exp),  # Convert returns to price series
                entries=aligned_positions > aligned_positions.shift(1),  # Increase position
                exits=aligned_positions < aligned_positions.shift(1),   # Decrease position
                size=aligned_positions,  # Position size
                size_type='percent',     # Percentage of capital
                freq='W'                 # Weekly frequency
            )
            
            logger.info(f"VectorBT backtest completed for {strategy_name}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error running VectorBT backtest for {strategy_name}: {str(e)}")
            return None

    def analyze_with_quantstats(self, portfolio, strategy_name):
        """
        Analyze portfolio performance using QuantStats
        
        Args:
            portfolio (vbt.Portfolio): VectorBT portfolio object
            strategy_name (str): Name of the strategy
            
        Returns:
            dict: Performance metrics dictionary
        """
        try:
            logger.info(f"Analyzing {strategy_name} with QuantStats")
            
            # Get portfolio returns
            portfolio_returns = portfolio.returns()
            
            if len(portfolio_returns) == 0 or portfolio_returns.isnull().all():
                logger.warning(f"No valid returns for {strategy_name}")
                return {}
            
            # Calculate key metrics using QuantStats with corrected function calls
            total_return = qs.stats.comp(portfolio_returns)
            cagr = qs.stats.cagr(portfolio_returns, periods=52)  # Weekly data
            volatility = qs.stats.volatility(portfolio_returns, periods=52)
            sharpe = qs.stats.sharpe(portfolio_returns, periods=52)
            sortino = qs.stats.sortino(portfolio_returns, periods=52)
            max_dd = qs.stats.max_drawdown(portfolio_returns)
            
            # Calculate Calmar ratio manually as it doesn't support periods parameter
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            
            # Additional metrics
            win_rate = qs.stats.win_rate(portfolio_returns)
            avg_win = qs.stats.avg_win(portfolio_returns)
            avg_loss = qs.stats.avg_loss(portfolio_returns)
            skew = qs.stats.skew(portfolio_returns)
            kurtosis = qs.stats.kurtosis(portfolio_returns)
            
            metrics = {
                'strategy_name': strategy_name,
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'skewness': skew,
                'kurtosis': kurtosis,
                'total_trades': len(portfolio_returns),
                'portfolio_obj': portfolio  # Store for later analysis
            }
            
            logger.info(f"QuantStats analysis completed for {strategy_name}")
            logger.info(f"CAGR: {cagr:.2%}, Sharpe: {sharpe:.3f}, Max DD: {max_dd:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing {strategy_name} with QuantStats: {str(e)}")
            return {}

    def run_comprehensive_iteration(self):
        """
        Run comprehensive strategy iteration using VectorBT and QuantStats
        
        Tests multiple strategies with different parameters to find optimal approach
        """
        try:
            logger.info("Starting comprehensive strategy iteration with VectorBT")
            
            all_results = []
            
            # Strategy 1: Mean Reversion Parameter Grid
            logger.info("Testing Mean Reversion strategies...")
            for lookback in [8, 12, 16, 20, 26]:
                for threshold in [0.5, 1.0, 1.5, 2.0]:
                    try:
                        strategy_name = f"mean_reversion_{lookback}_{threshold}"
                        positions = self.generate_mean_reversion_signals(lookback, threshold)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                # Track best strategy
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"New best strategy: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f})")
                                    
                    except Exception as e:
                        logger.warning(f"Failed mean reversion test {lookback}_{threshold}: {str(e)}")
                        continue
            
            # Strategy 2: Momentum Parameter Grid
            logger.info("Testing Momentum strategies...")
            for momentum_window in [4, 8, 12, 16]:
                for smoothing in [2, 4, 6, 8]:
                    try:
                        strategy_name = f"momentum_{momentum_window}_{smoothing}"
                        positions = self.generate_momentum_signals(momentum_window, smoothing)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"New best strategy: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f})")
                                    
                    except Exception as e:
                        logger.warning(f"Failed momentum test {momentum_window}_{smoothing}: {str(e)}")
                        continue
            
            # Strategy 3: Regime-Based Parameter Grid
            logger.info("Testing Regime-Based strategies...")
            for regime_weight in [0.2, 0.4, 0.6, 0.8]:
                for vol_adj in [True, False]:
                    try:
                        strategy_name = f"regime_{regime_weight}_{vol_adj}"
                        positions = self.generate_regime_signals(regime_weight, vol_adj)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"New best strategy: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f})")
                                    
                    except Exception as e:
                        logger.warning(f"Failed regime test {regime_weight}_{vol_adj}: {str(e)}")
                        continue
            
            # Strategy 4: Relative Value Parameter Grid
            logger.info("Testing Relative Value strategies...")
            for ratio_window in [20, 26, 32, 40]:
                for sensitivity in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    try:
                        strategy_name = f"relative_value_{ratio_window}_{sensitivity}"
                        positions = self.generate_relative_value_signals(ratio_window, sensitivity)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"New best strategy: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f})")
                                    
                    except Exception as e:
                        logger.warning(f"Failed relative value test {ratio_window}_{sensitivity}: {str(e)}")
                        continue
            
            # EXTREME STRATEGY 5: Binary Extreme Allocation
            logger.info("Testing EXTREME Binary Allocation strategies...")
            for spread_thresh in [0.7, 0.8, 0.9]:
                for vix_thresh in [0.6, 0.7, 0.8]:
                    for momentum_thresh in [1.0, 1.5, 2.0]:
                        try:
                            strategy_name = f"extreme_binary_{spread_thresh}_{vix_thresh}_{momentum_thresh}"
                            positions = self.generate_extreme_binary_signals(spread_thresh, vix_thresh, momentum_thresh)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    
                                    if metrics['sharpe_ratio'] > self.best_sharpe:
                                        self.best_sharpe = metrics['sharpe_ratio']
                                        self.best_strategy = strategy_name
                                        logger.info(f"🚀 NEW EXTREME BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                        
                                        # Check if we hit the target
                                        if metrics['total_return'] * 100 > self.target_return:
                                            logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > {self.target_return:.2%}")
                                        
                        except Exception as e:
                            logger.warning(f"Failed extreme binary test {spread_thresh}_{vix_thresh}_{momentum_thresh}: {str(e)}")
                            continue
            
            # EXTREME STRATEGY 6: Volatility Carry
            logger.info("Testing EXTREME Volatility Carry strategies...")
            for vol_lookback in [20, 26, 40]:
                for low_thresh in [0.15, 0.25, 0.35]:
                    for high_thresh in [0.65, 0.75, 0.85]:
                        try:
                            strategy_name = f"extreme_vol_carry_{vol_lookback}_{low_thresh}_{high_thresh}"
                            positions = self.generate_volatility_carry_signals(vol_lookback, low_thresh, high_thresh)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    
                                    if metrics['sharpe_ratio'] > self.best_sharpe:
                                        self.best_sharpe = metrics['sharpe_ratio']
                                        self.best_strategy = strategy_name
                                        logger.info(f"🚀 NEW EXTREME BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                        
                                        if metrics['total_return'] * 100 > self.target_return:
                                            logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > {self.target_return:.2%}")
                                        
                        except Exception as e:
                            logger.warning(f"Failed vol carry test {vol_lookback}_{low_thresh}_{high_thresh}: {str(e)}")
                            continue
            
            # EXTREME STRATEGY 7: Amplified Momentum
            logger.info("Testing EXTREME Amplified Momentum strategies...")
            for momentum_window in [4, 8, 12]:
                for amplification in [2.0, 3.0, 4.0, 5.0]:
                    for extreme_thresh in [1.0, 1.5, 2.0, 2.5]:
                        try:
                            strategy_name = f"extreme_momentum_{momentum_window}_{amplification}_{extreme_thresh}"
                            positions = self.generate_extreme_momentum_signals(momentum_window, amplification, extreme_thresh)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    
                                    if metrics['sharpe_ratio'] > self.best_sharpe:
                                        self.best_sharpe = metrics['sharpe_ratio']
                                        self.best_strategy = strategy_name
                                        logger.info(f"🚀 NEW EXTREME BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                        
                                        if metrics['total_return'] * 100 > self.target_return:
                                            logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > {self.target_return:.2%}")
                                        
                        except Exception as e:
                            logger.warning(f"Failed extreme momentum test {momentum_window}_{amplification}_{extreme_thresh}: {str(e)}")
                            continue
            
            # EXTREME STRATEGY 8: Crisis Opportunity
            logger.info("Testing EXTREME Crisis Opportunity strategies...")
            for crisis_lookback in [40, 52, 65]:
                for crisis_thresh in [1.5, 2.0, 2.5]:
                    for recovery_thresh in [0.3, 0.5, 0.7]:
                        try:
                            strategy_name = f"extreme_crisis_{crisis_lookback}_{crisis_thresh}_{recovery_thresh}"
                            positions = self.generate_crisis_opportunity_signals(crisis_lookback, crisis_thresh, recovery_thresh)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    
                                    if metrics['sharpe_ratio'] > self.best_sharpe:
                                        self.best_sharpe = metrics['sharpe_ratio']
                                        self.best_strategy = strategy_name
                                        logger.info(f"🚀 NEW EXTREME BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                        
                                        if metrics['total_return'] * 100 > self.target_return:
                                            logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > {self.target_return:.2%}")
                                        
                        except Exception as e:
                            logger.warning(f"Failed crisis opportunity test {crisis_lookback}_{crisis_thresh}_{recovery_thresh}: {str(e)}")
                            continue
            
            # EXTREME STRATEGY 9: Economic Regime Extreme
            logger.info("Testing EXTREME Economic Regime strategies...")
            for growth_weight in [0.4, 0.6, 0.8]:
                for inflation_weight in [0.2, 0.4, 0.6]:
                    for extreme_thresh in [0.8, 1.2, 1.6]:
                        try:
                            strategy_name = f"extreme_regime_{growth_weight}_{inflation_weight}_{extreme_thresh}"
                            positions = self.generate_economic_regime_extreme_signals(growth_weight, inflation_weight, extreme_thresh)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    
                                    if metrics['sharpe_ratio'] > self.best_sharpe:
                                        self.best_sharpe = metrics['sharpe_ratio']
                                        self.best_strategy = strategy_name
                                        logger.info(f"🚀 NEW EXTREME BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                        
                                        if metrics['total_return'] * 100 > self.target_return:
                                            logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > {self.target_return:.2%}")
                                        
                        except Exception as e:
                            logger.warning(f"Failed extreme regime test {growth_weight}_{inflation_weight}_{extreme_thresh}: {str(e)}")
                            continue

            # ULTRA STRATEGY 10: Ultra Aggressive Concentration
            logger.info("Testing ULTRA-AGGRESSIVE strategies...")
            for window in [8, 12, 20]:
                for concentration in [0.95, 0.98, 0.99]:
                    try:
                        strategy_name = f"ultra_aggressive_{window}_{concentration}"
                        positions = self.generate_ultra_aggressive_signals(window, concentration)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"🚀 NEW ULTRA BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                    
                                    if metrics['total_return'] * 100 > 64.07:
                                        logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                        
                    except Exception as e:
                        logger.warning(f"Failed ultra aggressive test {window}_{concentration}: {str(e)}")
                        continue

            # ULTRA STRATEGY 11: Crisis Alpha
            for crisis_lookback in [30, 40, 52]:
                try:
                    strategy_name = f"crisis_alpha_{crisis_lookback}"
                    positions = self.generate_crisis_alpha_signals(crisis_lookback)
                    portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            
                            if metrics['sharpe_ratio'] > self.best_sharpe:
                                self.best_sharpe = metrics['sharpe_ratio']
                                self.best_strategy = strategy_name
                                logger.info(f"🚀 NEW ULTRA BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                
                                if metrics['total_return'] * 100 > 64.07:
                                    logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                
                except Exception as e:
                    logger.warning(f"Failed crisis alpha test {crisis_lookback}: {str(e)}")
                    continue

            # ULTRA STRATEGY 12: Turbo Mean Reversion
            for lookback in [20, 26, 35]:
                for threshold in [2.0, 2.5, 3.0]:
                    try:
                        strategy_name = f"turbo_mean_reversion_{lookback}_{threshold}"
                        positions = self.generate_mean_reversion_turbo_signals(lookback, threshold)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"🚀 NEW ULTRA BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                    
                                    if metrics['total_return'] * 100 > 64.07:
                                        logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                    
                    except Exception as e:
                        logger.warning(f"Failed turbo mean reversion test {lookback}_{threshold}: {str(e)}")
                        continue

            # ULTRA STRATEGY 13: Momentum Breakout
            for momentum_window in [6, 8, 12]:
                for breakout_threshold in [2.5, 3.0, 3.5]:
                    try:
                        strategy_name = f"momentum_breakout_{momentum_window}_{breakout_threshold}"
                        positions = self.generate_momentum_breakout_signals(momentum_window, breakout_threshold)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"🚀 NEW ULTRA BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                    
                                    if metrics['total_return'] * 100 > 64.07:
                                        logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                    
                    except Exception as e:
                        logger.warning(f"Failed momentum breakout test {momentum_window}_{breakout_threshold}: {str(e)}")
                        continue

            # ULTRA STRATEGY 14: Volatility Timing
            for vol_window in [15, 20, 26]:
                for timing_sensitivity in [2.0, 2.5, 3.0]:
                    try:
                        strategy_name = f"volatility_timing_{vol_window}_{timing_sensitivity}"
                        positions = self.generate_volatility_timing_signals(vol_window, timing_sensitivity)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                
                                if metrics['sharpe_ratio'] > self.best_sharpe:
                                    self.best_sharpe = metrics['sharpe_ratio']
                                    self.best_strategy = strategy_name
                                    logger.info(f"🚀 NEW ULTRA BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                    
                                    if metrics['total_return'] * 100 > 64.07:
                                        logger.info(f"🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                    
                    except Exception as e:
                        logger.warning(f"Failed volatility timing test {vol_window}_{timing_sensitivity}: {str(e)}")
                        continue
            
            # Compile and analyze results
            if all_results:
                self.results = pd.DataFrame(all_results)
                
                # Sort by total return to find strategies closest to target
                self.results = self.results.sort_values('total_return', ascending=False)
                
                logger.info(f"Iteration completed! Tested {len(all_results)} strategy variants")
                
                # Check if any strategy hit the target
                target_threshold = 64.07  # 2x buy-and-hold target
                target_achieved = (self.results['total_return'] * 100 > target_threshold).any()
                if target_achieved:
                    winning_strategies = self.results[self.results['total_return'] * 100 > target_threshold]
                    logger.info(f"🎯🎯🎯 TARGET ACHIEVED! {len(winning_strategies)} strategies exceeded {target_threshold:.2f}% target!")
                    
                    for i, (_, row) in enumerate(winning_strategies.iterrows()):
                        logger.info(f"🏆 WINNER #{i+1}: {row['strategy_name']} | "
                                  f"Total Return: {row['total_return']:8.2%} | "
                                  f"Sharpe: {row['sharpe_ratio']:6.3f} | "
                                  f"CAGR: {row['cagr']:7.2%} | "
                                  f"MaxDD: {row['max_drawdown']:7.2%}")
                else:
                    best_return = self.results.iloc[0]['total_return'] * 100
                    shortfall = target_threshold - best_return
                    logger.info(f"❌ Target not achieved. Best: {best_return:.2f}%, Target: {target_threshold:.2f}%, Shortfall: {shortfall:.2f}%")
                
                logger.info(f"Best overall strategy: {self.results.iloc[0]['strategy_name']} with total return: {self.results.iloc[0]['total_return']:.2%}")
                
                # Display top 15 strategies by total return
                logger.info("\n" + "="*100)
                logger.info("TOP 15 STRATEGIES BY TOTAL RETURN (TARGET: >64.07%):")
                logger.info("="*100)
                logger.info(f"{'Rank':4s} {'Strategy Name':40s} {'Total Return':12s} {'Target Gap':10s} {'Sharpe':8s} {'CAGR':8s} {'MaxDD':8s}")
                logger.info("-" * 100)
                
                for i, (_, row) in enumerate(self.results.head(15).iterrows()):
                    total_ret_pct = row['total_return'] * 100
                    gap = total_ret_pct - target_threshold
                    gap_str = f"+{gap:.1f}%" if gap > 0 else f"{gap:.1f}%"
                    
                    logger.info(f"{i+1:4d} {row['strategy_name']:40s} "
                              f"{row['total_return']:11.2%} "
                              f"{gap_str:>9s} "
                              f"{row['sharpe_ratio']:7.3f} "
                              f"{row['cagr']:7.2%} "
                              f"{row['max_drawdown']:7.2%}")
                
                return self.results
            else:
                logger.warning("No successful strategy iterations completed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in comprehensive iteration: {str(e)}")
            raise

    def generate_detailed_report(self, top_n=5):
        """
        Generate detailed performance report for top strategies using QuantStats
        
        Args:
            top_n (int): Number of top strategies to analyze in detail
        """
        try:
            if self.results.empty:
                logger.warning("No results available for detailed report")
                return
                
            logger.info(f"Generating detailed report for top {top_n} strategies")
            
            top_strategies = self.results.head(top_n)
            
            for i, (_, strategy) in enumerate(top_strategies.iterrows()):
                logger.info(f"\n{'='*60}")
                logger.info(f"DETAILED ANALYSIS #{i+1}: {strategy['strategy_name']}")
                logger.info(f"{'='*60}")
                
                # Performance Summary - tabular format only
                logger.info("Performance Summary:")
                logger.info(f"Total Return:     {strategy['total_return']:8.2%}")
                logger.info(f"CAGR:            {strategy['cagr']:8.2%}")
                logger.info(f"Volatility:      {strategy['volatility']:8.2%}")
                logger.info(f"Sharpe Ratio:    {strategy['sharpe_ratio']:8.3f}")
                logger.info(f"Sortino Ratio:   {strategy['sortino_ratio']:8.3f}")
                logger.info(f"Calmar Ratio:    {strategy['calmar_ratio']:8.3f}")
                logger.info(f"Max Drawdown:    {strategy['max_drawdown']:8.2%}")
                logger.info(f"Win Rate:        {strategy['win_rate']:8.1%}")
                logger.info(f"Avg Win:         {strategy['avg_win']:8.2%}")
                logger.info(f"Avg Loss:        {strategy['avg_loss']:8.2%}")
                
        except Exception as e:
            logger.error(f"Error generating detailed report: {str(e)}")

    def generate_extreme_binary_signals(self, spread_threshold=0.8, vix_threshold=0.7, momentum_threshold=1.5):
        """
        EXTREME STRATEGY 1: Binary Extreme Allocation
        
        Allocates extreme positions (5% vs 95%) based on market conditions
        to maximize capture of favorable periods while minimizing exposure during bad periods.
        
        Args:
            spread_threshold (float): Percentile threshold for spread signals
            vix_threshold (float): Percentile threshold for VIX signals  
            momentum_threshold (float): Z-score threshold for momentum signals
        """
        try:
            logger.info(f"Generating extreme binary signals (spread={spread_threshold}, vix={vix_threshold}, momentum={momentum_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.0, index=data.index)
            
            # Start with 50% base allocation
            positions[:] = 0.5
            
            # Extreme high allocation conditions (95%)
            high_allocation_conditions = (
                # Spreads are wide (high percentile = opportunity)
                (data['extreme_spread_percentile'] > spread_threshold) |
                # VIX is low (low vol = good environment)
                (data['extreme_vix_percentile'] < (1 - vix_threshold)) |
                # Strong positive momentum
                (data['extreme_momentum_signal'] < -momentum_threshold)
            )
            
            # Extreme low allocation conditions (5%)
            low_allocation_conditions = (
                # Spreads are tight (low percentile = expensive)
                (data['extreme_spread_percentile'] < (1 - spread_threshold)) |
                # VIX is high (high vol = bad environment)
                (data['extreme_vix_percentile'] > vix_threshold) |
                # Strong negative momentum
                (data['extreme_momentum_signal'] > momentum_threshold)
            )
            
            # Apply extreme positioning
            positions[high_allocation_conditions] = 0.95
            positions[low_allocation_conditions] = 0.05
            
            # Ensure positions are within [0, 1] bounds
            positions = positions.clip(0, 1)
            
            allocation_stats = {
                'high_allocation_periods': high_allocation_conditions.sum(),
                'low_allocation_periods': low_allocation_conditions.sum(),
                'neutral_periods': len(positions) - high_allocation_conditions.sum() - low_allocation_conditions.sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Extreme binary allocation stats: {allocation_stats}")
            
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating extreme binary signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_volatility_carry_signals(self, vol_lookback=26, low_vol_threshold=0.25, high_vol_threshold=0.75):
        """
        EXTREME STRATEGY 2: Volatility Carry Strategy
        
        Maximizes allocation during low volatility periods (when credit performs well)
        and minimizes during high volatility periods (when credit underperforms).
        
        Args:
            vol_lookback (int): Lookback period for volatility calculation
            low_vol_threshold (float): Percentile threshold for low volatility
            high_vol_threshold (float): Percentile threshold for high volatility
        """
        try:
            logger.info(f"Generating volatility carry signals (lookback={vol_lookback}, low_thresh={low_vol_threshold}, high_thresh={high_vol_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Calculate rolling volatility percentiles
            vol_percentile = data['cad_ig_volatility_4'].rolling(vol_lookback).rank(pct=True)
            
            # Extreme high allocation during low volatility periods
            low_vol_mask = vol_percentile < low_vol_threshold
            positions[low_vol_mask] = 0.98
            
            # Extreme low allocation during high volatility periods  
            high_vol_mask = vol_percentile > high_vol_threshold
            positions[high_vol_mask] = 0.02
            
            # Gradual scaling in middle range
            mid_vol_mask = (~low_vol_mask) & (~high_vol_mask)
            positions[mid_vol_mask] = 0.5 - (vol_percentile[mid_vol_mask] - 0.5) * 0.8
            
            # Ensure bounds
            positions = positions.clip(0, 1)
            
            vol_stats = {
                'low_vol_periods': low_vol_mask.sum(),
                'high_vol_periods': high_vol_mask.sum(),
                'mid_vol_periods': mid_vol_mask.sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Volatility carry allocation stats: {vol_stats}")
            
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating volatility carry signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_extreme_momentum_signals(self, momentum_window=8, amplification=3.0, extreme_threshold=1.5):
        """
        EXTREME STRATEGY 3: Amplified Momentum Strategy
        
        Uses extreme position sizing based on momentum strength to capture
        strong trending periods with maximum allocation.
        
        Args:
            momentum_window (int): Window for momentum calculation
            amplification (float): Amplification factor for momentum signals
            extreme_threshold (float): Threshold for extreme positioning
        """
        try:
            logger.info(f"Generating extreme momentum signals (window={momentum_window}, amp={amplification}, thresh={extreme_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Calculate normalized momentum
            momentum = data['cad_oas_momentum_4'].fillna(0)
            momentum_zscore = (momentum - momentum.rolling(26).mean()) / momentum.rolling(26).std()
            
            # Amplified momentum signal
            amplified_signal = momentum_zscore * amplification
            
            # Convert to position sizes with extreme ranges
            # Strong negative momentum (spreads tightening) = HIGH allocation
            # Strong positive momentum (spreads widening) = LOW allocation
            
            extreme_high_mask = amplified_signal < -extreme_threshold
            extreme_low_mask = amplified_signal > extreme_threshold
            
            positions[extreme_high_mask] = 0.99
            positions[extreme_low_mask] = 0.01
            
            # Gradual scaling for moderate signals
            moderate_mask = (~extreme_high_mask) & (~extreme_low_mask)
            positions[moderate_mask] = 0.5 - (amplified_signal[moderate_mask] * 0.3)
            
            # Ensure bounds
            positions = positions.clip(0, 1)
            
            momentum_stats = {
                'extreme_high_periods': extreme_high_mask.sum(),
                'extreme_low_periods': extreme_low_mask.sum(),
                'moderate_periods': moderate_mask.sum(),
                'avg_momentum_zscore': momentum_zscore.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Extreme momentum allocation stats: {momentum_stats}")
            
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating extreme momentum signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_crisis_opportunity_signals(self, crisis_lookback=52, crisis_threshold=2.0, recovery_threshold=0.5):
        """
        EXTREME STRATEGY 4: Crisis Opportunity Strategy
        
        Maximizes allocation during crisis periods when spreads are wide
        (contrarian approach) and reduces allocation during normal/tight periods.
        
        Args:
            crisis_lookback (int): Lookback period for crisis detection
            crisis_threshold (float): Z-score threshold for crisis detection
            recovery_threshold (float): Threshold for recovery detection
        """
        try:
            logger.info(f"Generating crisis opportunity signals (lookback={crisis_lookback}, crisis_thresh={crisis_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Calculate market stress Z-score
            stress_zscore = (
                (data['market_stress'] - data['market_stress'].rolling(crisis_lookback).mean()) /
                data['market_stress'].rolling(crisis_lookback).std()
            ).fillna(0)
            
            # Crisis periods = high allocation opportunity
            crisis_mask = stress_zscore > crisis_threshold
            positions[crisis_mask] = 0.97
            
            # Recovery periods = moderate high allocation
            recovery_mask = (stress_zscore > recovery_threshold) & (~crisis_mask)
            positions[recovery_mask] = 0.75
            
            # Normal/tight periods = low allocation
            normal_mask = stress_zscore < -recovery_threshold
            positions[normal_mask] = 0.15
            
            # Neutral periods
            neutral_mask = (~crisis_mask) & (~recovery_mask) & (~normal_mask)
            positions[neutral_mask] = 0.5
            
            # Ensure bounds
            positions = positions.clip(0, 1)
            
            crisis_stats = {
                'crisis_periods': crisis_mask.sum(),
                'recovery_periods': recovery_mask.sum(),
                'normal_periods': normal_mask.sum(),
                'neutral_periods': neutral_mask.sum(),
                'avg_stress_zscore': stress_zscore.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Crisis opportunity allocation stats: {crisis_stats}")
            
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating crisis opportunity signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_economic_regime_extreme_signals(self, growth_weight=0.6, inflation_weight=0.4, extreme_threshold=1.2):
        """
        EXTREME STRATEGY 5: Economic Regime Extreme Positioning
        
        Uses economic surprise data to position extremely in favorable regimes
        and minimize exposure in unfavorable regimes.
        
        Args:
            growth_weight (float): Weight for growth surprises
            inflation_weight (float): Weight for inflation surprises  
            extreme_threshold (float): Threshold for extreme positioning
        """
        try:
            logger.info(f"Generating economic regime extreme signals (growth_w={growth_weight}, inflation_w={inflation_weight})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Normalize economic surprises
            growth_norm = (
                (data['growth_momentum'] - data['growth_momentum'].rolling(26).mean()) /
                data['growth_momentum'].rolling(26).std()
            ).fillna(0)
            
            inflation_norm = (
                (data['inflation_momentum'] - data['inflation_momentum'].rolling(26).mean()) /
                data['inflation_momentum'].rolling(26).std()
            ).fillna(0)
            
            # Combined regime score
            # Positive growth surprises + negative inflation surprises = favorable for credit
            regime_score = growth_weight * growth_norm - inflation_weight * inflation_norm
            
            # Extreme positioning based on regime
            extreme_favorable_mask = regime_score > extreme_threshold
            extreme_unfavorable_mask = regime_score < -extreme_threshold
            
            positions[extreme_favorable_mask] = 0.96
            positions[extreme_unfavorable_mask] = 0.04
            
            # Gradual scaling for moderate regimes
            moderate_mask = (~extreme_favorable_mask) & (~extreme_unfavorable_mask)
            positions[moderate_mask] = 0.5 + (regime_score[moderate_mask] * 0.3)
            
            # Ensure bounds
            positions = positions.clip(0, 1)
            
            regime_stats = {
                'extreme_favorable_periods': extreme_favorable_mask.sum(),
                'extreme_unfavorable_periods': extreme_unfavorable_mask.sum(),
                'moderate_periods': moderate_mask.sum(),
                'avg_regime_score': regime_score.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Economic regime allocation stats: {regime_stats}")
            
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating economic regime extreme signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_ultra_aggressive_signals(self, window=12, concentration_factor=0.98):
        """
        ULTRA STRATEGY 1: Ultra Aggressive Concentration
        
        Concentrates 98%+ allocation during the most favorable periods
        and stays at minimum allocation otherwise.
        """
        try:
            logger.info(f"Generating ultra aggressive concentration signals (window={window}, concentration={concentration_factor})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.02, index=data.index)  # Start very low
            
            # Multiple signal combination for ultra concentration
            spread_signal = (data['cad_oas'] - data['cad_oas'].rolling(window).mean()) / data['cad_oas'].rolling(window).std()
            vol_signal = (data['cad_ig_volatility_4'] - data['cad_ig_volatility_4'].rolling(window).mean()) / data['cad_ig_volatility_4'].rolling(window).std()
            momentum_signal = data['cad_oas_momentum_4'] / data['cad_oas_momentum_4'].rolling(window).std()
            
            # Ultra favorable conditions (max allocation)
            ultra_favorable = (
                (spread_signal > 1.5) |  # Spreads very wide
                (vol_signal < -1.0) |    # Volatility very low
                (momentum_signal < -2.0) # Strong tightening momentum
            )
            
            positions[ultra_favorable] = concentration_factor
            
            # Favorable conditions (high allocation)
            favorable = (
                (spread_signal > 0.75) |
                (vol_signal < -0.5) |
                (momentum_signal < -1.0)
            ) & (~ultra_favorable)
            
            positions[favorable] = 0.85
            
            stats = {
                'ultra_favorable_periods': ultra_favorable.sum(),
                'favorable_periods': favorable.sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Ultra aggressive stats: {stats}")
            return positions.fillna(0.02)
            
        except Exception as e:
            logger.error(f"Error generating ultra aggressive signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_crisis_alpha_signals(self, crisis_lookback=40):
        """
        ULTRA STRATEGY 2: Crisis Alpha Strategy
        
        Goes max allocation during crisis periods when spreads blow out
        (when there's maximum alpha opportunity).
        """
        try:
            logger.info(f"Generating crisis alpha signals (lookback={crisis_lookback})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.05, index=data.index)
            
            # Crisis detection - multiple standard deviations above normal
            spread_zscore = (data['cad_oas'] - data['cad_oas'].rolling(crisis_lookback).mean()) / data['cad_oas'].rolling(crisis_lookback).std()
            vix_zscore = (data['vix'] - data['vix'].rolling(crisis_lookback).mean()) / data['vix'].rolling(crisis_lookback).std()
            
            # Extreme crisis = max allocation (where alpha is highest)
            extreme_crisis = (spread_zscore > 3.0) | (vix_zscore > 3.0)
            positions[extreme_crisis] = 0.99
            
            # Moderate crisis = high allocation
            moderate_crisis = ((spread_zscore > 2.0) | (vix_zscore > 2.0)) & (~extreme_crisis)
            positions[moderate_crisis] = 0.90
            
            # Recovery phases = moderate allocation
            recovery = ((spread_zscore > 1.0) | (vix_zscore > 1.0)) & (~extreme_crisis) & (~moderate_crisis)
            positions[recovery] = 0.60
            
            stats = {
                'extreme_crisis_periods': extreme_crisis.sum(),
                'moderate_crisis_periods': moderate_crisis.sum(),
                'recovery_periods': recovery.sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Crisis alpha stats: {stats}")
            return positions.fillna(0.05)
            
        except Exception as e:
            logger.error(f"Error generating crisis alpha signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_mean_reversion_turbo_signals(self, lookback=26, extreme_threshold=2.5):
        """
        ULTRA STRATEGY 3: Turbo Mean Reversion
        
        Uses extreme mean reversion with maximum concentration during
        the most stretched periods.
        """
        try:
            logger.info(f"Generating turbo mean reversion signals (lookback={lookback}, threshold={extreme_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.03, index=data.index)
            
            # Calculate multiple mean reversion signals
            spread_ma = data['cad_oas'].rolling(lookback).mean()
            spread_std = data['cad_oas'].rolling(lookback).std()
            spread_zscore = (data['cad_oas'] - spread_ma) / spread_std
            
            # Economic cycle mean reversion
            growth_ma = data['us_growth_surprises'].rolling(lookback).mean()
            growth_std = data['us_growth_surprises'].rolling(lookback).std()
            growth_zscore = (data['us_growth_surprises'] - growth_ma) / growth_std
            
            # Extreme wide spreads + good economic conditions = max allocation
            extreme_opportunity = (spread_zscore > extreme_threshold) & (growth_zscore > 0)
            positions[extreme_opportunity] = 0.98
            
            # Very wide spreads = high allocation
            high_opportunity = (spread_zscore > extreme_threshold * 0.7) & (~extreme_opportunity)
            positions[high_opportunity] = 0.85
            
            # Moderate wide spreads = moderate allocation
            moderate_opportunity = (spread_zscore > extreme_threshold * 0.4) & (~extreme_opportunity) & (~high_opportunity)
            positions[moderate_opportunity] = 0.65
            
            stats = {
                'extreme_opportunity_periods': extreme_opportunity.sum(),
                'high_opportunity_periods': high_opportunity.sum(),
                'moderate_opportunity_periods': moderate_opportunity.sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Turbo mean reversion stats: {stats}")
            return positions.fillna(0.03)
            
        except Exception as e:
            logger.error(f"Error generating turbo mean reversion signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_momentum_breakout_signals(self, momentum_window=8, breakout_threshold=3.0):
        """
        ULTRA STRATEGY 4: Momentum Breakout Strategy
        
        Captures momentum breakouts with maximum allocation during
        strong trend continuations.
        """
        try:
            logger.info(f"Generating momentum breakout signals (window={momentum_window}, threshold={breakout_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.05, index=data.index)
            
            # Multiple momentum signals
            spread_momentum = data['cad_oas_momentum_4'].fillna(0)
            spread_momentum_zscore = (spread_momentum - spread_momentum.rolling(26).mean()) / spread_momentum.rolling(26).std()
            
            # Rate of change acceleration
            spread_roc = data['cad_oas'].pct_change(momentum_window)
            spread_roc_zscore = (spread_roc - spread_roc.rolling(26).mean()) / spread_roc.rolling(26).std()
            
            # Combine momentum signals
            combined_momentum = (spread_momentum_zscore + spread_roc_zscore) / 2
            
            # Strong tightening momentum (spreads falling fast) = max allocation
            strong_tightening = combined_momentum < -breakout_threshold
            positions[strong_tightening] = 0.97
            
            # Moderate tightening momentum = high allocation
            moderate_tightening = (combined_momentum < -breakout_threshold * 0.6) & (~strong_tightening)
            positions[moderate_tightening] = 0.80
            
            # Mild tightening momentum = moderate allocation
            mild_tightening = (combined_momentum < -breakout_threshold * 0.3) & (~strong_tightening) & (~moderate_tightening)
            positions[mild_tightening] = 0.60
            
            stats = {
                'strong_tightening_periods': strong_tightening.sum(),
                'moderate_tightening_periods': moderate_tightening.sum(),
                'mild_tightening_periods': mild_tightening.sum(),
                'avg_combined_momentum': combined_momentum.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Momentum breakout stats: {stats}")
            return positions.fillna(0.05)
            
        except Exception as e:
            logger.error(f"Error generating momentum breakout signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_volatility_timing_signals(self, vol_window=20, timing_sensitivity=2.5):
        """
        ULTRA STRATEGY 5: Volatility Timing Strategy
        
        Times allocations based on volatility regimes with maximum
        allocation during optimal volatility conditions.
        """
        try:
            logger.info(f"Generating volatility timing signals (window={vol_window}, sensitivity={timing_sensitivity})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.08, index=data.index)
            
            # Volatility timing signals
            vol_zscore = (data['cad_ig_volatility_4'] - data['cad_ig_volatility_4'].rolling(vol_window).mean()) / data['cad_ig_volatility_4'].rolling(vol_window).std()
            vix_zscore = (data['vix'] - data['vix'].rolling(vol_window).mean()) / data['vix'].rolling(vol_window).std()
            
            # Volatility mean reversion - buy when vol is high but declining
            vol_momentum = data['cad_ig_volatility_4'].pct_change(4)
            vol_momentum_zscore = (vol_momentum - vol_momentum.rolling(vol_window).mean()) / vol_momentum.rolling(vol_window).std()
            
            # Optimal timing = high vol that's declining
            optimal_timing = (vol_zscore > timing_sensitivity * 0.8) & (vol_momentum_zscore < -0.5)
            positions[optimal_timing] = 0.95
            
            # Good timing = low vol environment  
            good_timing = (vol_zscore < -timing_sensitivity * 0.5) & (~optimal_timing)
            positions[good_timing] = 0.85
            
            # Moderate timing = transitional periods
            moderate_timing = (abs(vol_zscore) < timing_sensitivity * 0.3) & (~optimal_timing) & (~good_timing)
            positions[moderate_timing] = 0.55
            
            stats = {
                'optimal_timing_periods': optimal_timing.sum(),
                'good_timing_periods': good_timing.sum(),
                'moderate_timing_periods': moderate_timing.sum(),
                'avg_vol_zscore': vol_zscore.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Volatility timing stats: {stats}")
            return positions.fillna(0.08)
            
        except Exception as e:
            logger.error(f"Error generating volatility timing signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_ultra_extreme_final_signals(self, perfect_threshold=0.99):
        """
        FINAL ULTRA STRATEGY: Maximum Concentration Strategy
        
        Uses near-maximum allocation (99.5%) during absolutely perfect conditions
        and minimum allocation (0.5%) otherwise to maximize total return potential.
        """
        try:
            logger.info(f"Generating FINAL ultra extreme signals (perfect_threshold={perfect_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.005, index=data.index)  # Start at minimum
            
            # Multiple layered signal approach for perfect timing
            spread_signal = (data['cad_oas'] - data['cad_oas'].rolling(12).mean()) / data['cad_oas'].rolling(12).std()
            vol_signal = (data['cad_ig_volatility_4'] - data['cad_ig_volatility_4'].rolling(12).mean()) / data['cad_ig_volatility_4'].rolling(12).std()
            momentum_signal = data['cad_oas_momentum_4'] / data['cad_oas_momentum_4'].rolling(12).std()
            vix_signal = (data['vix'] - data['vix'].rolling(12).mean()) / data['vix'].rolling(12).std()
            growth_signal = (data['us_growth_surprises'] - data['us_growth_surprises'].rolling(12).mean()) / data['us_growth_surprises'].rolling(12).std()
            
            # PERFECT conditions (99.5% allocation) - all stars aligned
            perfect_conditions = (
                (spread_signal > 2.0) &  # Spreads very wide
                (vol_signal < -1.5) &    # Volatility very low
                (momentum_signal < -3.0) & # Strong tightening momentum
                (vix_signal < -1.0) &    # VIX low
                (growth_signal > 0.5)    # Growth positive
            )
            
            positions[perfect_conditions] = perfect_threshold
            
            # EXCELLENT conditions (95% allocation) - most stars aligned
            excellent_conditions = (
                (
                    ((spread_signal > 1.5) & (vol_signal < -1.0) & (momentum_signal < -2.0)) |
                    ((spread_signal > 2.0) & (momentum_signal < -2.5)) |
                    ((vol_signal < -1.5) & (momentum_signal < -3.0) & (vix_signal < -0.5))
                ) & (~perfect_conditions)
            )
            
            positions[excellent_conditions] = 0.95
            
            # VERY GOOD conditions (85% allocation)
            very_good_conditions = (
                (
                    ((spread_signal > 1.0) & (vol_signal < -0.5) & (momentum_signal < -1.5)) |
                    ((spread_signal > 1.5) & (momentum_signal < -2.0)) |
                    ((momentum_signal < -2.5) & (vix_signal < -0.5))
                ) & (~perfect_conditions) & (~excellent_conditions)
            )
            
            positions[very_good_conditions] = 0.85
            
            # GOOD conditions (65% allocation)
            good_conditions = (
                (
                    ((spread_signal > 0.75) & (momentum_signal < -1.0)) |
                    ((vol_signal < -0.5) & (momentum_signal < -1.5)) |
                    (momentum_signal < -2.0)
                ) & (~perfect_conditions) & (~excellent_conditions) & (~very_good_conditions)
            )
            
            positions[good_conditions] = 0.65
            
            # MODERATE conditions (35% allocation)
            moderate_conditions = (
                (
                    (spread_signal > 0.5) |
                    (momentum_signal < -0.75) |
                    (vol_signal < 0.0)
                ) & (~perfect_conditions) & (~excellent_conditions) & (~very_good_conditions) & (~good_conditions)
            )
            
            positions[moderate_conditions] = 0.35
            
            stats = {
                'perfect_periods': perfect_conditions.sum(),
                'excellent_periods': excellent_conditions.sum(),
                'very_good_periods': very_good_conditions.sum(),
                'good_periods': good_conditions.sum(),
                'moderate_periods': moderate_conditions.sum(),
                'minimum_periods': (~(perfect_conditions | excellent_conditions | very_good_conditions | good_conditions | moderate_conditions)).sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"FINAL ultra extreme stats: {stats}")
            return positions.fillna(0.005)
            
        except Exception as e:
            logger.error(f"Error generating final ultra extreme signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_leveraged_like_signals(self, base_amplification=8.0):
        """
        FINAL STRATEGY 2: Leveraged-Like Approach (within constraints)
        
        Simulates leveraged exposure by using extreme concentration during 
        favorable periods to achieve leverage-like returns without actual leverage.
        """
        try:
            logger.info(f"Generating leveraged-like signals (amplification={base_amplification})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.01, index=data.index)  # Start very low
            
            # Create a composite "leverage-like" signal
            spread_momentum = data['cad_oas_momentum_4'].fillna(0)
            spread_level = (data['cad_oas'] - data['cad_oas'].rolling(26).mean()) / data['cad_oas'].rolling(26).std()
            vol_regime = (data['cad_ig_volatility_4'] - data['cad_ig_volatility_4'].rolling(26).mean()) / data['cad_ig_volatility_4'].rolling(26).std()
            
            # Normalize signals
            momentum_norm = (spread_momentum - spread_momentum.rolling(52).mean()) / spread_momentum.rolling(52).std()
            
            # Composite signal (negative is good for credit)
            composite_signal = -(momentum_norm + spread_level * 0.5 - vol_regime * 0.3)
            
            # Apply extreme amplification
            amplified_signal = composite_signal * base_amplification
            
            # Convert to positions with extreme concentration
            # Use sigmoid-like function to create extreme concentration
            positions = 1 / (1 + np.exp(-amplified_signal))
            
            # Apply floor and ceiling with extreme ranges
            positions = positions.clip(0.01, 0.995)
            
            # Additional boost for extreme positive signals
            extreme_boost_mask = composite_signal > 2.0
            positions[extreme_boost_mask] = 0.999
            
            stats = {
                'extreme_boost_periods': extreme_boost_mask.sum(),
                'high_allocation_periods': (positions > 0.8).sum(),
                'moderate_allocation_periods': ((positions > 0.4) & (positions <= 0.8)).sum(),
                'low_allocation_periods': (positions <= 0.4).sum(),
                'avg_composite_signal': composite_signal.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Leveraged-like allocation stats: {stats}")
            return positions.fillna(0.01)
            
        except Exception as e:
            logger.error(f"Error generating leveraged-like signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def run_final_ultra_iteration(self):
        """
        Final iteration focused on ultra-extreme strategies to reach 64.07% target
        """
        try:
            logger.info("="*80)
            logger.info("FINAL ULTRA-EXTREME ITERATION - TARGET: >64.07%")
            logger.info("="*80)
            
            all_results = []
            self.best_sharpe = 0
            self.best_strategy = ""
            
            # Test the current best extreme momentum strategies with higher amplification
            logger.info("Testing FINAL extreme momentum strategies...")
            for window in [4, 8, 12]:
                for amplification in [6.0, 7.0, 8.0, 10.0]:
                    for threshold in [1.0, 1.5, 2.0]:
                        try:
                            strategy_name = f"final_extreme_momentum_{window}_{amplification}_{threshold}"
                            positions = self.generate_extreme_momentum_signals(window, amplification, threshold)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    
                                    if metrics['sharpe_ratio'] > self.best_sharpe:
                                        self.best_sharpe = metrics['sharpe_ratio']
                                        self.best_strategy = strategy_name
                                        logger.info(f"🚀 NEW FINAL BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                        
                                        if metrics['total_return'] * 100 > 64.07:
                                            logger.info(f"🎯🎯🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                        
                        except Exception as e:
                            logger.warning(f"Failed final extreme momentum test {window}_{amplification}_{threshold}: {str(e)}")
                            continue

            # Test ultra extreme final strategies
            logger.info("Testing ULTRA EXTREME FINAL strategies...")
            for perfect_threshold in [0.995, 0.998, 0.999]:
                try:
                    strategy_name = f"ultra_extreme_final_{perfect_threshold}"
                    positions = self.generate_ultra_extreme_final_signals(perfect_threshold)
                    portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            
                            if metrics['sharpe_ratio'] > self.best_sharpe:
                                self.best_sharpe = metrics['sharpe_ratio']
                                self.best_strategy = strategy_name
                                logger.info(f"🚀 NEW FINAL BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                
                                if metrics['total_return'] * 100 > 64.07:
                                    logger.info(f"🎯🎯🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                
                except Exception as e:
                    logger.warning(f"Failed ultra extreme final test {perfect_threshold}: {str(e)}")
                    continue

            # Test leveraged-like strategies
            logger.info("Testing LEVERAGED-LIKE strategies...")
            for amplification in [8.0, 10.0, 12.0, 15.0]:
                try:
                    strategy_name = f"leveraged_like_{amplification}"
                    positions = self.generate_leveraged_like_signals(amplification)
                    portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            
                            if metrics['sharpe_ratio'] > self.best_sharpe:
                                self.best_sharpe = metrics['sharpe_ratio']
                                self.best_strategy = strategy_name
                                logger.info(f"🚀 NEW FINAL BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
                                
                                if metrics['total_return'] * 100 > 64.07:
                                    logger.info(f"🎯🎯🎯 TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")
                                
                except Exception as e:
                    logger.warning(f"Failed leveraged-like test {amplification}: {str(e)}")
                    continue

            # Compile and analyze results
            if all_results:
                self.results = pd.DataFrame(all_results)
                self.results = self.results.sort_values('total_return', ascending=False)
                
                logger.info(f"FINAL iteration completed! Tested {len(all_results)} ultra-extreme variants")
                
                target_threshold = 64.07
                target_achieved = (self.results['total_return'] * 100 > target_threshold).any()
                
                if target_achieved:
                    winning_strategies = self.results[self.results['total_return'] * 100 > target_threshold]
                    logger.info(f"🎯🎯🎯 MISSION ACCOMPLISHED! {len(winning_strategies)} strategies exceeded {target_threshold:.2f}% target!")
                    
                    for i, (_, row) in enumerate(winning_strategies.iterrows()):
                        logger.info(f"🏆 CHAMPION #{i+1}: {row['strategy_name']} | "
                                  f"Total Return: {row['total_return']:8.2%} | "
                                  f"Sharpe: {row['sharpe_ratio']:6.3f} | "
                                  f"CAGR: {row['cagr']:7.2%} | "
                                  f"MaxDD: {row['max_drawdown']:7.2%}")
                else:
                    best_return = self.results.iloc[0]['total_return'] * 100
                    shortfall = target_threshold - best_return
                    logger.info(f"Final attempt - Best: {best_return:.2f}%, Target: {target_threshold:.2f}%, Shortfall: {shortfall:.2f}%")
                
                logger.info(f"FINAL best strategy: {self.results.iloc[0]['strategy_name']} with total return: {self.results.iloc[0]['total_return']:.2%}")
                
                # Display top 10 FINAL strategies by total return
                logger.info("\n" + "="*100)
                logger.info("FINAL TOP 10 STRATEGIES BY TOTAL RETURN:")
                logger.info("="*100)
                logger.info(f"{'Rank':4s} {'Strategy Name':45s} {'Total Return':12s} {'Target Gap':10s} {'Sharpe':8s} {'CAGR':8s} {'MaxDD':8s}")
                logger.info("-" * 100)
                
                for i, (_, row) in enumerate(self.results.head(10).iterrows()):
                    total_ret_pct = row['total_return'] * 100
                    gap = total_ret_pct - target_threshold
                    gap_str = f"+{gap:.1f}%" if gap > 0 else f"{gap:.1f}%"
                    
                    logger.info(f"{i+1:4d} {row['strategy_name']:45s} "
                              f"{row['total_return']:11.2%} "
                              f"{gap_str:>9s} "
                              f"{row['sharpe_ratio']:7.3f} "
                              f"{row['cagr']:7.2%} "
                              f"{row['max_drawdown']:7.2%}")
                
                return self.results
            else:
                logger.warning("No successful FINAL strategy iterations completed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in final ultra iteration: {str(e)}")
            raise

    def generate_volatility_regime_signals(self, short_vol_window=12, long_vol_window=52, regime_threshold=1.2):
        """
        STRATEGY: Volatility Regime Switching
        
        Adjusts allocation based on current volatility regime relative to historical norms.
        High allocation during low vol regimes, low allocation during high vol regimes.
        """
        try:
            logger.info(f"Generating volatility regime signals (short={short_vol_window}, long={long_vol_window}, threshold={regime_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Calculate short and long-term volatility
            short_vol = data['cad_ig_volatility_4'].rolling(short_vol_window).mean()
            long_vol = data['cad_ig_volatility_4'].rolling(long_vol_window).mean()
            
            # Volatility regime indicator
            vol_ratio = short_vol / long_vol
            
            # Low volatility regime = high allocation
            low_vol_mask = vol_ratio < (1 / regime_threshold)
            positions[low_vol_mask] = 0.85
            
            # High volatility regime = low allocation
            high_vol_mask = vol_ratio > regime_threshold
            positions[high_vol_mask] = 0.15
            
            # Transitional regime = moderate allocation with momentum bias
            transition_mask = (~low_vol_mask) & (~high_vol_mask)
            momentum_signal = data['cad_oas_momentum_4'].fillna(0)
            momentum_zscore = (momentum_signal - momentum_signal.rolling(26).mean()) / momentum_signal.rolling(26).std()
            positions[transition_mask] = 0.5 - (momentum_zscore[transition_mask] * 0.2)
            
            positions = positions.clip(0, 1)
            
            stats = {
                'low_vol_periods': low_vol_mask.sum(),
                'high_vol_periods': high_vol_mask.sum(),
                'transition_periods': transition_mask.sum(),
                'avg_vol_ratio': vol_ratio.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Volatility regime stats: {stats}")
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating volatility regime signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_carry_strategy_signals(self, carry_window=26, carry_threshold=0.5):
        """
        STRATEGY: Credit Carry Strategy
        
        Uses term structure and carry indicators to determine allocation.
        High allocation when carry is attractive, low when carry is poor.
        """
        try:
            logger.info(f"Generating carry strategy signals (window={carry_window}, threshold={carry_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Carry proxy: spread level relative to history
            spread_percentile = data['cad_oas'].rolling(carry_window).rank(pct=True)
            
            # Volatility-adjusted carry
            vol_adj_carry = spread_percentile / (data['cad_ig_volatility_4'].rolling(carry_window).rank(pct=True) + 0.1)
            
            # High carry = high allocation
            high_carry_mask = vol_adj_carry > (1 - carry_threshold)
            positions[high_carry_mask] = 0.80
            
            # Low carry = low allocation
            low_carry_mask = vol_adj_carry < carry_threshold
            positions[low_carry_mask] = 0.20
            
            # Medium carry with momentum overlay
            med_carry_mask = (~high_carry_mask) & (~low_carry_mask)
            momentum = data['cad_oas_momentum_4'].fillna(0)
            momentum_norm = (momentum - momentum.rolling(carry_window).mean()) / momentum.rolling(carry_window).std()
            positions[med_carry_mask] = 0.5 - (momentum_norm[med_carry_mask] * 0.25)
            
            positions = positions.clip(0, 1)
            
            stats = {
                'high_carry_periods': high_carry_mask.sum(),
                'low_carry_periods': low_carry_mask.sum(),
                'medium_carry_periods': med_carry_mask.sum(),
                'avg_carry_score': vol_adj_carry.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Carry strategy stats: {stats}")
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating carry strategy signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_cross_asset_momentum_signals(self, momentum_window=12, signal_threshold=1.0):
        """
        STRATEGY: Cross-Asset Momentum
        
        Uses momentum across multiple assets (spreads, VIX, growth surprises) 
        to determine optimal allocation timing.
        """
        try:
            logger.info(f"Generating cross-asset momentum signals (window={momentum_window}, threshold={signal_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Multiple momentum signals
            spread_mom = data['cad_oas_momentum_4'].fillna(0)
            vix_mom = data['vix'].pct_change(momentum_window).fillna(0)
            growth_mom = data['us_growth_surprises'].pct_change(momentum_window).fillna(0)
            
            # Normalize all momentum signals
            spread_mom_z = (spread_mom - spread_mom.rolling(26).mean()) / spread_mom.rolling(26).std()
            vix_mom_z = (vix_mom - vix_mom.rolling(26).mean()) / vix_mom.rolling(26).std()
            growth_mom_z = (growth_mom - growth_mom.rolling(26).mean()) / growth_mom.rolling(26).std()
            
            # Composite momentum score (credit friendly when negative spread momentum, negative VIX momentum, positive growth)
            composite_momentum = -spread_mom_z + (-vix_mom_z) + growth_mom_z
            
            # High positive composite momentum = high allocation
            high_momentum_mask = composite_momentum > signal_threshold
            positions[high_momentum_mask] = 0.90
            
            # High negative composite momentum = low allocation
            low_momentum_mask = composite_momentum < -signal_threshold
            positions[low_momentum_mask] = 0.10
            
            # Moderate momentum with volatility adjustment
            mod_momentum_mask = (~high_momentum_mask) & (~low_momentum_mask)
            vol_adj = 1 / (1 + data['cad_ig_volatility_4'].rolling(12).mean())
            positions[mod_momentum_mask] = 0.5 + (composite_momentum[mod_momentum_mask] * 0.3 * vol_adj[mod_momentum_mask])
            
            positions = positions.clip(0, 1)
            
            stats = {
                'high_momentum_periods': high_momentum_mask.sum(),
                'low_momentum_periods': low_momentum_mask.sum(),
                'moderate_momentum_periods': mod_momentum_mask.sum(),
                'avg_composite_momentum': composite_momentum.mean(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Cross-asset momentum stats: {stats}")
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating cross-asset momentum signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_seasonal_calendar_signals(self, base_allocation=0.5, seasonal_amplitude=0.3):
        """
        STRATEGY: Seasonal/Calendar Effects
        
        Exploits known seasonal patterns in credit markets.
        Higher allocation during favorable seasons, lower during unfavorable.
        """
        try:
            logger.info(f"Generating seasonal calendar signals (base={base_allocation}, amplitude={seasonal_amplitude})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(base_allocation, index=data.index)
            
            # Extract calendar features
            months = data.index.month
            quarters = data.index.quarter
            week_of_year = data.index.isocalendar().week
            
            # Seasonal adjustments based on historical credit patterns
            # Q1: Usually weak for credit (year-end effects, supply)
            q1_mask = quarters == 1
            positions[q1_mask] *= (1 - seasonal_amplitude * 0.5)
            
            # Q2: Typically strong for credit
            q2_mask = quarters == 2
            positions[q2_mask] *= (1 + seasonal_amplitude * 0.7)
            
            # Q3: Summer doldrums, moderate
            q3_mask = quarters == 3
            positions[q3_mask] *= (1 + seasonal_amplitude * 0.2)
            
            # Q4: Usually strong, but watch December
            q4_mask = quarters == 4
            december_mask = months == 12
            positions[q4_mask & ~december_mask] *= (1 + seasonal_amplitude * 0.6)
            positions[december_mask] *= (1 - seasonal_amplitude * 0.3)
            
            # Month-end effects (reduce allocation in last week of month for supply)
            # This is a rough approximation
            month_end_mask = week_of_year % 4 == 0  # Approximate month-end weeks
            positions[month_end_mask] *= (1 - seasonal_amplitude * 0.2)
            
            # Overlay with momentum for confirmation
            momentum = data['cad_oas_momentum_4'].fillna(0)
            momentum_z = (momentum - momentum.rolling(12).mean()) / momentum.rolling(12).std()
            momentum_adj = np.tanh(momentum_z) * 0.1  # Small momentum overlay
            positions = positions - momentum_adj
            
            positions = positions.clip(0, 1)
            
            stats = {
                'q1_periods': q1_mask.sum(),
                'q2_periods': q2_mask.sum(),
                'q3_periods': q3_mask.sum(),
                'q4_periods': q4_mask.sum(),
                'month_end_periods': month_end_mask.sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Seasonal calendar stats: {stats}")
            return positions.fillna(base_allocation)
            
        except Exception as e:
            logger.error(f"Error generating seasonal calendar signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_risk_parity_signals(self, lookback_window=26, rebalance_threshold=0.05):
        """
        STRATEGY: Risk-Adjusted Allocation
        
        Adjusts allocation based on recent volatility to maintain consistent risk exposure.
        Lower allocation during high volatility periods, higher during low volatility.
        """
        try:
            logger.info(f"Generating risk parity signals (window={lookback_window}, threshold={rebalance_threshold})")
            
            data = self.weekly_data.copy()
            positions = pd.Series(0.5, index=data.index)
            
            # Target volatility (average historical volatility)
            target_vol = data['cad_ig_volatility_4'].rolling(lookback_window * 4).mean()
            current_vol = data['cad_ig_volatility_4'].rolling(lookback_window).mean()
            
            # Risk parity allocation: inverse of volatility
            risk_parity_weight = target_vol / (current_vol + 1e-6)  # Avoid division by zero
            
            # Scale to reasonable range
            risk_parity_weight = np.clip(risk_parity_weight, 0.1, 2.0)
            base_positions = 0.5 * risk_parity_weight
            
            # Momentum overlay for direction
            momentum = data['cad_oas_momentum_4'].fillna(0)
            momentum_z = (momentum - momentum.rolling(lookback_window).mean()) / momentum.rolling(lookback_window).std()
            momentum_adj = -np.tanh(momentum_z) * 0.2  # Negative because negative momentum is good for credit
            
            # Combine risk parity with momentum
            positions = base_positions + momentum_adj
            positions = positions.clip(0, 1)
            
            stats = {
                'avg_target_vol': target_vol.mean(),
                'avg_current_vol': current_vol.mean(),
                'avg_risk_parity_weight': risk_parity_weight.mean(),
                'high_vol_periods': (current_vol > target_vol * 1.2).sum(),
                'low_vol_periods': (current_vol < target_vol * 0.8).sum(),
                'avg_allocation': positions.mean()
            }
            
            logger.info(f"Risk parity stats: {stats}")
            return positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating risk parity signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def generate_ensemble_signals(self, momentum_weight=0.3, regime_weight=0.3, carry_weight=0.4):
        """
        STRATEGY: Ensemble Strategy
        
        Combines multiple signal sources with weighted averaging for more robust signals.
        """
        try:
            logger.info(f"Generating ensemble signals (momentum={momentum_weight}, regime={regime_weight}, carry={carry_weight})")
            
            data = self.weekly_data.copy()
            
            # Generate component signals
            momentum_pos = self.generate_extreme_momentum_signals(8, 3.0, 1.5)
            regime_pos = self.generate_volatility_regime_signals(12, 52, 1.2)
            carry_pos = self.generate_carry_strategy_signals(26, 0.5)
            
            # Weighted ensemble
            ensemble_positions = (
                momentum_pos * momentum_weight +
                regime_pos * regime_weight +
                carry_pos * carry_weight
            )
            
            # Normalize weights (they should sum to 1)
            total_weight = momentum_weight + regime_weight + carry_weight
            ensemble_positions = ensemble_positions / total_weight
            
            # Confidence-based adjustment (higher allocation when signals agree)
            signal_agreement = 1 - np.abs(momentum_pos - regime_pos) - np.abs(regime_pos - carry_pos) - np.abs(momentum_pos - carry_pos)
            confidence_adj = signal_agreement * 0.2
            
            final_positions = ensemble_positions + confidence_adj
            final_positions = final_positions.clip(0, 1)
            
            stats = {
                'avg_momentum_signal': momentum_pos.mean(),
                'avg_regime_signal': regime_pos.mean(),
                'avg_carry_signal': carry_pos.mean(),
                'avg_signal_agreement': signal_agreement.mean(),
                'avg_ensemble_allocation': final_positions.mean(),
                'high_confidence_periods': (signal_agreement > 0.8).sum()
            }
            
            logger.info(f"Ensemble strategy stats: {stats}")
            return final_positions.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Error generating ensemble signals: {str(e)}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def run_comprehensive_strategy_exploration(self):
        """
        Comprehensive exploration of all strategies within constraints
        """
        try:
            logger.info("="*80)
            logger.info("COMPREHENSIVE STRATEGY EXPLORATION - TARGET: >64.07%")
            logger.info("="*80)
            
            all_results = []
            self.best_sharpe = 0
            self.best_strategy = ""
            
            # CATEGORY 1: Enhanced Momentum Strategies
            logger.info("Testing Enhanced Momentum Strategies...")
            for window in [4, 8, 12, 16]:
                for amplification in [3.0, 4.0, 5.0, 6.0]:
                    for threshold in [1.0, 1.5, 2.0]:
                        try:
                            strategy_name = f"enhanced_momentum_{window}_{amplification}_{threshold}"
                            positions = self.generate_extreme_momentum_signals(window, amplification, threshold)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    self._update_best_strategy(metrics, strategy_name)
                                        
                        except Exception as e:
                            logger.warning(f"Failed enhanced momentum test {window}_{amplification}_{threshold}: {str(e)}")
                            continue

            # CATEGORY 2: Volatility Regime Strategies
            logger.info("Testing Volatility Regime Strategies...")
            for short_window in [8, 12, 16]:
                for long_window in [39, 52, 65]:
                    for threshold in [1.1, 1.2, 1.3, 1.5]:
                        try:
                            strategy_name = f"vol_regime_{short_window}_{long_window}_{threshold}"
                            positions = self.generate_volatility_regime_signals(short_window, long_window, threshold)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    self._update_best_strategy(metrics, strategy_name)
                                        
                        except Exception as e:
                            logger.warning(f"Failed vol regime test {short_window}_{long_window}_{threshold}: {str(e)}")
                            continue

            # CATEGORY 3: Carry Strategies
            logger.info("Testing Carry Strategies...")
            for carry_window in [20, 26, 39, 52]:
                for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    try:
                        strategy_name = f"carry_{carry_window}_{threshold}"
                        positions = self.generate_carry_strategy_signals(carry_window, threshold)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                self._update_best_strategy(metrics, strategy_name)
                                    
                    except Exception as e:
                        logger.warning(f"Failed carry test {carry_window}_{threshold}: {str(e)}")
                        continue

            # CATEGORY 4: Cross-Asset Momentum
            logger.info("Testing Cross-Asset Momentum Strategies...")
            for momentum_window in [8, 12, 16, 20]:
                for threshold in [0.5, 1.0, 1.5, 2.0]:
                    try:
                        strategy_name = f"cross_momentum_{momentum_window}_{threshold}"
                        positions = self.generate_cross_asset_momentum_signals(momentum_window, threshold)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                self._update_best_strategy(metrics, strategy_name)
                                    
                    except Exception as e:
                        logger.warning(f"Failed cross momentum test {momentum_window}_{threshold}: {str(e)}")
                        continue

            # CATEGORY 5: Seasonal/Calendar Strategies
            logger.info("Testing Seasonal/Calendar Strategies...")
            for base_allocation in [0.4, 0.5, 0.6]:
                for amplitude in [0.2, 0.3, 0.4, 0.5]:
                    try:
                        strategy_name = f"seasonal_{base_allocation}_{amplitude}"
                        positions = self.generate_seasonal_calendar_signals(base_allocation, amplitude)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                self._update_best_strategy(metrics, strategy_name)
                                    
                    except Exception as e:
                        logger.warning(f"Failed seasonal test {base_allocation}_{amplitude}: {str(e)}")
                        continue

            # CATEGORY 6: Risk Parity Strategies
            logger.info("Testing Risk Parity Strategies...")
            for window in [20, 26, 39, 52]:
                for threshold in [0.03, 0.05, 0.08, 0.10]:
                    try:
                        strategy_name = f"risk_parity_{window}_{threshold}"
                        positions = self.generate_risk_parity_signals(window, threshold)
                        portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                        
                        if portfolio is not None:
                            metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                            if metrics:
                                all_results.append(metrics)
                                self._update_best_strategy(metrics, strategy_name)
                                    
                    except Exception as e:
                        logger.warning(f"Failed risk parity test {window}_{threshold}: {str(e)}")
                        continue

            # CATEGORY 7: Ensemble Strategies
            logger.info("Testing Ensemble Strategies...")
            weight_combinations = [
                (0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.3, 0.4),
                (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6),
                (0.33, 0.33, 0.34), (0.7, 0.15, 0.15), (0.15, 0.7, 0.15)
            ]
            
            for i, (mom_w, reg_w, car_w) in enumerate(weight_combinations):
                try:
                    strategy_name = f"ensemble_{i+1}_{mom_w}_{reg_w}_{car_w}"
                    positions = self.generate_ensemble_signals(mom_w, reg_w, car_w)
                    portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            self._update_best_strategy(metrics, strategy_name)
                                
                except Exception as e:
                    logger.warning(f"Failed ensemble test {i+1}: {str(e)}")
                    continue

            # CATEGORY 8: Binary and Leveraged-like (Best performers from previous runs)
            logger.info("Testing Binary and Leveraged-like Strategies...")
            
            # Binary strategies
            for spread_thresh in [0.7, 0.8, 0.9]:
                for vix_thresh in [0.6, 0.7, 0.8]:
                    for mom_thresh in [1.0, 1.5, 2.0]:
                        try:
                            strategy_name = f"binary_{spread_thresh}_{vix_thresh}_{mom_thresh}"
                            positions = self.generate_extreme_binary_signals(spread_thresh, vix_thresh, mom_thresh)
                            portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                            
                            if portfolio is not None:
                                metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                                if metrics:
                                    all_results.append(metrics)
                                    self._update_best_strategy(metrics, strategy_name)
                                        
                        except Exception as e:
                            logger.warning(f"Failed binary test {spread_thresh}_{vix_thresh}_{mom_thresh}: {str(e)}")
                            continue
            
            # Leveraged-like strategies
            for amplification in [6.0, 8.0, 10.0, 12.0]:
                try:
                    strategy_name = f"leveraged_like_{amplification}"
                    positions = self.generate_leveraged_like_signals(amplification)
                    portfolio = self.run_vectorbt_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_with_quantstats(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            self._update_best_strategy(metrics, strategy_name)
                                
                except Exception as e:
                    logger.warning(f"Failed leveraged-like test {amplification}: {str(e)}")
                    continue

            # Compile and analyze results
            return self._compile_results(all_results, 64.07)
                
        except Exception as e:
            logger.error(f"Error in comprehensive strategy exploration: {str(e)}")
            raise

    def _update_best_strategy(self, metrics, strategy_name):
        """Helper function to update best strategy tracking"""
        if metrics['sharpe_ratio'] > self.best_sharpe:
            self.best_sharpe = metrics['sharpe_ratio']
            self.best_strategy = strategy_name
            logger.info(f"NEW BEST: {strategy_name} (Sharpe: {metrics['sharpe_ratio']:.3f}, Total Return: {metrics['total_return']:.2%})")
            
            if metrics['total_return'] * 100 > 64.07:
                logger.info(f"TARGET ACHIEVED! {metrics['total_return']:.2%} > 64.07%")

    def _compile_results(self, all_results, target_threshold):
        """Helper function to compile and display results"""
        if all_results:
            self.results = pd.DataFrame(all_results)
            self.results = self.results.sort_values('total_return', ascending=False)
            
            logger.info(f"Strategy exploration completed! Tested {len(all_results)} strategy variants")
            
            target_achieved = (self.results['total_return'] * 100 > target_threshold).any()
            
            if target_achieved:
                winning_strategies = self.results[self.results['total_return'] * 100 > target_threshold]
                logger.info(f"TARGET ACHIEVED! {len(winning_strategies)} strategies exceeded {target_threshold:.2f}% target!")
                
                for i, (_, row) in enumerate(winning_strategies.iterrows()):
                    logger.info(f"WINNER #{i+1}: {row['strategy_name']} | "
                              f"Total Return: {row['total_return']:8.2%} | "
                              f"Sharpe: {row['sharpe_ratio']:6.3f} | "
                              f"CAGR: {row['cagr']:7.2%} | "
                              f"MaxDD: {row['max_drawdown']:7.2%}")
            else:
                best_return = self.results.iloc[0]['total_return'] * 100
                shortfall = target_threshold - best_return
                logger.info(f"Best Result: {best_return:.2f}%, Target: {target_threshold:.2f}%, Shortfall: {shortfall:.2f}%")
            
            logger.info(f"Best overall strategy: {self.results.iloc[0]['strategy_name']} with total return: {self.results.iloc[0]['total_return']:.2%}")
            
            # Display top 20 strategies by total return
            self._display_top_strategies(target_threshold, 20)
            
            # Display top strategies by category
            self._display_category_analysis()
            
            return self.results
        else:
            logger.warning("No successful strategy iterations completed")
            return pd.DataFrame()

    def _display_top_strategies(self, target_threshold, top_n=20):
        """Display top strategies in tabular format"""
        logger.info("\n" + "="*120)
        logger.info(f"TOP {top_n} STRATEGIES BY TOTAL RETURN:")
        logger.info("="*120)
        logger.info(f"{'Rank':4s} {'Strategy Name':45s} {'Total Return':12s} {'Target Gap':10s} {'Sharpe':8s} {'CAGR':8s} {'MaxDD':8s} {'Win Rate':9s}")
        logger.info("-" * 120)
        
        for i, (_, row) in enumerate(self.results.head(top_n).iterrows()):
            total_ret_pct = row['total_return'] * 100
            gap = total_ret_pct - target_threshold
            gap_str = f"+{gap:.1f}%" if gap > 0 else f"{gap:.1f}%"
            
            logger.info(f"{i+1:4d} {row['strategy_name']:45s} "
                      f"{row['total_return']:11.2%} "
                      f"{gap_str:>9s} "
                      f"{row['sharpe_ratio']:7.3f} "
                      f"{row['cagr']:7.2%} "
                      f"{row['max_drawdown']:7.2%} "
                      f"{row['win_rate']:8.1%}")

    def _display_category_analysis(self):
        """Display performance analysis by strategy category"""
        logger.info("\n" + "="*100)
        logger.info("STRATEGY CATEGORY ANALYSIS:")
        logger.info("="*100)
        
        # Define category mappings
        categories = {
            'Enhanced Momentum': 'enhanced_momentum',
            'Volatility Regime': 'vol_regime',
            'Carry Strategy': 'carry_',
            'Cross-Asset Momentum': 'cross_momentum',
            'Seasonal/Calendar': 'seasonal',
            'Risk Parity': 'risk_parity',
            'Ensemble': 'ensemble',
            'Binary Extreme': 'binary_',
            'Leveraged-like': 'leveraged_like'
        }
        
        logger.info(f"{'Category':20s} {'Count':6s} {'Best Return':12s} {'Best Sharpe':11s} {'Avg Return':11s} {'Avg Sharpe':10s}")
        logger.info("-" * 100)
        
        for category_name, category_prefix in categories.items():
            category_strategies = self.results[self.results['strategy_name'].str.contains(category_prefix)]
            
            if not category_strategies.empty:
                best_return = category_strategies['total_return'].max()
                best_sharpe = category_strategies['sharpe_ratio'].max()
                avg_return = category_strategies['total_return'].mean()
                avg_sharpe = category_strategies['sharpe_ratio'].mean()
                count = len(category_strategies)
                
                logger.info(f"{category_name:20s} {count:6d} "
                          f"{best_return:11.2%} "
                          f"{best_sharpe:10.3f} "
                          f"{avg_return:10.2%} "
                          f"{avg_sharpe:9.3f}")


# Main execution function
def main():
    """
    Main execution function implementing the iterative backtesting process
    
    This function runs the complete pipeline:
    1. Initialize backtester with data loading
    2. Run comprehensive strategy iteration
    3. Generate detailed performance analysis
    4. Continue iterating until target performance is reached
    """
    try:
        logger.info("="*80)
        logger.info("WEEKLY CAD IG BACKTESTING - COMPREHENSIVE STRATEGY EXPLORATION")
        logger.info("="*80)
        
        # Initialize backtester
        backtester = WeeklyCADIGBacktester()
        
        # Run comprehensive strategy exploration
        results = backtester.run_comprehensive_strategy_exploration()
        
        if not results.empty:
            # Generate detailed report for top strategies
            backtester.generate_detailed_report(top_n=3)
            
            # Check if we've reached our target (e.g., Sharpe > 1.0)
            target_sharpe = 1.0
            if backtester.best_sharpe >= target_sharpe:
                logger.info(f"\n🎯 TARGET REACHED! Best Sharpe ratio: {backtester.best_sharpe:.3f}")
                logger.info(f"Best strategy: {backtester.best_strategy}")
            else:
                logger.info(f"\n📈 Current best Sharpe: {backtester.best_sharpe:.3f} (Target: {target_sharpe:.1f})")
                logger.info("Consider additional strategy refinements or parameter tuning")
                
                # Suggestions for further iteration
                logger.info("\nSuggestions for next iteration:")
                logger.info("1. Combine top-performing strategies")
                logger.info("2. Add risk management overlays")
                logger.info("3. Test regime-switching approaches")
                logger.info("4. Implement ensemble methods")
        
        logger.info("\n" + "="*80)
        logger.info("BACKTESTING ITERATION COMPLETED")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main() 