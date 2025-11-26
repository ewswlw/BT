"""
Machine Learning Based CDX Trading Strategies
Use ML to discover predictive patterns in the data
Goal: 3.81% CAGR
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from strategies.base_strategy import BaseStrategy
from core.backtest_engine import BacktestEngine

OUTPUT_DIR = Path(__file__).parent / "outputs" / "ml_iterations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import from previous script
exec(open('iterative_strategy_development.py').read().split('def main():')[0])


class Iteration13Strategy(BaseStrategy):
    """
    Iteration 13: Random Forest - Predict positive return periods
    Train RF to predict when next week will be positive
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration13_RandomForest"
        self.description = "RF model predicting positive periods"
        self.model = None
        self.scaler = StandardScaler()

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """ML-based signals"""

        # Create target: forward return > 0
        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        # Select numeric features only
        feature_cols = [col for col in train_features.columns if train_features[col].dtype in [np.float64, np.int64, np.float32, np.int32]]

        # Remove inf and fill NaN
        train_features_clean = train_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        # Align
        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        # Scale
        X_train_scaled = self.scaler.fit_transform(train_features_clean)
        X_test_scaled = self.scaler.transform(test_features_clean)

        # Train RF
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )

        self.model.fit(X_train_scaled, train_target_clean)

        # Predict
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]

        # Entry when probability > 0.5
        entry_signals = pd.Series(predictions > 0.5, index=test_features.index)

        # Exit when probability < 0.45
        exit_signals = pd.Series(predictions < 0.45, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration14Strategy(BaseStrategy):
    """
    Iteration 14: Gradient Boosting - More aggressive ML
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration14_GradientBoosting"
        self.description = "GB model with feature selection"
        self.model = None
        self.scaler = StandardScaler()

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """GB-based signals"""

        # Create target: forward return > 0
        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        # Select key features based on domain knowledge
        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'us_2s10s',
                                      'fci', 'rate_vol', 'us_lei']):
                if train_features[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    key_features.append(col)

        if len(key_features) == 0:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:20]

        # Clean data
        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        # Align
        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        # Scale
        X_train_scaled = self.scaler.fit_transform(train_features_clean)
        X_test_scaled = self.scaler.transform(test_features_clean)

        # Train GB
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )

        self.model.fit(X_train_scaled, train_target_clean)

        # Predict
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]

        # Entry when probability > 0.52
        entry_signals = pd.Series(predictions > 0.52, index=test_features.index)

        # Exit when probability < 0.48
        exit_signals = pd.Series(predictions < 0.48, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration15Strategy(BaseStrategy):
    """
    Iteration 15: Simple but extreme - only invest on strongest days
    Use ensemble of signals to find highest conviction periods
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration15_HighConviction"
        self.description = "Only invest in highest conviction periods"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """High conviction only"""

        features = test_features.copy()

        # Calculate conviction score (0-10)
        conviction = pd.Series(0, index=features.index)

        # Score 1: Spread momentum very positive
        if 'ig_cdx_ret_21d' in features.columns:
            spread_mom_strong = features['ig_cdx_ret_21d'] < -0.02  # Tightening > 2%
            conviction += spread_mom_strong.astype(int) * 2  # Weight 2

        # Score 2: VIX very low
        if 'vix' in features.columns:
            vix_low = features['vix'] < 15
            conviction += vix_low.astype(int) * 2

        # Score 3: SPX strong momentum
        if 'spx_ret_21d' in features.columns:
            spx_mom = features['spx_ret_21d'] > 0.02
            conviction += spx_mom.astype(int) * 2

        # Score 4: Growth surprises positive
        if 'us_growth_surprises' in features.columns:
            growth_pos = features['us_growth_surprises'] > 0.2
            conviction += growth_pos.astype(int)

        # Score 5: Curve steep
        if 'us_2s10s_spread' in features.columns:
            curve_steep = features['us_2s10s_spread'] > 1.0
            conviction += curve_steep.astype(int)

        # Score 6: FCI easing
        if 'us_bloomberg_fci' in features.columns:
            fci_easy = features['us_bloomberg_fci'].diff(21) > 0
            conviction += fci_easy.astype(int)

        # Score 7: IG spreads tight
        if 'ig_cdx_zscore_63d' in features.columns:
            spreads_tight = features['ig_cdx_zscore_63d'] < -0.5
            conviction += spreads_tight.astype(int)

        # Only enter on VERY high conviction (>= 6 out of 10)
        entry_signals = conviction >= 6

        # Exit when conviction drops below 3
        exit_signals = conviction < 3

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration16Strategy(BaseStrategy):
    """
    Iteration 16: Statistical Arbitrage - exploit mean reversion at extremes
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration16_StatArb"
        self.description = "Mean reversion at statistical extremes"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Statistical arbitrage signals"""

        features = test_features.copy()

        # Find extreme deviations in spread
        if 'ig_cdx' in features.columns:
            # Calculate percentile rank over 252 days
            spread_percentile = features['ig_cdx'].rolling(252).apply(
                lambda x: np.percentile(x, 75) if len(x) > 20 else x.iloc[-1]
            )

            # Buy when spreads spike above 75th percentile (mean revert down)
            extreme_wide = features['ig_cdx'] > spread_percentile

            # But also check if starting to compress
            spread_change_recent = features['ig_cdx'].diff(3)
            compressing = spread_change_recent < 0

            entry_signal = extreme_wide & compressing

        else:
            entry_signal = pd.Series(False, index=features.index)

        # Additional filter: VIX not exploding
        if 'vix' in features.columns:
            vix_ok = features['vix'] < 30
            entry_signal = entry_signal & vix_ok

        # Exit when spreads compress below median
        if 'ig_cdx' in features.columns:
            spread_median = features['ig_cdx'].rolling(252).median()
            exit_signal = features['ig_cdx'] < spread_median
        else:
            exit_signal = ~entry_signal

        return entry_signal.fillna(False), exit_signal.fillna(False)


class Iteration17Strategy(BaseStrategy):
    """
    Iteration 17: Just avoid the worst 10% of periods
    Stay invested 90% of the time, avoid worst periods
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration17_AvoidWorst10Pct"
        self.description = "Invest 90% of time, avoid worst periods"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Avoid worst 10%"""

        features = test_features.copy()

        # Create composite risk score
        risk_score = pd.Series(0, index=features.index)

        # Risk factor 1: VIX percentile
        if 'vix' in features.columns:
            vix_pct = features['vix'].rolling(252).apply(
                lambda x: np.percentile(x, 80) if len(x) > 20 else x.iloc[-1]
            )
            risk_score += (features['vix'] > vix_pct).astype(int) * 3

        # Risk factor 2: Spread widening percentile
        if 'ig_cdx' in features.columns:
            spread_chg = features['ig_cdx'].pct_change(5)
            spread_chg_pct90 = spread_chg.rolling(252).quantile(0.90)
            risk_score += (spread_chg > spread_chg_pct90).astype(int) * 3

        # Risk factor 3: SPX drawdown
        if 'spx_tr' in features.columns:
            spx_dd = (features['spx_tr'] / features['spx_tr'].rolling(252).max()) - 1
            risk_score += (spx_dd < -0.08).astype(int) * 2

        # Risk factor 4: Growth collapse
        if 'us_growth_surprises' in features.columns:
            risk_score += (features['us_growth_surprises'] < -0.6).astype(int) * 2

        # Exit only when risk score >= 5 (worst ~10% of periods)
        exit_signals = risk_score >= 5

        # Stay invested otherwise
        entry_signals = ~exit_signals

        return entry_signals.fillna(True), exit_signals.fillna(False)


def run_ml_iterations():
    """Run ML-based iterations"""

    print("\n" + "="*80)
    print("MACHINE LEARNING ITERATION STRATEGY DEVELOPMENT")
    print("="*80)

    # Load data
    data = DataLoader.load_data()
    baseline_results, baseline_prices = BaselineAnalysis.calculate_baseline(data)
    features = FeatureEngineering.create_features(data)
    features = features.fillna(method='ffill').fillna(method='bfill')

    common_idx = data.index.intersection(features.index)
    data = data.loc[common_idx]
    features = features.loc[common_idx]

    # Define ML iterations
    iterations = [
        (13, Iteration13Strategy, {'name': 'Iteration13_RandomForest', 'holding_period_days': 7}),
        (14, Iteration14Strategy, {'name': 'Iteration14_GradientBoosting', 'holding_period_days': 7}),
        (15, Iteration15Strategy, {'name': 'Iteration15_HighConviction', 'holding_period_days': 7}),
        (16, Iteration16Strategy, {'name': 'Iteration16_StatArb', 'holding_period_days': 7}),
        (17, Iteration17Strategy, {'name': 'Iteration17_AvoidWorst10Pct', 'holding_period_days': 7}),
    ]

    results = []

    for iter_num, strategy_class, config in iterations:
        try:
            result, validation, output_dir = run_iteration(
                iter_num, strategy_class, config, data, features,
                baseline_results, baseline_prices
            )

            results.append({
                'iteration': iter_num,
                'name': config['name'],
                'cagr': result.metrics['cagr'],
                'outperformance': validation['outperformance'],
                'sharpe': result.metrics['sharpe_ratio'],
                'max_dd': result.metrics['max_drawdown'],
                'trades': result.trades_count,
                'time_in_market': result.time_in_market,
                'target_achieved': validation['target_achieved']
            })
        except Exception as e:
            print(f"\n❌ Iteration {iter_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary report
    print("\n" + "="*80)
    print("SUMMARY OF ML ITERATIONS")
    print("="*80)
    print(f"\nBaseline CAGR: {baseline_results['cagr']*100:.2f}%")
    print(f"Target CAGR: {(baseline_results['cagr'] + 0.025)*100:.2f}%\n")

    summary_df = pd.DataFrame(results)

    if len(summary_df) > 0:
        summary_df['cagr_pct'] = summary_df['cagr'] * 100
        summary_df['outperf_pct'] = summary_df['outperformance'] * 100
        summary_df['max_dd_pct'] = summary_df['max_dd'] * 100
        summary_df['time_in_mkt_pct'] = summary_df['time_in_market'] * 100

        print(summary_df[['iteration', 'name', 'cagr_pct', 'outperf_pct',
                         'sharpe', 'max_dd_pct', 'trades', 'time_in_mkt_pct', 'target_achieved']].to_string(index=False))

        # Save summary
        summary_df.to_csv(OUTPUT_DIR / "ml_iterations_summary.csv", index=False)

        # Check if any achieved target
        winners = summary_df[summary_df['target_achieved']]

        if len(winners) > 0:
            print("\n" + "="*80)
            print("🎯 SUCCESS! The following strategies achieved the target:")
            print("="*80)
            for idx, row in winners.iterrows():
                print(f"  Iteration {row['iteration']}: {row['name']}")
                print(f"    CAGR: {row['cagr']*100:.2f}%")
                print(f"    Outperformance: {row['outperformance']*100:.2f}%")
                print(f"    Time in Market: {row['time_in_market']*100:.2f}%\n")

            # Save winning strategy details
            with open(OUTPUT_DIR / "WINNING_STRATEGIES.txt", 'w') as f:
                f.write("WINNING STRATEGIES THAT BEAT BASELINE BY 2.5%+\n")
                f.write("="*80 + "\n\n")
                for idx, row in winners.iterrows():
                    f.write(f"Iteration {row['iteration']}: {row['name']}\n")
                    f.write(f"  CAGR: {row['cagr']*100:.2f}%\n")
                    f.write(f"  Outperformance: {row['outperformance']*100:.2f}%\n")
                    f.write(f"  Sharpe Ratio: {row['sharpe']:.2f}\n")
                    f.write(f"  Max Drawdown: {row['max_dd']*100:.2f}%\n")
                    f.write(f"  Time in Market: {row['time_in_market']*100:.2f}%\n")
                    f.write(f"  Number of Trades: {row['trades']}\n\n")

        else:
            print("\n⚠️  No strategy achieved the target yet.")

            if len(summary_df) > 0:
                best_idx = summary_df['outperformance'].idxmax()
                best = summary_df.loc[best_idx]
                print(f"\nBest performer:")
                print(f"  Iteration {best['iteration']}: {best['name']}")
                print(f"  CAGR: {best['cagr']*100:.2f}%")
                print(f"  Outperformance: {best['outperformance']*100:.2f}%")
                print(f"  Still need: {(0.025 - best['outperformance'])*100:.2f}% more")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_ml_iterations()
