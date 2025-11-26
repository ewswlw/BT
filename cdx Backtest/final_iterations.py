"""
Final Advanced CDX Trading Strategies
Building on GB success (1.75% CAGR, 0.44% outperformance)
Goal: 3.81% CAGR (2.5% above baseline 1.31%)
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from strategies.base_strategy import BaseStrategy
from core.backtest_engine import BacktestEngine

OUTPUT_DIR = Path(__file__).parent / "outputs" / "final_iterations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import from previous script
exec(open('iterative_strategy_development.py').read().split('def main():')[0])


class Iteration18Strategy(BaseStrategy):
    """
    Iteration 18: Optimized GB with different threshold
    Lower entry threshold to capture more opportunities
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration18_GB_LowThreshold"
        self.description = "GB with lower entry threshold (0.48)"
        self.model = None
        self.scaler = StandardScaler()

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """GB with low threshold"""

        # Create target
        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        # Select key features
        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'us_2s10s',
                                      'fci', 'rate_vol', 'us_lei', 'hy_cdx']):
                if train_features[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    key_features.append(col)

        if len(key_features) == 0:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:30]

        # Clean
        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        # Align
        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        # Scale
        X_train_scaled = self.scaler.fit_transform(train_features_clean)
        X_test_scaled = self.scaler.transform(test_features_clean)

        # Train GB with more trees
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=40,
            min_samples_leaf=15,
            random_state=42
        )

        self.model.fit(X_train_scaled, train_target_clean)

        # Predict
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]

        # LOWER threshold - be more aggressive
        entry_signals = pd.Series(predictions > 0.48, index=test_features.index)
        exit_signals = pd.Series(predictions < 0.45, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration19Strategy(BaseStrategy):
    """
    Iteration 19: GB Ensemble - train multiple models and average
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration19_GB_Ensemble"
        self.description = "Ensemble of GB models"
        self.models = []
        self.scalers = []

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Ensemble GB"""

        # Create target
        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        # Select features
        feature_cols = [col for col in train_features.columns
                       if train_features[col].dtype in [np.float64, np.int64]][:40]

        train_features_clean = train_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        # Train 3 different GB models
        model_configs = [
            {'n_estimators': 80, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05},
            {'n_estimators': 120, 'max_depth': 5, 'learning_rate': 0.03},
        ]

        ensemble_predictions = []

        for config in model_configs:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_features_clean)
            X_test_scaled = scaler.transform(test_features_clean)

            model = GradientBoostingClassifier(
                min_samples_split=40,
                min_samples_leaf=15,
                random_state=42,
                **config
            )

            model.fit(X_train_scaled, train_target_clean)
            preds = model.predict_proba(X_test_scaled)[:, 1]
            ensemble_predictions.append(preds)

            self.models.append(model)
            self.scalers.append(scaler)

        # Average predictions
        avg_predictions = np.mean(ensemble_predictions, axis=0)

        # Signals
        entry_signals = pd.Series(avg_predictions > 0.50, index=test_features.index)
        exit_signals = pd.Series(avg_predictions < 0.47, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration20Strategy(BaseStrategy):
    """
    Iteration 20: GB + Rule Filter
    Combine GB predictions with rule-based risk filter
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration20_GB_RuleFilter"
        self.description = "GB predictions + risk filter"
        self.model = None
        self.scaler = StandardScaler()

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """GB + filter"""

        # Train GB part (same as Iteration 14)
        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'us_2s10s', 'fci']):
                if train_features[col].dtype in [np.float64, np.int64]:
                    key_features.append(col)

        if len(key_features) == 0:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:20]

        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        X_train_scaled = self.scaler.fit_transform(train_features_clean)
        X_test_scaled = self.scaler.transform(test_features_clean)

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=40,
            min_samples_leaf=15,
            random_state=42
        )

        self.model.fit(X_train_scaled, train_target_clean)
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]

        # ML signals
        ml_entry = predictions > 0.50

        # Rule-based filter: avoid extreme stress
        features = test_features.copy()

        stress_filter = pd.Series(True, index=features.index)

        # Not extreme VIX
        if 'vix' in features.columns:
            stress_filter &= features['vix'] < 40

        # Not extreme spread widening
        if 'ig_cdx' in features.columns:
            spread_chg = features['ig_cdx'].pct_change(5)
            stress_filter &= spread_chg < spread_chg.rolling(252).quantile(0.95)

        # Not deep equity drawdown
        if 'spx_tr' in features.columns:
            spx_dd = (features['spx_tr'] / features['spx_tr'].rolling(252).max()) - 1
            stress_filter &= spx_dd > -0.15

        # Combine: ML says yes AND not in extreme stress
        entry_signals = pd.Series(ml_entry & stress_filter, index=test_features.index)

        # Exit when ML says no OR in stress
        exit_signals = pd.Series((predictions < 0.47) | (~stress_filter), index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration21Strategy(BaseStrategy):
    """
    Iteration 21: Longer Holding Period (14 days)
    Reduce trading frequency, capture longer trends
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration21_GB_LongHold"
        self.description = "GB with 14-day holding period"
        self.model = None
        self.scaler = StandardScaler()

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """GB with longer horizon"""

        # Create target for 14-day forward return
        train_target = train_data['us_ig_cdx_er_index'].pct_change(14).shift(-14) > 0

        # Features
        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'us_2s10s', 'fci']):
                if train_features[col].dtype in [np.float64, np.int64]:
                    key_features.append(col)

        if len(key_features) == 0:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:20]

        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        X_train_scaled = self.scaler.fit_transform(train_features_clean)
        X_test_scaled = self.scaler.transform(test_features_clean)

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )

        self.model.fit(X_train_scaled, train_target_clean)
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]

        entry_signals = pd.Series(predictions > 0.51, index=test_features.index)
        exit_signals = pd.Series(predictions < 0.49, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration22Strategy(BaseStrategy):
    """
    Iteration 22: Ultra Aggressive - Maximum Time in Market
    GB with very low threshold to maximize exposure
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration22_GB_MaxExposure"
        self.description = "GB with maximum exposure (threshold 0.40)"
        self.model = None
        self.scaler = StandardScaler()

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """GB with very low threshold"""

        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'us_2s10s', 'fci']):
                if train_features[col].dtype in [np.float64, np.int64]:
                    key_features.append(col)

        if len(key_features) == 0:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:25]

        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        X_train_scaled = self.scaler.fit_transform(train_features_clean)
        X_test_scaled = self.scaler.transform(test_features_clean)

        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            min_samples_split=30,
            min_samples_leaf=10,
            random_state=42
        )

        self.model.fit(X_train_scaled, train_target_clean)
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]

        # VERY LOW threshold to maximize time in market
        entry_signals = pd.Series(predictions > 0.40, index=test_features.index)
        exit_signals = pd.Series(predictions < 0.35, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


def run_final_iterations():
    """Run final advanced iterations"""

    print("\n" + "="*80)
    print("FINAL ADVANCED ITERATION STRATEGY DEVELOPMENT")
    print("Building on GB success (Iteration 14: 1.75% CAGR, 0.44% outperformance)")
    print("="*80)

    # Load data
    data = DataLoader.load_data()
    baseline_results, baseline_prices = BaselineAnalysis.calculate_baseline(data)
    features = FeatureEngineering.create_features(data)
    features = features.fillna(method='ffill').fillna(method='bfill')

    common_idx = data.index.intersection(features.index)
    data = data.loc[common_idx]
    features = features.loc[common_idx]

    # Define final iterations
    iterations = [
        (18, Iteration18Strategy, {'name': 'Iteration18_GB_LowThreshold', 'holding_period_days': 7}),
        (19, Iteration19Strategy, {'name': 'Iteration19_GB_Ensemble', 'holding_period_days': 7}),
        (20, Iteration20Strategy, {'name': 'Iteration20_GB_RuleFilter', 'holding_period_days': 7}),
        (21, Iteration21Strategy, {'name': 'Iteration21_GB_LongHold', 'holding_period_days': 14}),
        (22, Iteration22Strategy, {'name': 'Iteration22_GB_MaxExposure', 'holding_period_days': 7}),
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

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF FINAL ITERATIONS")
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
        summary_df.to_csv(OUTPUT_DIR / "final_iterations_summary.csv", index=False)

        # Check winners
        winners = summary_df[summary_df['target_achieved']]

        if len(winners) > 0:
            print("\n" + "="*80)
            print("🎯 SUCCESS! The following strategies ACHIEVED THE TARGET:")
            print("="*80)
            for idx, row in winners.iterrows():
                print(f"\nIteration {row['iteration']}: {row['name']}")
                print(f"  CAGR: {row['cagr']*100:.2f}%")
                print(f"  Outperformance vs Baseline: +{row['outperformance']*100:.2f}%")
                print(f"  Sharpe Ratio: {row['sharpe']:.2f}")
                print(f"  Max Drawdown: {row['max_dd']*100:.2f}%")
                print(f"  Time in Market: {row['time_in_market']*100:.2f}%")
                print(f"  Number of Trades: {row['trades']}")

            # Save winning strategies
            with open(OUTPUT_DIR / "WINNING_STRATEGIES.txt", 'w') as f:
                f.write("🎯 WINNING STRATEGIES - Beat Baseline by 2.5%+ Annualized\n")
                f.write("="*80 + "\n\n")
                f.write(f"Baseline Buy-and-Hold CAGR: {baseline_results['cagr']*100:.2f}%\n")
                f.write(f"Target CAGR: {(baseline_results['cagr'] + 0.025)*100:.2f}%\n\n")

                for idx, row in winners.iterrows():
                    f.write(f"Iteration {row['iteration']}: {row['name']}\n")
                    f.write(f"  CAGR: {row['cagr']*100:.2f}%\n")
                    f.write(f"  Outperformance: +{row['outperformance']*100:.2f}%\n")
                    f.write(f"  Sharpe Ratio: {row['sharpe']:.2f}\n")
                    f.write(f"  Max Drawdown: {row['max_dd']*100:.2f}%\n")
                    f.write(f"  Time in Market: {row['time_in_market']*100:.2f}%\n")
                    f.write(f"  Number of Trades: {row['trades']}\n\n")

            print(f"\n✅ Winning strategy details saved to: {OUTPUT_DIR / 'WINNING_STRATEGIES.txt'}")

        else:
            print("\n⚠️  No strategy achieved the 2.5% outperformance target yet.")

            if len(summary_df) > 0:
                best_idx = summary_df['outperformance'].idxmax()
                best = summary_df.loc[best_idx]
                print(f"\nBest performer so far:")
                print(f"  Iteration {best['iteration']}: {best['name']}")
                print(f"  CAGR: {best['cagr']*100:.2f}%")
                print(f"  Outperformance: +{best['outperformance']*100:.2f}%")
                print(f"  Still need: {(0.025 - best['outperformance'])*100:.2f}% more to hit target")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_final_iterations()
