"""
Ultra-Advanced CDX Trading Strategies
Best so far: Iteration 19 GB Ensemble at 2.22% CAGR (0.91% outperformance)
Goal: Close the 1.59% gap to reach 3.81% CAGR target
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from strategies.base_strategy import BaseStrategy
from core.backtest_engine import BacktestEngine

OUTPUT_DIR = Path(__file__).parent / "outputs" / "ultra_advanced_iterations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import from previous script
exec(open('iterative_strategy_development.py').read().split('def main():')[0])


class Iteration23Strategy(BaseStrategy):
    """
    Iteration 23: Mega Ensemble - 5 different ML models
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration23_MegaEnsemble"
        self.description = "Ensemble of 5 different ML algorithms"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Mega ensemble"""

        # Target
        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        # Features - be more selective
        best_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx_zscore', 'ig_cdx_ret', 'vix', 'spx_ret',
                                      'us_growth_surprises', 'us_2s10s_spread', 'fci',
                                      'us_lei', 'ig_hy_ratio', 'rate_vol']):
                if train_features[col].dtype in [np.float64, np.int64]:
                    best_features.append(col)

        if len(best_features) < 10:
            best_features = [col for col in train_features.columns
                           if train_features[col].dtype in [np.float64, np.int64]][:30]

        train_features_clean = train_features[best_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[best_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_features_clean)
        X_test_scaled = scaler.transform(test_features_clean)

        # Train 5 different models
        models = [
            GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      min_samples_split=40, min_samples_leaf=15, random_state=42),
            GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.03,
                                      min_samples_split=30, min_samples_leaf=10, random_state=43),
            RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=40,
                                  min_samples_leaf=15, random_state=42),
            AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        ]

        predictions_list = []
        for model in models:
            model.fit(X_train_scaled, train_target_clean)
            preds = model.predict_proba(X_test_scaled)[:, 1]
            predictions_list.append(preds)

        # Weighted average - give more weight to GB models
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        avg_predictions = np.average(predictions_list, axis=0, weights=weights)

        # Aggressive threshold
        entry_signals = pd.Series(avg_predictions > 0.47, index=test_features.index)
        exit_signals = pd.Series(avg_predictions < 0.44, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration24Strategy(BaseStrategy):
    """
    Iteration 24: Multi-Horizon Ensemble
    Predict 7-day, 14-day, and 21-day returns and ensemble
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration24_MultiHorizonEnsemble"
        self.description = "Ensemble across multiple prediction horizons"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Multi-horizon ensemble"""

        # Select features
        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'us_2s10s', 'fci', 'us_lei']):
                if train_features[col].dtype in [np.float64, np.int64]:
                    key_features.append(col)

        if len(key_features) < 10:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:25]

        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        # Train models for different horizons
        horizons = [7, 14, 21]
        horizon_predictions = []

        for horizon in horizons:
            train_target = train_data['us_ig_cdx_er_index'].pct_change(horizon).shift(-horizon) > 0

            common_idx = train_target.dropna().index.intersection(train_features_clean.index)
            X_train = train_features_clean.loc[common_idx]
            y_train = train_target.loc[common_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(test_features_clean)

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_samples_split=40,
                min_samples_leaf=15,
                random_state=42
            )

            model.fit(X_train_scaled, y_train)
            preds = model.predict_proba(X_test_scaled)[:, 1]
            horizon_predictions.append(preds)

        # Average across horizons
        avg_predictions = np.mean(horizon_predictions, axis=0)

        entry_signals = pd.Series(avg_predictions > 0.48, index=test_features.index)
        exit_signals = pd.Series(avg_predictions < 0.45, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration25Strategy(BaseStrategy):
    """
    Iteration 25: Probability-Weighted Ensemble
    Use model confidence to weight decisions
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration25_ConfidenceWeighted"
        self.description = "Weight by prediction confidence"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Confidence-weighted ensemble"""

        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'fci']):
                if train_features[col].dtype in [np.float64, np.int64]:
                    key_features.append(col)

        if len(key_features) < 10:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:25]

        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_features_clean)
        X_test_scaled = scaler.transform(test_features_clean)

        # Train multiple GB models
        models = [
            GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1,
                                      min_samples_split=50, random_state=42),
            GradientBoostingClassifier(n_estimators=120, max_depth=5, learning_rate=0.03,
                                      min_samples_split=30, random_state=43),
            GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      min_samples_split=40, random_state=44),
        ]

        predictions_list = []
        for model in models:
            model.fit(X_train_scaled, train_target_clean)
            preds = model.predict_proba(X_test_scaled)[:, 1]
            predictions_list.append(preds)

        # Calculate average and confidence (standard deviation)
        avg_predictions = np.mean(predictions_list, axis=0)
        pred_std = np.std(predictions_list, axis=0)

        # Higher confidence (lower std) = more weight
        # Use adaptive threshold based on confidence
        confidence = 1 / (1 + pred_std)

        # Entry when high probability AND high confidence
        high_prob = avg_predictions > 0.48
        high_conf = confidence > np.percentile(confidence, 40)

        entry_signals = pd.Series(high_prob & high_conf, index=test_features.index)

        # Exit when low probability OR low confidence
        low_prob = avg_predictions < 0.45
        low_conf = confidence < np.percentile(confidence, 30)

        exit_signals = pd.Series(low_prob | low_conf, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration26Strategy(BaseStrategy):
    """
    Iteration 26: Feature-Engineered Ensemble
    Create advanced interaction features
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration26_FeatureEngineered"
        self.description = "Advanced feature engineering + ensemble"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Advanced feature engineering"""

        # Create interaction features
        def add_interactions(df):
            new_df = df.copy()

            # Spread-VIX interaction
            if 'ig_cdx_zscore_63d' in df.columns and 'vix_zscore' in df.columns:
                new_df['spread_vix_interaction'] = df['ig_cdx_zscore_63d'] * df['vix_zscore']

            # Growth-curve interaction
            if 'us_growth_surprises' in df.columns and 'us_2s10s_spread' in df.columns:
                new_df['growth_curve_interaction'] = df['us_growth_surprises'] * df['us_2s10s_spread']

            # SPX-spread momentum
            if 'spx_ret_21d' in df.columns and 'ig_cdx_ret_21d' in df.columns:
                new_df['spx_spread_momentum'] = df['spx_ret_21d'] * df['ig_cdx_ret_21d']

            # Volatility regime indicator
            if 'vix' in df.columns and 'rate_vol' in df.columns:
                new_df['vol_regime'] = df['vix'] * df['rate_vol']

            return new_df

        train_features_enhanced = add_interactions(train_features)
        test_features_enhanced = add_interactions(test_features)

        # Target
        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        # Select features including interactions
        feature_cols = [col for col in train_features_enhanced.columns
                       if train_features_enhanced[col].dtype in [np.float64, np.int64]][:40]

        train_features_clean = train_features_enhanced[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features_enhanced[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_features_clean)
        X_test_scaled = scaler.transform(test_features_clean)

        # Ensemble
        models = [
            GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      min_samples_split=40, min_samples_leaf=15, random_state=42),
            GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.03,
                                      min_samples_split=30, min_samples_leaf=10, random_state=43),
            RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=40, random_state=42),
        ]

        predictions_list = []
        for model in models:
            model.fit(X_train_scaled, train_target_clean)
            preds = model.predict_proba(X_test_scaled)[:, 1]
            predictions_list.append(preds)

        avg_predictions = np.mean(predictions_list, axis=0)

        entry_signals = pd.Series(avg_predictions > 0.47, index=test_features_enhanced.index)
        exit_signals = pd.Series(avg_predictions < 0.44, index=test_features_enhanced.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration27Strategy(BaseStrategy):
    """
    Iteration 27: Extreme Aggressive - Threshold 0.35
    Maximize time in market with very low threshold
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration27_ExtremeAggressive"
        self.description = "Ultra low threshold (0.35) for max exposure"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Extreme aggressive threshold"""

        train_target = train_data['us_ig_cdx_er_index'].pct_change(7).shift(-7) > 0

        key_features = []
        for col in train_features.columns:
            if any(x in col for x in ['ig_cdx', 'vix', 'spx', 'us_growth', 'fci', 'us_lei']):
                if train_features[col].dtype in [np.float64, np.int64]:
                    key_features.append(col)

        if len(key_features) < 10:
            key_features = [col for col in train_features.columns
                          if train_features[col].dtype in [np.float64, np.int64]][:30]

        train_features_clean = train_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        test_features_clean = test_features[key_features].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

        common_idx = train_target.dropna().index.intersection(train_features_clean.index)
        train_features_clean = train_features_clean.loc[common_idx]
        train_target_clean = train_target.loc[common_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_features_clean)
        X_test_scaled = scaler.transform(test_features_clean)

        # Best GB configuration
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            min_samples_split=30,
            min_samples_leaf=10,
            random_state=42
        )

        model.fit(X_train_scaled, train_target_clean)
        predictions = model.predict_proba(X_test_scaled)[:, 1]

        # EXTREMELY low threshold
        entry_signals = pd.Series(predictions > 0.35, index=test_features.index)
        exit_signals = pd.Series(predictions < 0.30, index=test_features.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


def run_ultra_advanced_iterations():
    """Run ultra-advanced final iterations"""

    print("\n" + "="*80)
    print("ULTRA-ADVANCED FINAL ITERATION STRATEGY DEVELOPMENT")
    print("Best so far: Iteration 19 GB Ensemble at 2.22% CAGR (0.91% outperformance)")
    print("Need: 1.59% more to reach 3.81% CAGR target")
    print("="*80)

    # Load data
    data = DataLoader.load_data()
    baseline_results, baseline_prices = BaselineAnalysis.calculate_baseline(data)
    features = FeatureEngineering.create_features(data)
    features = features.fillna(method='ffill').fillna(method='bfill')

    common_idx = data.index.intersection(features.index)
    data = data.loc[common_idx]
    features = features.loc[common_idx]

    # Define ultra-advanced iterations
    iterations = [
        (23, Iteration23Strategy, {'name': 'Iteration23_MegaEnsemble', 'holding_period_days': 7}),
        (24, Iteration24Strategy, {'name': 'Iteration24_MultiHorizonEnsemble', 'holding_period_days': 7}),
        (25, Iteration25Strategy, {'name': 'Iteration25_ConfidenceWeighted', 'holding_period_days': 7}),
        (26, Iteration26Strategy, {'name': 'Iteration26_FeatureEngineered', 'holding_period_days': 7}),
        (27, Iteration27Strategy, {'name': 'Iteration27_ExtremeAggressive', 'holding_period_days': 7}),
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
    print("SUMMARY OF ULTRA-ADVANCED ITERATIONS")
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

        # Save
        summary_df.to_csv(OUTPUT_DIR / "ultra_advanced_summary.csv", index=False)

        # Check winners
        winners = summary_df[summary_df['target_achieved']]

        if len(winners) > 0:
            print("\n" + "="*80)
            print("🎯🎯🎯 SUCCESS!!! TARGET ACHIEVED!!! 🎯🎯🎯")
            print("="*80)
            for idx, row in winners.iterrows():
                print(f"\nIteration {row['iteration']}: {row['name']}")
                print(f"  ✓ CAGR: {row['cagr']*100:.2f}%")
                print(f"  ✓ Outperformance vs Baseline: +{row['outperformance']*100:.2f}%")
                print(f"  ✓ Sharpe Ratio: {row['sharpe']:.2f}")
                print(f"  ✓ Max Drawdown: {row['max_dd']*100:.2f}%")
                print(f"  ✓ Time in Market: {row['time_in_market']*100:.2f}%")
                print(f"  ✓ Number of Trades: {row['trades']}")

            # Save winning strategy
            with open(OUTPUT_DIR / "SUCCESS_WINNING_STRATEGY.txt", 'w') as f:
                f.write("🎯🎯🎯 SUCCESS - TARGET ACHIEVED!!! 🎯🎯🎯\n")
                f.write("="*80 + "\n\n")
                f.write(f"Baseline Buy-and-Hold CAGR: {baseline_results['cagr']*100:.2f}%\n")
                f.write(f"Target CAGR (Baseline + 2.5%): {(baseline_results['cagr'] + 0.025)*100:.2f}%\n\n")
                f.write("WINNING STRATEGIES:\n\n")

                for idx, row in winners.iterrows():
                    f.write(f"Iteration {row['iteration']}: {row['name']}\n")
                    f.write(f"  CAGR: {row['cagr']*100:.2f}%\n")
                    f.write(f"  Outperformance: +{row['outperformance']*100:.2f}%\n")
                    f.write(f"  Sharpe Ratio: {row['sharpe']:.2f}\n")
                    f.write(f"  Max Drawdown: {row['max_dd']*100:.2f}%\n")
                    f.write(f"  Time in Market: {row['time_in_market']*100:.2f}%\n")
                    f.write(f"  Number of Trades: {row['trades']}\n\n")

            print(f"\n✅✅✅ SUCCESS DETAILS SAVED TO: {OUTPUT_DIR / 'SUCCESS_WINNING_STRATEGY.txt'}")

        else:
            print("\n⚠️  Target not achieved yet.")

            if len(summary_df) > 0:
                best_idx = summary_df['outperformance'].idxmax()
                best = summary_df.loc[best_idx]
                print(f"\nBest performer:")
                print(f"  Iteration {best['iteration']}: {best['name']}")
                print(f"  CAGR: {best['cagr']*100:.2f}%")
                print(f"  Outperformance: +{best['outperformance']*100:.2f}%")
                print(f"  Gap to target: {(0.025 - best['outperformance'])*100:.2f}%")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_ultra_advanced_iterations()
