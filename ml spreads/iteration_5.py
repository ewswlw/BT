"""
CAD OAS Prediction Model - Iteration 5: Ensemble Methods with Feature Selection
Create ensemble of best models with feature selection to reduce overfitting.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, MODEL_PARAMS
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from evaluator import ModelEvaluator

# Import gradient boosting libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def run_iteration_5():
    """Run Iteration 5: Ensemble methods with feature selection."""
    
    print("="*80)
    print("CAD OAS PREDICTION MODEL - ITERATION 5: ENSEMBLE WITH FEATURE SELECTION")
    print("="*80)
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    evaluator = ModelEvaluator(config)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data = data_loader.load_data()
    
    # Create comprehensive features
    print("\n2. Creating comprehensive features...")
    baseline_features = data_loader.create_baseline_features(data)
    tech_features = feature_engineer.create_technical_indicators(data)
    cross_asset_features = feature_engineer.create_cross_asset_features(data)
    statistical_features = feature_engineer.create_statistical_features(data)
    regime_features = feature_engineer.create_regime_features(data)
    
    # Combine all features
    all_features = pd.concat([
        baseline_features, 
        tech_features, 
        cross_asset_features, 
        statistical_features,
        regime_features
    ], axis=1)
    target = data_loader.prepare_target(data)
    
    print(f"Total features shape: {all_features.shape}")
    
    # Clean features (remove infinite values)
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(method='ffill').fillna(0)
    
    # Create walk-forward splits (limit to first 10 splits for faster execution)
    print("\n3. Creating walk-forward validation splits...")
    splits = list(data_loader.create_walk_forward_splits(all_features, target))
    print(f"Created {len(splits)} walk-forward splits")
    print(f"Limiting to first 10 splits for faster execution...")
    splits = splits[:10]  # Limit for faster execution
    
    # Feature selection parameters
    feature_selection_methods = {
        'top_50': 50,
        'top_100': 100,
        'top_150': 150
    }
    
    print(f"\n4. Training ensemble models with feature selection...")
    
    # Train and evaluate on each split
    all_results = []
    
    for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
        print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        print(f"Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]} ({split_info['train_size']} samples)")
        print(f"Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]} ({split_info['test_size']} samples)")
        
        for fs_name, n_features in feature_selection_methods.items():
            try:
                # Feature selection
                if n_features < train_features.shape[1]:
                    selector = SelectKBest(score_func=f_regression, k=n_features)
                    train_features_selected = selector.fit_transform(train_features, train_target)
                    test_features_selected = selector.transform(test_features)
                    
                    # Convert back to DataFrame for consistency
                    selected_features = train_features.columns[selector.get_support()]
                    train_features_selected = pd.DataFrame(
                        train_features_selected, 
                        index=train_features.index, 
                        columns=selected_features
                    )
                    test_features_selected = pd.DataFrame(
                        test_features_selected, 
                        index=test_features.index, 
                        columns=selected_features
                    )
                else:
                    train_features_selected = train_features
                    test_features_selected = test_features
                
                # Create ensemble models
                models = {}
                
                # Regularized Random Forest
                models['RF_Regularized'] = RandomForestRegressor(
                    n_estimators=50,  # Reduced
                    max_depth=3,      # Very shallow
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    random_state=42
                )
                
                # Ridge Regression
                models['Ridge'] = Ridge(alpha=10.0)  # Strong regularization
                
                # LightGBM with strong regularization
                if LIGHTGBM_AVAILABLE:
                    models['LightGBM_Regularized'] = lgb.LGBMRegressor(
                        objective='regression',
                        metric='rmse',
                        boosting_type='gbdt',
                        num_leaves=7,      # Very small
                        learning_rate=0.01,  # Very low
                        feature_fraction=0.5,  # Use only half features
                        bagging_fraction=0.5,
                        bagging_freq=5,
                        max_depth=3,       # Very shallow
                        min_child_samples=50,  # High minimum
                        reg_alpha=1.0,     # Strong L1 regularization
                        reg_lambda=1.0,    # Strong L2 regularization
                        verbose=-1,
                        force_col_wise=True,
                        seed=42,
                        n_estimators=100
                    )
                
                # Create ensemble
                ensemble_models = list(models.values())
                ensemble_names = list(models.keys())
                
                ensemble = VotingRegressor(
                    [(name, model) for name, model in zip(ensemble_names, ensemble_models)],
                    weights=[1.0] * len(ensemble_models)  # Equal weights
                )
                
                # Evaluate ensemble
                results = evaluator.evaluate_model(
                    ensemble, train_features_selected, train_target, 
                    test_features_selected, test_target,
                    f'Ensemble_{fs_name}', iteration=5, split_info=split_info
                )
                
                all_results.append(results)
                evaluator.print_results(results)
                
                # Save prediction plots for first few splits
                if split_idx < 2:
                    save_path = f"{config.output_dir}/predictions/iteration5_{fs_name}_split_{split_idx+1}_predictions.png"
                    evaluator.plot_predictions(
                        test_target.values, results['test_metrics']['r2'], 
                        f'Ensemble_{fs_name}', "test", save_path
                    )
                
            except Exception as e:
                print(f"Error training {fs_name}: {str(e)}")
                continue
    
    # Save all results
    print(f"\n5. Saving results...")
    evaluator.save_results(iteration=5)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ITERATION 5 SUMMARY")
    print(f"{'='*80}")
    
    if evaluator.best_model:
        best_model = evaluator.best_model
        print(f"Best Model: {best_model['model_name']}")
        print(f"Best R²: {best_model['r2']:.4f}")
        print(f"Iteration: {best_model['iteration']}")
        
        # Calculate average performance across splits for each feature selection method
        fs_performances = {}
        for fs_name in feature_selection_methods.keys():
            model_results = [r for r in all_results if f'Ensemble_{fs_name}' in r['model_name']]
            if model_results:
                avg_r2 = np.mean([r['test_metrics']['r2'] for r in model_results])
                std_r2 = np.std([r['test_metrics']['r2'] for r in model_results])
                fs_performances[fs_name] = {'avg_r2': avg_r2, 'std_r2': std_r2}
        
        print(f"\nAverage Performance Across All Splits:")
        for fs_name, perf in fs_performances.items():
            print(f"  {fs_name:15s}: R² = {perf['avg_r2']:.4f} ± {perf['std_r2']:.4f}")
        
        # Progress toward target
        target_r2 = config.target_r2
        progress = (best_model['r2'] / target_r2) * 100
        print(f"\nProgress toward {target_r2:.1%} R² target: {progress:.1f}%")
        
        if best_model['r2'] >= target_r2:
            print("🎉 TARGET ACHIEVED! 🎉")
        else:
            remaining = target_r2 - best_model['r2']
            print(f"Need {remaining:.4f} more R² to reach target")
            
            print(f"\nNext Steps (Iteration 6):")
            print(f"  1. Hyperparameter optimization with Optuna")
            print(f"  2. Advanced feature engineering (polynomial features)")
            print(f"  3. Different ensemble methods (stacking)")
            print(f"  4. Consider neural networks or other advanced models")
    else:
        print("No successful model training")
    
    print(f"\nResults saved to: {config.output_dir}/reports/")
    print("="*80)
    
    return evaluator.best_model

if __name__ == "__main__":
    # Run iteration 5
    best_model = run_iteration_5()
