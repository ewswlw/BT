"""
CAD OAS Prediction Model - Iteration 4: Gradient Boosting Models
Implement LightGBM and XGBoost with proper regularization to reduce overfitting.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
    print("Warning: LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

def run_iteration_4():
    """Run Iteration 4: Gradient boosting models with regularization."""
    
    print("="*80)
    print("CAD OAS PREDICTION MODEL - ITERATION 4: GRADIENT BOOSTING MODELS")
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
    print(f"Baseline: {baseline_features.shape[1]}, Technical: {tech_features.shape[1]}")
    print(f"Cross-asset: {cross_asset_features.shape[1]}, Statistical: {statistical_features.shape[1]}")
    print(f"Regime: {regime_features.shape[1]}, Total: {all_features.shape[1]}")
    
    # Clean features (remove infinite values)
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(method='ffill').fillna(0)
    
    # Create walk-forward splits (limit to first 15 splits for faster execution)
    print("\n3. Creating walk-forward validation splits...")
    splits = list(data_loader.create_walk_forward_splits(all_features, target))
    print(f"Created {len(splits)} walk-forward splits")
    print(f"Limiting to first 15 splits for faster execution...")
    splits = splits[:15]  # Limit for faster execution
    
    # Initialize models with regularization
    models = {}
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=15,  # Reduced to prevent overfitting
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            max_depth=5,  # Reduced depth
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
            force_col_wise=True,
            seed=42,
            n_estimators=200
        )
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.05,
            max_depth=5,  # Reduced depth
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_estimators=200
        )
    
    print(f"\n4. Training {len(models)} gradient boosting models...")
    
    # Train and evaluate on each split
    all_results = []
    
    for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
        print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        print(f"Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]} ({split_info['train_size']} samples)")
        print(f"Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]} ({split_info['test_size']} samples)")
        
        for model_name, model in models.items():
            try:
                # Evaluate model
                results = evaluator.evaluate_model(
                    model, train_features, train_target, test_features, test_target,
                    f'{model_name} + All Features', iteration=4, split_info=split_info
                )
                
                all_results.append(results)
                evaluator.print_results(results)
                
                # Save prediction plots for first few splits
                if split_idx < 2:
                    save_path = f"{config.output_dir}/predictions/iteration4_{model_name.lower()}_split_{split_idx+1}_predictions.png"
                    evaluator.plot_predictions(
                        test_target.values, results['test_metrics']['r2'], 
                        f'{model_name} + All Features', "test", save_path
                    )
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
    
    # Save all results
    print(f"\n5. Saving results...")
    evaluator.save_results(iteration=4)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ITERATION 4 SUMMARY")
    print(f"{'='*80}")
    
    if evaluator.best_model:
        best_model = evaluator.best_model
        print(f"Best Model: {best_model['model_name']}")
        print(f"Best R²: {best_model['r2']:.4f}")
        print(f"Iteration: {best_model['iteration']}")
        
        # Calculate average performance across splits for each model
        model_performances = {}
        for model_name in models.keys():
            model_results = [r for r in all_results if model_name in r['model_name']]
            if model_results:
                avg_r2 = np.mean([r['test_metrics']['r2'] for r in model_results])
                std_r2 = np.std([r['test_metrics']['r2'] for r in model_results])
                model_performances[model_name] = {'avg_r2': avg_r2, 'std_r2': std_r2}
        
        print(f"\nAverage Performance Across All Splits:")
        for model_name, perf in model_performances.items():
            print(f"  {model_name:15s}: R² = {perf['avg_r2']:.4f} ± {perf['std_r2']:.4f}")
        
        # Progress toward target
        target_r2 = config.target_r2
        progress = (best_model['r2'] / target_r2) * 100
        print(f"\nProgress toward {target_r2:.1%} R² target: {progress:.1f}%")
        
        if best_model['r2'] >= target_r2:
            print("🎉 TARGET ACHIEVED! 🎉")
        else:
            remaining = target_r2 - best_model['r2']
            print(f"Need {remaining:.4f} more R² to reach target")
            
            print(f"\nNext Steps (Iteration 5):")
            print(f"  1. Create ensemble of best models")
            print(f"  2. Implement feature selection to reduce overfitting")
            print(f"  3. Try hyperparameter optimization")
            print(f"  4. Consider advanced techniques (neural networks, etc.)")
    else:
        print("No successful model training")
    
    print(f"\nResults saved to: {config.output_dir}/reports/")
    print("="*80)
    
    return evaluator.best_model

if __name__ == "__main__":
    # Run iteration 4
    best_model = run_iteration_4()
