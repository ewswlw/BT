"""
Main execution script for CAD OAS prediction model.
Phase 1: Foundation and baseline models.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, MODEL_PARAMS
from data_loader import DataLoader
from evaluator import ModelEvaluator

def run_baseline_models():
    """Run baseline models to establish performance baseline."""
    
    print("="*80)
    print("CAD OAS PREDICTION MODEL - PHASE 1: FOUNDATION")
    print("="*80)
    
    # Initialize components
    data_loader = DataLoader(config)
    evaluator = ModelEvaluator(config)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data = data_loader.load_data()
    data_loader.save_data_summary()
    
    # Create baseline features
    print("\n2. Creating baseline features...")
    features = data_loader.create_baseline_features(data)
    target = data_loader.prepare_target(data)
    
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Feature columns: {len(features.columns)}")
    
    # Create walk-forward splits
    print("\n3. Creating walk-forward validation splits...")
    splits = list(data_loader.create_walk_forward_splits(features, target))
    print(f"Created {len(splits)} walk-forward splits")
    
    # Initialize baseline models
    models = {
        'Linear Regression': LinearRegression(**MODEL_PARAMS['linear_regression']),
        'Ridge': Ridge(**MODEL_PARAMS['ridge']),
        'Lasso': Lasso(**MODEL_PARAMS['lasso']),
        'Random Forest': RandomForestRegressor(**MODEL_PARAMS['random_forest'])
    }
    
    print(f"\n4. Training {len(models)} baseline models...")
    
    # Train and evaluate models on each split
    all_results = []
    
    for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
        print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        print(f"Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]} ({split_info['train_size']} samples)")
        print(f"Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]} ({split_info['test_size']} samples)")
        
        # Scale features for linear models
        scaler = StandardScaler()
        train_features_scaled = pd.DataFrame(
            scaler.fit_transform(train_features),
            index=train_features.index,
            columns=train_features.columns
        )
        test_features_scaled = pd.DataFrame(
            scaler.transform(test_features),
            index=test_features.index,
            columns=test_features.columns
        )
        
        for model_name, model in models.items():
            try:
                # Use scaled features for linear models, original for tree-based
                if model_name in ['Linear Regression', 'Ridge', 'Lasso']:
                    train_feat = train_features_scaled
                    test_feat = test_features_scaled
                else:
                    train_feat = train_features
                    test_feat = test_features
                
                # Evaluate model
                results = evaluator.evaluate_model(
                    model, train_feat, train_target, test_feat, test_target,
                    model_name, iteration=0, split_info=split_info
                )
                
                all_results.append(results)
                evaluator.print_results(results)
                
                # Save prediction plots for first split and best performing model
                if split_idx == 0 and model_name == 'Random Forest':
                    save_path = f"{config.output_dir}/predictions/baseline_predictions_{model_name.lower().replace(' ', '_')}.png"
                    evaluator.plot_predictions(
                        test_target.values, results['test_metrics']['r2'], 
                        model_name, "test", save_path
                    )
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
    
    # Save all results
    print(f"\n5. Saving results...")
    evaluator.save_results(iteration=0)
    
    # Print summary
    print(f"\n{'='*80}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*80}")
    
    if evaluator.best_model:
        best_model = evaluator.best_model
        print(f"Best Model: {best_model['model_name']}")
        print(f"Best R²: {best_model['r2']:.4f}")
        print(f"Iteration: {best_model['iteration']}")
        
        # Calculate average performance across splits
        model_performances = {}
        for model_name in models.keys():
            model_results = [r for r in all_results if r['model_name'] == model_name]
            if model_results:
                avg_r2 = np.mean([r['test_metrics']['r2'] for r in model_results])
                std_r2 = np.std([r['test_metrics']['r2'] for r in model_results])
                model_performances[model_name] = {'avg_r2': avg_r2, 'std_r2': std_r2}
        
        print(f"\nAverage Performance Across All Splits:")
        for model_name, perf in model_performances.items():
            print(f"  {model_name:20s}: R² = {perf['avg_r2']:.4f} ± {perf['std_r2']:.4f}")
        
        # Progress toward target
        target_r2 = config.target_r2
        progress = (best_model['r2'] / target_r2) * 100
        print(f"\nProgress toward {target_r2:.1%} R² target: {progress:.1f}%")
        
        if best_model['r2'] >= target_r2:
            print("🎉 TARGET ACHIEVED IN BASELINE! 🎉")
        else:
            remaining = target_r2 - best_model['r2']
            print(f"Need {remaining:.4f} more R² to reach target")
            print(f"\nNext Steps:")
            print(f"  1. Add technical indicators (RSI, MACD, Bollinger Bands)")
            print(f"  2. Add cross-asset features (spread differentials, VIX regimes)")
            print(f"  3. Add statistical features (z-scores, percentiles)")
            print(f"  4. Try gradient boosting models (LightGBM, XGBoost)")
    else:
        print("No successful model training")
    
    print(f"\nResults saved to: {config.output_dir}/reports/")
    print("="*80)
    
    return evaluator.best_model

if __name__ == "__main__":
    # Run baseline models
    best_model = run_baseline_models()
