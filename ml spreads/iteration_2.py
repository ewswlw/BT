"""
CAD OAS Prediction Model - Iteration 2: Cross-Asset and Statistical Features
Add cross-asset features and statistical transformations to improve performance.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, MODEL_PARAMS
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from evaluator import ModelEvaluator

def run_iteration_2():
    """Run Iteration 2: Add cross-asset and statistical features."""
    
    print("="*80)
    print("CAD OAS PREDICTION MODEL - ITERATION 2: CROSS-ASSET & STATISTICAL FEATURES")
    print("="*80)
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    evaluator = ModelEvaluator(config)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data = data_loader.load_data()
    
    # Create baseline features
    print("\n2. Creating baseline features...")
    baseline_features = data_loader.create_baseline_features(data)
    
    # Create technical indicators
    print("\n3. Adding technical indicators...")
    tech_features = feature_engineer.create_technical_indicators(data)
    
    # Create cross-asset features
    print("\n4. Adding cross-asset features...")
    cross_asset_features = feature_engineer.create_cross_asset_features(data)
    
    # Create statistical features
    print("\n5. Adding statistical features...")
    statistical_features = feature_engineer.create_statistical_features(data)
    
    # Combine all features
    all_features = pd.concat([
        baseline_features, 
        tech_features, 
        cross_asset_features, 
        statistical_features
    ], axis=1)
    target = data_loader.prepare_target(data)
    
    print(f"Total features shape: {all_features.shape}")
    print(f"Baseline features: {baseline_features.shape[1]}")
    print(f"Technical features: {tech_features.shape[1]}")
    print(f"Cross-asset features: {cross_asset_features.shape[1]}")
    print(f"Statistical features: {statistical_features.shape[1]}")
    print(f"Total features: {all_features.shape[1]}")
    
    # Clean features (remove infinite values)
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(method='ffill').fillna(0)
    
    # Create walk-forward splits (limit to first 20 splits for faster execution)
    print("\n6. Creating walk-forward validation splits...")
    splits = list(data_loader.create_walk_forward_splits(all_features, target))
    print(f"Created {len(splits)} walk-forward splits")
    print(f"Limiting to first 20 splits for faster execution...")
    splits = splits[:20]  # Limit for faster execution
    
    # Use Random Forest with regularization to reduce overfitting
    model_params = MODEL_PARAMS['random_forest'].copy()
    model_params.update({
        'max_depth': 5,  # Reduce depth to prevent overfitting
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt'  # Use sqrt features to reduce overfitting
    })
    model = RandomForestRegressor(**model_params)
    
    print(f"\n7. Training Random Forest with cross-asset and statistical features...")
    print(f"Model parameters: {model_params}")
    
    # Train and evaluate on each split
    all_results = []
    
    for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
        print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        print(f"Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]} ({split_info['train_size']} samples)")
        print(f"Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]} ({split_info['test_size']} samples)")
        
        try:
            # Evaluate model
            results = evaluator.evaluate_model(
                model, train_features, train_target, test_features, test_target,
                'Random Forest + Cross-Asset + Statistical', iteration=2, split_info=split_info
            )
            
            all_results.append(results)
            evaluator.print_results(results)
            
            # Save prediction plots for first few splits
            if split_idx < 3:
                save_path = f"{config.output_dir}/predictions/iteration2_split_{split_idx+1}_predictions.png"
                evaluator.plot_predictions(
                    test_target.values, results['test_metrics']['r2'], 
                    'Random Forest + Cross-Asset + Statistical', "test", save_path
                )
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            continue
    
    # Save all results
    print(f"\n8. Saving results...")
    evaluator.save_results(iteration=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ITERATION 2 SUMMARY")
    print(f"{'='*80}")
    
    if evaluator.best_model:
        best_model = evaluator.best_model
        print(f"Best Model: {best_model['model_name']}")
        print(f"Best R²: {best_model['r2']:.4f}")
        print(f"Iteration: {best_model['iteration']}")
        
        # Calculate average performance across splits
        model_results = [r for r in all_results if r['model_name'] == 'Random Forest + Cross-Asset + Statistical']
        if model_results:
            avg_r2 = np.mean([r['test_metrics']['r2'] for r in model_results])
            std_r2 = np.std([r['test_metrics']['r2'] for r in model_results])
            print(f"\nAverage Performance Across All Splits:")
            print(f"  Random Forest + Cross-Asset + Statistical: R² = {avg_r2:.4f} ± {std_r2:.4f}")
        
        # Progress toward target
        target_r2 = config.target_r2
        progress = (best_model['r2'] / target_r2) * 100
        print(f"\nProgress toward {target_r2:.1%} R² target: {progress:.1f}%")
        
        if best_model['r2'] >= target_r2:
            print("🎉 TARGET ACHIEVED! 🎉")
        else:
            remaining = target_r2 - best_model['r2']
            print(f"Need {remaining:.4f} more R² to reach target")
            
            print(f"\nNext Steps (Iteration 3):")
            print(f"  1. Add regime and interaction features")
            print(f"  2. Try gradient boosting models (LightGBM, XGBoost)")
            print(f"  3. Implement feature selection to reduce overfitting")
    else:
        print("No successful model training")
    
    print(f"\nResults saved to: {config.output_dir}/reports/")
    print("="*80)
    
    return evaluator.best_model

if __name__ == "__main__":
    # Run iteration 2
    best_model = run_iteration_2()
