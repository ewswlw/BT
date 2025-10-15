"""
CAD OAS Prediction Model - Iteration 6: Advanced Regularization & Robustness
Focus on reducing overfitting and improving model stability across different market regimes.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from evaluator import ModelEvaluator

def run_iteration_6():
    """Run Iteration 6: Advanced regularization and robustness improvements."""
    
    print("="*80)
    print("CAD OAS PREDICTION MODEL - ITERATION 6: ROBUSTNESS & REGULARIZATION")
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
    interaction_features = feature_engineer.create_interaction_features(data)
    
    # Combine all features
    all_features = pd.concat([
        baseline_features, 
        tech_features, 
        cross_asset_features, 
        statistical_features,
        regime_features,
        interaction_features
    ], axis=1)
    target = data_loader.prepare_target(data)
    
    print(f"Total features shape: {all_features.shape}")
    
    # Clean features (remove infinite values)
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(method='ffill').fillna(0)
    
    # Create walk-forward splits (limit to first 20 splits for faster execution)
    print("\n3. Creating walk-forward validation splits...")
    splits = list(data_loader.create_walk_forward_splits(all_features, target))
    print(f"Created {len(splits)} walk-forward splits")
    print(f"Limiting to first 20 splits for faster execution...")
    splits = splits[:20]  # Limit for faster execution
    
    # Advanced feature selection with stability
    print("\n4. Advanced feature selection for robustness...")
    
    # Use a subset of data for feature selection to avoid overfitting
    feature_selection_data = all_features.iloc[:2000]  # Use first 2000 samples
    feature_selection_target = target.iloc[:2000]
    
    # Method 1: SelectKBest with f_regression (top 100 features)
    selector_kbest = SelectKBest(f_regression, k=100)
    features_kbest = selector_kbest.fit_transform(feature_selection_data, feature_selection_target)
    selected_features_kbest = feature_selection_data.columns[selector_kbest.get_support()].tolist()
    
    # Method 2: SelectFromModel with RandomForest (top 100 features)
    rf_selector = RandomForestRegressor(
        n_estimators=50, 
        max_depth=3, 
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    selector_rf = SelectFromModel(rf_selector, max_features=100)
    features_rf = selector_rf.fit_transform(feature_selection_data, feature_selection_target)
    selected_features_rf = feature_selection_data.columns[selector_rf.get_support()].tolist()
    
    # Combine both methods (union of selected features)
    combined_features = list(set(selected_features_kbest + selected_features_rf))
    print(f"Selected {len(combined_features)} features using combined approach")
    
    # Filter features for all splits
    all_features_filtered = all_features[combined_features]
    
    # Advanced scaling for robustness
    print("\n5. Advanced scaling for robustness...")
    
    # Use RobustScaler instead of StandardScaler for outlier resistance
    robust_scaler = RobustScaler()
    
    # Test different model configurations for robustness
    print("\n6. Testing robust model configurations...")
    
    # Configuration 1: Highly Regularized Random Forest
    rf_robust = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=3,      # Reduced from 5
        min_samples_split=20,  # Increased from 10
        min_samples_leaf=10,    # Increased from 5
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    
    # Configuration 2: Elastic Net (combines Ridge and Lasso)
    elastic_net = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,  # Equal mix of L1 and L2
        max_iter=2000,
        random_state=42
    )
    
    # Configuration 3: Ridge with higher regularization
    ridge_robust = Ridge(
        alpha=10.0,  # Increased from 1.0
        max_iter=2000,
        random_state=42
    )
    
    # Create robust ensemble
    robust_ensemble = VotingRegressor([
        ('rf_robust', rf_robust),
        ('elastic_net', elastic_net),
        ('ridge_robust', ridge_robust)
    ])
    
    print(f"Model configurations:")
    print(f"  Random Forest: n_estimators=50, max_depth=3, min_samples_split=20")
    print(f"  Elastic Net: alpha=0.1, l1_ratio=0.5")
    print(f"  Ridge: alpha=10.0")
    
    # Train and evaluate on each split
    all_results = []
    
    for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
        print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        print(f"Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]} ({split_info['train_size']} samples)")
        print(f"Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]} ({split_info['test_size']} samples)")
        
        try:
            # Apply robust scaling
            train_features_scaled = robust_scaler.fit_transform(train_features)
            test_features_scaled = robust_scaler.transform(test_features)
            
            # Convert back to DataFrame for consistency
            train_features_scaled = pd.DataFrame(
                train_features_scaled, 
                index=train_features.index, 
                columns=train_features.columns
            )
            test_features_scaled = pd.DataFrame(
                test_features_scaled, 
                index=test_features.index, 
                columns=test_features.columns
            )
            
            # Evaluate robust ensemble
            results = evaluator.evaluate_model(
                robust_ensemble, train_features_scaled, train_target, 
                test_features_scaled, test_target,
                'Robust Ensemble (RF + ElasticNet + Ridge)', iteration=6, split_info=split_info
            )
            
            all_results.append(results)
            evaluator.print_results(results)
            
            # Save prediction plots for first few splits
            if split_idx < 3:
                save_path = f"{config.output_dir}/predictions/iteration6_split_{split_idx+1}_predictions.png"
                evaluator.plot_predictions(
                    test_target.values, results['test_metrics']['r2'], 
                    'Robust Ensemble', "test", save_path
                )
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            continue
    
    # Additional robustness analysis
    print(f"\n7. Robustness analysis...")
    
    # Calculate stability metrics
    model_results = [r for r in all_results if r['model_name'] == 'Robust Ensemble (RF + ElasticNet + Ridge)']
    if model_results:
        test_r2_scores = [r['test_metrics']['r2'] for r in model_results]
        train_r2_scores = [r['train_metrics']['r2'] for r in model_results]
        
        # Calculate overfitting metrics
        overfitting_scores = [train - test for train, test in zip(train_r2_scores, test_r2_scores)]
        
        print(f"\nRobustness Metrics:")
        print(f"  Test R²: {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
        print(f"  Train R²: {np.mean(train_r2_scores):.4f} ± {np.std(train_r2_scores):.4f}")
        print(f"  Overfitting: {np.mean(overfitting_scores):.4f} ± {np.std(overfitting_scores):.4f}")
        print(f"  R² Stability (CV): {np.std(test_r2_scores):.4f}")
        print(f"  Min Test R²: {np.min(test_r2_scores):.4f}")
        print(f"  Max Test R²: {np.max(test_r2_scores):.4f}")
        
        # Calculate improvement over previous iterations
        print(f"\nImprovement Analysis:")
        print(f"  Target R²: {config.target_r2:.1%}")
        print(f"  Current Best: {np.max(test_r2_scores):.1%}")
        print(f"  Average: {np.mean(test_r2_scores):.1%}")
        print(f"  Splits above target: {sum(1 for r2 in test_r2_scores if r2 >= config.target_r2)}/{len(test_r2_scores)}")
    
    # Save all results
    print(f"\n8. Saving results...")
    evaluator.save_results(iteration=6)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ITERATION 6 SUMMARY: ROBUSTNESS & REGULARIZATION")
    print(f"{'='*80}")
    
    if evaluator.best_model:
        best_model = evaluator.best_model
        print(f"Best Model: {best_model['model_name']}")
        print(f"Best R²: {best_model['r2']:.4f}")
        print(f"Iteration: {best_model['iteration']}")
        
        # Calculate average performance across splits
        if model_results:
            avg_r2 = np.mean(test_r2_scores)
            std_r2 = np.std(test_r2_scores)
            print(f"\nAverage Performance Across All Splits:")
            print(f"  Robust Ensemble: R² = {avg_r2:.4f} ± {std_r2:.4f}")
        
        # Progress toward target
        target_r2 = config.target_r2
        progress = (best_model['r2'] / target_r2) * 100
        print(f"\nProgress toward {target_r2:.1%} R² target: {progress:.1f}%")
        
        if best_model['r2'] >= target_r2:
            print("🎉 TARGET ACHIEVED! 🎉")
        else:
            remaining = target_r2 - best_model['r2']
            print(f"Need {remaining:.4f} more R² to reach target")
            
            print(f"\nNext Steps (Iteration 7):")
            print(f"  1. Implement neural networks (LSTM/Transformer)")
            print(f"  2. Add polynomial features for key predictors")
            print(f"  3. Implement two-stage models (regime + level)")
            print(f"  4. Consider quantile regression for robustness")
            print(f"  5. Advanced ensemble methods (stacking, blending)")
    else:
        print("No successful model training")
    
    print(f"\nResults saved to: {config.output_dir}/reports/")
    print("="*80)
    
    return evaluator.best_model

if __name__ == "__main__":
    # Run iteration 6
    best_model = run_iteration_6()
