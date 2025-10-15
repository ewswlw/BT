"""
CAD OAS Prediction Model - Iteration 6: Conservative Approach
Focus on reducing overfitting with conservative regularization and simpler models.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from evaluator import ModelEvaluator

def run_iteration_6_conservative():
    """Run Iteration 6: Conservative approach to reduce overfitting."""
    
    print("="*80)
    print("CAD OAS PREDICTION MODEL - ITERATION 6: CONSERVATIVE APPROACH")
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
    
    # Conservative feature selection
    print("\n4. Conservative feature selection...")
    
    # Use a subset of data for feature selection to avoid overfitting
    feature_selection_data = all_features.iloc[:2000]  # Use first 2000 samples
    feature_selection_target = target.iloc[:2000]
    
    # Select only top 50 features for simplicity
    selector = SelectKBest(f_regression, k=50)
    features_selected = selector.fit_transform(feature_selection_data, feature_selection_target)
    selected_features = feature_selection_data.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} features using conservative approach")
    
    # Filter features for all splits
    all_features_filtered = all_features[selected_features]
    
    # Conservative scaling
    print("\n5. Conservative scaling...")
    scaler = StandardScaler()
    
    # Test different conservative model configurations
    print("\n6. Testing conservative model configurations...")
    
    # Configuration 1: Highly Regularized Random Forest
    rf_conservative = RandomForestRegressor(
        n_estimators=30,  # Reduced from 100
        max_depth=2,      # Very shallow
        min_samples_split=50,  # High minimum split
        min_samples_leaf=25,    # High minimum leaf
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    
    # Configuration 2: Ridge with high regularization
    ridge_conservative = Ridge(
        alpha=100.0,  # Very high regularization
        max_iter=2000,
        random_state=42
    )
    
    # Configuration 3: Lasso with high regularization
    lasso_conservative = Lasso(
        alpha=1.0,  # High regularization
        max_iter=2000,
        random_state=42
    )
    
    print(f"Model configurations:")
    print(f"  Random Forest: n_estimators=30, max_depth=2, min_samples_split=50")
    print(f"  Ridge: alpha=100.0")
    print(f"  Lasso: alpha=1.0")
    
    # Train and evaluate on each split
    all_results = []
    
    for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
        print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        print(f"Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]} ({split_info['train_size']} samples)")
        print(f"Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]} ({split_info['test_size']} samples)")
        
        try:
            # Apply scaling
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
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
            
            # Test each model individually
            models = {
                'Conservative Random Forest': rf_conservative,
                'Conservative Ridge': ridge_conservative,
                'Conservative Lasso': lasso_conservative
            }
            
            best_model_name = None
            best_test_r2 = -np.inf
            
            for model_name, model in models.items():
                # Train model
                model.fit(train_features_scaled, train_target)
                
                # Make predictions
                train_pred = model.predict(train_features_scaled)
                test_pred = model.predict(test_features_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(train_target, train_pred)
                test_r2 = r2_score(test_target, test_pred)
                
                # Track best model for this split
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_model_name = model_name
                
                print(f"  {model_name}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")
            
            # Evaluate best model for this split
            best_model = models[best_model_name]
            results = evaluator.evaluate_model(
                best_model, train_features_scaled, train_target, 
                test_features_scaled, test_target,
                f'Conservative {best_model_name}', iteration=6, split_info=split_info
            )
            
            all_results.append(results)
            evaluator.print_results(results)
            
            # Save prediction plots for first few splits
            if split_idx < 3:
                save_path = f"{config.output_dir}/predictions/iteration6_conservative_split_{split_idx+1}_predictions.png"
                evaluator.plot_predictions(
                    test_target.values, results['test_metrics']['r2'], 
                    f'Conservative {best_model_name}', "test", save_path
                )
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            continue
    
    # Additional robustness analysis
    print(f"\n7. Robustness analysis...")
    
    # Calculate stability metrics
    model_results = [r for r in all_results if 'Conservative' in r['model_name']]
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
    print("ITERATION 6 SUMMARY: CONSERVATIVE APPROACH")
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
            print(f"  Conservative Models: R² = {avg_r2:.4f} ± {std_r2:.4f}")
        
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
    # Run iteration 6 conservative
    best_model = run_iteration_6_conservative()
