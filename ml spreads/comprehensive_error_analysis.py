"""
Comprehensive error analysis across all model iterations to find the best model
and analyze its biggest prediction errors.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

def comprehensive_error_analysis():
    """Comprehensive error analysis to find best model and biggest errors."""
    
    print("="*80)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("="*80)
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    
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
    
    # Clean features
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(method='ffill').fillna(0)
    
    # Ultra-conservative feature selection
    print("\n3. Applying ultra-conservative feature selection...")
    feature_selection_data = all_features.iloc[:1500]
    feature_selection_target = target.iloc[:1500]
    
    selector = SelectKBest(f_regression, k=30)
    features_selected = selector.fit_transform(feature_selection_data, feature_selection_target)
    selected_features = feature_selection_data.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} features")
    
    # Filter features
    all_features_filtered = all_features[selected_features]
    
    # Apply scaling
    scaler = RobustScaler()
    
    # Train multiple model configurations to find the best one
    print("\n4. Training multiple model configurations...")
    
    # Model configurations
    models = {
        'Ultra Ridge': Ridge(alpha=1000.0, max_iter=2000, random_state=42),
        'Conservative Ridge': Ridge(alpha=100.0, max_iter=2000, random_state=42),
        'Moderate Ridge': Ridge(alpha=10.0, max_iter=2000, random_state=42),
        'Light Ridge': Ridge(alpha=1.0, max_iter=2000, random_state=42)
    }
    
    # Create walk-forward splits (same as final robust model)
    splits = list(data_loader.create_walk_forward_splits(all_features_filtered, target))
    splits = splits[:30]  # Use first 30 splits
    
    best_model_name = None
    best_r2 = -np.inf
    best_predictions = None
    best_test_dates = None
    best_test_target = None
    
    all_results = []
    
    for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
        print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
        
        try:
            # Apply scaling
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            # Test each model
            for model_name, model in models.items():
                # Train model
                model.fit(train_features_scaled, train_target)
                
                # Make predictions
                test_pred = model.predict(test_features_scaled)
                
                # Calculate R²
                from sklearn.metrics import r2_score
                r2 = r2_score(test_target, test_pred)
                
                # Track best model
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_predictions = test_pred
                    best_test_dates = test_features.index
                    best_test_target = test_target
                
                all_results.append({
                    'split': split_idx + 1,
                    'model': model_name,
                    'r2': r2,
                    'test_dates': test_features.index,
                    'test_target': test_target,
                    'predictions': test_pred
                })
                
                print(f"  {model_name}: R² = {r2:.4f}")
            
        except Exception as e:
            print(f"Error in split {split_idx + 1}: {str(e)}")
            continue
    
    print(f"\n5. Best model identified: {best_model_name} with R² = {best_r2:.4f}")
    
    # Analyze the best model's predictions
    print(f"\n6. Analyzing best model's predictions...")
    
    # Create results DataFrame for the best model
    results_df = pd.DataFrame({
        'Date': best_test_dates,
        'Actual': best_test_target.values,
        'Predicted': best_predictions,
        'Deviation': best_test_target.values - best_predictions
    })
    
    # Sort by absolute deviation to find biggest errors
    results_df['Abs_Deviation'] = np.abs(results_df['Deviation'])
    results_df = results_df.sort_values('Abs_Deviation', ascending=False)
    
    # Get top 10 biggest deviations
    top_10 = results_df.head(10).copy()
    
    # Add forward-looking actual values
    print("\n7. Adding forward-looking actual values...")
    
    for idx, row in top_10.iterrows():
        prediction_date = row['Date']
        
        # Find actual values at t+10, t+30, t+60, t+90
        for horizon in [10, 30, 60, 90]:
            future_date = prediction_date + timedelta(days=horizon)
            
            # Find the closest date in the data
            future_mask = data.index >= future_date
            if future_mask.any():
                closest_date = data.index[future_mask][0]
                future_value = data.loc[closest_date, 'cad_oas']
                top_10.loc[idx, f'Actual_t+{horizon}'] = future_value
            else:
                top_10.loc[idx, f'Actual_t+{horizon}'] = np.nan
    
    # Format the results table
    print("\n8. Creating results table...")
    
    # Select and rename columns for the final table
    final_table = top_10[['Date', 'Actual', 'Predicted', 'Deviation', 
                         'Actual_t+10', 'Actual_t+30', 'Actual_t+60', 'Actual_t+90']].copy()
    
    # Round numerical columns
    numerical_cols = ['Actual', 'Predicted', 'Deviation', 'Actual_t+10', 'Actual_t+30', 'Actual_t+60', 'Actual_t+90']
    for col in numerical_cols:
        final_table[col] = final_table[col].round(4)
    
    # Format dates
    final_table['Date'] = final_table['Date'].dt.strftime('%Y-%m-%d')
    
    # Display the table
    print("\n" + "="*120)
    print(f"TOP 10 BIGGEST DEVIATIONS FROM ACTUAL - PREDICTED")
    print(f"Best Model: {best_model_name} (R² = {best_r2:.4f})")
    print("="*120)
    print(final_table.to_string(index=False))
    
    # Save to CSV
    output_path = f"{config.output_dir}/top_10_deviations_comprehensive.csv"
    final_table.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Additional analysis
    print("\n9. Additional analysis...")
    
    # Calculate summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean Absolute Deviation: {results_df['Abs_Deviation'].mean():.4f}")
    print(f"  Max Absolute Deviation: {results_df['Abs_Deviation'].max():.4f}")
    print(f"  Min Absolute Deviation: {results_df['Abs_Deviation'].min():.4f}")
    print(f"  Std Deviation: {results_df['Abs_Deviation'].std():.4f}")
    
    # Model performance
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(best_test_target, best_predictions)
    rmse = np.sqrt(mean_squared_error(best_test_target, best_predictions))
    mae = mean_absolute_error(best_test_target, best_predictions)
    
    print(f"\nBest Model Performance:")
    print(f"  Model: {best_model_name}")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Analyze patterns in biggest errors
    print(f"\nPattern Analysis:")
    print(f"  Positive deviations (over-predicted): {sum(results_df['Deviation'] < 0)}")
    print(f"  Negative deviations (under-predicted): {sum(results_df['Deviation'] > 0)}")
    
    # Check if biggest errors are clustered in time
    top_10_dates = pd.to_datetime(top_10['Date'])
    date_range = (top_10_dates.max() - top_10_dates.min()).days
    print(f"  Date range of top 10 errors: {date_range} days")
    
    # Analyze forward-looking patterns
    print(f"\nForward-Looking Analysis:")
    for horizon in [10, 30, 60, 90]:
        col_name = f'Actual_t+{horizon}'
        if col_name in top_10.columns:
            # Calculate how many times the actual moved in the predicted direction
            direction_correct = 0
            for idx, row in top_10.iterrows():
                if not pd.isna(row[col_name]):
                    actual_change = row[col_name] - row['Actual']
                    predicted_change = row['Predicted'] - row['Actual']
                    if (actual_change > 0 and predicted_change > 0) or (actual_change < 0 and predicted_change < 0):
                        direction_correct += 1
            
            print(f"  Direction accuracy at t+{horizon}: {direction_correct}/{len(top_10)} ({direction_correct/len(top_10)*100:.1f}%)")
    
    return final_table, best_model_name, best_r2

if __name__ == "__main__":
    # Run comprehensive analysis
    results_table, best_model, best_r2 = comprehensive_error_analysis()
