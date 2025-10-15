"""
Analyze the best model's predictions to find top 10 biggest deviations
and their forward-looking outcomes.
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

def analyze_best_model():
    """Analyze the best model's predictions and find biggest deviations."""
    
    print("="*80)
    print("ANALYZING BEST MODEL: Ultra Robust Ultra Ridge")
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
    
    # Ultra-conservative feature selection (same as best model)
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
    
    # Train the best model configuration
    print("\n4. Training best model configuration...")
    
    # Use the same split as the best model (Split 21)
    train_start = pd.to_datetime('2005-07-01')
    train_end = pd.to_datetime('2008-05-13')
    test_start = pd.to_datetime('2008-05-14')
    test_end = pd.to_datetime('2008-08-07')
    
    # Get the data for this split
    train_mask = (data.index >= train_start) & (data.index <= train_end)
    test_mask = (data.index >= test_start) & (data.index <= test_end)
    
    train_features = all_features_filtered[train_mask]
    train_target = target[train_mask]
    test_features = all_features_filtered[test_mask]
    test_target = target[test_mask]
    
    # Apply scaling
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Train Ridge model with ultra-high regularization
    ridge_model = Ridge(alpha=1000.0, max_iter=2000, random_state=42)
    ridge_model.fit(train_features_scaled, train_target)
    
    # Make predictions
    train_pred = ridge_model.predict(train_features_scaled)
    test_pred = ridge_model.predict(test_features_scaled)
    
    print(f"Model trained on {len(train_features)} samples")
    print(f"Predictions made on {len(test_features)} samples")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': test_features.index,
        'Actual': test_target.values,
        'Predicted': test_pred,
        'Deviation': test_target.values - test_pred
    })
    
    # Sort by absolute deviation to find biggest errors
    results_df['Abs_Deviation'] = np.abs(results_df['Deviation'])
    results_df = results_df.sort_values('Abs_Deviation', ascending=False)
    
    print(f"\n5. Finding top 10 biggest deviations...")
    
    # Get top 10 biggest deviations
    top_10 = results_df.head(10).copy()
    
    # Add forward-looking actual values
    print("\n6. Adding forward-looking actual values...")
    
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
    print("\n7. Creating results table...")
    
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
    print("TOP 10 BIGGEST DEVIATIONS FROM ACTUAL - PREDICTED")
    print("="*120)
    print(final_table.to_string(index=False))
    
    # Save to CSV
    output_path = f"{config.output_dir}/top_10_deviations.csv"
    final_table.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Additional analysis
    print("\n8. Additional analysis...")
    
    # Calculate summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean Absolute Deviation: {results_df['Abs_Deviation'].mean():.4f}")
    print(f"  Max Absolute Deviation: {results_df['Abs_Deviation'].max():.4f}")
    print(f"  Min Absolute Deviation: {results_df['Abs_Deviation'].min():.4f}")
    print(f"  Std Deviation: {results_df['Abs_Deviation'].std():.4f}")
    
    # Model performance
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(test_target, test_pred)
    rmse = np.sqrt(mean_squared_error(test_target, test_pred))
    mae = mean_absolute_error(test_target, test_pred)
    
    print(f"\nModel Performance:")
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
    
    return final_table

if __name__ == "__main__":
    # Run analysis
    results_table = analyze_best_model()
