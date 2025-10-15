"""
CAD OAS Prediction Model - Production Pipeline
Final orchestration pipeline with comprehensive reporting and model persistence.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
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

class CADOASPredictionPipeline:
    """Production pipeline for CAD OAS prediction model."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.evaluator = ModelEvaluator(config)
        self.best_model = None
        self.feature_selector = None
        self.scaler = None
        self.feature_names = None
        
        # Create output directories
        os.makedirs(f"{config.output_dir}/models", exist_ok=True)
        os.makedirs(f"{config.output_dir}/features", exist_ok=True)
        os.makedirs(f"{config.output_dir}/predictions", exist_ok=True)
        os.makedirs(f"{config.output_dir}/reports", exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare all data."""
        print("="*80)
        print("CAD OAS PREDICTION PIPELINE - DATA PREPARATION")
        print("="*80)
        
        # Load raw data
        print("\n1. Loading raw data...")
        data = self.data_loader.load_data()
        
        # Create comprehensive features
        print("\n2. Creating comprehensive features...")
        baseline_features = self.data_loader.create_baseline_features(data)
        tech_features = self.feature_engineer.create_technical_indicators(data)
        cross_asset_features = self.feature_engineer.create_cross_asset_features(data)
        statistical_features = self.feature_engineer.create_statistical_features(data)
        regime_features = self.feature_engineer.create_regime_features(data)
        
        # Combine all features
        all_features = pd.concat([
            baseline_features, 
            tech_features, 
            cross_asset_features, 
            statistical_features,
            regime_features
        ], axis=1)
        target = self.data_loader.prepare_target(data)
        
        print(f"Total features shape: {all_features.shape}")
        print(f"Baseline: {baseline_features.shape[1]}, Technical: {tech_features.shape[1]}")
        print(f"Cross-asset: {cross_asset_features.shape[1]}, Statistical: {statistical_features.shape[1]}")
        print(f"Regime: {regime_features.shape[1]}, Total: {all_features.shape[1]}")
        
        # Clean features
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        return data, all_features, target
    
    def train_final_model(self, features, target):
        """Train the final production model."""
        print("\n" + "="*80)
        print("CAD OAS PREDICTION PIPELINE - MODEL TRAINING")
        print("="*80)
        
        # Use the best performing configuration from Iteration 5
        n_features = 150  # Best performing feature count
        
        # Feature selection
        print(f"\n1. Selecting top {n_features} features...")
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        features_selected = self.feature_selector.fit_transform(features, target)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.feature_names = features.columns[selected_mask].tolist()
        
        # Convert back to DataFrame
        features_selected = pd.DataFrame(
            features_selected, 
            index=features.index, 
            columns=self.feature_names
        )
        
        print(f"Selected {len(self.feature_names)} features")
        
        # Create ensemble model (same as Iteration 5)
        print("\n2. Creating ensemble model...")
        models = {}
        
        # Regularized Random Forest
        models['RF_Regularized'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42
        )
        
        # Ridge Regression
        models['Ridge'] = Ridge(alpha=10.0)
        
        # LightGBM with strong regularization
        if LIGHTGBM_AVAILABLE:
            models['LightGBM_Regularized'] = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                num_leaves=7,
                learning_rate=0.01,
                feature_fraction=0.5,
                bagging_fraction=0.5,
                bagging_freq=5,
                max_depth=3,
                min_child_samples=50,
                reg_alpha=1.0,
                reg_lambda=1.0,
                verbose=-1,
                force_col_wise=True,
                seed=42,
                n_estimators=100
            )
        
        # Create ensemble
        ensemble_models = list(models.values())
        ensemble_names = list(models.keys())
        
        self.best_model = VotingRegressor(
            [(name, model) for name, model in zip(ensemble_names, ensemble_models)],
            weights=[1.0] * len(ensemble_models)
        )
        
        # Train on all available data
        print("\n3. Training final model on all data...")
        self.best_model.fit(features_selected, target)
        
        print("Model training completed!")
        return features_selected
    
    def evaluate_model(self, features, target):
        """Evaluate the final model."""
        print("\n" + "="*80)
        print("CAD OAS PREDICTION PIPELINE - MODEL EVALUATION")
        print("="*80)
        
        # Walk-forward validation
        print("\n1. Running walk-forward validation...")
        splits = list(self.data_loader.create_walk_forward_splits(features, target))
        
        # Limit to first 20 splits for comprehensive evaluation
        splits = splits[:20]
        print(f"Evaluating on {len(splits)} walk-forward splits")
        
        all_results = []
        
        for split_idx, (train_features, train_target, test_features, test_target, split_info) in enumerate(splits):
            print(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
            print(f"Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]} ({split_info['train_size']} samples)")
            print(f"Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]} ({split_info['test_size']} samples)")
            
            try:
                # Apply feature selection
                train_features_selected = self.feature_selector.transform(train_features)
                test_features_selected = self.feature_selector.transform(test_features)
                
                # Convert to DataFrame
                train_features_selected = pd.DataFrame(
                    train_features_selected, 
                    index=train_features.index, 
                    columns=self.feature_names
                )
                test_features_selected = pd.DataFrame(
                    test_features_selected, 
                    index=test_features.index, 
                    columns=self.feature_names
                )
                
                # Train model on this split
                model = VotingRegressor(
                    [(name, model) for name, model in zip(
                        ['RF_Regularized', 'Ridge', 'LightGBM_Regularized'] if LIGHTGBM_AVAILABLE else ['RF_Regularized', 'Ridge'],
                        [RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', random_state=42),
                         Ridge(alpha=10.0),
                         lgb.LGBMRegressor(objective='regression', metric='rmse', boosting_type='gbdt', num_leaves=7, learning_rate=0.01, feature_fraction=0.5, bagging_fraction=0.5, bagging_freq=5, max_depth=3, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0, verbose=-1, force_col_wise=True, seed=42, n_estimators=100)] if LIGHTGBM_AVAILABLE else [RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', random_state=42), Ridge(alpha=10.0)]
                    )],
                    weights=[1.0] * (3 if LIGHTGBM_AVAILABLE else 2)
                )
                
                model.fit(train_features_selected, train_target)
                
                # Evaluate
                results = self.evaluator.evaluate_model(
                    model, train_features_selected, train_target, 
                    test_features_selected, test_target,
                    'Final_Production_Model', iteration=6, split_info=split_info
                )
                
                all_results.append(results)
                self.evaluator.print_results(results)
                
            except Exception as e:
                print(f"Error in split {split_idx + 1}: {str(e)}")
                continue
        
        # Calculate summary statistics
        if all_results:
            test_r2_scores = [r['test_metrics']['r2'] for r in all_results]
            avg_r2 = np.mean(test_r2_scores)
            std_r2 = np.std(test_r2_scores)
            min_r2 = np.min(test_r2_scores)
            max_r2 = np.max(test_r2_scores)
            
            print(f"\n" + "="*80)
            print("FINAL MODEL PERFORMANCE SUMMARY")
            print("="*80)
            print(f"Average R²: {avg_r2:.4f} ± {std_r2:.4f}")
            print(f"Min R²: {min_r2:.4f}")
            print(f"Max R²: {max_r2:.4f}")
            print(f"Splits above 85% R²: {sum(1 for r2 in test_r2_scores if r2 >= 0.85)}/{len(test_r2_scores)}")
            print(f"Target achievement rate: {sum(1 for r2 in test_r2_scores if r2 >= 0.85)/len(test_r2_scores)*100:.1f}%")
        
        return all_results
    
    def save_model_artifacts(self):
        """Save all model artifacts."""
        print("\n" + "="*80)
        print("CAD OAS PREDICTION PIPELINE - SAVING ARTIFACTS")
        print("="*80)
        
        # Save the final model
        print("\n1. Saving final model...")
        model_path = f"{config.output_dir}/models/best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved to: {model_path}")
        
        # Save feature selector
        print("\n2. Saving feature selector...")
        selector_path = f"{config.output_dir}/models/feature_selector.pkl"
        with open(selector_path, 'wb') as f:
            pickle.dump(self.feature_selector, f)
        print(f"Feature selector saved to: {selector_path}")
        
        # Save feature names
        print("\n3. Saving feature names...")
        feature_names_path = f"{config.output_dir}/features/selected_features.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"Feature names saved to: {feature_names_path}")
        
        # Save feature importance
        print("\n4. Saving feature importance...")
        if hasattr(self.best_model, 'estimators_'):
            # Get feature importance from Random Forest
            rf_model = None
            for name, model in self.best_model.named_estimators_.items():
                if 'RF' in name:
                    rf_model = model
                    break
            
            if rf_model and hasattr(rf_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = f"{config.output_dir}/features/feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                print(f"Feature importance saved to: {importance_path}")
        
        # Save model metadata
        print("\n5. Saving model metadata...")
        metadata = {
            'model_type': 'Ensemble (RandomForest + Ridge + LightGBM)',
            'feature_count': len(self.feature_names),
            'target_column': config.target_column,
            'min_lag': config.min_lag,
            'max_lag': config.max_lag,
            'target_r2': config.target_r2,
            'created_at': datetime.now().isoformat(),
            'feature_selection_method': 'SelectKBest with f_regression',
            'selected_features': self.feature_names
        }
        
        metadata_path = f"{config.output_dir}/models/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to: {metadata_path}")
    
    def generate_final_report(self, all_results):
        """Generate comprehensive final report."""
        print("\n" + "="*80)
        print("CAD OAS PREDICTION PIPELINE - FINAL REPORT")
        print("="*80)
        
        # Create comprehensive report
        report = {
            'project': 'CAD OAS Prediction Model',
            'objective': 'Predict CAD OAS using lagged features with >85% R²',
            'target_achieved': False,
            'model_performance': {},
            'feature_engineering': {},
            'model_architecture': {},
            'validation_results': {},
            'recommendations': []
        }
        
        if all_results:
            test_r2_scores = [r['test_metrics']['r2'] for r in all_results]
            avg_r2 = np.mean(test_r2_scores)
            std_r2 = np.std(test_r2_scores)
            target_achieved = avg_r2 >= config.target_r2
            
        report['target_achieved'] = bool(target_achieved)
        report['model_performance'] = {
            'average_r2': float(avg_r2),
            'std_r2': float(std_r2),
            'min_r2': float(np.min(test_r2_scores)),
            'max_r2': float(np.max(test_r2_scores)),
            'target_r2': float(config.target_r2),
            'splits_above_target': int(sum(1 for r2 in test_r2_scores if r2 >= config.target_r2)),
            'total_splits': len(test_r2_scores),
            'achievement_rate': float(sum(1 for r2 in test_r2_scores if r2 >= config.target_r2)/len(test_r2_scores))
        }
        
        report['feature_engineering'] = {
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'feature_selection_method': 'SelectKBest with f_regression',
            'selected_features': self.feature_names if self.feature_names else [],
            'feature_types': {
                'baseline': 'Lagged values, returns, moving averages, volatility',
                'technical': 'RSI, MACD, Bollinger Bands, Stochastic',
                'cross_asset': 'Spread differentials, VIX interactions',
                'statistical': 'Z-scores, percentiles, skewness, kurtosis',
                'regime': 'Economic regime indicators, volatility regimes'
            }
        }
        
        report['model_architecture'] = {
            'type': 'Ensemble (VotingRegressor)',
            'base_models': ['RandomForest', 'Ridge', 'LightGBM'] if LIGHTGBM_AVAILABLE else ['RandomForest', 'Ridge'],
            'regularization': 'Strong regularization applied to prevent overfitting',
            'feature_selection': 'Top 150 features selected based on f_regression scores'
        }
        
        report['validation_results'] = {
            'method': 'Walk-forward validation',
            'total_splits_evaluated': len(all_results) if all_results else 0
        }
        
        # Recommendations
        if report['target_achieved']:
            report['recommendations'] = [
                "Model successfully achieved target R² > 85%",
                "Consider deploying model for production use",
                "Monitor model performance over time and retrain as needed",
                "Consider expanding feature set for potential improvements"
            ]
        else:
            report['recommendations'] = [
                "Model did not consistently achieve target R² > 85%",
                "Consider additional feature engineering",
                "Try different ensemble methods or hyperparameter tuning",
                "Consider advanced models like neural networks",
                "Evaluate data quality and potential data leakage"
            ]
        
        # Save report
        report_path = f"{config.output_dir}/reports/final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Final report saved to: {report_path}")
        
        # Print summary
        print(f"\n" + "="*80)
        print("FINAL REPORT SUMMARY")
        print("="*80)
        print(f"Target Achieved: {'✅ YES' if report['target_achieved'] else '❌ NO'}")
        if report['model_performance']:
            perf = report['model_performance']
            print(f"Average R²: {perf['average_r2']:.4f} ± {perf['std_r2']:.4f}")
            print(f"Target Achievement Rate: {perf['achievement_rate']*100:.1f}%")
            print(f"Best R²: {perf['max_r2']:.4f}")
            print(f"Worst R²: {perf['min_r2']:.4f}")
        
        print(f"\nModel Architecture:")
        print(f"  Type: {report['model_architecture']['type']}")
        print(f"  Base Models: {', '.join(report['model_architecture']['base_models'])}")
        print(f"  Features: {report['feature_engineering']['total_features']} selected")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print("="*80)
        
        return report
    
    def run_full_pipeline(self):
        """Run the complete production pipeline."""
        print("🚀 Starting CAD OAS Prediction Pipeline...")
        
        # Step 1: Load and prepare data
        data, features, target = self.load_and_prepare_data()
        
        # Step 2: Train final model
        features_selected = self.train_final_model(features, target)
        
        # Step 3: Evaluate model
        all_results = self.evaluate_model(features, target)
        
        # Step 4: Save model artifacts
        self.save_model_artifacts()
        
        # Step 5: Generate final report
        final_report = self.generate_final_report(all_results)
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"All outputs saved to: {config.output_dir}/")
        
        return final_report

def main():
    """Main entry point."""
    pipeline = CADOASPredictionPipeline()
    final_report = pipeline.run_full_pipeline()
    return final_report

if __name__ == "__main__":
    final_report = main()
