"""
Model evaluation and performance metrics for CAD OAS prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any
import json
import os
from datetime import datetime

class ModelEvaluator:
    """Handles model evaluation and performance tracking."""
    
    def __init__(self, config_obj=None):
        """Initialize evaluator with configuration."""
        # Import config here to avoid circular imports
        from config import config
        self.config = config_obj or config
        self.results_history = []
        self.best_model = None
        self.best_r2 = -np.inf
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         split_info: Dict = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'r2': -np.inf, 'rmse': np.inf, 'mae': np.inf, 'mape': np.inf}
        
        # Core regression metrics
        r2 = r2_score(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # Mean Absolute Percentage Error (robust to zero values)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + 1e-8))) * 100
        
        # Directional accuracy (sign prediction)
        direction_true = np.sign(np.diff(y_true_clean))
        direction_pred = np.sign(np.diff(y_pred_clean))
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        # Correlation
        correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'n_samples': len(y_true_clean)
        }
        
        # Add split information if provided
        if split_info:
            metrics.update(split_info)
        
        return metrics
    
    def evaluate_model(self, model, train_features: pd.DataFrame, train_target: pd.Series,
                      test_features: pd.DataFrame, test_target: pd.Series,
                      model_name: str, iteration: int = 0, split_info: Dict = None) -> Dict[str, Any]:
        """Evaluate a model and return comprehensive results."""
        
        # Train the model
        model.fit(train_features, train_target)
        
        # Make predictions
        train_pred = model.predict(train_features)
        test_pred = model.predict(test_features)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(train_target.values, train_pred, 
                                             {**split_info, 'split_type': 'train'} if split_info else {'split_type': 'train'})
        test_metrics = self.calculate_metrics(test_target.values, test_pred,
                                            {**split_info, 'split_type': 'test'} if split_info else {'split_type': 'test'})
        
        # Create results dictionary
        results = {
            'model_name': model_name,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting': train_metrics['r2'] - test_metrics['r2'],
            'split_info': split_info
        }
        
        # Track best model
        if test_metrics['r2'] > self.best_r2:
            self.best_r2 = test_metrics['r2']
            self.best_model = {
                'model': model,
                'model_name': model_name,
                'iteration': iteration,
                'r2': test_metrics['r2'],
                'metrics': test_metrics
            }
        
        # Add to history
        self.results_history.append(results)
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        model_name = results['model_name']
        iteration = results['iteration']
        train_r2 = results['train_metrics']['r2']
        test_r2 = results['test_metrics']['r2']
        overfitting = results['overfitting']
        
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} (Iteration {iteration})")
        print(f"{'='*60}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²:  {test_r2:.4f}")
        print(f"Overfitting: {overfitting:.4f}")
        
        if results['split_info']:
            split_info = results['split_info']
            print(f"\nSplit Info:")
            print(f"  Train: {split_info['train_dates'][0]} to {split_info['train_dates'][1]}")
            print(f"  Test:  {split_info['test_dates'][0]} to {split_info['test_dates'][1]}")
            print(f"  Train samples: {split_info['train_size']}, Test samples: {split_info['test_size']}")
        
        print(f"\nDetailed Test Metrics:")
        test_metrics = results['test_metrics']
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  Directional Accuracy: {test_metrics['directional_accuracy']:.2f}%")
        print(f"  Correlation: {test_metrics['correlation']:.4f}")
        
        # Progress toward target
        target_r2 = self.config.target_r2
        progress = (test_r2 / target_r2) * 100 if target_r2 > 0 else 0
        print(f"\nProgress toward {target_r2:.1%} R² target: {progress:.1f}%")
        
        if test_r2 >= target_r2:
            print("🎉 TARGET ACHIEVED! 🎉")
        else:
            remaining = target_r2 - test_r2
            print(f"Need {remaining:.4f} more R² to reach target")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str, split_type: str = "test", 
                        save_path: str = None):
        """Create prediction visualization plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - {split_type.title()} Predictions', fontsize=16)
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # 1. Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(y_true_clean, y_pred_clean, alpha=0.6, s=20)
        axes[0, 0].plot([y_true_clean.min(), y_true_clean.max()], 
                       [y_true_clean.min(), y_true_clean.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual CAD OAS')
        axes[0, 0].set_ylabel('Predicted CAD OAS')
        axes[0, 0].set_title('Predicted vs Actual')
        
        # Add R² to plot
        r2 = r2_score(y_true_clean, y_pred_clean)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Time series plot (if we have time index)
        if len(y_true_clean) > 100:  # Only if we have enough data points
            axes[0, 1].plot(y_true_clean[:100], label='Actual', alpha=0.8)
            axes[0, 1].plot(y_pred_clean[:100], label='Predicted', alpha=0.8)
            axes[0, 1].set_xlabel('Time Index')
            axes[0, 1].set_ylabel('CAD OAS')
            axes[0, 1].set_title('Time Series (First 100 points)')
            axes[0, 1].legend()
        else:
            axes[0, 1].plot(y_true_clean, label='Actual', alpha=0.8)
            axes[0, 1].plot(y_pred_clean, label='Predicted', alpha=0.8)
            axes[0, 1].set_xlabel('Time Index')
            axes[0, 1].set_ylabel('CAD OAS')
            axes[0, 1].set_title('Time Series')
            axes[0, 1].legend()
        
        # 3. Residuals plot
        residuals = y_true_clean - y_pred_clean
        axes[1, 0].scatter(y_pred_clean, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted CAD OAS')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Predicted')
        
        # 4. Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction plots saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                               top_n: int = 20, save_path: str = None):
        """Plot feature importance for tree-based models."""
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importance = np.abs(model.coef_)
        else:
            print("Model does not support feature importance")
            return
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Take top N features
        top_features = importance_df.tail(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
        
        return importance_df
    
    def save_results(self, iteration: int = 0):
        """Save evaluation results to files."""
        
        if not self.results_history:
            print("No results to save")
            return
        
        # Save results history
        results_path = f"{self.config.output_dir}/reports/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
        
        # Save best model info
        if self.best_model:
            best_model_path = f"{self.config.output_dir}/reports/best_model_info.json"
            best_model_info = {
                'model_name': self.best_model['model_name'],
                'iteration': self.best_model['iteration'],
                'r2': self.best_model['r2'],
                'metrics': self.best_model['metrics'],
                'timestamp': datetime.now().isoformat()
            }
            with open(best_model_path, 'w') as f:
                json.dump(best_model_info, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        for result in self.results_history:
            summary_data.append({
                'model_name': result['model_name'],
                'iteration': result['iteration'],
                'train_r2': result['train_metrics']['r2'],
                'test_r2': result['test_metrics']['r2'],
                'overfitting': result['overfitting'],
                'test_rmse': result['test_metrics']['rmse'],
                'test_mae': result['test_metrics']['mae'],
                'test_mape': result['test_metrics']['mape'],
                'directional_accuracy': result['test_metrics']['directional_accuracy'],
                'correlation': result['test_metrics']['correlation']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{self.config.output_dir}/reports/results_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Results saved to:")
        print(f"  - {results_path}")
        print(f"  - {summary_path}")
        if self.best_model:
            print(f"  - {best_model_path}")
    
    def get_iteration_summary(self, iteration: int) -> Dict[str, Any]:
        """Get summary of results for a specific iteration."""
        
        iteration_results = [r for r in self.results_history if r['iteration'] == iteration]
        
        if not iteration_results:
            return {}
        
        # Calculate averages across all splits for this iteration
        test_r2s = [r['test_metrics']['r2'] for r in iteration_results]
        train_r2s = [r['train_metrics']['r2'] for r in iteration_results]
        
        summary = {
            'iteration': iteration,
            'num_models': len(iteration_results),
            'avg_train_r2': np.mean(train_r2s),
            'avg_test_r2': np.mean(test_r2s),
            'std_test_r2': np.std(test_r2s),
            'max_test_r2': np.max(test_r2s),
            'min_test_r2': np.min(test_r2s),
            'best_model': max(iteration_results, key=lambda x: x['test_metrics']['r2'])['model_name']
        }
        
        return summary
