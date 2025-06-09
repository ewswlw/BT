#!/usr/bin/env python3
"""
Comprehensive Feature Importance Analysis
Detailed analysis of the 76 features in the trading strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and data from the previous script
exec(open('comprehensive_feature_analysis.py').read())

print("\n" + "="*80)
print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
print("="*80)

class FeatureImportanceAnalyzer:
    def __init__(self, model, X, y, feature_names, feature_categories, scaler=None):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.feature_categories = feature_categories
        self.scaler = scaler
        self.importance_results = {}
        
    def calculate_permutation_importance(self):
        """Calculate permutation importance for SVM model"""
        print("\n1. Calculating Permutation Importance...")
        
        # Scale data if needed
        X_scaled = self.scaler.transform(self.X) if self.scaler else self.X
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, X_scaled, self.y, 
            n_repeats=10, random_state=42, scoring='accuracy'
        )
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Add feature categories
        importance_df['category'] = importance_df['feature'].apply(self.get_feature_category)
        
        self.importance_results['permutation'] = importance_df
        
        print(f"Permutation importance calculated for {len(importance_df)} features")
        return importance_df
    
    def calculate_rf_importance(self):
        """Calculate Random Forest feature importance for comparison"""
        print("\n2. Calculating Random Forest Importance for comparison...")
        
        # Train Random Forest for feature importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X, self.y)
        
        # Get feature importance
        rf_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': rf_model.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        rf_importance_df['category'] = rf_importance_df['feature'].apply(self.get_feature_category)
        
        self.importance_results['random_forest'] = rf_importance_df
        
        print(f"Random Forest importance calculated for {len(rf_importance_df)} features")
        return rf_importance_df
    
    def get_feature_category(self, feature_name):
        """Get category for a feature"""
        for category, features in self.feature_categories.items():
            if feature_name in features:
                return category
        return 'other'
    
    def create_top_features_visualization(self):
        """Create visualization of top 20 most important features"""
        print("\n3. Creating Top Features Visualization...")
        
        perm_df = self.importance_results['permutation']
        top_20 = perm_df.head(20)
        
        # Create plotly bar chart
        fig = go.Figure()
        
        # Color map for categories
        category_colors = {
            'momentum': '#FF6B6B',
            'volatility': '#4ECDC4', 
            'trend': '#45B7D1',
            'higher_order': '#96CEB4',
            'regime': '#FFEAA7',
            'cross_asset': '#DDA0DD',
            'composite': '#98D8C8',
            'interaction': '#F7DC6F'
        }
        
        colors = [category_colors.get(cat, '#BDC3C7') for cat in top_20['category']]
        
        fig.add_trace(go.Bar(
            x=top_20['importance_mean'],
            y=top_20['feature'],
            orientation='h',
            marker_color=colors,
            error_x=dict(type='data', array=top_20['importance_std']),
            text=[f"{imp:.4f}" for imp in top_20['importance_mean']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 20 Most Important Features (Permutation Importance)",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=800,
            width=1000,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )
        
        fig.write_html('top_20_features_importance.html')
        print("✓ Top 20 features chart saved as 'top_20_features_importance.html'")
        
        return fig
    
    def create_category_analysis(self):
        """Analyze importance by feature category"""
        print("\n4. Creating Category Analysis...")
        
        perm_df = self.importance_results['permutation']
        
        # Calculate category statistics
        category_stats = perm_df.groupby('category').agg({
            'importance_mean': ['mean', 'sum', 'count', 'max'],
            'importance_std': 'mean'
        }).round(4)
        
        category_stats.columns = ['avg_importance', 'total_importance', 'feature_count', 'max_importance', 'avg_std']
        category_stats = category_stats.sort_values('total_importance', ascending=False)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Importance by Category', 'Average Importance by Category',
                          'Feature Count by Category', 'Max Importance by Category'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        categories = category_stats.index
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
        # Total importance
        fig.add_trace(go.Bar(x=categories, y=category_stats['total_importance'], 
                            marker_color=colors[:len(categories)], name='Total'), row=1, col=1)
        
        # Average importance
        fig.add_trace(go.Bar(x=categories, y=category_stats['avg_importance'], 
                            marker_color=colors[:len(categories)], name='Average'), row=1, col=2)
        
        # Feature count
        fig.add_trace(go.Bar(x=categories, y=category_stats['feature_count'], 
                            marker_color=colors[:len(categories)], name='Count'), row=2, col=1)
        
        # Max importance
        fig.add_trace(go.Bar(x=categories, y=category_stats['max_importance'], 
                            marker_color=colors[:len(categories)], name='Max'), row=2, col=2)
        
        fig.update_layout(height=800, width=1200, title_text="Feature Importance Analysis by Category",
                         showlegend=False, template='plotly_white')
        
        fig.write_html('category_importance_analysis.html')
        print("✓ Category analysis saved as 'category_importance_analysis.html'")
        
        return category_stats
    
    def create_correlation_analysis(self):
        """Analyze correlations between top features"""
        print("\n5. Creating Correlation Analysis...")
        
        perm_df = self.importance_results['permutation']
        top_15_features = perm_df.head(15)['feature'].tolist()
        
        # Calculate correlation matrix for top features
        corr_matrix = self.X[top_15_features].corr()
        
        # Create correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Matrix: Top 15 Most Important Features",
            width=800,
            height=800,
            template='plotly_white'
        )
        
        fig.write_html('top_features_correlation.html')
        print("✓ Correlation analysis saved as 'top_features_correlation.html'")
        
        return corr_matrix
    
    def create_importance_distribution(self):
        """Analyze distribution of feature importance"""
        print("\n6. Creating Importance Distribution Analysis...")
        
        perm_df = self.importance_results['permutation']
        
        # Create distribution plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Importance Distribution', 'Importance by Category (Box Plot)',
                          'Cumulative Importance', 'Top Features by Category'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Histogram of importance scores
        fig.add_trace(go.Histogram(x=perm_df['importance_mean'], nbinsx=20, 
                                  name='Distribution'), row=1, col=1)
        
        # Box plot by category
        for i, category in enumerate(perm_df['category'].unique()):
            cat_data = perm_df[perm_df['category'] == category]['importance_mean']
            fig.add_trace(go.Box(y=cat_data, name=category), row=1, col=2)
        
        # Cumulative importance
        perm_df_sorted = perm_df.sort_values('importance_mean', ascending=False)
        cumulative_importance = perm_df_sorted['importance_mean'].cumsum()
        fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_importance)+1)), 
                                y=cumulative_importance, mode='lines+markers',
                                name='Cumulative'), row=2, col=1)
        
        # Top feature per category
        top_per_category = perm_df.groupby('category')['importance_mean'].max().sort_values(ascending=False)
        fig.add_trace(go.Bar(x=top_per_category.index, y=top_per_category.values,
                            name='Top per Category'), row=2, col=2)
        
        fig.update_layout(height=800, width=1200, title_text="Feature Importance Distribution Analysis",
                         showlegend=False, template='plotly_white')
        
        fig.write_html('importance_distribution_analysis.html')
        print("✓ Distribution analysis saved as 'importance_distribution_analysis.html'")
        
        return perm_df
    
    def generate_feature_insights(self):
        """Generate detailed insights about feature importance"""
        print("\n7. Generating Feature Insights...")
        
        perm_df = self.importance_results['permutation']
        top_20 = perm_df.head(20)
        
        insights = {
            'top_features': top_20,
            'category_analysis': perm_df.groupby('category').agg({
                'importance_mean': ['count', 'mean', 'sum', 'max']
            }).round(4),
            'statistical_summary': {
                'total_features': len(perm_df),
                'mean_importance': perm_df['importance_mean'].mean(),
                'std_importance': perm_df['importance_mean'].std(),
                'top_10_contribution': perm_df.head(10)['importance_mean'].sum(),
                'top_20_contribution': perm_df.head(20)['importance_mean'].sum(),
            }
        }
        
        return insights

# Initialize the analyzer with our trained model
feature_analyzer = FeatureImportanceAnalyzer(
    model=analyzer.best_model,
    X=analyzer.X,
    y=analyzer.y,
    feature_names=analyzer.feature_cols,
    feature_categories=analyzer.feature_categories,
    scaler=analyzer.scaler
)

# Run all analyses
print("Starting comprehensive feature importance analysis...")

# 1. Calculate permutation importance
perm_importance = feature_analyzer.calculate_permutation_importance()

# 2. Calculate RF importance for comparison
rf_importance = feature_analyzer.calculate_rf_importance()

# 3. Create visualizations
top_features_fig = feature_analyzer.create_top_features_visualization()
category_stats = feature_analyzer.create_category_analysis()
correlation_matrix = feature_analyzer.create_correlation_analysis()
distribution_analysis = feature_analyzer.create_importance_distribution()

# 4. Generate insights
insights = feature_analyzer.generate_feature_insights()

print("\n" + "="*80)
print("ANALYSIS COMPLETE - KEY FINDINGS")
print("="*80)

print(f"\n📊 SUMMARY STATISTICS:")
print(f"   • Total Features Analyzed: {insights['statistical_summary']['total_features']}")
print(f"   • Mean Importance Score: {insights['statistical_summary']['mean_importance']:.4f}")
print(f"   • Standard Deviation: {insights['statistical_summary']['std_importance']:.4f}")
print(f"   • Top 10 Features Contribution: {insights['statistical_summary']['top_10_contribution']:.4f}")
print(f"   • Top 20 Features Contribution: {insights['statistical_summary']['top_20_contribution']:.4f}")

print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
for i, (_, row) in enumerate(perm_importance.head(10).iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:<25} | {row['importance_mean']:.4f} ± {row['importance_std']:.4f} | {row['category']}")

print(f"\n📈 CATEGORY RANKINGS (by total importance):")
for i, (category, stats) in enumerate(category_stats.iterrows(), 1):
    print(f"   {i}. {category.title():<15} | Total: {stats['total_importance']:.4f} | Avg: {stats['avg_importance']:.4f} | Count: {int(stats['feature_count'])}")

print(f"\n💡 KEY INSIGHTS:")
print(f"   • Most important category: {category_stats.index[0].title()}")
print(f"   • Most features from: {category_stats.sort_values('feature_count', ascending=False).index[0].title()}")
print(f"   • Highest individual feature: {perm_importance.iloc[0]['feature']}")
print(f"   • Most consistent category: {category_stats.sort_values('avg_std').index[0].title()}")

print(f"\n📁 FILES CREATED:")
print(f"   • top_20_features_importance.html")
print(f"   • category_importance_analysis.html") 
print(f"   • top_features_correlation.html")
print(f"   • importance_distribution_analysis.html")

print(f"\n✅ Feature importance analysis completed successfully!")
