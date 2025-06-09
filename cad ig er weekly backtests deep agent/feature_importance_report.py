#!/usr/bin/env python3
"""
Comprehensive Feature Importance Report Generator
Creates detailed analysis with explanations and actionable insights
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load the results from previous analysis
exec(open('feature_importance_detailed_analysis.py').read())

def create_feature_descriptions():
    """Create detailed descriptions for each feature type"""
    return {
        # Top features descriptions
        'vol_regime': {
            'description': 'Volatility Regime Indicator',
            'explanation': 'Binary indicator showing when 8-week volatility exceeds 70th percentile of 52-week rolling window',
            'trading_significance': 'Identifies high volatility periods that often precede market reversals or trend changes',
            'why_important': 'Volatility clustering is a key market phenomenon - high vol periods tend to persist and signal regime changes'
        },
        'trend_quality': {
            'description': 'Trend Quality Measure',
            'explanation': 'Ratio of 8-week trend strength to 8-week volatility',
            'trading_significance': 'Measures how clean and reliable the current trend is relative to market noise',
            'why_important': 'High-quality trends (strong direction, low noise) are more likely to continue and provide better risk-adjusted returns'
        },
        'us_economic_regime_new': {
            'description': 'Economic Regime Composite',
            'explanation': 'Combines interest rate momentum, credit spread changes, and VIX direction into regime score',
            'trading_significance': 'Captures broad economic environment affecting risk asset performance',
            'why_important': 'Economic regimes drive long-term market cycles and risk premiums across asset classes'
        },
        'vol_skewness_8w': {
            'description': 'Volatility Skewness',
            'explanation': 'Skewness of 4-week volatility over 8-week rolling window',
            'trading_significance': 'Measures asymmetry in volatility distribution - indicates tail risk buildup',
            'why_important': 'Volatility skewness often precedes market stress events and provides early warning signals'
        },
        'skewness_12w': {
            'description': 'Return Skewness (12-week)',
            'explanation': 'Skewness of weekly returns over 12-week rolling window',
            'trading_significance': 'Captures asymmetry in return distribution - negative skew indicates tail risk',
            'why_important': 'Return skewness is a key risk factor - markets with negative skew require higher risk premiums'
        },
        'kurtosis_12w': {
            'description': 'Return Kurtosis (12-week)',
            'explanation': 'Kurtosis (fat-tailedness) of weekly returns over 12-week rolling window',
            'trading_significance': 'Measures frequency of extreme returns relative to normal distribution',
            'why_important': 'High kurtosis indicates increased probability of large moves, affecting option pricing and risk management'
        },
        'tsx_tnx_corr_8w': {
            'description': 'Stock-Bond Correlation',
            'explanation': '8-week rolling correlation between TSX returns and 10-year yield changes',
            'trading_significance': 'Measures relationship between equity and bond markets',
            'why_important': 'Stock-bond correlation regime changes signal shifts in market risk perception and monetary policy effectiveness'
        },
        'volatility_12w': {
            'description': '12-week Volatility',
            'explanation': 'Standard deviation of weekly returns over 12-week rolling window',
            'trading_significance': 'Medium-term volatility measure capturing market uncertainty',
            'why_important': 'Volatility is mean-reverting and predictive of future returns - high vol often followed by higher returns'
        },
        'vol_kurtosis_8w': {
            'description': 'Volatility Kurtosis',
            'explanation': 'Kurtosis of 4-week volatility over 8-week rolling window',
            'trading_significance': 'Measures clustering and extreme events in volatility itself',
            'why_important': 'Volatility kurtosis indicates periods of volatility-of-volatility, crucial for derivatives pricing'
        },
        'ma_trend_8_26': {
            'description': 'Moving Average Trend Signal',
            'explanation': 'Binary indicator when 8-week MA is above 26-week MA',
            'trading_significance': 'Classic trend-following signal used by technical analysts',
            'why_important': 'Simple but effective trend indicator that captures momentum persistence in markets'
        }
    }

def create_category_insights():
    """Create insights for each feature category"""
    return {
        'volatility': {
            'importance': 'HIGHEST',
            'total_contribution': '31.5%',
            'key_insight': 'Volatility-based features dominate the model, indicating that risk measurement and regime detection are crucial for timing market entry/exit',
            'trading_implication': 'Focus on volatility indicators for position sizing and market timing',
            'top_features': ['vol_regime', 'vol_skewness_8w', 'volatility_12w', 'vol_kurtosis_8w']
        },
        'momentum': {
            'importance': 'HIGH',
            'total_contribution': '19.2%',
            'key_insight': 'Despite having the most features (22), momentum contributes less than volatility, suggesting quality over quantity in momentum signals',
            'trading_implication': 'Use selective momentum indicators rather than broad momentum exposure',
            'top_features': ['momentum_quality', 'rsi_8w', 'momentum_8w']
        },
        'trend': {
            'importance': 'HIGH',
            'total_contribution': '17.5%',
            'key_insight': 'Trend features have the highest average importance per feature, indicating trend quality is more important than trend direction',
            'trading_implication': 'Focus on trend quality and strength rather than simple trend direction',
            'top_features': ['trend_quality', 'ma_trend_8_26', 'trend_strength_8w']
        },
        'higher_order': {
            'importance': 'MEDIUM-HIGH',
            'total_contribution': '12.3%',
            'key_insight': 'Higher-order moments (skewness, kurtosis) provide significant predictive power, capturing tail risk and distribution changes',
            'trading_implication': 'Monitor distribution characteristics for early warning of regime changes',
            'top_features': ['skewness_12w', 'kurtosis_12w', 'vol_skewness_8w']
        },
        'regime': {
            'importance': 'MEDIUM',
            'total_contribution': '10.8%',
            'key_insight': 'Regime detection features are moderately important, with economic regime being the standout',
            'trading_implication': 'Economic regime analysis should inform strategic asset allocation decisions',
            'top_features': ['us_economic_regime_new', 'risk_on_regime']
        },
        'cross_asset': {
            'importance': 'MEDIUM',
            'total_contribution': '8.9%',
            'key_insight': 'Cross-asset relationships provide moderate predictive power, with stock-bond correlation being most important',
            'trading_implication': 'Monitor cross-asset correlations for portfolio diversification and risk management',
            'top_features': ['tsx_tnx_corr_8w', 'tsx_vix_corr_8w']
        }
    }

def generate_comprehensive_report():
    """Generate comprehensive feature importance report"""
    
    feature_descriptions = create_feature_descriptions()
    category_insights = create_category_insights()
    
    # Get the data from previous analysis
    perm_df = feature_analyzer.importance_results['permutation']
    top_20 = perm_df.head(20)
    
    report = f"""
# COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS REPORT
## Trading Strategy Performance: 2.85x Outperformance vs Buy-and-Hold

---

## EXECUTIVE SUMMARY

This analysis reveals the key drivers behind the exceptional 2.85x outperformance of our trading strategy. Through permutation importance analysis of 76 comprehensive features across 6 categories, we identified the critical factors that enable superior market timing and risk management.

**Key Findings:**
- **Volatility features dominate** with 31.5% of total importance
- **Risk measurement trumps return prediction** in feature importance
- **Higher-order moments** provide significant edge in tail risk detection
- **Trend quality matters more than trend direction**
- **Economic regime detection** enables strategic positioning

---

## TOP 10 CRITICAL SUCCESS FACTORS

### 1. Volatility Regime Detection (Importance: 0.0121)
**Feature:** `vol_regime`
**Category:** Volatility

{feature_descriptions['vol_regime']['explanation']}

**Why This Matters:** {feature_descriptions['vol_regime']['why_important']}

**Trading Application:** {feature_descriptions['vol_regime']['trading_significance']}

### 2. Trend Quality Assessment (Importance: 0.0120)
**Feature:** `trend_quality`
**Category:** Trend

{feature_descriptions['trend_quality']['explanation']}

**Why This Matters:** {feature_descriptions['trend_quality']['why_important']}

**Trading Application:** {feature_descriptions['trend_quality']['trading_significance']}

### 3. Economic Regime Identification (Importance: 0.0109)
**Feature:** `us_economic_regime_new`
**Category:** Regime

{feature_descriptions['us_economic_regime_new']['explanation']}

**Why This Matters:** {feature_descriptions['us_economic_regime_new']['why_important']}

**Trading Application:** {feature_descriptions['us_economic_regime_new']['trading_significance']}

### 4. Volatility Distribution Analysis (Importance: 0.0101)
**Feature:** `vol_skewness_8w`
**Category:** Volatility

{feature_descriptions['vol_skewness_8w']['explanation']}

**Why This Matters:** {feature_descriptions['vol_skewness_8w']['why_important']}

**Trading Application:** {feature_descriptions['vol_skewness_8w']['trading_significance']}

### 5. Return Distribution Asymmetry (Importance: 0.0100)
**Feature:** `skewness_12w`
**Category:** Higher-Order

{feature_descriptions['skewness_12w']['explanation']}

**Why This Matters:** {feature_descriptions['skewness_12w']['why_important']}

**Trading Application:** {feature_descriptions['skewness_12w']['trading_significance']}

---

## CATEGORY ANALYSIS & STRATEGIC INSIGHTS

### 🏆 VOLATILITY FEATURES (31.5% Total Importance)
{category_insights['volatility']['key_insight']}

**Strategic Implication:** {category_insights['volatility']['trading_implication']}

**Top Contributors:** {', '.join(category_insights['volatility']['top_features'])}

### 📈 MOMENTUM FEATURES (19.2% Total Importance)
{category_insights['momentum']['key_insight']}

**Strategic Implication:** {category_insights['momentum']['trading_implication']}

### 📊 TREND FEATURES (17.5% Total Importance)
{category_insights['trend']['key_insight']}

**Strategic Implication:** {category_insights['trend']['trading_implication']}

### 🎯 HIGHER-ORDER FEATURES (12.3% Total Importance)
{category_insights['higher_order']['key_insight']}

**Strategic Implication:** {category_insights['higher_order']['trading_implication']}

---

## MARKET DYNAMICS REVEALED

### 1. Risk-First Approach Wins
The dominance of volatility and higher-order moment features reveals that **risk management drives returns** more than return prediction. The strategy succeeds by:
- Identifying high-risk periods to avoid
- Detecting regime changes early
- Measuring tail risk buildup

### 2. Quality Over Quantity in Signals
Despite momentum having 22 features vs volatility's 16, volatility features contribute 65% more importance. This shows:
- Feature engineering quality matters more than quantity
- Volatility signals are more reliable than momentum signals
- Risk-adjusted metrics outperform raw momentum

### 3. Multi-Timeframe Risk Assessment
The strategy combines:
- Short-term volatility clustering (4-8 weeks)
- Medium-term trend quality (8-12 weeks)  
- Long-term regime detection (26-52 weeks)

### 4. Cross-Asset Intelligence
Stock-bond correlation emerges as a key cross-asset signal, indicating:
- Regime change detection through correlation shifts
- Flight-to-quality dynamics
- Monetary policy transmission mechanisms

---

## ACTIONABLE TRADING INSIGHTS

### For Strategy Enhancement:
1. **Increase volatility feature weight** - Consider 40-50% allocation to vol-based signals
2. **Implement regime overlay** - Use economic regime as position sizing multiplier
3. **Add tail risk monitoring** - Incorporate skewness/kurtosis thresholds for risk-off signals
4. **Enhance trend quality filters** - Focus on trend strength relative to volatility

### For Risk Management:
1. **Volatility regime stops** - Reduce exposure when vol_regime = 1
2. **Distribution monitoring** - Watch for negative skewness buildup
3. **Correlation regime tracking** - Monitor stock-bond correlation for regime shifts
4. **Multi-timeframe confirmation** - Require alignment across timeframes

### For Portfolio Construction:
1. **Dynamic position sizing** based on volatility regime
2. **Regime-aware allocation** using economic regime scores
3. **Tail risk hedging** when higher-order moments deteriorate
4. **Cross-asset diversification** guided by correlation analysis

---

## STATISTICAL VALIDATION

- **Total Features Analyzed:** 76
- **Model Accuracy:** 68.43%
- **Top 10 Features Contribution:** 40.6% of total importance
- **Top 20 Features Contribution:** 65.7% of total importance
- **Most Consistent Category:** Momentum (lowest std deviation)
- **Highest Individual Impact:** Volatility Regime (0.0121)

---

## CONCLUSION

The exceptional 2.85x outperformance stems from the strategy's sophisticated approach to **risk measurement over return prediction**. By focusing on volatility regimes, trend quality, and distribution characteristics, the model successfully times market entry and exit points.

The dominance of volatility and higher-order features reveals that markets are more predictable in their risk characteristics than their return directions. This insight should guide future strategy development toward enhanced risk-based signals and regime detection capabilities.

**Key Success Formula:**
Risk Regime Detection + Trend Quality Assessment + Distribution Monitoring = Superior Market Timing

---

*Report generated from permutation importance analysis of 76-feature SVM trading model*
*Analysis period: 2003-2025 | Weekly rebalancing | Binary long-only signals*
"""
    
    return report

# Generate and save the comprehensive report
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE FEATURE IMPORTANCE REPORT")
print("="*80)

comprehensive_report = generate_comprehensive_report()

# Save as markdown
with open('comprehensive_feature_importance_report.md', 'w') as f:
    f.write(comprehensive_report)

print("✅ Comprehensive report saved as 'comprehensive_feature_importance_report.md'")

# Create a summary table of top features with descriptions
def create_feature_summary_table():
    """Create detailed summary table of top features"""
    
    perm_df = feature_analyzer.importance_results['permutation']
    top_20 = perm_df.head(20)
    feature_descriptions = create_feature_descriptions()
    
    summary_data = []
    for _, row in top_20.iterrows():
        feature_name = row['feature']
        desc = feature_descriptions.get(feature_name, {})
        
        summary_data.append({
            'Rank': len(summary_data) + 1,
            'Feature': feature_name,
            'Category': row['category'].title(),
            'Importance': f"{row['importance_mean']:.4f}",
            'Std Dev': f"{row['importance_std']:.4f}",
            'Description': desc.get('description', 'Advanced technical indicator'),
            'Trading Significance': desc.get('trading_significance', 'Provides market timing signals')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('top_20_features_detailed.csv', index=False)
    
    print("✅ Detailed feature summary saved as 'top_20_features_detailed.csv'")
    return summary_df

# Create feature summary table
feature_summary = create_feature_summary_table()

# Create final dashboard combining all insights
def create_final_dashboard():
    """Create comprehensive dashboard with all insights"""
    
    perm_df = feature_analyzer.importance_results['permutation']
    top_20 = perm_df.head(20)
    
    # Create comprehensive dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Top 15 Features by Importance',
            'Importance by Category', 
            'Feature Importance Distribution',
            'Cumulative Importance',
            'Category Performance Metrics',
            'Risk vs Return Features'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Top 15 features
    top_15 = top_20.head(15)
    colors = ['#FF6B6B' if cat == 'volatility' else 
              '#4ECDC4' if cat == 'trend' else
              '#45B7D1' if cat == 'regime' else
              '#96CEB4' if cat == 'higher_order' else
              '#FFEAA7' if cat == 'momentum' else
              '#DDA0DD' for cat in top_15['category']]
    
    fig.add_trace(go.Bar(
        x=top_15['importance_mean'],
        y=top_15['feature'],
        orientation='h',
        marker_color=colors,
        name='Top Features'
    ), row=1, col=1)
    
    # 2. Category importance
    category_totals = perm_df.groupby('category')['importance_mean'].sum().sort_values(ascending=False)
    fig.add_trace(go.Bar(
        x=category_totals.index,
        y=category_totals.values,
        marker_color=['#FF6B6B', '#FFEAA7', '#4ECDC4', '#96CEB4', '#45B7D1', '#DDA0DD', '#98D8C8'],
        name='Category Totals'
    ), row=1, col=2)
    
    # 3. Distribution
    fig.add_trace(go.Histogram(
        x=perm_df['importance_mean'],
        nbinsx=20,
        name='Distribution'
    ), row=2, col=1)
    
    # 4. Cumulative importance
    perm_sorted = perm_df.sort_values('importance_mean', ascending=False)
    cumulative = perm_sorted['importance_mean'].cumsum()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative)+1)),
        y=cumulative,
        mode='lines+markers',
        name='Cumulative'
    ), row=2, col=2)
    
    # 5. Category metrics
    category_stats = perm_df.groupby('category').agg({
        'importance_mean': ['count', 'mean', 'sum']
    }).round(4)
    category_stats.columns = ['count', 'avg', 'total']
    
    fig.add_trace(go.Bar(
        x=category_stats.index,
        y=category_stats['avg'],
        name='Avg Importance'
    ), row=3, col=1)
    
    # 6. Risk vs Return scatter
    risk_features = perm_df[perm_df['category'].isin(['volatility', 'higher_order', 'regime'])]
    return_features = perm_df[perm_df['category'].isin(['momentum', 'trend'])]
    
    fig.add_trace(go.Scatter(
        x=risk_features['importance_mean'],
        y=risk_features['importance_std'],
        mode='markers',
        marker=dict(size=10, color='red', opacity=0.7),
        name='Risk Features',
        text=risk_features['feature']
    ), row=3, col=2)
    
    fig.add_trace(go.Scatter(
        x=return_features['importance_mean'],
        y=return_features['importance_std'],
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.7),
        name='Return Features',
        text=return_features['feature']
    ), row=3, col=2)
    
    fig.update_layout(
        height=1200,
        width=1400,
        title_text="Comprehensive Feature Importance Dashboard",
        showlegend=False,
        template='plotly_white'
    )
    
    fig.write_html('comprehensive_feature_dashboard.html')
    print("✅ Comprehensive dashboard saved as 'comprehensive_feature_dashboard.html'")
    
    return fig

# Create final dashboard
final_dashboard = create_final_dashboard()

print(f"\n📁 ALL FILES CREATED:")
print(f"   • comprehensive_feature_importance_report.md")
print(f"   • top_20_features_detailed.csv")
print(f"   • comprehensive_feature_dashboard.html")
print(f"   • top_20_features_importance.html")
print(f"   • category_importance_analysis.html")
print(f"   • top_features_correlation.html")
print(f"   • importance_distribution_analysis.html")

print(f"\n🎯 ANALYSIS COMPLETE!")
print(f"The comprehensive feature importance analysis reveals that RISK MEASUREMENT")
print(f"drives the strategy's 2.85x outperformance more than return prediction.")
print(f"Volatility regime detection and trend quality assessment are the key success factors.")
