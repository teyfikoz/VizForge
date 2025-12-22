"""
VizForge v1.0.0 - Smart Charts Demo

Demonstrates intelligent chart selection and auto-optimization features.
Shows the USP features that make VizForge superior to competitors.

NEW v1.0.0 Features Demonstrated:
- Automatic chart type selection (local ML, no API costs)
- Data quality scoring
- Automatic insights generation
- Color optimization for accessibility
- Best practices recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import VizForge v1.0.0 features
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

from vizforge.intelligence.chart_selector import ChartSelector
from vizforge.intelligence.data_profiler import DataProfiler, DataQualityScorer
from vizforge.intelligence.insights_engine import InsightsEngine
from vizforge.intelligence.color_optimizer import ColorOptimizer
from vizforge.intelligence.recommendation import RecommendationEngine


# ==================== Sample Data ====================

def create_sample_sales_data():
    """Create sample sales data for demos."""
    dates = pd.date_range('2024-01-01', periods=365, freq='D')

    return pd.DataFrame({
        'date': dates,
        'sales': np.random.uniform(1000, 5000, 365) + np.linspace(0, 2000, 365),  # Upward trend
        'revenue': np.random.uniform(5000, 20000, 365),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 365),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    })


def create_sample_geographic_data():
    """Create sample geographic data."""
    return pd.DataFrame({
        'country': ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia'],
        'city': ['New York', 'Toronto', 'London', 'Berlin', 'Paris', 'Tokyo', 'Sydney'],
        'latitude': [40.7, 43.7, 51.5, 52.5, 48.9, 35.7, -33.9],
        'longitude': [-74.0, -79.4, -0.1, 13.4, 2.4, 139.7, 151.2],
        'sales': [50000, 35000, 42000, 38000, 41000, 55000, 32000],
        'population': [8.3, 2.9, 9.0, 3.7, 2.2, 13.9, 5.3],
    })


def create_sample_correlation_data():
    """Create sample data with correlations."""
    x = np.random.randn(100)

    return pd.DataFrame({
        'marketing_spend': x * 1000 + 5000,
        'sales': x * 5000 + 20000 + np.random.randn(100) * 500,  # Correlated with marketing
        'temperature': np.random.uniform(10, 35, 100),  # Uncorrelated
        'customer_satisfaction': x * 2 + 75 + np.random.randn(100) * 5,  # Correlated
    })


# ==================== Demo 1: Automatic Chart Selection ====================

def demo_auto_chart_selection():
    """
    Demo: Automatic chart type selection.

    VizForge's USP: No competitor offers free automatic chart selection!
    """
    print("\n" + "="*60)
    print("DEMO 1: Automatic Chart Selection (USP Feature)")
    print("="*60)

    selector = ChartSelector()

    # Test 1: Time series data
    print("\n📊 Test 1: Time Series Data")
    sales_data = create_sample_sales_data()
    result = selector.recommend(sales_data, x='date', y='sales')

    print(f"✅ Recommended Chart: {result['primary']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Reasoning: {result['reasoning']}")
    print(f"   Alternatives: {', '.join(result['alternatives'][:3])}")

    # Test 2: Geographic data
    print("\n🗺️ Test 2: Geographic Data")
    geo_data = create_sample_geographic_data()
    result = selector.recommend(geo_data, x='latitude', y='longitude')

    print(f"✅ Recommended Chart: {result['primary']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Reasoning: {result['reasoning']}")

    # Test 3: Categorical data
    print("\n📈 Test 3: Categorical Data")
    sales_data = create_sample_sales_data()
    result = selector.recommend(sales_data, x='category', y='sales')

    print(f"✅ Recommended Chart: {result['primary']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Reasoning: {result['reasoning']}")

    # Test 4: Correlation analysis
    print("\n🔗 Test 4: Correlation Analysis")
    corr_data = create_sample_correlation_data()
    result = selector.recommend(corr_data, x='marketing_spend', y='sales', intent='correlation')

    print(f"✅ Recommended Chart: {result['primary']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Reasoning: {result['reasoning']}")


# ==================== Demo 2: Data Quality Scoring ====================

def demo_data_quality_scoring():
    """
    Demo: Automatic data quality assessment.

    Tableau charges $70/user/month for this. VizForge: FREE!
    """
    print("\n" + "="*60)
    print("DEMO 2: Data Quality Scoring (Tableau $$$$ vs VizForge FREE)")
    print("="*60)

    scorer = DataQualityScorer()

    # Test 1: Clean data
    print("\n✨ Test 1: Clean Data")
    clean_data = pd.DataFrame({
        'id': range(1000),
        'value': np.random.uniform(10, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
    })

    report = scorer.score(clean_data)

    print(f"📊 Overall Quality Score: {report.score:.1f}/100")
    print(f"   Completeness: {report.completeness:.1f}/100 (missing values)")
    print(f"   Consistency: {report.consistency:.1f}/100 (format consistency)")
    print(f"   Accuracy: {report.accuracy:.1f}/100 (outliers)")
    print(f"   Uniqueness: {report.uniqueness:.1f}/100 (duplicates)")

    if report.issues:
        print(f"\n⚠️ Issues Found: {len(report.issues)}")
        for issue in report.issues[:3]:
            print(f"   - {issue}")

    # Test 2: Dirty data
    print("\n🚨 Test 2: Dirty Data")
    dirty_data = pd.DataFrame({
        'id': [1, 2, 3, None, 5, None, 7, 8, 9, 10] * 10,  # Missing values
        'value': [10, 10, 10, 10, 10, 200, 10, 10, 10, 10] * 10,  # Outlier
        'category': ['A'] * 100,  # No variance
    })

    report = scorer.score(dirty_data)

    print(f"📊 Overall Quality Score: {report.score:.1f}/100")
    print(f"   Completeness: {report.completeness:.1f}/100")
    print(f"   Consistency: {report.consistency:.1f}/100")
    print(f"   Accuracy: {report.accuracy:.1f}/100")
    print(f"   Uniqueness: {report.uniqueness:.1f}/100")

    if report.issues:
        print(f"\n⚠️ Issues Found: {len(report.issues)}")
        for issue in report.issues:
            print(f"   - {issue}")

    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"   - {rec}")


# ==================== Demo 3: Automatic Insights ====================

def demo_automatic_insights():
    """
    Demo: Automatic insights generation.

    Tableau's "Explain Data" feature. VizForge: FREE!
    """
    print("\n" + "="*60)
    print("DEMO 3: Automatic Insights (Tableau's 'Explain Data' Clone)")
    print("="*60)

    engine = InsightsEngine()

    # Create data with interesting patterns
    data = create_sample_sales_data()

    # Generate insights
    insights = engine.generate_insights(data, target_column='sales')

    print(f"\n💡 Found {len(insights)} insights:")
    print("-" * 60)

    for i, insight in enumerate(insights[:5], 1):
        severity_emoji = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢',
            'INFO': 'ℹ️'
        }

        print(f"\n{i}. {severity_emoji.get(insight.severity.name, 'ℹ️')} {insight.title}")
        print(f"   Type: {insight.insight_type.value.upper()}")
        print(f"   Severity: {insight.severity.name}")
        print(f"   Confidence: {insight.confidence:.1%}")
        print(f"   Description: {insight.description}")

        if insight.affected_columns:
            print(f"   Affected: {', '.join(insight.affected_columns)}")

        if insight.recommended_action:
            print(f"   ➡️ Action: {insight.recommended_action}")


# ==================== Demo 4: Color Optimization ====================

def demo_color_optimization():
    """
    Demo: Accessible color palette generation.

    WCAG 2.1 AA+ compliance automatically.
    """
    print("\n" + "="*60)
    print("DEMO 4: Color Optimization for Accessibility")
    print("="*60)

    optimizer = ColorOptimizer()

    # Show all available palettes
    print("\n🎨 Available Palettes:")
    palettes = ['colorblind_safe', 'high_contrast', 'tableau', 'material', 'pastel']

    for palette_name in palettes:
        palette = optimizer.get_palette(palette_name)
        print(f"\n   {palette_name.upper()}: {len(palette)} colors")
        print(f"   Colors: {', '.join(palette[:5])}")

    # Test WCAG contrast
    print("\n✅ WCAG Contrast Testing:")

    test_pairs = [
        ('#000000', '#FFFFFF', 'Black on White'),
        ('#FFFFFF', '#000000', 'White on Black'),
        ('#0173B2', '#FFFFFF', 'Blue on White'),
        ('#FF0000', '#FFFFFF', 'Red on White'),
    ]

    for color1, color2, label in test_pairs:
        contrast = optimizer.calculate_contrast(color1, color2)

        # WCAG standards
        aa_normal = contrast >= 4.5  # WCAG AA for normal text
        aa_large = contrast >= 3.0   # WCAG AA for large text
        aaa_normal = contrast >= 7.0  # WCAG AAA for normal text

        status = "AAA ✅" if aaa_normal else ("AA ✅" if aa_normal else "FAIL ❌")

        print(f"   {label}: {contrast:.2f}:1 - {status}")

    # Generate custom accessible palette
    print("\n🎨 Generate Custom Accessible Palette:")
    custom_palette = optimizer.generate_palette(n_colors=5, colorblind_safe=True)
    print(f"   Generated: {', '.join(custom_palette)}")


# ==================== Demo 5: Best Practices Recommendations ====================

def demo_best_practices():
    """
    Demo: Automatic best practices recommendations.

    Get recommendations for improving your visualizations.
    """
    print("\n" + "="*60)
    print("DEMO 5: Best Practices Recommendations")
    print("="*60)

    engine = RecommendationEngine()

    # Test with different datasets
    datasets = [
        ("Time Series", create_sample_sales_data()),
        ("Geographic", create_sample_geographic_data()),
        ("Correlation", create_sample_correlation_data()),
    ]

    for name, data in datasets:
        print(f"\n📊 {name} Data:")
        recommendations = engine.generate_recommendations(data)

        if recommendations:
            print(f"   Found {len(recommendations)} recommendations:")
            for rec in recommendations[:3]:
                print(f"\n   • {rec.title}")
                print(f"     {rec.description}")
                print(f"     Priority: {rec.priority.name}")
                print(f"     Impact: {rec.impact}")


# ==================== Demo 6: Complete Workflow ====================

def demo_complete_workflow():
    """
    Demo: Complete intelligent visualization workflow.

    Shows how all features work together.
    """
    print("\n" + "="*60)
    print("DEMO 6: Complete Intelligent Workflow")
    print("="*60)

    # Step 1: Load data
    print("\n📥 Step 1: Load Data")
    data = create_sample_sales_data()
    print(f"   Loaded {len(data)} rows, {len(data.columns)} columns")

    # Step 2: Profile data
    print("\n🔍 Step 2: Profile Data")
    profiler = DataProfiler()
    profile = profiler.profile(data)

    print(f"   Temporal columns: {len(profile.temporal_cols)}")
    print(f"   Numeric columns: {len(profile.numeric_cols)}")
    print(f"   Categorical columns: {len(profile.categorical_cols)}")

    # Step 3: Check quality
    print("\n✅ Step 3: Check Data Quality")
    scorer = DataQualityScorer()
    quality = scorer.score(data)

    print(f"   Quality Score: {quality.score:.1f}/100")
    if quality.score >= 80:
        print("   ✅ Data quality is good!")
    else:
        print("   ⚠️ Data quality needs improvement")

    # Step 4: Get chart recommendation
    print("\n📊 Step 4: Get Chart Recommendation")
    selector = ChartSelector()
    chart_rec = selector.recommend(data, x='date', y='sales')

    print(f"   Recommended: {chart_rec['primary']}")
    print(f"   Confidence: {chart_rec['confidence']:.1%}")

    # Step 5: Generate insights
    print("\n💡 Step 5: Generate Insights")
    insights_engine = InsightsEngine()
    insights = insights_engine.generate_insights(data, target_column='sales')

    print(f"   Found {len(insights)} insights")
    if insights:
        print(f"   Top insight: {insights[0].title}")

    # Step 6: Get color palette
    print("\n🎨 Step 6: Get Accessible Colors")
    optimizer = ColorOptimizer()
    palette = optimizer.get_palette('colorblind_safe')

    print(f"   Using colorblind-safe palette ({len(palette)} colors)")

    # Step 7: Get recommendations
    print("\n📋 Step 7: Get Best Practices")
    rec_engine = RecommendationEngine()
    recommendations = rec_engine.generate_recommendations(data)

    print(f"   Found {len(recommendations)} recommendations")

    print("\n✅ Complete workflow finished!")
    print("   Ready to create an optimized, accessible, insightful visualization!")


# ==================== Main ====================

def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🚀 VizForge v1.0.0 - Smart Charts Demo")
    print("="*60)
    print("\nDemonstrating features that make VizForge SUPERIOR to:")
    print("  • Tableau ($$$$$ expensive)")
    print("  • Streamlit (no intelligence)")
    print("  • Plotly (manual everything)")
    print("\nAll features use LOCAL ML - NO API COSTS! 🎉")

    try:
        # Run all demos
        demo_auto_chart_selection()
        demo_data_quality_scoring()
        demo_automatic_insights()
        demo_color_optimization()
        demo_best_practices()
        demo_complete_workflow()

        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        print("\n💡 Key Takeaways:")
        print("  1. Smart chart selection - NO competitor has this for free")
        print("  2. Data quality scoring - Tableau charges $70/user/month")
        print("  3. Auto insights - Tableau's 'Explain Data' clone")
        print("  4. Accessible colors - WCAG 2.1 AA+ automatically")
        print("  5. Best practices - Expert recommendations for free")
        print("  6. All LOCAL - No API costs, no data privacy issues")

        print("\n🎯 Next Steps:")
        print("  - Run: python examples/interactive_dashboard.py")
        print("  - Read: MIGRATION_GUIDE.md")
        print("  - Explore: vizforge.intelligence module")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
