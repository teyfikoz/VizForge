"""
VizForge v1.2.0 - Smart Chart Recommender v2 Demo

Demonstrates intelligent chart recommendation with multi-criteria scoring.
NO API required - pure statistical analysis + best practices!

Run this file to see smart recommendations in action!
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

from vizforge.intelligence import (
    SmartChartRecommender,
    recommend_chart,
    get_recommendation_report
)

print("=" * 80)
print("VizForge v1.2.0 - Smart Chart Recommender v2")
print("=" * 80)
print("\n🎨 NO API Required - Multi-Criteria Intelligent Recommendations!\n")


# ==================== Example 1: Time Series Data ====================

def example_1_time_series():
    """Example 1: Time series recommendation."""
    print("\n" + "=" * 80)
    print("Example 1: Time Series Data - Revenue Over Time")
    print("=" * 80)

    # Create time series data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    revenue = 10000 + np.cumsum(np.random.normal(100, 500, 365))

    df = pd.DataFrame({
        'date': dates,
        'revenue': revenue,
        'profit': revenue * 0.3 + np.random.normal(0, 500, 365)
    })

    print(f"\n📊 Data: {len(df)} days, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Types: 1 datetime, 2 numeric")

    # Get recommendations
    print("\n🔍 Getting chart recommendations...\n")

    recommender = SmartChartRecommender(df, verbose=True)
    recommendations = recommender.recommend(top_n=3)

    print(f"\n✅ Top 3 Recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.chart_type.value.upper()} - {rec.confidence:.0%} confidence")
        print(f"   {rec.reasoning[:100]}...")
        print()

    # Detailed explanation for top recommendation
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS - Top Recommendation")
    print("=" * 80)
    print(recommender.explain_recommendation(recommendations[0]))


# ==================== Example 2: Categorical Comparison ====================

def example_2_categorical():
    """Example 2: Categorical comparison recommendation."""
    print("\n" + "=" * 80)
    print("Example 2: Categorical Data - Sales by Region")
    print("=" * 80)

    # Create categorical data
    np.random.seed(42)
    regions = ['North', 'South', 'East', 'West', 'Central']

    df = pd.DataFrame({
        'region': np.random.choice(regions, 100),
        'sales': np.random.normal(50000, 10000, 100),
        'customers': np.random.randint(100, 500, 100)
    })

    print(f"\n📊 Data: {len(df)} records, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Types: 1 categorical, 2 numeric")

    # Get recommendations
    print("\n🔍 Analyzing best chart types...\n")

    recommendations = recommend_chart(df, top_n=3, verbose=True)

    print(f"\n✅ Recommended Charts:\n")

    for i, rec in enumerate(recommendations, 1):
        confidence_bar = "█" * int(rec.confidence * 20)
        print(f"{i}. {rec.chart_type.value.upper()}")
        print(f"   Confidence: {confidence_bar} {rec.confidence:.0%}")
        print(f"   Category: {rec.category.value}")
        print(f"   Pros: {', '.join(rec.pros[:2])}")
        print()


# ==================== Example 3: Relationship Analysis ====================

def example_3_relationship():
    """Example 3: Relationship/correlation recommendation."""
    print("\n" + "=" * 80)
    print("Example 3: Relationship Data - Marketing vs Sales")
    print("=" * 80)

    # Create correlated data
    np.random.seed(42)
    marketing_spend = np.random.normal(10000, 2000, 200)
    sales = 2.5 * marketing_spend + np.random.normal(0, 3000, 200)
    profit = sales - marketing_spend - np.random.normal(5000, 1000, 200)

    df = pd.DataFrame({
        'marketing_spend': marketing_spend,
        'sales': sales,
        'profit': profit,
        'roi': (profit / marketing_spend) * 100
    })

    print(f"\n📊 Data: {len(df)} observations, {len(df.columns)} numeric columns")
    print(f"   Perfect for correlation analysis!")

    # Get comprehensive report
    print("\n🔍 Generating comprehensive recommendation report...\n")

    report = get_recommendation_report(df, top_n=5)
    print(report)


# ==================== Example 4: Distribution Analysis ====================

def example_4_distribution():
    """Example 4: Distribution data recommendation."""
    print("\n" + "=" * 80)
    print("Example 4: Distribution Data - Customer Ages")
    print("=" * 80)

    # Create distribution data
    np.random.seed(42)
    ages = np.random.normal(35, 12, 1000)

    df = pd.DataFrame({
        'age': ages,
        'income': ages * 1500 + np.random.normal(30000, 10000, 1000)
    })

    print(f"\n📊 Data: {len(df)} customers, {len(df.columns)} numeric columns")

    # Get recommendations
    recommender = SmartChartRecommender(df, verbose=True)
    recommendations = recommender.recommend(top_n=5)

    print(f"\n✅ Top 5 Recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.chart_type.value.upper()} ({rec.category.value})")
        print(f"   Confidence: {rec.confidence:.0%}")
        print(f"   Best for: {rec.reasoning[:80]}...")
        code_lines = rec.example_code.split('\n')
        code_snippet = code_lines[1] if len(code_lines) > 1 else code_lines[0]
        print(f"   Quick Start: {code_snippet}")
        print()


# ==================== Example 5: Large Dataset ====================

def example_5_large_dataset():
    """Example 5: Large dataset recommendation."""
    print("\n" + "=" * 80)
    print("Example 5: Large Dataset - 100,000 Records")
    print("=" * 80)

    # Create large dataset
    np.random.seed(42)
    n = 100000

    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='min'),
        'sensor_1': np.random.normal(25, 5, n),
        'sensor_2': np.random.normal(50, 10, n),
        'status': np.random.choice(['OK', 'Warning', 'Error'], n)
    })

    print(f"\n📊 Data: {len(df):,} records (LARGE!)")
    print(f"   Columns: {len(df.columns)} ({df.select_dtypes(include=[np.number]).shape[1]} numeric)")

    # Get recommendations
    print("\n🔍 Analyzing (considering performance)...\n")

    recommender = SmartChartRecommender(df, verbose=True)
    recommendations = recommender.recommend(top_n=3)

    print(f"\n✅ Performance-Optimized Recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.chart_type.value.upper()}")
        print(f"   Confidence: {rec.confidence:.0%}")
        print(f"   Performance Score: {rec.performance_score:.0%}")
        print(f"   Note: {rec.cons[0] if rec.cons else 'No performance issues'}")
        print()


# ==================== Example 6: Mixed Data Types ====================

def example_6_mixed_types():
    """Example 6: Mixed data types recommendation."""
    print("\n" + "=" * 80)
    print("Example 6: Mixed Data - E-Commerce Dashboard")
    print("=" * 80)

    # Create mixed data
    np.random.seed(42)

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=90, freq='D'),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 90),
        'sales': np.random.normal(5000, 1000, 90),
        'units_sold': np.random.randint(50, 200, 90),
        'rating': np.random.uniform(3.5, 5.0, 90)
    })

    print(f"\n📊 Data: {len(df)} days, {len(df.columns)} columns")
    print(f"   Types: 1 datetime, 1 categorical, 3 numeric")
    print(f"   Complex data - multiple visualization options!")

    # Get comprehensive analysis
    recommender = SmartChartRecommender(df, verbose=True)
    recommendations = recommender.recommend(top_n=5)

    print(f"\n✅ Top 5 Chart Options:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.chart_type.value.upper()} ({rec.confidence:.0%})")
        print(f"   Category: {rec.category.value}")
        print(f"   Scoring Breakdown:")
        print(f"     • Data Fit:        {rec.data_fit_score:.0%}")
        print(f"     • Best Practices:  {rec.best_practice_score:.0%}")
        print(f"     • Performance:     {rec.performance_score:.0%}")
        print(f"     • Accessibility:   {rec.accessibility_score:.0%}")
        print(f"     • Aesthetics:      {rec.aesthetic_score:.0%}")
        print()


# ==================== Example 7: Quick One-Liner ====================

def example_7_quick():
    """Example 7: Quick one-liner recommendation."""
    print("\n" + "=" * 80)
    print("Example 7: Quick One-Liner Recommendation")
    print("=" * 80)

    # Simple data
    np.random.seed(42)
    df = pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'],
        'sales': [15000, 23000, 18000, 31000, 12000]
    })

    print(f"\n📊 Data: {len(df)} products, {len(df.columns)} columns")

    # One-liner!
    print("\n🔍 Quick recommendation (one line!):")
    print("\n>>> recommendations = recommend_chart(df, top_n=1)")

    recommendations = recommend_chart(df, top_n=1)

    print(f"\n✅ Best Chart: {recommendations[0].chart_type.value.upper()}")
    print(f"   Confidence: {recommendations[0].confidence:.0%}")
    print(f"   Reason: {recommendations[0].reasoning}")


# ==================== Example 8: Composition/Part-to-Whole ====================

def example_8_composition():
    """Example 8: Composition data recommendation."""
    print("\n" + "=" * 80)
    print("Example 8: Composition Data - Market Share")
    print("=" * 80)

    # Create composition data
    df = pd.DataFrame({
        'company': ['Apple', 'Samsung', 'Google', 'Others'],
        'market_share': [28, 24, 15, 33]
    })

    print(f"\n📊 Data: {len(df)} companies, {len(df.columns)} columns")
    print(f"   Type: Part-to-whole relationship")

    # Get recommendations
    recommender = SmartChartRecommender(df, verbose=True)
    recommendations = recommender.recommend(top_n=3)

    print(f"\n✅ Recommendations for Part-to-Whole:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.chart_type.value.upper()}")
        print(f"   Confidence: {rec.confidence:.0%}")
        print(f"   ✅ Pros:")
        for pro in rec.pros[:2]:
            print(f"      • {pro}")
        print(f"   ⚠️  Cons:")
        for con in rec.cons[:1]:
            print(f"      • {con}")
        print()


# ==================== Main Runner ====================

def main():
    """Run all chart recommender examples."""
    print("\n" + "=" * 80)
    print("🚀 VizForge Smart Chart Recommender v2 - Complete Demo")
    print("=" * 80)

    try:
        example_1_time_series()
        example_2_categorical()
        example_3_relationship()
        example_4_distribution()
        example_5_large_dataset()
        example_6_mixed_types()
        example_7_quick()
        example_8_composition()

        print("\n" + "=" * 80)
        print("✅ All Smart Chart Recommender Examples Completed!")
        print("=" * 80)

        print("\n💡 Key Takeaways:")
        print("  ✅ NO API required - 100% local intelligence")
        print("  ✅ Multi-criteria scoring (5 factors)")
        print("  ✅ Fast - instant recommendations")
        print("  ✅ Comprehensive - 15+ chart types")
        print("  ✅ Context-aware - considers data size, types, cardinality")
        print("  ✅ Best practices - built-in visualization guidelines")
        print("  ✅ Accessible - WCAG considerations")
        print("  ✅ Performance-optimized - handles large datasets")

        print("\n🎯 Scoring Criteria:")
        print("  1. Data Fit:        How well the chart matches data characteristics")
        print("  2. Best Practices:  Follows visualization guidelines")
        print("  3. Performance:     Rendering speed & efficiency")
        print("  4. Accessibility:   WCAG compliance & readability")
        print("  5. Aesthetics:      Visual appeal & clarity")

        print("\n📚 Usage:")
        print("  # Quick recommendation")
        print("  from vizforge.intelligence import recommend_chart")
        print("  recommendations = recommend_chart(df, top_n=3)")
        print()
        print("  # Detailed analysis")
        print("  from vizforge.intelligence import SmartChartRecommender")
        print("  recommender = SmartChartRecommender(df)")
        print("  recs = recommender.recommend(top_n=5)")
        print("  print(recommender.explain_recommendation(recs[0]))")
        print()
        print("  # Comprehensive report")
        print("  from vizforge.intelligence import get_recommendation_report")
        print("  report = get_recommendation_report(df)")
        print("  print(report)")

        print("\n" + "=" * 80)
        print("🎉 VizForge v1.2.0 - Smart Recommendations Without APIs!")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
