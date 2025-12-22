"""
VizForge v1.3.0 - Auto Data Storytelling Demo

AUTOMATIC INSIGHTS + NARRATIVES FROM DATA!
NO API required - intelligent pattern discovery!

This is REVOLUTIONARY! Automatic story generation from data!
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

import vizforge as vz
from vizforge.storytelling import (
    discover_insights,
    generate_story,
    generate_report,
    InsightType,
    ReportFormat,
)

print("=" * 80)
print("VizForge v1.3.0 - Auto Data Storytelling Engine")
print("=" * 80)
print("\n📖 AUTOMATIC INSIGHTS + NARRATIVES - NO API REQUIRED!")
print("🚀 Discover patterns, Generate stories, Export reports!\n")


# ==================== Example 1: Insight Discovery ====================

def example_1_discovery():
    """Example 1: Automatic insight discovery."""
    print("\n" + "=" * 80)
    print("Example 1: Automatic Insight Discovery")
    print("=" * 80)

    # Create sample e-commerce data
    np.random.seed(42)
    n = 365  # 1 year

    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    trend = 5000 + np.arange(n) * 10  # Growing trend
    weekly = 1000 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly pattern
    noise = np.random.normal(0, 500, n)

    df = pd.DataFrame({
        'date': dates,
        'sales': trend + weekly + noise,
        'profit': (trend + weekly + noise) * 0.3,
        'customers': np.random.randint(100, 500, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food'], n),
    })

    # Add some anomalies
    df.loc[100, 'sales'] *= 0.3  # Big drop
    df.loc[200, 'sales'] *= 2.5  # Big spike

    print(f"\n📊 Sample Data: E-Commerce Sales")
    print(f"   Period: {n} days")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Total Revenue: ${df['sales'].sum():,.2f}")

    print("\n🔍 Discovering insights...")

    # DISCOVER INSIGHTS - ONE LINE!
    insights = discover_insights(df, time_column='date', max_insights=10)

    print(f"\n✅ Discovery complete! Found {len(insights)} insights:\n")

    # Display insights
    for i, insight in enumerate(insights, 1):
        emoji = {
            'summary': '📋',
            'trend': '📈',
            'correlation': '🔗',
            'anomaly': '⚠️',
            'extreme': '🎯',
            'distribution': '📊',
            'seasonality': '🔄',
            'comparison': '⚖️',
            'change': '📉',
        }.get(insight.type.value, '💡')

        print(f"{i}. [{emoji} {insight.type.value.upper()}] {insight.title}")
        print(f"   Importance: {insight.importance:.0%}")
        print(f"   {insight.description}")

        if insight.recommendation:
            print(f"   💡 Recommendation: {insight.recommendation}")

        print()


# ==================== Example 2: Story Generation ====================

def example_2_story():
    """Example 2: Automatic narrative generation."""
    print("\n" + "=" * 80)
    print("Example 2: Automatic Story Generation")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    n = 180

    df = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=n, freq='D'),
        'revenue': 10000 + np.arange(n) * 50 + np.random.normal(0, 1000, n),
        'expenses': 7000 + np.arange(n) * 30 + np.random.normal(0, 500, n),
        'users': 1000 + np.arange(n) * 10 + np.random.randint(-50, 50, n),
    })

    df['profit'] = df['revenue'] - df['expenses']

    print(f"\n📊 Sample Data: Business Metrics (6 months)")
    print(f"   Average Revenue: ${df['revenue'].mean():,.2f}/day")
    print(f"   Average Profit: ${df['profit'].mean():,.2f}/day")

    print("\n📝 Generating data story...")

    # GENERATE STORY - ONE LINE!
    story = generate_story(df, max_insights=8)

    print(f"\n✅ Story generated!\n")
    print("=" * 80)
    print(f"TITLE: {story.title}")
    print("=" * 80)
    print(f"\nSUMMARY:\n{story.summary}")
    print(f"\nINSIGHTS: {len(story.insights)}")
    print(f"RECOMMENDATIONS: {len(story.recommendations)}")

    print("\n" + "-" * 80)
    print("FULL NARRATIVE (first 500 chars):")
    print("-" * 80)
    print(story.narrative[:500] + "...\n")

    if story.recommendations:
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(story.recommendations[:3], 1):
            print(f"  {i}. {rec}")


# ==================== Example 3: Report Export ====================

def example_3_export():
    """Example 3: Report generation and export."""
    print("\n" + "=" * 80)
    print("Example 3: Report Generation & Export")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=90, freq='D'),
        'metric_a': 100 + np.cumsum(np.random.normal(2, 10, 90)),
        'metric_b': 50 + np.cumsum(np.random.normal(1, 5, 90)),
        'category': np.random.choice(['A', 'B', 'C'], 90),
    })

    print("\n📝 Generating complete report...")

    # Generate story
    story = generate_story(df, title="Quarterly Performance Report", max_insights=5)

    print(f"✅ Story generated with {len(story.insights)} insights\n")

    # Export to different formats
    print("💾 Exporting to multiple formats...")

    try:
        # Markdown
        md_path = generate_report(story, "/tmp/vizforge_report.md")
        print(f"   ✅ Markdown: {md_path}")

        # HTML
        html_path = generate_report(story, "/tmp/vizforge_report.html")
        print(f"   ✅ HTML: {html_path}")

        # Text
        txt_path = generate_report(story, "/tmp/vizforge_report.txt")
        print(f"   ✅ Text: {txt_path}")

        print("\n📄 Report preview (Markdown, first 800 chars):")
        print("-" * 80)
        with open(md_path, 'r') as f:
            print(f.read()[:800] + "...")

    except Exception as e:
        print(f"   ❌ Export error: {e}")


# ==================== Example 4: Real-World Scenario ====================

def example_4_realworld():
    """Example 4: Real-world business scenario."""
    print("\n" + "=" * 80)
    print("Example 4: Real-World Business Analysis")
    print("=" * 80)

    # Create realistic sales data
    np.random.seed(42)
    n = 365

    # Simulate seasonal business (summer peak)
    seasonal = 50 * np.sin(2 * np.pi * (np.arange(n) - 180) / 365)  # Peak in summer
    trend = 100 + np.arange(n) * 0.5  # Slow growth
    weekly = 20 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly pattern
    noise = np.random.normal(0, 15, n)

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n, freq='D'),
        'daily_sales': 1000 + seasonal + trend + weekly + noise,
        'visitors': 500 + (seasonal * 2) + (trend * 0.5) + np.random.normal(0, 50, n),
        'conversion_rate': 0.05 + (seasonal / 1000) + np.random.normal(0, 0.01, n),
        'product_line': np.random.choice(['Premium', 'Standard', 'Budget'], n),
        'channel': np.random.choice(['Online', 'In-Store', 'Mobile'], n),
    })

    # Add marketing campaign effect
    df.loc[200:230, 'daily_sales'] *= 1.3  # Campaign boost

    print(f"\n📊 Real-World Dataset: Annual E-Commerce Performance")
    print(f"   Period: Full year (365 days)")
    print(f"   Total Sales: ${df['daily_sales'].sum():,.2f}")
    print(f"   Avg Conversion: {df['conversion_rate'].mean():.1%}")

    print("\n🔍 Running complete analysis...")

    # Discover insights
    insights = discover_insights(
        df,
        time_column='date',
        max_insights=15,
        min_importance=0.6
    )

    print(f"\n✅ Analysis complete! Key findings:\n")

    # Show top 5 insights
    for i, insight in enumerate(insights[:5], 1):
        print(f"{i}. {insight.title} (importance: {insight.importance:.0%})")
        print(f"   {insight.description}\n")

    # Generate executive report
    print("📝 Generating executive report...")

    story = generate_story(
        df,
        title="2024 Annual E-Commerce Performance Review",
        max_insights=10
    )

    print(f"\n✅ Executive report ready!")
    print(f"   Insights: {len(story.insights)}")
    print(f"   Recommendations: {len(story.recommendations)}")

    # Save report
    try:
        report_path = generate_report(story, "/tmp/executive_report.html")
        print(f"   💾 Saved to: {report_path}")
    except Exception as e:
        print(f"   ⚠️ Could not save: {e}")


# ==================== Example 5: Custom Analysis ====================

def example_5_custom():
    """Example 5: Custom-focused analysis."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Focused Analysis")
    print("=" * 80)

    # Marketing campaign data
    np.random.seed(42)
    campaigns = ['Email', 'Social', 'Display', 'Search']
    n = 100

    df = pd.DataFrame({
        'campaign': np.random.choice(campaigns, n),
        'spend': np.random.uniform(1000, 10000, n),
        'impressions': np.random.randint(10000, 100000, n),
        'clicks': np.random.randint(100, 5000, n),
        'conversions': np.random.randint(10, 500, n),
        'revenue': np.random.uniform(5000, 50000, n),
    })

    df['ctr'] = df['clicks'] / df['impressions'] * 100
    df['cpa'] = df['spend'] / df['conversions']
    df['roi'] = (df['revenue'] - df['spend']) / df['spend'] * 100

    print(f"\n📊 Marketing Campaign Data")
    print(f"   Campaigns: {n} runs across {len(campaigns)} channels")
    print(f"   Total Spend: ${df['spend'].sum():,.2f}")
    print(f"   Total Revenue: ${df['revenue'].sum():,.2f}")
    print(f"   Overall ROI: {((df['revenue'].sum() - df['spend'].sum()) / df['spend'].sum() * 100):.1f}%")

    print("\n🎯 Analyzing campaign performance...")

    # Discover insights
    insights = discover_insights(df, max_insights=8)

    # Filter for specific types
    comparison_insights = [i for i in insights if i.type == InsightType.COMPARISON]
    correlation_insights = [i for i in insights if i.type == InsightType.CORRELATION]

    print(f"\n✅ Found {len(insights)} total insights:")
    print(f"   - {len(comparison_insights)} comparisons between campaigns")
    print(f"   - {len(correlation_insights)} correlations between metrics")

    if comparison_insights:
        print(f"\n📊 Top Campaign Comparison:")
        insight = comparison_insights[0]
        print(f"   {insight.description}")

    # Generate marketing report
    story = generate_story(
        df,
        title="Marketing Campaign Performance Analysis",
        max_insights=6
    )

    print(f"\n📝 Marketing Report Generated:")
    print(f"   {story.summary}")


# ==================== Main ====================

def main():
    """Run all storytelling examples."""
    try:
        example_1_discovery()
        example_2_story()
        example_3_export()
        example_4_realworld()
        example_5_custom()

        print("\n" + "=" * 80)
        print("✅ All Auto Data Storytelling Examples Completed!")
        print("=" * 80)

        print("\n💡 Key Features:")
        print("  ✅ NO API required - 100% pattern recognition")
        print("  ✅ Automatic insight discovery (9 types)")
        print("  ✅ Natural language narratives")
        print("  ✅ Multiple export formats (Markdown, HTML, Text)")
        print("  ✅ Actionable recommendations")
        print("  ✅ Fast - instant analysis")

        print("\n🎯 Insight Types Detected:")
        print("  • Summary: Dataset overview and statistics")
        print("  • Trend: Growing/declining patterns")
        print("  • Correlation: Relationships between variables")
        print("  • Anomaly: Unusual data points")
        print("  • Extreme: Maximum/minimum values")
        print("  • Distribution: Data distribution characteristics")
        print("  • Seasonality: Periodic patterns")
        print("  • Comparison: Group comparisons")
        print("  • Change: Significant changes over time")

        print("\n📚 Usage:")
        print("  from vizforge.storytelling import discover_insights, generate_story, generate_report")
        print("  insights = discover_insights(df)")
        print("  story = generate_story(df)")
        print("  generate_report(story, 'report.html')")

        print("\n" + "=" * 80)
        print("🎉 VizForge v1.3.0 - Auto Data Storytelling!")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
