"""
VizForge v1.2.0 - Pattern Detection Demo

Demonstrates intelligent pattern detection capabilities WITHOUT any API!
Pure statistical and mathematical pattern recognition.

Run this file to see pattern detection in action!
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

from vizforge.intelligence import PatternDetector, detect_patterns, get_pattern_summary

print("=" * 80)
print("VizForge v1.2.0 - Intelligent Pattern Detection")
print("=" * 80)
print("\n🧠 NO API Required - Pure Statistical Intelligence!\n")


# ==================== Example 1: Time Series Patterns ====================

def example_1_time_series():
    """Example 1: Detect patterns in time series data."""
    print("\n" + "=" * 80)
    print("Example 1: Time Series Pattern Detection")
    print("=" * 80)

    # Create time series with trend + seasonality + noise
    np.random.seed(42)
    n = 365

    time = np.arange(n)
    trend = 0.5 * time  # Linear trend
    seasonality = 10 * np.sin(2 * np.pi * time / 30)  # Monthly seasonality
    noise = np.random.normal(0, 5, n)

    values = trend + seasonality + noise

    # Add some spikes
    values[50] += 50  # Spike
    values[150] -= 40  # Dip
    values[250] += 60  # Another spike

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'sales': values
    })

    print("\n📊 Data: 1 year of daily sales data")
    print(f"   - {n} days")
    print(f"   - Range: {values.min():.2f} to {values.max():.2f}")

    # Detect patterns
    print("\n🔍 Detecting patterns...")
    detector = PatternDetector(df)
    patterns = detector.detect_all_patterns(verbose=True)

    print(f"\n✅ Found {len(patterns)} patterns!\n")

    # Show top patterns
    print("Top 5 Patterns:")
    for i, pattern in enumerate(patterns[:5], 1):
        print(f"{i}. [{pattern.pattern_type.value}] {pattern.description}")
        print(f"   Confidence: {pattern.confidence:.2%}")

    print("\n" + "=" * 80)


# ==================== Example 2: Correlation Analysis ====================

def example_2_correlation():
    """Example 2: Detect correlations between variables."""
    print("\n" + "=" * 80)
    print("Example 2: Correlation Detection")
    print("=" * 80)

    # Create correlated data
    np.random.seed(42)
    n = 1000

    x1 = np.random.normal(100, 15, n)
    x2 = 2 * x1 + np.random.normal(0, 10, n)  # Strong positive correlation
    x3 = -1.5 * x1 + np.random.normal(0, 20, n)  # Strong negative correlation
    x4 = np.random.normal(50, 10, n)  # No correlation

    df = pd.DataFrame({
        'advertising': x1,
        'sales': x2,
        'costs': x3,
        'weather': x4
    })

    print("\n📊 Data: 1000 observations, 4 variables")
    print("   - advertising, sales, costs, weather")

    # Detect correlations
    print("\n🔍 Detecting correlations...")
    detector = PatternDetector(df)
    patterns = detector.detect_correlations(threshold=0.7)

    print(f"\n✅ Found {len(patterns)} strong correlations!\n")

    for pattern in patterns:
        print(f"• {pattern.description}")
        print(f"  Confidence: {pattern.confidence:.2%}")
        print()


# ==================== Example 3: Anomaly Detection ====================

def example_3_anomalies():
    """Example 3: Detect anomalies/outliers."""
    print("\n" + "=" * 80)
    print("Example 3: Anomaly Detection")
    print("=" * 80)

    # Create data with anomalies
    np.random.seed(42)
    n = 500

    # Normal data
    normal_data = np.random.normal(100, 10, n)

    # Add anomalies
    anomaly_indices = [50, 150, 250, 350, 450]
    for idx in anomaly_indices:
        normal_data[idx] += np.random.choice([-50, 50])  # Large deviation

    df = pd.DataFrame({'value': normal_data})

    print("\n📊 Data: 500 observations with 5 planted anomalies")

    # Detect anomalies
    print("\n🔍 Detecting anomalies (z-score method)...")
    detector = PatternDetector(df)
    patterns = detector.detect_anomalies(method='zscore')

    print(f"\n✅ Found {len(patterns)} anomalies!\n")

    for pattern in patterns[:5]:  # Show first 5
        print(f"• Index {pattern.location}: {pattern.description}")
        print(f"  Confidence: {pattern.confidence:.2%}")
        print(f"  Value: {pattern.metadata['value']:.2f}")
        print()

    # Try IQR method
    print("\n🔍 Detecting anomalies (IQR method)...")
    patterns_iqr = detector.detect_anomalies(method='iqr')

    print(f"✅ IQR method found {len(patterns_iqr)} anomalies!")


# ==================== Example 4: Cluster Detection ====================

def example_4_clusters():
    """Example 4: Detect natural clusters."""
    print("\n" + "=" * 80)
    print("Example 4: Cluster Detection")
    print("=" * 80)

    # Create clustered data
    np.random.seed(42)

    # Cluster 1: High sales, low cost
    cluster1_sales = np.random.normal(150, 10, 100)
    cluster1_cost = np.random.normal(50, 5, 100)

    # Cluster 2: Medium sales, medium cost
    cluster2_sales = np.random.normal(100, 10, 100)
    cluster2_cost = np.random.normal(80, 5, 100)

    # Cluster 3: Low sales, high cost
    cluster3_sales = np.random.normal(60, 10, 100)
    cluster3_cost = np.random.normal(110, 5, 100)

    sales = np.concatenate([cluster1_sales, cluster2_sales, cluster3_sales])
    cost = np.concatenate([cluster1_cost, cluster2_cost, cluster3_cost])

    df = pd.DataFrame({
        'sales': sales,
        'cost': cost,
        'profit': sales - cost
    })

    print("\n📊 Data: 300 observations with 3 natural clusters")
    print("   - sales, cost, profit")

    # Detect clusters
    print("\n🔍 Detecting clusters...")
    detector = PatternDetector(df)
    patterns = detector.detect_clusters(columns=['sales', 'cost'])

    if patterns:
        print(f"\n✅ Found natural clustering!\n")
        for pattern in patterns:
            print(f"• {pattern.description}")
            print(f"  Confidence: {pattern.confidence:.2%}")
            print(f"  Number of clusters: {pattern.metadata['n_clusters']}")
    else:
        print("\n❌ No significant clusters detected")


# ==================== Example 5: Distribution Analysis ====================

def example_5_distribution():
    """Example 5: Detect distribution characteristics."""
    print("\n" + "=" * 80)
    print("Example 5: Distribution Analysis")
    print("=" * 80)

    # Create different distributions
    np.random.seed(42)

    # Normal distribution
    normal = np.random.normal(100, 15, 1000)

    # Skewed distribution
    skewed = np.random.exponential(2, 1000)

    df_normal = pd.DataFrame({'value': normal})
    df_skewed = pd.DataFrame({'value': skewed})

    # Test normal distribution
    print("\n📊 Dataset 1: Normal distribution")
    detector1 = PatternDetector(df_normal)
    patterns1 = detector1.detect_distribution()

    if patterns1:
        print(f"✅ {patterns1[0].description}")
        print(f"   Confidence: {patterns1[0].confidence:.2%}")

    # Test skewed distribution
    print("\n📊 Dataset 2: Skewed distribution")
    detector2 = PatternDetector(df_skewed)
    patterns2 = detector2.detect_distribution()

    if patterns2:
        print(f"✅ {patterns2[0].description}")
        print(f"   Confidence: {patterns2[0].confidence:.2%}")


# ==================== Example 6: Complete Analysis ====================

def example_6_complete_analysis():
    """Example 6: Complete pattern analysis."""
    print("\n" + "=" * 80)
    print("Example 6: Complete Pattern Analysis")
    print("=" * 80)

    # Create realistic dataset
    np.random.seed(42)
    n = 365

    # Sales with trend + seasonality
    time = np.arange(n)
    trend = 0.3 * time + 100
    seasonality = 20 * np.sin(2 * np.pi * time / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, n)

    sales = trend + seasonality + noise

    # Add spikes for holidays
    sales[50] += 40  # Holiday 1
    sales[150] += 35  # Holiday 2
    sales[250] += 45  # Holiday 3

    # Marketing spend (correlated with sales)
    marketing = 0.5 * sales + np.random.normal(0, 10, n)

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'sales': sales,
        'marketing': marketing,
        'profit': sales * 0.3 - np.random.normal(20, 5, n)
    })

    print("\n📊 Data: 1 year of business metrics")
    print("   - 365 days")
    print("   - 3 variables: sales, marketing, profit")

    # Get complete summary
    print("\n🔍 Running complete pattern analysis...\n")

    summary = get_pattern_summary(df, confidence_threshold=0.7)
    print(summary)


# ==================== Example 7: Quick Analysis ====================

def example_7_quick_analysis():
    """Example 7: Quick one-liner analysis."""
    print("\n" + "=" * 80)
    print("Example 7: Quick Pattern Detection (One-Liner!)")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'revenue': np.random.normal(1000, 200, 100),
        'costs': np.random.normal(600, 100, 100)
    })

    # Add correlation
    df['profit'] = df['revenue'] - df['costs']

    print("\n📊 Data: 100 observations of revenue, costs, profit")

    # One-liner pattern detection!
    print("\n🔍 Quick analysis (one line!):")
    print("\n>>> patterns = detect_patterns(df, verbose=True)\n")

    patterns = detect_patterns(df, verbose=True)

    print(f"\n✅ Found {len(patterns)} patterns in milliseconds!\n")

    # Show top 3
    print("Top 3 Patterns:")
    for i, p in enumerate(patterns[:3], 1):
        print(f"{i}. {p.description}")


# ==================== Main Runner ====================

def main():
    """Run all pattern detection examples."""
    print("\n" + "=" * 80)
    print("🚀 VizForge Pattern Detection - Complete Demo")
    print("=" * 80)

    try:
        example_1_time_series()
        example_2_correlation()
        example_3_anomalies()
        example_4_clusters()
        example_5_distribution()
        example_6_complete_analysis()
        example_7_quick_analysis()

        print("\n" + "=" * 80)
        print("✅ All Pattern Detection Examples Completed!")
        print("=" * 80)

        print("\n💡 Key Takeaways:")
        print("  ✅ NO API required - 100% local processing")
        print("  ✅ Fast - milliseconds for 1000s of rows")
        print("  ✅ Accurate - statistical & mathematical methods")
        print("  ✅ Comprehensive - 14 pattern types detected")
        print("  ✅ Easy - one-liner quick analysis")

        print("\n🎯 Pattern Types Detected:")
        print("  1. Time series trends (increasing, decreasing, stable)")
        print("  2. Seasonality & cycles")
        print("  3. Spikes & dips")
        print("  4. Volatility patterns")
        print("  5. Correlations (positive, negative)")
        print("  6. Natural clusters")
        print("  7. Anomalies & outliers")
        print("  8. Distribution characteristics")

        print("\n📚 Usage:")
        print("  from vizforge.intelligence import detect_patterns")
        print("  patterns = detect_patterns(df, confidence_threshold=0.7)")
        print("  for p in patterns:")
        print("      print(p.description)")

        print("\n" + "=" * 80)
        print("🎉 VizForge v1.2.0 - Intelligence Without APIs!")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
