"""
VizForge v1.3.0 - Predictive Analytics Demo

PREDICT THE FUTURE WITH ONE LINE!
NO API required - pure statistical intelligence!

This is REVOLUTIONARY! Time series forecasting, trend detection,
anomaly detection, and seasonality analysis - all built-in!
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

import vizforge as vz
from vizforge.predictive import (
    forecast,
    detect_trend,
    detect_anomalies,
    analyze_seasonality
)

print("=" * 80)
print("VizForge v1.3.0 - Predictive Analytics Engine")
print("=" * 80)
print("\n🔮 PREDICT THE FUTURE - NO API REQUIRED!")
print("🚀 Statistical Forecasting, Trend Detection, Anomaly Detection, Seasonality!\n")


# ==================== Example 1: Forecasting ====================

def example_1_forecasting():
    """Example 1: Time series forecasting."""
    print("\n" + "=" * 80)
    print("Example 1: Time Series Forecasting")
    print("=" * 80)

    # Create sample sales data with trend
    np.random.seed(42)
    n = 100
    time = np.arange(n)
    trend = 1000 + time * 5  # Growing trend
    noise = np.random.normal(0, 50, n)
    sales = trend + noise

    df = pd.DataFrame({
        'day': np.arange(n),
        'sales': sales
    })

    print(f"\n📊 Data: {len(df)} days of sales data")
    print(f"   Mean: ${df['sales'].mean():.2f}")
    print(f"   Std: ${df['sales'].std():.2f}")

    print("\n🔮 Forecasting next 30 days...")

    # FORECAST - ONE LINE!
    result = forecast(df['sales'], periods=30, method='auto')

    print(f"\n✅ Forecast complete!")
    print(f"   Method: {result.method}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   MSE: {result.mse:.2f}")
    print(f"   MAE: {result.mae:.2f}")

    print(f"\n📈 Predictions (first 5 days):")
    for i in range(min(5, len(result.predictions))):
        print(f"   Day {n+i+1}: ${result.predictions.iloc[i]:.2f} "
              f"[${result.lower_bound.iloc[i]:.2f} - ${result.upper_bound.iloc[i]:.2f}]")

    print(f"\n   ... (total {len(result.predictions)} predictions)")

    # Test different methods
    print("\n🔬 Testing different forecasting methods:")
    methods = ['linear', 'exponential_smoothing', 'moving_average']

    for method in methods:
        result = forecast(df['sales'], periods=10, method=method)
        print(f"   {method:25s} | MAE: {result.mae:.2f} | Confidence: {result.confidence:.0%}")


# ==================== Example 2: Trend Detection ====================

def example_2_trend_detection():
    """Example 2: Trend detection."""
    print("\n" + "=" * 80)
    print("Example 2: Trend Detection")
    print("=" * 80)

    # Create datasets with different trends
    np.random.seed(42)
    n = 50

    datasets = {
        'Strong Upward': 100 + np.arange(n) * 3 + np.random.normal(0, 5, n),
        'Weak Downward': 200 - np.arange(n) * 0.5 + np.random.normal(0, 10, n),
        'Flat': 150 + np.random.normal(0, 5, n),
        'Volatile': 100 + np.random.normal(0, 50, n),
    }

    print("\n📊 Analyzing trends in 4 different datasets:")

    for name, data in datasets.items():
        print(f"\n📈 Dataset: {name}")
        print(f"   Values: {data[:3]} ... {data[-3:]}")

        # DETECT TREND - ONE LINE!
        result = detect_trend(data)

        emoji = {
            'strong_upward': '📈',
            'moderate_upward': '↗️',
            'weak_upward': '⤴️',
            'flat': '➡️',
            'weak_downward': '⤵️',
            'moderate_downward': '↘️',
            'strong_downward': '📉',
            'volatile': '📊',
            'cyclical': '🔄',
        }.get(result.trend_type.value, '❓')

        print(f"\n   {emoji} Trend Type: {result.trend_type.value}")
        print(f"   Slope: {result.slope:.4f} per period")
        print(f"   Strength: {result.strength:.0%}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   R²: {result.r_squared:.3f}")
        print(f"   Volatility: {result.volatility:.3f}")
        print(f"   Turning points: {len(result.turning_points)}")


# ==================== Example 3: Anomaly Detection ====================

def example_3_anomaly_detection():
    """Example 3: Anomaly detection."""
    print("\n" + "=" * 80)
    print("Example 3: Anomaly Detection")
    print("=" * 80)

    # Create data with anomalies
    np.random.seed(42)
    n = 100

    # Normal data
    data = np.random.normal(100, 10, n)

    # Inject anomalies
    anomaly_indices = [10, 25, 50, 75, 90]
    for idx in anomaly_indices:
        data[idx] = data[idx] * 2  # Make it 2x the normal value

    print(f"\n📊 Data: {len(data)} points")
    print(f"   Mean: {np.mean(data):.2f}")
    print(f"   Std: {np.std(data):.2f}")
    print(f"   Actual anomalies injected: {len(anomaly_indices)}")

    print("\n🔍 Detecting anomalies...")

    # DETECT ANOMALIES - ONE LINE!
    anomalies = detect_anomalies(data, method='auto', sensitivity=2.0)

    print(f"\n✅ Detection complete!")
    print(f"   Found {len(anomalies)} anomalies\n")

    # Show first 5 anomalies
    for i, anomaly in enumerate(anomalies[:5]):
        severity_emoji = {
            'low': '🟡',
            'medium': '🟠',
            'high': '🔴',
            'critical': '🚨'
        }.get(anomaly.severity, '❓')

        print(f"   {severity_emoji} Index {anomaly.index}:")
        print(f"      Value: {anomaly.value:.2f}")
        print(f"      Expected: {anomaly.expected:.2f}")
        print(f"      Score: {anomaly.score:.2f}")
        print(f"      Severity: {anomaly.severity}")
        print()

    # Test different methods
    print("🔬 Testing different detection methods:")
    methods = ['zscore', 'iqr', 'mad', 'moving_average']

    for method in methods:
        anomalies = detect_anomalies(data, method=method, sensitivity=2.0)
        print(f"   {method:20s} | Found: {len(anomalies)} anomalies")


# ==================== Example 4: Seasonality Analysis ====================

def example_4_seasonality():
    """Example 4: Seasonality analysis."""
    print("\n" + "=" * 80)
    print("Example 4: Seasonality Analysis")
    print("=" * 80)

    # Create data with weekly seasonality
    np.random.seed(42)
    n = 365  # 1 year of daily data

    # Weekly pattern (7-day cycle)
    time = np.arange(n)
    trend = 1000 + time * 2  # Growing trend
    weekly_pattern = 50 * np.sin(2 * np.pi * time / 7)  # Weekly cycle
    noise = np.random.normal(0, 20, n)

    sales = trend + weekly_pattern + noise

    print(f"\n📊 Data: {n} days of sales data")
    print(f"   Mean: ${np.mean(sales):.2f}")
    print(f"   Std: ${np.std(sales):.2f}")

    print("\n🔍 Analyzing seasonality...")

    # ANALYZE SEASONALITY - ONE LINE!
    pattern = analyze_seasonality(sales)

    print(f"\n✅ Analysis complete!")
    print(f"   Seasonality Type: {pattern.type.value}")
    print(f"   Period: {pattern.period} days")
    print(f"   Strength: {pattern.strength:.0%}")
    print(f"   Confidence: {pattern.confidence:.0%}")
    print(f"   Peaks found: {len(pattern.peak_indices)}")
    print(f"   Troughs found: {len(pattern.trough_indices)}")

    # Show decomposition stats
    print(f"\n📊 Decomposition:")
    print(f"   Trend component: Mean={np.mean(pattern.decomposition['trend']):.2f}")
    print(f"   Seasonal component: Amplitude={np.max(np.abs(pattern.decomposition['seasonal'])):.2f}")
    print(f"   Residual component: Std={np.std(pattern.decomposition['residual']):.2f}")

    # Test with different data (monthly pattern)
    print("\n🔬 Testing monthly seasonality:")

    # Monthly pattern (30-day cycle)
    monthly_pattern = 100 * np.sin(2 * np.pi * time / 30)
    monthly_sales = trend + monthly_pattern + noise

    pattern = analyze_seasonality(monthly_sales)
    print(f"   Detected: {pattern.type.value} (period: {pattern.period} days)")
    print(f"   Strength: {pattern.strength:.0%}")


# ==================== Example 5: Combined Analysis ====================

def example_5_combined():
    """Example 5: Combined predictive analysis."""
    print("\n" + "=" * 80)
    print("Example 5: Complete Predictive Analysis Pipeline")
    print("=" * 80)

    # Create realistic e-commerce data
    np.random.seed(42)
    n = 180  # 6 months

    time = np.arange(n)
    trend = 5000 + time * 20  # Growing trend
    weekly_pattern = 1000 * np.sin(2 * np.pi * time / 7)  # Weekly seasonality
    noise = np.random.normal(0, 200, n)

    sales = trend + weekly_pattern + noise

    # Add some anomalies
    anomaly_days = [30, 60, 120, 150]
    for day in anomaly_days:
        sales[day] *= 0.3  # Drop to 30%

    df = pd.DataFrame({
        'day': time,
        'date': pd.date_range('2024-01-01', periods=n, freq='D'),
        'sales': sales
    })

    print(f"\n📊 E-Commerce Sales Data:")
    print(f"   Period: {n} days (6 months)")
    print(f"   Mean daily sales: ${df['sales'].mean():.2f}")
    print(f"   Total revenue: ${df['sales'].sum():,.2f}")

    print("\n" + "-" * 80)
    print("🔍 Step 1: Trend Analysis")
    print("-" * 80)

    trend_result = detect_trend(df['sales'])
    print(f"   Trend: {trend_result.trend_type.value}")
    print(f"   Growth rate: {trend_result.slope:.4f} per day")
    print(f"   Strength: {trend_result.strength:.0%}")

    print("\n" + "-" * 80)
    print("🔍 Step 2: Seasonality Detection")
    print("-" * 80)

    seasonality = analyze_seasonality(df['sales'])
    print(f"   Pattern: {seasonality.type.value}")
    print(f"   Period: {seasonality.period} days")
    print(f"   Strength: {seasonality.strength:.0%}")

    print("\n" + "-" * 80)
    print("🔍 Step 3: Anomaly Detection")
    print("-" * 80)

    anomalies = detect_anomalies(df['sales'], sensitivity=2.5)
    print(f"   Anomalies found: {len(anomalies)}")

    if anomalies:
        print(f"\n   Top 3 anomalies:")
        for i, a in enumerate(sorted(anomalies, key=lambda x: x.score, reverse=True)[:3], 1):
            date = df.loc[a.index, 'date'].strftime('%Y-%m-%d')
            print(f"   {i}. {date}: ${a.value:.2f} (expected ${a.expected:.2f}) - {a.severity}")

    print("\n" + "-" * 80)
    print("🔮 Step 4: Forecasting")
    print("-" * 80)

    forecast_result = forecast(df['sales'], periods=30, method='auto')
    print(f"   Method: {forecast_result.method}")
    print(f"   Forecast horizon: 30 days")
    print(f"   Average prediction: ${forecast_result.predictions.mean():.2f}")
    print(f"   Confidence: {forecast_result.confidence:.0%}")

    print("\n✅ Complete Analysis Done!")


# ==================== Main ====================

def main():
    """Run all predictive analytics examples."""
    try:
        example_1_forecasting()
        example_2_trend_detection()
        example_3_anomaly_detection()
        example_4_seasonality()
        example_5_combined()

        print("\n" + "=" * 80)
        print("✅ All Predictive Analytics Examples Completed!")
        print("=" * 80)

        print("\n💡 Key Features:")
        print("  ✅ NO API required - 100% statistical methods")
        print("  ✅ Time series forecasting (multiple methods)")
        print("  ✅ Trend detection (9 trend types)")
        print("  ✅ Anomaly detection (4 methods)")
        print("  ✅ Seasonality analysis (decomposition)")
        print("  ✅ Confidence intervals")
        print("  ✅ Fast - instant analysis")

        print("\n🎯 Supported Features:")
        print("  • Forecasting: ARIMA, Exponential Smoothing, Moving Average, Linear, Polynomial")
        print("  • Trend: Strong/Moderate/Weak Up/Down, Flat, Volatile, Cyclical")
        print("  • Anomalies: Z-Score, IQR, MAD, Moving Average deviation")
        print("  • Seasonality: Daily, Weekly, Monthly, Quarterly, Yearly, Custom")

        print("\n📚 Usage:")
        print("  from vizforge.predictive import forecast, detect_trend, detect_anomalies, analyze_seasonality")
        print("  result = forecast(df['sales'], periods=30)")
        print("  trend = detect_trend(df['revenue'])")
        print("  anomalies = detect_anomalies(df['metrics'])")
        print("  pattern = analyze_seasonality(df['traffic'])")

        print("\n" + "=" * 80)
        print("🎉 VizForge v1.3.0 - Predictive Analytics Engine!")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
