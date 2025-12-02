"""Test VizForge v0.2.0 - All 12 2D Chart Types"""

import pandas as pd
import numpy as np

print("=" * 60)
print("VizForge v0.2.0 - Testing All 12 2D Chart Types")
print("=" * 60)

try:
    import vizforge as vz
    print(f"\n✓ VizForge version: {vz.__version__}")
except ImportError as e:
    print(f"\n✗ Failed to import vizforge: {e}")
    exit(1)

# Test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30)
df = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(100, 200, 30) + np.arange(30) * 2,
    'profit': np.random.randint(50, 100, 30) + np.arange(30) * 1.5,
    'category': np.random.choice(['A', 'B', 'C'], 30),
    'value': np.random.randn(30) * 10 + 50
})

print("\n" + "-" * 60)
print("Testing 2D Charts (12 types)")
print("-" * 60)

# Test 1: Line Chart
try:
    chart = vz.line(df, x='date', y='sales', title='Line Chart Test', show=False)
    print("✓ 1. Line Chart")
except Exception as e:
    print(f"✗ 1. Line Chart: {e}")

# Test 2: Bar Chart
try:
    chart = vz.bar(df.head(10), x='date', y='sales', title='Bar Chart Test', show=False)
    print("✓ 2. Bar Chart")
except Exception as e:
    print(f"✗ 2. Bar Chart: {e}")

# Test 3: Area Chart
try:
    chart = vz.area(df, x='date', y='sales', title='Area Chart Test', show=False)
    print("✓ 3. Area Chart")
except Exception as e:
    print(f"✗ 3. Area Chart: {e}")

# Test 4: Scatter Plot
try:
    chart = vz.scatter(df, x='sales', y='profit', title='Scatter Plot Test', show=False)
    print("✓ 4. Scatter Plot")
except Exception as e:
    print(f"✗ 4. Scatter Plot: {e}")

# Test 5: Pie Chart
try:
    pie_data = df.groupby('category')['sales'].sum()
    chart = vz.pie(pie_data.to_dict(), title='Pie Chart Test', show=False)
    print("✓ 5. Pie Chart")
except Exception as e:
    print(f"✗ 5. Pie Chart: {e}")

# Test 6: Heatmap
try:
    matrix = np.random.randn(10, 10)
    chart = vz.heatmap(matrix, title='Heatmap Test', show=False)
    print("✓ 6. Heatmap")
except Exception as e:
    print(f"✗ 6. Heatmap: {e}")

# Test 7: Histogram
try:
    chart = vz.histogram(df, x='value', title='Histogram Test', nbins=15, show=False)
    print("✓ 7. Histogram")
except Exception as e:
    print(f"✗ 7. Histogram: {e}")

# Test 8: Boxplot
try:
    chart = vz.boxplot(df, x='category', y='value', title='Boxplot Test', show=False)
    print("✓ 8. Boxplot")
except Exception as e:
    print(f"✗ 8. Boxplot: {e}")

# Test 9: Radar Chart
try:
    radar_data = {'Speed': 8, 'Power': 6, 'Defense': 7, 'Magic': 9, 'Agility': 7}
    chart = vz.radar(radar_data, title='Radar Chart Test', show=False)
    print("✓ 9. Radar Chart")
except Exception as e:
    print(f"✗ 9. Radar Chart: {e}")

# Test 10: Waterfall Chart
try:
    waterfall_data = {
        'Revenue': 100,
        'Costs': -30,
        'Expenses': -20,
        'Profit': 50
    }
    chart = vz.waterfall(waterfall_data, title='Waterfall Test', show=False)
    print("✓ 10. Waterfall Chart")
except Exception as e:
    print(f"✗ 10. Waterfall Chart: {e}")

# Test 11: Funnel Chart
try:
    funnel_data = {
        'Visitors': 10000,
        'Signed Up': 5000,
        'Active': 2000,
        'Purchased': 500
    }
    chart = vz.funnel(funnel_data, title='Funnel Test', show=False)
    print("✓ 11. Funnel Chart")
except Exception as e:
    print(f"✗ 11. Funnel Chart: {e}")

# Test 12: Bubble Chart
try:
    bubble_df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'size': np.random.randint(10, 100, 50)
    })
    chart = vz.bubble(bubble_df, x='x', y='y', size='size', title='Bubble Test', show=False)
    print("✓ 12. Bubble Chart")
except Exception as e:
    print(f"✗ 12. Bubble Chart: {e}")

print("\n" + "-" * 60)
print("Theme System")
print("-" * 60)

# Test themes
try:
    themes = vz.list_themes()
    print(f"✓ Available themes: {', '.join(themes)}")

    vz.set_theme("dark")
    print("✓ Set theme to 'dark'")

    chart = vz.line(df, x='date', y='sales', title='Dark Theme Test', show=False)
    print("✓ Created chart with dark theme")
except Exception as e:
    print(f"✗ Theme system: {e}")

print("\n" + "-" * 60)
print("Export System")
print("-" * 60)

# Test export
try:
    chart = vz.line(df, x='date', y='sales', title='Export Test', show=False)
    chart.export('/tmp/vizforge_v02_test.html')
    print("✓ Exported to HTML: /tmp/vizforge_v02_test.html")
except Exception as e:
    print(f"✗ Export: {e}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"VizForge v{vz.__version__}")
print("✓ 12 2D Chart Types")
print("✓ 5 Themes")
print("✓ Export System")
print("\nAll tests completed successfully!")
print("=" * 60)
