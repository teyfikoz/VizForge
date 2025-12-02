"""Quick test of VizForge basic functionality."""

import pandas as pd
import numpy as np
import vizforge as vz

print("VizForge v" + vz.__version__)
print("=" * 50)

# Test 1: Line chart
print("✓ Testing line chart...")
data = pd.DataFrame({
    'x': list(range(10)),
    'y': [2, 4, 3, 5, 7, 6, 8, 9, 11, 10]
})
chart = vz.line(data, x='x', y='y', title='Line Test', show=False)
print(f"  Created: {type(chart).__name__}")

# Test 2: Bar chart
print("✓ Testing bar chart...")
data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'value': [23, 45, 56, 32]
})
chart = vz.bar(data, x='category', y='value', title='Bar Test', show=False)
print(f"  Created: {type(chart).__name__}")

# Test 3: Scatter plot
print("✓ Testing scatter plot...")
data = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50)
})
chart = vz.scatter(data, x='x', y='y', title='Scatter Test', show=False)
print(f"  Created: {type(chart).__name__}")

# Test 4: Pie chart
print("✓ Testing pie chart...")
data = {'A': 30, 'B': 25, 'C': 45}
chart = vz.pie(data, title='Pie Test', show=False)
print(f"  Created: {type(chart).__name__}")

# Test 5: Themes
print("✓ Testing themes...")
themes = vz.list_themes()
print(f"  Available themes: {', '.join(themes)}")

vz.set_theme("dark")
current = vz.get_theme()
print(f"  Current theme: {current.name}")

# Test 6: Export
print("✓ Testing export...")
data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
chart = vz.line(data, x='x', y='y', title='Export Test', show=False)
chart.export('/tmp/vizforge_test.html')
print("  Exported to /tmp/vizforge_test.html")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("\nVizForge is ready to use!")
print("Run: python vizforge/examples/basic_examples.py")
