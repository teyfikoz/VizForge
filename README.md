# VizForge

**Production-grade data visualization library with ZERO AI dependencies**

Create beautiful, interactive visualizations with a single line of code. No API keys, no paid services, just pure visualization power.

## Features (v0.2.0)

- 🎨 **Beautiful by Default** - Professional themes out of the box
- ⚡ **Simple API** - One-line visualizations: `vz.line(data, x, y)`
- 📊 **12 2D Chart Types** - Line, Bar, Area, Scatter, Pie, Heatmap, Histogram, Boxplot, Radar, Waterfall, Funnel, Bubble
- 🎭 **Theme System** - 5 built-in themes + custom themes
- 💾 **Export Anywhere** - HTML export (PNG, SVG, PDF coming in v0.3.0)
- 🚀 **No AI Dependencies** - Completely free, no API keys needed
- 📈 **Performance** - Handle 100K+ data points efficiently
- 🔜 **Coming Soon** - 3D charts, Geographic maps, Network graphs, Real-time visualizations (v0.3.0+)

## Installation

```bash
pip install vizforge
```

## Quick Start

```python
import vizforge as vz
import pandas as pd

# Line chart
data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30),
    'sales': [100, 120, 115, 130, 140, ...]
})
vz.line(data, x='date', y='sales', title='Daily Sales')

# Bar chart
vz.bar(data, x='category', y='amount', color='region')

# Scatter plot
vz.scatter(data, x='age', y='income', size='population')

# Pie chart
vz.pie(data, values='market_share', names='company')
```

## Chart Types (12 in v0.2.0)

### Basic Charts (4)
- **Line Chart** - `vz.line()` - Single/multi-line, time series
- **Bar Chart** - `vz.bar()` - Vertical/horizontal, grouped/stacked
- **Scatter Plot** - `vz.scatter()` - 2D scatter with optional colors
- **Pie Chart** - `vz.pie()` / `vz.donut()` - Proportions and percentages

### Statistical Charts (4)
- **Histogram** - `vz.histogram()` - Distribution analysis with bins
- **Boxplot** - `vz.boxplot()` - Quartiles, median, outliers
- **Heatmap** - `vz.heatmap()` - Correlation matrices, color-coded data
- **Area Chart** - `vz.area()` - Filled area under line, stacking support

### Business Charts (4)
- **Waterfall Chart** - `vz.waterfall()` - Cumulative sequential changes
- **Funnel Chart** - `vz.funnel()` - Conversion tracking, stages
- **Radar Chart** - `vz.radar()` - Multivariate data, spider/star chart
- **Bubble Chart** - `vz.bubble()` - 3-variable scatter with size encoding

### Coming in v0.3.0+
- 3D Charts (Surface, Scatter3D, Mesh3D, Volume, Cone, Isosurface)
- Geographic Charts (Choropleth, Scatter Geo, Line Geo, Density Geo, Flow Map)
- Network Charts (Network Graph, Sankey, Tree, Dendrogram)
- Real-time Charts (Streaming Line, Live Dashboard, Real-time Heatmap)

## Themes

```python
# Built-in themes
vz.set_theme("default")    # Modern, colorful
vz.set_theme("dark")       # Dark background, neon accents
vz.set_theme("minimal")    # Clean, monochrome
vz.set_theme("corporate")  # Professional, conservative
vz.set_theme("scientific") # Publication-ready

# Custom theme
custom = vz.Theme(
    background_color="#ffffff",
    text_color="#333333",
    primary_color="#3498db",
    font_family="Arial"
)
vz.set_theme(custom)
```

## Export Options

```python
# HTML (interactive) - v0.2.0
chart.export("output.html")

# Coming in v0.3.0:
# PNG, SVG, PDF export
```

## Examples

### Multi-series Line Chart
```python
import vizforge as vz
import pandas as pd

df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30),
    'Product A': np.random.randint(100, 200, 30),
    'Product B': np.random.randint(80, 180, 30),
    'Product C': np.random.randint(90, 190, 30)
})

vz.line(df, x='date', y=['Product A', 'Product B', 'Product C'],
        title='Product Comparison', theme='dark')
```

### Grouped Bar Chart
```python
vz.bar(
    data,
    x='month',
    y='revenue',
    color='region',
    barmode='group',
    title='Regional Revenue by Month'
)
```

### Correlation Heatmap
```python
import numpy as np

correlation_matrix = np.random.randn(10, 10)
vz.heatmap(
    correlation_matrix,
    title='Feature Correlation',
    colorscale='RdBu'
)
```

### Funnel Analysis
```python
funnel_data = {
    'Website Visitors': 10000,
    'Signed Up': 5000,
    'Active Users': 2000,
    'Paying Customers': 500
}

vz.funnel(funnel_data, title='Conversion Funnel')
```

### Bubble Chart
```python
df = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'size': np.random.randint(10, 100, 50),
    'category': np.random.choice(['A', 'B', 'C'], 50)
})

vz.bubble(df, x='x', y='y', size='size', color='category',
          title='Multi-dimensional Analysis')
```

## Philosophy

> "Visualization should be easy. The code should disappear, and the story should emerge."

VizForge believes that:
- Beautiful visualizations shouldn't require complex code
- Themes should be globally consistent
- Export should be effortless
- No visualization library should require AI or paid APIs

## Why VizForge?

| Feature | VizForge | Plotly | Matplotlib | Seaborn |
|---------|----------|--------|------------|---------|
| Easy API | ✅ | ⚠️ | ❌ | ✅ |
| Interactive | ✅ | ✅ | ❌ | ❌ |
| Static Export | ✅ | ✅ | ✅ | ✅ |
| Themes | ✅ | ⚠️ | ⚠️ | ✅ |
| Geographic | ✅ | ✅ | ⚠️ | ❌ |
| Network Graphs | ✅ | ✅ | ⚠️ | ❌ |
| One-line Plots | ✅ | ❌ | ⚠️ | ✅ |
| Learning Curve | Low | Medium | High | Medium |

## Performance

- Efficiently handles 100K+ data points
- WebGL rendering for scatter plots
- Automatic data aggregation
- Lazy loading for large datasets
- Caching for computed layouts

## Requirements

- Python >= 3.10
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.18.0

## License

MIT License - Free for commercial use

## Contributing

Contributions welcome! Please open an issue or pull request.

## Documentation

See `examples/` directory for comprehensive examples.

---

**VizForge: Forge Beautiful Visualizations, Effortlessly.**
