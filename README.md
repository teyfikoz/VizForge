# VizForge

**Production-grade data visualization library with ZERO AI dependencies**

Create beautiful, interactive visualizations with a single line of code. No API keys, no paid services, just pure visualization power.

## Features

- 🎨 **Beautiful by Default** - Professional themes out of the box
- ⚡ **Simple API** - One-line visualizations: `vz.line(data, x, y)`
- 📊 **20+ Chart Types** - From basics to advanced (Sankey, Network, Geographic)
- 🎭 **Theme System** - 5 built-in themes + custom themes
- 💾 **Export Anywhere** - PNG, SVG, HTML, PDF support
- 🚀 **No AI Dependencies** - Completely free, no API keys needed
- 📈 **Performance** - Handle 100K+ data points efficiently
- 🌍 **Geographic Maps** - Choropleth, scatter, and route maps
- 🔗 **Network Graphs** - Force-directed layouts, community detection
- 📐 **Statistical Plots** - Distributions, correlations, regressions

## Installation

```bash
pip install vizforge

# With all features
pip install vizforge[full]

# Only specific features
pip install vizforge[geo]      # Geographic mapping
pip install vizforge[stats]    # Statistical visualizations
pip install vizforge[network]  # Network graphs
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

## Chart Types

### Basic Charts
- **Line Chart** - Single/multi-line, area charts
- **Bar Chart** - Vertical/horizontal, grouped/stacked
- **Scatter Plot** - 2D/3D, bubble charts
- **Pie Chart** - Pie/donut, sunburst

### Advanced Charts
- **Heatmap** - Correlation matrices, custom data
- **Treemap** - Hierarchical data visualization
- **Sankey Diagram** - Flow visualization
- **Network Graph** - Nodes/edges, force-directed
- **Waterfall Chart** - Cumulative effect
- **Funnel Chart** - Conversion tracking
- **Gauge Chart** - KPI display
- **Candlestick** - Financial data

### Statistical Charts
- **Histogram** - Distribution analysis
- **Box Plot** - Quartiles and outliers
- **Violin Plot** - Distribution shape
- **KDE Plot** - Kernel density estimation
- **Correlation Matrix** - Feature relationships
- **Regression Plot** - With confidence intervals

### Geographic Charts
- **Choropleth Map** - Regions colored by value
- **Scatter Map** - Points on map
- **Route Map** - Lines between locations
- **Heat Map** - Density visualization

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
# PNG (high resolution)
chart.export("output.png", width=1920, height=1080)

# SVG (vector graphics)
chart.export("output.svg")

# HTML (interactive)
chart.export("output.html")

# PDF (publication-ready)
chart.export("output.pdf")
```

## Examples

### Multi-line Comparison
```python
from vizforge import Chart

chart = Chart(chart_type='line', theme='dark')
chart.add_line(x=dates, y=product_a, name='Product A')
chart.add_line(x=dates, y=product_b, name='Product B')
chart.add_line(x=dates, y=product_c, name='Product C')
chart.update_layout(title='Product Comparison')
chart.show()
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
vz.heatmap(
    correlation_matrix,
    title='Feature Correlation',
    colorscale='RdBu',
    annotations=True
)
```

### Geographic Choropleth
```python
from vizforge.geo import choropleth_map

choropleth_map(
    locations=countries,
    values=gdp_data,
    title='GDP by Country',
    colorscale='Greens'
)
```

### Network Graph
```python
from vizforge.charts import network_graph

network_graph(
    nodes=node_list,
    edges=edge_list,
    layout='force_directed',
    node_size='degree',
    node_color='community'
)
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
