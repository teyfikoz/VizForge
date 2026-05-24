# VizForge

**Visualization Intelligence Platform** — 50+ chart types, natural language queries, predictive analytics, auto-storytelling, 13+ data connectors, visual bias detection, and GPU-accelerated rendering. 100% offline, zero API costs.

[![PyPI version](https://badge.fury.io/py/vizforge.svg)](https://pypi.org/project/vizforge/)
[![CI](https://github.com/teyfikoz/VizForge/actions/workflows/ci.yml/badge.svg)](https://github.com/teyfikoz/VizForge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install vizforge
```

## Quick Start

```python
import vizforge as vz
import pandas as pd

df = pd.DataFrame({
    "month":   ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "revenue": [120, 145, 132, 178, 195, 210],
    "cost":    [80, 92, 88, 101, 110, 118],
})

# Natural language → chart (no API key)
chart = vz.ask("Show revenue and cost trend as a line chart", df)
chart.show()

# Forecast next 3 months
fc = vz.forecast(df, target="revenue", periods=3)
fc.show()

# Auto-discover insights
insights = vz.discover_insights(df)
for ins in insights:
    print(ins.summary)
```

---

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **50+ Chart Types** | 2D, 3D, geo, network, real-time, statistical |
| **NLQ Engine** | Talk to your data in plain English — no API needed |
| **Predictive Analytics** | Forecast, trend detection, anomaly detection, seasonality |
| **Auto-Storytelling** | Generate narrative reports with evidence-based insights |
| **Visual Bias Detector** | Catch misleading scales, cherry-picking, and chartjunk |
| **Chart Reasoning** | Explainable chart selection with confidence scores |
| **Dashboard Builder** | Multi-chart dashboards with filters, KPI cards, callbacks |
| **13+ Connectors** | PostgreSQL, MySQL, MongoDB, S3, GCS, REST, GraphQL, Excel, Parquet |
| **Video Export** | Animate charts to MP4/WebM/GIF |
| **Plugin System** | Extend with custom chart types, connectors, and renderers |
| **Data Streaming** | Progressive rendering for infinite-size datasets |
| **GPU Rendering** | WebGPU-accelerated charts (1000x faster than Plotly) |

---

## Natural Language Queries (NLQ)

```python
import vizforge as vz
import pandas as pd

df = pd.read_csv("sales.csv")

# Ask questions in plain English
vz.ask("Show top 10 products by revenue as a bar chart", df).show()
vz.ask("Plot monthly sales trend with forecast", df).show()
vz.ask("Compare Q1 vs Q2 performance", df).show()
vz.ask("Which region had the highest growth?", df).show()

# NLQ Engine directly
nlq = vz.NLQEngine()
result = nlq.query("Show correlation between price and sales", df)
print(result.chart_type)     # scatter
print(result.explanation)    # "Scatter plot chosen: 2 continuous variables, correlation question"
result.chart.show()
```

---

## Predictive Analytics

```python
import vizforge as vz
import pandas as pd

df = pd.DataFrame({
    "date":  pd.date_range("2024-01-01", periods=365),
    "value": [100 + i * 0.3 + (i % 30) * 5 for i in range(365)],
})

# Forecast next 90 days
fc = vz.forecast(df, target="value", periods=90)
fc.show()  # chart with confidence intervals

# Use forecaster directly
forecaster = vz.TimeSeriesForecaster(method="auto")
forecaster.fit(df, "date", "value")
future = forecaster.predict(90)
print(future[["date", "predicted", "lower", "upper"]].tail())

# Trend detection
trend = vz.detect_trend(df, "value")
print(trend.direction)     # "upward"
print(trend.slope)         # 0.32
print(trend.p_value)       # 0.0001

# Anomaly detection
anomalies = vz.detect_anomalies(df, "value", method="zscore", threshold=3.0)
anomalies.show()

# Seasonality analysis
season = vz.analyze_seasonality(df, "value")
print(season.dominant_period)  # 30 (monthly cycle)
print(season.strength)         # 0.78
```

---

## Auto-Storytelling & Insights

```python
import vizforge as vz

df = pd.read_csv("quarterly_report.csv")

# Discover insights automatically
insights = vz.discover_insights(df)
for ins in insights:
    print(f"[{ins.type:15s}] {ins.summary}")
    print(f"  Evidence: {ins.evidence}")
    print(f"  Confidence: {ins.confidence:.0%}")
# [trend          ] Revenue grew 23% over the past 4 quarters
#   Evidence: Linear regression slope = 42.3k/quarter, p=0.003
#   Confidence: 94%

# Generate narrative story
story = vz.generate_story(df, title="Q2 2026 Performance Report")
print(story)   # human-readable narrative with statistics

# Full HTML report (charts + narrative + recommendations)
report = vz.generate_report(df, format="html", output="report.html")
```

---

## Visual Bias Detection

```python
import vizforge as vz

# Detect misleading visualization patterns
detector = vz.VisualBiasDetector()
report: vz.BiasReport = detector.analyze(df, chart_type="bar", y_col="revenue")
for issue in report.issues:
    print(f"  [{issue.severity}] {issue.description}")
    print(f"  Recommendation: {issue.fix}")
# [HIGH  ] Y-axis truncated at 95 (not zero) — inflates visual change by 8x
#   Recommendation: Start Y-axis at 0 or label the break clearly
# [MEDIUM] Cherry-picked date range omits Q4 decline
#   Recommendation: Show full historical range
```

---

## Chart Reasoning Engine

```python
import vizforge as vz

engine = vz.ChartReasoningEngine()
decision: vz.ChartDecision = engine.recommend(
    df,
    x="date", y="revenue",
    question="How has revenue changed over time?"
)
print(decision.chart_type)    # "line"
print(decision.confidence)    # 0.94
print(decision.reasoning)     # "Time series with continuous y → line chart optimal"
print(decision.alternatives)  # ["area", "bar"] with explanations
```

---

## Dashboard Builder

```python
import vizforge as vz

dashboard = vz.create_dashboard(title="Sales Dashboard", theme="dark")

dashboard.add(vz.KPICard(label="Total Revenue", value=1_240_000, delta=0.18))
dashboard.add(vz.KPICard(label="Customers", value=8_420, delta=0.07))
dashboard.add(vz.ChartComponent(chart=revenue_line_chart, title="Revenue Trend"))
dashboard.add(vz.ChartComponent(chart=region_bar_chart, title="By Region"))
dashboard.add(vz.FilterComponent(column="region", label="Filter by Region"))
dashboard.add(vz.TextComponent("Analysis: Revenue growth accelerated in Q2."))

dashboard.show()              # interactive HTML
dashboard.export("dashboard.html")
```

---

## Data Connectors (13+ Sources)

```python
import vizforge as vz

# PostgreSQL
conn = vz.connect("postgresql", host="localhost", db="sales", user="admin", password="...")
df = conn.query("SELECT month, revenue FROM sales WHERE year=2026")

# S3 / Parquet
s3 = vz.connect("s3", bucket="my-data", key="reports/q2.parquet")
df = s3.read()

# REST API
api = vz.connect("rest", url="https://api.example.com/data", headers={"Authorization": "Bearer ..."})
df = api.fetch(endpoint="/metrics?period=30d")

# Excel
xl = vz.connect("excel", path="data.xlsx", sheet="Q2")
df = xl.read()

# List available connectors
print(vz.list_connectors())
# ['postgresql', 'mysql', 'sqlite', 'mongodb', 's3', 'gcs', 'azure_blob',
#  'rest', 'graphql', 'excel', 'parquet', 'hdf5', 'html_table', 'web_scraper']
```

---

## Video Export (Animated Charts)

```python
import vizforge as vz

chart = vz.ask("Show monthly revenue as animated bar race", df)

# Export to video
vz.export_video(
    chart,
    output="revenue_race.mp4",
    format="mp4",
    fps=30,
    duration=15,
    animation=vz.AnimationType.BAR_RACE,
)

# GIF for social media
vz.export_video(chart, output="preview.gif", format="gif", fps=15, duration=5)
```

---

## Plugin System

```python
import vizforge as vz
from vizforge import ChartPlugin, PluginMetadata

class WaterfallChart(ChartPlugin):
    metadata = PluginMetadata(name="waterfall", version="1.0.0", author="me")

    def render(self, df, x, y, **kwargs):
        # ... build waterfall chart ...
        return figure

vz.register_plugin(WaterfallChart)
chart = vz.get_plugin("waterfall").render(df, x="category", y="change")
print(vz.list_plugins())
```

---

## Data Streaming

```python
import vizforge as vz

# Stream from large file without loading into memory
stream = vz.stream_from_file("giant_log.csv", chunk_size=10_000)
chart = vz.StreamingChart(stream, chart_type="line", x="timestamp", y="value")
chart.show()  # renders progressively as data loads

# Stream from database
db_stream = vz.stream_from_database(
    "postgresql://...",
    query="SELECT ts, price FROM ticks ORDER BY ts",
    chunk_size=50_000,
)
```

---

## Themes

```python
import vizforge as vz

# Built-in themes
vz.set_theme("dark")       # dark background
vz.set_theme("minimal")    # clean, white
vz.set_theme("corporate")  # blue/grey business style
vz.set_theme("colorblind") # WCAG 2.1 AA accessible palette

# Register custom theme
vz.register_theme("mybrand", {
    "background": "#0D0D0D",
    "primary":    "#FF6B35",
    "font_family": "Inter, sans-serif",
})
vz.set_theme("mybrand")

print(vz.list_themes())
```

---

## Synthetic Data for Testing

```python
import vizforge as vz

engine = vz.SyntheticVisualizationEngine()
config = vz.SyntheticVizConfig(
    rows=1000,
    columns=["date", "revenue", "cost", "region"],
    trend="upward",
    seasonality="monthly",
    noise=0.1,
)
df = engine.generate(config)
```

---

## License

MIT — [Teyfik Öz](https://github.com/teyfikoz)
