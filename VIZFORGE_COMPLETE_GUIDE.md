# 📘 VizForge v1.3.0 - Complete User Guide

## The Ultimate AI-Powered Data Visualization Platform

**Version**: 1.3.0  
**Author**: Teyfik OZ  
**License**: MIT  
**Python**: 3.8+

---

## 🌟 Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Feature Guide](#feature-guide)
   - [Natural Language Queries](#natural-language-queries)
   - [Predictive Analytics](#predictive-analytics)
   - [Auto Data Storytelling](#auto-data-storytelling)
   - [Visual Designer](#visual-designer)
   - [Universal Data Connectors](#universal-data-connectors)
   - [Video Export](#video-export)
5. [Real-World Scenarios](#real-world-scenarios)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## 📖 Introduction

VizForge is a **revolutionary AI-powered data visualization platform** that combines:

- 🧠 **Local AI** - Zero API costs, complete privacy
- 🗣️ **Natural Language** - Talk to your data  
- 📈 **Predictive Analytics** - Built-in forecasting
- 📝 **Auto Storytelling** - Insights that write themselves
- 🎨 **Visual Designer** - Drag & drop UI
- 🔌 **13+ Connectors** - One API for all data sources
- 🎬 **Video Export** - Professional MP4/WebM/GIF

**What makes it unique**: No other Python library offers ALL of these capabilities in ONE package, completely FREE with NO API costs.

---

## 🚀 Installation

### Basic Installation

```bash
pip install vizforge
```

### Full Installation (All Features)

```bash
pip install "vizforge[full]"
```

### Feature-Specific Installation

```bash
# Natural Language & AI
pip install "vizforge[nlp]"

# Predictive Analytics
pip install "vizforge[predictive]"

# Data Connectors  
pip install "vizforge[connectors]"

# Visual Designer
pip install "vizforge[designer]"

# Video Export
pip install "vizforge[video]"

# Note: Video export also requires ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: Download from ffmpeg.org
```

---

## ⚡ Quick Start

```python
import vizforge as vz
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=12, freq='M'),
    'revenue': [45000, 52000, 48000, 61000, 67000, 71000, 
                68000, 75000, 82000, 79000, 88000, 95000],
    'costs': [35000, 38000, 36000, 42000, 45000, 48000,
              46000, 50000, 54000, 52000, 57000, 60000]
})

# 1. Simple chart
chart = vz.line(df, x='date', y='revenue', title='Revenue Growth 2024')
chart.show()

# 2. Natural language query
chart = vz.ask("Show revenue trend over time", df)
chart.show()

# 3. Forecast future values
forecast_chart, predictions = vz.forecast(df, 'date', 'revenue', periods=6)
forecast_chart.show()

# 4. Discover insights
insights = vz.discover_insights(df)
for insight in insights:
    print(f"💡 {insight.description}")

# 5. Export as video
data_frames = [df.iloc[:i+1] for i in range(len(df))]
vz.export_video(chart, 'revenue_growth.mp4', data_frames=data_frames)
```

**Result**: In just 5 lines, you've created charts, analyzed data, forecasted future, discovered insights, and created a professional video!

---

## 🎯 Feature Guide

### 1. Natural Language Queries

**Talk to your data in plain English!**

#### Basic Queries

```python
import vizforge as vz

# Load data
df = pd.read_csv('sales_data.csv')

# Ask questions
chart = vz.ask("Show sales by region", df)
chart = vz.ask("Compare revenue across products", df)
chart = vz.ask("What's the distribution of customer ages?", df)
chart = vz.ask("Show correlation between price and sales", df)
```

#### Supported Query Types

1. **Trend Analysis**
```python
vz.ask("Show sales trend over time", df)
vz.ask("What's the trend in revenue?", df)
```

2. **Comparisons**
```python
vz.ask("Compare sales by region", df)
vz.ask("Compare revenue across products", df)
```

3. **Distributions**
```python
vz.ask("Show distribution of ages", df)
vz.ask("What's the distribution of prices?", df)
```

4. **Correlations**
```python
vz.ask("Show correlation between price and sales", df)
vz.ask("How does age relate to income?", df)
```

5. **Aggregations**
```python
vz.ask("Show total sales by month", df)
vz.ask("What's average revenue per region?", df)
```

6. **Breakdowns**
```python
vz.ask("Break down sales by category", df)
vz.ask("Split revenue by product and region", df)
```

#### Advanced Features

```python
from vizforge.nlq import NLQEngine

# Create engine
nlq = NLQEngine()

# Query with options
result = nlq.query(
    "Show sales trend",
    df,
    auto_detect_columns=True,
    suggest_chart_type=True
)

# Access components
print(f"Chart type: {result.chart_type}")
print(f"Columns used: {result.columns}")
print(f"Confidence: {result.confidence}")
```

#### Real-World Example: E-Commerce Dashboard

```python
# Load e-commerce data
df = pd.read_csv('ecommerce_sales.csv')

# Quick exploration
charts = [
    vz.ask("Show daily sales trend", df),
    vz.ask("Compare sales by product category", df),
    vz.ask("What's the distribution of order values?", df),
    vz.ask("Show correlation between price and quantity", df),
    vz.ask("Total sales by region and product", df)
]

# Create dashboard
dashboard = vz.Dashboard(rows=3, cols=2)
for i, chart in enumerate(charts):
    row = i // 2
    col = i % 2
    dashboard.add_chart(chart, row=row, col=col)

dashboard.show()
```

---

### 2. Predictive Analytics

**Built-in machine learning for forecasting and anomaly detection**

#### Time Series Forecasting

```python
# Load historical data
df = pd.read_csv('sales_history.csv')

# Forecast next 30 days
forecast_chart, predictions = vz.forecast(
    df,
    date_col='date',
    value_col='sales',
    periods=30,
    model='arima',  # auto, arima, prophet, exponential, linear
    confidence_level=0.95
)

forecast_chart.show()

# Access predictions
print(predictions)
# Output:
#         date  predicted_sales  lower_bound  upper_bound
# 0 2024-07-01           125000       118000       132000
# 1 2024-07-02           127000       119000       135000
# ...
```

#### Anomaly Detection

```python
# Detect unusual transactions
anomalies_chart, anomalies_df = vz.detect_anomalies(
    df,
    value_col='transaction_amount',
    method='isolation_forest',  # isolation_forest, zscore, iqr, lof, svm
    contamination=0.05
)

anomalies_chart.show()

# Flag suspicious transactions
print(f"Found {len(anomalies_df)} anomalies")
print(anomalies_df[['transaction_id', 'amount', 'anomaly_score']])
```

#### Trend Detection

```python
# Analyze revenue trend
trend_chart, trend_info = vz.detect_trend(
    df,
    x='date',
    y='revenue',
    degree=1  # 1=linear, 2=quadratic, 3=cubic
)

trend_chart.show()

print(f"Trend direction: {trend_info.direction}")  # up, down, flat
print(f"Trend strength: {trend_info.strength:.2f}")  # 0-1
print(f"R-squared: {trend_info.r_squared:.3f}")
print(f"Growth rate: {trend_info.slope:.2f} per period")
```

#### Seasonality Analysis

```python
# Detect seasonal patterns
seasonal_chart, patterns = vz.analyze_seasonality(
    df,
    date_col='date',
    value_col='sales',
    periods=['daily', 'weekly', 'monthly', 'yearly']
)

seasonal_chart.show()

print(f"Strongest pattern: {patterns.strongest}")
print(f"Seasonal strength: {patterns.strength:.2f}")
```

#### Real-World Example: Sales Forecasting System

```python
import vizforge as vz
import pandas as pd

# Load 2 years of sales data
df = pd.read_csv('sales_2022_2024.csv')

# 1. Detect trends
trend_chart, trend_info = vz.detect_trend(df, 'date', 'sales')
print(f"📈 Trend: {trend_info.direction} ({trend_info.strength:.0%} strength)")

# 2. Find seasonality
seasonal_chart, patterns = vz.analyze_seasonality(df, 'date', 'sales')
print(f"📅 Seasonal pattern: {patterns.strongest}")

# 3. Detect anomalies (unusual days)
anomalies_chart, anomalies = vz.detect_anomalies(df, 'sales')
print(f"⚠️ Found {len(anomalies)} unusual days")

# 4. Forecast next quarter (90 days)
forecast_chart, predictions = vz.forecast(
    df, 'date', 'sales',
    periods=90,
    model='prophet',  # Best for seasonal data
    confidence_level=0.95
)

# 5. Generate report
report = vz.generate_report(
    df,
    title="Q3 2024 Sales Forecast",
    sections=['trend', 'seasonality', 'anomalies', 'forecast']
)

print(report)

# 6. Create executive dashboard
dashboard = vz.Dashboard(rows=2, cols=2)
dashboard.add_chart(trend_chart, row=0, col=0)
dashboard.add_chart(seasonal_chart, row=0, col=1)
dashboard.add_chart(anomalies_chart, row=1, col=0)
dashboard.add_chart(forecast_chart, row=1, col=1)
dashboard.export('forecast_dashboard.html')
```

---

### 3. Auto Data Storytelling

**Insights that write themselves**

#### Discover Insights

```python
# Auto-discover all insights
insights = vz.discover_insights(
    df,
    max_insights=10,
    min_impact=5.0,  # Only show high-impact insights (0-10)
    insight_types=['trends', 'outliers', 'correlations', 'clusters']
)

for insight in insights:
    print(f"\n{'='*60}")
    print(f"Type: {insight.type}")
    print(f"Impact: {insight.impact_score}/10")
    print(f"Confidence: {insight.confidence}%")
    print(f"\n{insight.description}")
    print(f"\nRecommendation: {insight.recommendation}")
    
    # Show chart
    insight.chart.show()
```

#### Generate Narrative

```python
# Generate executive summary
story = vz.generate_story(
    df,
    title="Q2 2024 Performance Review",
    style='executive',  # executive, technical, storytelling
    max_length=500,  # words
    include_recommendations=True
)

print(story)

# Output:
# Q2 2024 Performance Review
# 
# Executive Summary
# -----------------
# Revenue showed strong growth in Q2 2024, increasing 23% compared to Q1.
# The upward trend is driven primarily by Enterprise segment (+45%) and
# Product Category A (+38%). However, Small Business segment declined 12%,
# requiring immediate attention.
#
# Key Findings
# ------------
# 1. Revenue reached $2.4M, exceeding target by 15%
# 2. Customer acquisition cost decreased 18%
# 3. Identified 3 high-value customer segments...
#
# [continues...]
```

#### Generate Full Report

```python
# Create comprehensive report
report = vz.generate_report(
    df,
    title="2024 Annual Performance Report",
    format='markdown',  # markdown, html, pdf
    include_charts=True,
    include_statistics=True,
    include_recommendations=True,
    sections=[
        'executive_summary',
        'key_metrics',
        'trends',
        'insights',
        'forecasts',
        'recommendations'
    ]
)

# Save report
with open('annual_report_2024.md', 'w') as f:
    f.write(report)

# Convert to PDF (requires additional package)
import markdown
from weasyprint import HTML

html_content = markdown.markdown(report)
HTML(string=html_content).write_pdf('annual_report_2024.pdf')
```

#### Real-World Example: Marketing Campaign Analysis

```python
# Load campaign data
df = pd.read_csv('marketing_campaigns.csv')

# 1. Discover all insights
insights = vz.discover_insights(df, max_insights=15)

# 2. Generate executive summary
executive_summary = vz.generate_story(
    df,
    title="Campaign Performance Analysis",
    style='executive'
)

# 3. Generate detailed report
full_report = vz.generate_report(
    df,
    title="Marketing Analytics Report - Q2 2024",
    include_charts=True
)

# 4. Create presentation slides
slides = []
for insight in insights[:5]:  # Top 5 insights
    slide = {
        'title': insight.description,
        'chart': insight.chart,
        'impact': insight.impact_score,
        'recommendation': insight.recommendation
    }
    slides.append(slide)

# 5. Export everything
with open('campaign_analysis.md', 'w') as f:
    f.write(full_report)

print("✅ Report generated successfully!")
```

---

### 4. Visual Designer

**Web-based drag & drop chart designer**

#### Launch Designer

```python
import vizforge as vz

# Launch web interface
vz.launch_designer()

# Opens http://localhost:5000 in browser
```

#### Custom Configuration

```python
from vizforge.visual_designer import DesignerApp

# Custom host and port
app = DesignerApp(host='0.0.0.0', port=8080)
app.run()

# Access from network: http://your-ip:8080
```

#### Features

1. **28+ Chart Types** - Line, Bar, Scatter, Pie, Heatmap, 3D, Geographic, Network, etc.
2. **Live Preview** - See changes instantly
3. **Property Editor** - Dynamic configuration panel
4. **Code Generation** - Copy-paste ready Python code
5. **CSV Upload** - Drag & drop your data
6. **Export** - PNG, SVG, PDF, HTML

#### Workflow

1. **Upload Data**: Drag CSV file or use sample data
2. **Select Chart Type**: Choose from 28+ options
3. **Configure**: Set title, colors, axes, etc.
4. **Preview**: See live preview
5. **Generate Code**: Copy Python code
6. **Export**: Download chart or code

#### Generated Code Example

```python
# Code generated by Visual Designer
import vizforge as vz
import pandas as pd

# Load data
df = pd.read_csv('sales_data.csv')

# Create chart
chart = vz.bar(
    df,
    x='product',
    y='revenue',
    color='region',
    title='Revenue by Product and Region',
    theme='professional',
    width=1200,
    height=600
)

# Show chart
chart.show()

# Export
chart.export('revenue_chart.png', width=1920, height=1080)
```

#### Real-World Example: Client Presentation

```python
# Scenario: Create charts for client meeting

# 1. Launch designer
vz.launch_designer()

# 2. Upload client_data.csv
# 3. Create 5 different charts using the UI
# 4. Copy generated code for each chart

# 5. Combine in dashboard
import vizforge as vz

charts = [
    # Chart 1: Revenue trend
    vz.line(df, x='month', y='revenue', title='Revenue Trend'),
    
    # Chart 2: Product comparison
    vz.bar(df, x='product', y='sales', title='Sales by Product'),
    
    # Chart 3: Geographic distribution
    vz.choropleth(df, locations='state', values='customers'),
    
    # Chart 4: Correlation heatmap
    vz.correlation_matrix(df[['price', 'sales', 'profit']]),
    
    # Chart 5: Customer segments
    vz.pie(df, values='customers', names='segment')
]

# Create presentation dashboard
dashboard = vz.Dashboard(rows=3, cols=2)
for i, chart in enumerate(charts):
    row = i // 2
    col = i % 2
    dashboard.add_chart(chart, row=row, col=col)

dashboard.export('client_presentation.html')
```

---

### 5. Universal Data Connectors

**Connect to 13+ data sources with ONE unified API**

#### Database Connectors

##### PostgreSQL

```python
# Connect to PostgreSQL
db = vz.connect('postgresql',
    host='localhost',
    port=5432,
    database='sales_db',
    username='analyst',
    password='secure_password'
)

# Query data
df = db.query("SELECT * FROM sales WHERE year = 2024")

# Use with context manager
with vz.connect('postgresql', **config) as db:
    df = db.query("SELECT product, SUM(revenue) FROM sales GROUP BY product")
    chart = vz.bar(df, x='product', y='sum')
    chart.show()
```

##### MySQL

```python
# Connect to MySQL
db = vz.connect('mysql',
    host='mysql.example.com',
    database='analytics',
    username='user',
    password='pass'
)

# Read table
df = db.read(table='customers', limit=1000)

# Custom query
df = db.query("""
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as new_customers
    FROM customers
    WHERE created_at >= '2024-01-01'
    GROUP BY DATE(created_at)
""")

chart = vz.line(df, x='date', y='new_customers')
```

##### SQLite

```python
# Connect to SQLite
db = vz.connect('sqlite', path='data/app.db')

df = db.query("SELECT * FROM users")
```

##### MongoDB

```python
# Connect to MongoDB
mongo = vz.connect('mongodb',
    host='localhost',
    port=27017,
    database='app_db',
    username='user',
    password='pass'
)

# Query with filter
df = mongo.read('orders', {
    'status': 'completed',
    'total': {'$gt': 100}
})

# Aggregation pipeline
df = mongo.aggregate('sales', [
    {'$match': {'year': 2024}},
    {'$group': {
        '_id': '$product',
        'total': {'$sum': '$amount'}
    }}
])
```

#### Cloud Storage Connectors

##### AWS S3

```python
# Connect to S3
s3 = vz.connect('s3',
    bucket='my-data-bucket',
    username='AWS_ACCESS_KEY',
    password='AWS_SECRET_KEY',
    options={'region': 'us-east-1'}
)

# Read CSV from S3
df = s3.read('data/sales_2024.csv', file_type='csv')

# Read Parquet
df = s3.read('data/analytics.parquet', file_type='parquet')

# List files
files = s3.list_tables(prefix='data/')
```

##### Google Cloud Storage

```python
# Connect to GCS
gcs = vz.connect('gcs',
    bucket='my-gcs-bucket',
    options={'project': 'my-gcp-project'}
)

df = gcs.read('data/sales.csv', file_type='csv')
```

##### Azure Blob Storage

```python
# Connect to Azure
azure = vz.connect('azure',
    bucket='my-container',
    username='storage_account_name',
    password='storage_account_key'
)

df = azure.read('data/metrics.json', file_type='json')
```

#### API Connectors

##### REST API

```python
# Connect to REST API
api = vz.connect('rest',
    url='https://api.example.com',
    api_key='YOUR_API_KEY'
)

# GET request
df = api.read('/v1/users')

# With parameters
df = api.read('/v1/sales', params={'year': 2024})

# Custom headers
df = api.read('/v1/data', headers={'X-Custom': 'value'})
```

##### GraphQL

```python
# Connect to GraphQL API
graphql = vz.connect('graphql',
    url='https://api.example.com/graphql',
    api_key='YOUR_API_KEY'
)

# Query
query = '''
    query {
        users(limit: 100) {
            id
            name
            email
            created_at
        }
    }
'''

df = graphql.read(query)
```

#### File Connectors

```python
# Excel
excel = vz.connect('excel', path='data.xlsx')
df = excel.read(sheet_name='Sales')

# Parquet
parquet = vz.connect('parquet', path='data.parquet')
df = parquet.read()

# HDF5
hdf5 = vz.connect('hdf5', path='data.h5')
df = hdf5.read(key='dataset1')
```

#### Web Connectors

```python
# HTML Tables
html = vz.connect('html', url='https://example.com/data.html')
df = html.read(table_index=0)

# Web Scraper
web = vz.connect('web', url='https://example.com')
df = web.read(selector='.data-table tr')
```

#### Real-World Example: Multi-Source Dashboard

```python
import vizforge as vz

# 1. Connect to multiple sources
postgres_db = vz.connect('postgresql', **pg_config)
s3 = vz.connect('s3', **s3_config)
api = vz.connect('rest', url='https://api.example.com', api_key=KEY)

# 2. Load data from each source
sales_df = postgres_db.query("SELECT * FROM sales WHERE year = 2024")
inventory_df = s3.read('inventory/current.csv', file_type='csv')
customer_df = api.read('/v1/customers')

# 3. Create charts
charts = [
    vz.line(sales_df, x='date', y='revenue', title='Sales Trend'),
    vz.bar(inventory_df, x='product', y='stock', title='Inventory'),
    vz.pie(customer_df, values='count', names='segment', title='Customers')
]

# 4. Combine in dashboard
dashboard = vz.Dashboard(rows=2, cols=2)
for i, chart in enumerate(charts):
    dashboard.add_chart(chart, row=i//2, col=i%2)

dashboard.show()

# 5. Auto-refresh every hour
import schedule
import time

def update_dashboard():
    # Reload data
    sales_df = postgres_db.query("SELECT * FROM sales WHERE year = 2024")
    inventory_df = s3.read('inventory/current.csv', file_type='csv')
    customer_df = api.read('/v1/customers')
    
    # Update dashboard
    dashboard.refresh()

schedule.every().hour.do(update_dashboard)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

### 6. Video Export

**Export charts as professional MP4/WebM/GIF animations**

#### Basic Video Export

```python
# Create animated data
data_frames = [
    df[df['month'] == m] for m in range(1, 13)
]

# Create chart
chart = vz.line(df, x='day', y='sales')

# Export as MP4
vz.export_video(
    chart,
    'sales_animation.mp4',
    data_frames=data_frames,
    fps=30,
    width=1920,
    height=1080,
    quality='high'
)
```

#### GIF Export

```python
# Export as GIF
vz.export_video(
    chart,
    'sales.gif',
    data_frames=data_frames,
    format='gif',
    fps=10,
    width=800,
    height=600,
    quality='high'
)
```

#### WebM Export

```python
# Export as WebM (for web)
vz.export_video(
    chart,
    'sales.webm',
    data_frames=data_frames,
    format='webm',
    fps=24,
    quality='medium'
)
```

#### Advanced Features

```python
from vizforge.video_export import (
    VideoExporter, VideoConfig, VideoFormat,
    AnimationEngine, AnimationType, FrameGenerator
)

# Configure video
config = VideoConfig(
    format=VideoFormat.MP4,
    fps=30,
    duration=5.0,
    width=1920,
    height=1080,
    quality='high',
    loop=True,
    optimize=True
)

# Create exporter
exporter = VideoExporter(chart, config)

# Export with progress callback
def progress(pct):
    print(f"Export: {int(pct*100)}%", end='\r')

exporter.export(
    'output.mp4',
    data_frames=data_frames,
    transition='smooth',
    progress_callback=progress
)
```

#### Animation Types

```python
# Different animation styles
animations = {
    'smooth': 'Ease in-out cubic',
    'elastic': 'Spring-like bounce',
    'bounce': 'Bouncy animation',
    'ease_in': 'Accelerate',
    'ease_out': 'Decelerate'
}

for anim_type, description in animations.items():
    vz.export_video(
        chart,
        f'animation_{anim_type}.mp4',
        data_frames=data_frames,
        animation_type=anim_type
    )
```

#### Frame Generator

```python
# Advanced frame control
generator = FrameGenerator(chart, width=1920, height=1080)

# Generate frames
frames = generator.generate_from_data_frames(data_frames)

# Add watermark
generator.add_watermark("Company Name", opacity=180)

# Optimize
generator.optimize_frames(quality=85, resize_factor=0.8)

# Save individual frames
frame_paths = generator.save_frames('output_frames/')
```

#### Real-World Example: Social Media Content

```python
import vizforge as vz

# Load monthly data
df = pd.read_csv('monthly_metrics.csv')

# Create growth animation (month by month)
data_frames = [df.iloc[:i+1] for i in range(len(df))]

# 1. Create chart
chart = vz.line(
    df,
    x='month',
    y=['revenue', 'users', 'engagement'],
    title='Company Growth 2024',
    theme='modern'
)

# 2. Export for different platforms

# Instagram (square, 1:1)
vz.export_video(
    chart, 'instagram_growth.mp4',
    data_frames=data_frames,
    fps=30, width=1080, height=1080,
    quality='high'
)

# Twitter/X (16:9)
vz.export_video(
    chart, 'twitter_growth.mp4',
    data_frames=data_frames,
    fps=30, width=1280, height=720,
    quality='high'
)

# LinkedIn (1200x627)
vz.export_video(
    chart, 'linkedin_growth.mp4',
    data_frames=data_frames,
    fps=30, width=1200, height=627,
    quality='high'
)

# Email newsletter (GIF, small size)
vz.export_video(
    chart, 'newsletter.gif',
    data_frames=data_frames,
    format='gif',
    fps=10, width=600, height=400
)

print("✅ All social media videos exported!")
```

---

## 🎯 Real-World Scenarios

### Scenario 1: E-Commerce Analytics

```python
import vizforge as vz
import pandas as pd

# Connect to database
db = vz.connect('postgresql', **db_config)

# Load sales data
sales_df = db.query("""
    SELECT 
        DATE(order_date) as date,
        product_category,
        SUM(amount) as revenue,
        COUNT(*) as orders
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY DATE(order_date), product_category
""")

# 1. Quick exploration with NLP
overview = vz.ask("Show daily revenue trend", sales_df)
overview.show()

# 2. Forecast next month
forecast_chart, predictions = vz.forecast(
    sales_df, 'date', 'revenue', periods=30
)

# 3. Find anomalies
anomalies_chart, anomalies = vz.detect_anomalies(
    sales_df, 'revenue', method='isolation_forest'
)

# 4. Generate insights
insights = vz.discover_insights(sales_df)

# 5. Create executive report
report = vz.generate_report(
    sales_df,
    title="E-Commerce Performance Report",
    include_charts=True
)

# 6. Save everything
overview.export('daily_revenue.png')
forecast_chart.export('30day_forecast.png')
anomalies_chart.export('anomalies.png')

with open('ecommerce_report.md', 'w') as f:
    f.write(report)

print("✅ Complete e-commerce analysis done!")
```

### Scenario 2: Marketing Campaign ROI

```python
# Load campaign data from multiple sources
google_ads = vz.connect('rest', url='https://api.google.com/ads', api_key=KEY)
facebook_ads = s3.read('facebook_ads/2024.csv')
crm_data = db.query("SELECT * FROM leads WHERE source LIKE 'campaign%'")

# Combine data
campaigns_df = pd.concat([
    google_ads.read('/campaigns'),
    facebook_ads,
    crm_data
])

# Calculate ROI
campaigns_df['roi'] = (
    (campaigns_df['revenue'] - campaigns_df['cost']) / campaigns_df['cost'] * 100
)

# 1. Compare campaign performance
comparison = vz.ask("Compare ROI by campaign", campaigns_df)

# 2. Find best-performing segments
segments_chart = vz.ask("Show revenue by audience segment", campaigns_df)

# 3. Predict campaign performance
forecast_chart, predictions = vz.forecast(
    campaigns_df, 'date', 'revenue', periods=14
)

# 4. Generate insights
insights = vz.discover_insights(campaigns_df, insight_types=['correlations'])

# 5. Create stakeholder presentation
dashboard = vz.Dashboard(rows=2, cols=2)
dashboard.add_chart(comparison, row=0, col=0)
dashboard.add_chart(segments_chart, row=0, col=1)
dashboard.add_chart(forecast_chart, row=1, col=0)

# 6. Export as video for meeting
data_frames = [campaigns_df[campaigns_df['week'] == w] for w in range(1, 13)]
vz.export_video(
    comparison, 'campaign_roi.mp4',
    data_frames=data_frames,
    fps=2  # Slow animation for presentation
)

dashboard.export('campaign_dashboard.html')
print("✅ Campaign ROI analysis complete!")
```

### Scenario 3: Financial Reporting

```python
# Connect to financial database
db = vz.connect('postgresql', **config)

# Load financial data
df = db.query("""
    SELECT 
        fiscal_quarter,
        revenue,
        operating_expenses,
        net_income,
        cash_flow
    FROM financials
    WHERE fiscal_year >= 2022
""")

# 1. Trend analysis
revenue_trend = vz.detect_trend(df, 'fiscal_quarter', 'revenue')
print(f"Revenue trend: {revenue_trend[1].direction} ({revenue_trend[1].strength:.0%})")

# 2. Forecast next quarter
forecast_chart, predictions = vz.forecast(
    df, 'fiscal_quarter', 'revenue', periods=4
)

# 3. Generate financial narrative
story = vz.generate_story(
    df,
    title="Q2 2024 Financial Results",
    style='executive'
)

# 4. Create comprehensive report
report = vz.generate_report(
    df,
    title="Quarterly Financial Report Q2 2024",
    sections=[
        'executive_summary',
        'key_metrics',
        'trends',
        'forecasts',
        'recommendations'
    ]
)

# 5. Create charts
charts = [
    vz.line(df, x='fiscal_quarter', y='revenue', title='Revenue'),
    vz.line(df, x='fiscal_quarter', y='net_income', title='Net Income'),
    vz.bar(df, x='fiscal_quarter', y='cash_flow', title='Cash Flow'),
    forecast_chart
]

# 6. Export for board meeting
dashboard = vz.Dashboard(rows=2, cols=2)
for i, chart in enumerate(charts):
    dashboard.add_chart(chart, row=i//2, col=i%2)

dashboard.export('board_presentation.html')

# Save report
with open('financial_report_q2_2024.md', 'w') as f:
    f.write(report)

print("✅ Financial reporting complete!")
```

### Scenario 4: Customer Behavior Analysis

```python
# Load customer data
customers_df = vz.connect('mongodb', **mongo_config).read('customers')
events_df = s3.read('user_events/2024/*.parquet')

# Merge data
df = customers_df.merge(events_df, on='customer_id')

# 1. Segment customers using NLP
segments = vz.ask("Break down customers by behavior segment", df)

# 2. Find patterns
insights = vz.discover_insights(
    df,
    insight_types=['clusters', 'correlations']
)

# 3. Predict churn
df['days_since_last_purchase'] = (
    pd.Timestamp.now() - df['last_purchase_date']
).dt.days

churn_chart, predictions = vz.detect_anomalies(
    df, 'days_since_last_purchase',
    method='isolation_forest'
)

# 4. Generate customer insights report
story = vz.generate_story(
    df,
    title="Customer Behavior Insights",
    style='storytelling'
)

# 5. Create retention dashboard
retention_dashboard = vz.Dashboard(rows=3, cols=2)
retention_dashboard.add_chart(segments, row=0, col=0)
retention_dashboard.add_chart(churn_chart, row=0, col=1)

for i, insight in enumerate(insights[:4]):
    row = (i + 2) // 2
    col = (i + 2) % 2
    retention_dashboard.add_chart(insight.chart, row=row, col=col)

retention_dashboard.export('retention_dashboard.html')

print(f"✅ Found {len(insights)} customer insights!")
print(story)
```

### Scenario 5: Real-Time Monitoring Dashboard

```python
import vizforge as vz
import time
import schedule

# Connect to live data sources
db = vz.connect('postgresql', **db_config)
metrics_api = vz.connect('rest', url='https://api.metrics.com', api_key=KEY)

# Create dashboard
dashboard = vz.Dashboard(rows=3, cols=3, refresh_rate=60)

def update_dashboard():
    # Fetch latest data
    sales_df = db.query("""
        SELECT * FROM sales 
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
    """)
    
    metrics_df = metrics_api.read('/realtime')
    
    # Update charts
    dashboard.update_chart(
        'sales_trend',
        vz.line(sales_df, x='timestamp', y='amount')
    )
    
    dashboard.update_chart(
        'metrics',
        vz.ask("Show current metrics", metrics_df)
    )
    
    # Detect anomalies in real-time
    anomalies_chart, anomalies = vz.detect_anomalies(
        sales_df, 'amount', method='zscore'
    )
    
    if len(anomalies) > 0:
        print(f"⚠️ ALERT: {len(anomalies)} anomalies detected!")
        # Send notification
        send_alert(anomalies)
    
    dashboard.update_chart('anomalies', anomalies_chart)

# Update every minute
schedule.every().minute.do(update_dashboard)

# Initial update
update_dashboard()

# Serve dashboard
dashboard.serve(port=8050)

# Keep running
while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## 📚 API Reference

### Core Functions

```python
# Chart creation (48+ types)
vz.line(df, x, y, ...)
vz.bar(df, x, y, ...)
vz.scatter(df, x, y, ...)
vz.pie(df, values, names, ...)
# ... and 44 more

# Natural Language
vz.ask(question, df, ...)

# Predictive Analytics
vz.forecast(df, date_col, value_col, periods, ...)
vz.detect_trend(df, x, y, ...)
vz.detect_anomalies(df, value_col, method, ...)
vz.analyze_seasonality(df, date_col, value_col, ...)

# Storytelling
vz.discover_insights(df, max_insights, ...)
vz.generate_story(df, title, style, ...)
vz.generate_report(df, title, format, ...)

# Data Connectors
vz.connect(source_type, **config)
vz.list_connectors()

# Video Export
vz.export_video(chart, output_path, data_frames, ...)

# Visual Designer
vz.launch_designer(host, port)

# Dashboard
vz.Dashboard(rows, cols, ...)
dashboard.add_chart(chart, row, col)
dashboard.show()
```

### Chart Methods

```python
# All charts support
chart.show()
chart.export(path, format, width, height)
chart.to_html()
chart.to_image(format)
chart.update_layout(...)
chart.add_annotation(...)
```

---

## 💡 Best Practices

### 1. Data Quality

```python
# Always clean data first
from vizforge.utils import clean_data

df = clean_data(df,
    drop_duplicates=True,
    handle_missing='interpolate',
    remove_outliers=True
)
```

### 2. Performance

```python
# For large datasets (>100k rows), sample first
if len(df) > 100000:
    df_sample = df.sample(n=10000, random_state=42)
    chart = vz.scatter(df_sample, x='x', y='y')
else:
    chart = vz.scatter(df, x='x', y='y')
```

### 3. Reproducibility

```python
# Set random seeds
import numpy as np
np.random.seed(42)

# Save configuration
config = {
    'model': 'arima',
    'periods': 30,
    'confidence': 0.95
}

forecast_chart, predictions = vz.forecast(df, 'date', 'sales', **config)

# Save for documentation
with open('forecast_config.json', 'w') as f:
    json.dump(config, f)
```

### 4. Error Handling

```python
try:
    db = vz.connect('postgresql', **config)
    df = db.query("SELECT * FROM sales")
except Exception as e:
    print(f"Database error: {e}")
    # Fallback to CSV
    df = pd.read_csv('backup_sales.csv')
```

### 5. Resource Management

```python
# Use context managers
with vz.connect('postgresql', **config) as db:
    df = db.query("SELECT * FROM large_table LIMIT 1000")
    # Connection auto-closes
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# Issue: ModuleNotFoundError
# Solution: Install dependencies
pip install "vizforge[full]"
```

#### 2. Database Connection Errors

```python
# Issue: Connection refused
# Solution: Check host, port, credentials
db = vz.connect('postgresql',
    host='localhost',  # Correct host
    port=5432,         # Correct port
    database='mydb',
    username='user',
    password='correct_password'
)

# Test connection
if db.test_connection():
    print("✅ Connected!")
else:
    print("❌ Connection failed")
```

#### 3. Video Export Issues

```bash
# Issue: ffmpeg not found
# Solution: Install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### 4. Memory Issues with Large Datasets

```python
# Issue: MemoryError
# Solution: Use chunking or sampling

# Option 1: Sample data
df_sample = df.sample(frac=0.1, random_state=42)

# Option 2: Process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    chart = vz.line(chunk, x='date', y='value')
    chart.export(f'chunk_{i}.png')
```

#### 5. NLQ Not Understanding Query

```python
# Issue: Query not understood
# Solution: Be more specific

# Instead of:
vz.ask("show data", df)  # Too vague

# Use:
vz.ask("Show sales trend by month", df)  # Specific

# Or specify columns:
from vizforge.nlq import NLQEngine

nlq = NLQEngine()
result = nlq.query(
    "show trend",
    df,
    x_column='date',  # Explicit column mapping
    y_column='sales'
)
```

---

## 🎓 Learning Resources

### Video Tutorials

1. **Getting Started** (5 min) - Basic charts and customization
2. **Natural Language Queries** (10 min) - Talk to your data
3. **Predictive Analytics** (15 min) - Forecasting and anomalies
4. **Building Dashboards** (20 min) - Interactive dashboards
5. **Data Connectors** (10 min) - Connect to databases and APIs
6. **Video Creation** (10 min) - Export professional videos

### Example Notebooks

- `01_quick_start.ipynb` - 5-minute introduction
- `02_nlp_queries.ipynb` - Natural language examples
- `03_predictive_analytics.ipynb` - Forecasting and ML
- `04_storytelling.ipynb` - Auto insights and reports
- `05_connectors.ipynb` - Database and API connections
- `06_video_export.ipynb` - Video creation workflows

### Community

- GitHub: https://github.com/teyfikoz/VizForge
- Issues: https://github.com/teyfikoz/VizForge/issues
- Discussions: https://github.com/teyfikoz/VizForge/discussions

---

## 🚀 Next Steps

### After Reading This Guide

1. ✅ **Install VizForge**: `pip install "vizforge[full]"`
2. ✅ **Run Examples**: Try the code snippets in this guide
3. ✅ **Explore Features**: Test each of the 6 revolutionary features
4. ✅ **Build Something**: Create your first dashboard
5. ✅ **Share**: Show your work to colleagues

### Advanced Topics

- Building custom chart types
- Creating VizForge plugins
- Deploying dashboards to production
- Integrating with existing BI tools
- Performance optimization for big data

---

## 📄 Changelog

### v1.3.0 (December 2024)

**Revolutionary AI-Powered Features**:
- ✨ Natural Language Query Engine
- ✨ Predictive Analytics (Forecasting, Anomalies, Trends)
- ✨ Auto Data Storytelling
- ✨ Visual Designer (Web UI)
- ✨ Universal Data Connectors (13+ sources)
- ✨ Video Export Engine (MP4/WebM/GIF)

**Stats**: 31 new files, ~6,500 lines of code, 6 comprehensive demos

### Previous Versions

- v1.2.0 - ULTRA Intelligence (NO API)
- v1.1.0 - Super AGI 3D Features
- v1.0.0 - Intelligence & Interactivity
- v0.5.0 - Core visualization (48 chart types)

---

## 📝 License

MIT License - Free for commercial and personal use

---

## 👨‍💻 Author

**Teyfik OZ**  
Data Visualization & AI Engineer

---

**VizForge v1.3.0** - Intelligence Without APIs, Power Without Complexity

*The Ultimate AI-Powered Data Visualization Platform for Python* 🚀
