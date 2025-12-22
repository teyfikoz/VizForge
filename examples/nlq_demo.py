"""
VizForge v1.3.0 - Natural Language Query (NLQ) Demo

ASK QUESTIONS IN ENGLISH → GET AUTOMATIC VISUALIZATIONS!
NO API required - pure intelligence!

This is REVOLUTIONARY! No other Python library has this!
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

import vizforge as vz

print("=" * 80)
print("VizForge v1.3.0 - Natural Language Query Engine")
print("=" * 80)
print("\n🎤 ASK QUESTIONS IN ENGLISH - GET AUTOMATIC VISUALIZATIONS!")
print("🚀 NO API Required - Pure Intelligence!\n")


# ==================== Example 1: Sales Data ====================

def example_1_sales():
    """Example 1: Sales trend analysis."""
    print("\n" + "=" * 80)
    print("Example 1: Sales Data - E-Commerce")
    print("=" * 80)

    # Create realistic sales data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'sales': 1000 + np.cumsum(np.random.normal(10, 50, 365)),
        'profit': 300 + np.cumsum(np.random.normal(3, 15, 365)),
        'customers': np.random.randint(50, 200, 365),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
        'product': np.random.choice(['A', 'B', 'C', 'D', 'E'], 365)
    })

    print(f"\n📊 Data: {len(df)} days of sales data")
    print(f"   Columns: {list(df.columns)}")

    # Example queries
    queries = [
        "Show me sales trend by date",
        "Compare sales vs profit",
        "Find top 10 days by sales",
        "Show sales by region",
    ]

    print("\n🎤 Asking questions...")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Q: '{query}'")
        try:
            chart = vz.ask(query, df, verbose=True)
            print(f"   ✅ Chart created: {type(chart).__name__}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


# ==================== Example 2: Marketing Data ====================

def example_2_marketing():
    """Example 2: Marketing campaign analysis."""
    print("\n" + "=" * 80)
    print("Example 2: Marketing Campaign Data")
    print("=" * 80)

    # Create marketing data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'campaign': np.random.choice(['Email', 'Social', 'Search', 'Display'], n),
        'spend': np.random.normal(5000, 1000, n),
        'impressions': np.random.randint(10000, 100000, n),
        'clicks': np.random.randint(100, 5000, n),
        'conversions': np.random.randint(10, 200, n),
        'revenue': np.random.normal(15000, 3000, n)
    })

    df['ctr'] = (df['clicks'] / df['impressions']) * 100
    df['roi'] = ((df['revenue'] - df['spend']) / df['spend']) * 100

    print(f"\n📊 Data: {len(df)} marketing campaigns")
    print(f"   Columns: {list(df.columns)}")

    # Example queries
    queries = [
        "Compare spend vs revenue",
        "Show ROI by campaign",
        "Find top 5 campaigns by conversions",
        "What is the correlation between spend and revenue",
    ]

    print("\n🎤 Asking marketing questions...")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Q: '{query}'")
        try:
            chart = vz.ask(query, df, verbose=False)
            print(f"   ✅ {type(chart).__name__} generated!")
        except Exception as e:
            print(f"   ❌ Error: {e}")


# ==================== Example 3: Product Performance ====================

def example_3_products():
    """Example 3: Product performance analysis."""
    print("\n" + "=" * 80)
    print("Example 3: Product Performance Data")
    print("=" * 80)

    # Create product data
    np.random.seed(42)

    products = ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones',
                'Keyboard', 'Mouse', 'Monitor', 'Camera', 'Speaker']

    df = pd.DataFrame({
        'product': products,
        'units_sold': np.random.randint(100, 1000, 10),
        'revenue': np.random.randint(50000, 500000, 10),
        'rating': np.random.uniform(3.5, 5.0, 10),
        'reviews': np.random.randint(50, 500, 10),
    })

    print(f"\n📊 Data: {len(df)} products")

    # Example queries
    queries = [
        "Show top 5 products by revenue",
        "Compare revenue vs rating",
        "Find products with highest ratings",
    ]

    print("\n🎤 Asking product questions...")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Q: '{query}'")
        try:
            chart = vz.ask(query, df)
            print(f"   ✅ Success!")
        except Exception as e:
            print(f"   ❌ Error: {e}")


# ==================== Example 4: Time Series ====================

def example_4_timeseries():
    """Example 4: Time series analysis."""
    print("\n" + "=" * 80)
    print("Example 4: Time Series - Stock Prices")
    print("=" * 80)

    # Create time series data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='B')  # Business days

    price = 100
    prices = []
    for _ in range(252):
        price += np.random.normal(0.2, 2)
        prices.append(price)

    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000000, 10000000, 252),
    })

    print(f"\n📊 Data: {len(df)} trading days")

    # Example queries
    queries = [
        "Show price trend over time",
        "Compare price vs volume",
    ]

    print("\n🎤 Asking time series questions...")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Q: '{query}'")
        try:
            chart = vz.ask(query, df)
            print(f"   ✅ Success!")
        except Exception as e:
            print(f"   ❌ Error: {e}")


# ==================== Example 5: Interactive Demo ====================

def example_5_interactive():
    """Example 5: Interactive question answering."""
    print("\n" + "=" * 80)
    print("Example 5: TRY YOUR OWN QUESTIONS!")
    print("=" * 80)

    # Sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=12, freq='MS'),
        'revenue': np.random.randint(50000, 150000, 12),
        'expenses': np.random.randint(30000, 80000, 12),
        'customers': np.random.randint(500, 2000, 12),
    })

    df['profit'] = df['revenue'] - df['expenses']

    print(f"\n📊 Sample Dataset:")
    print(df.head())

    print("\n💡 Example questions you can ask:")
    print("  • Show revenue trend by month")
    print("  • Compare revenue vs profit")
    print("  • Find month with highest profit")
    print("  • Show customer growth over time")

    # Ask a few questions
    questions = [
        "Show revenue trend by month",
        "Compare revenue vs profit",
    ]

    print("\n🎤 Demo questions:")
    for q in questions:
        print(f"\nQ: '{q}'")
        try:
            chart = vz.ask(q, df, verbose=True)
            print(f"✅ Chart type: {type(chart).__name__}")
        except Exception as e:
            print(f"❌ Error: {e}")


# ==================== Main ====================

def main():
    """Run all NLQ examples."""
    print("\n" + "=" * 80)
    print("🚀 VizForge NLQ Engine - Complete Demo")
    print("=" * 80)

    try:
        example_1_sales()
        example_2_marketing()
        example_3_products()
        example_4_timeseries()
        example_5_interactive()

        print("\n" + "=" * 80)
        print("✅ All NLQ Examples Completed!")
        print("=" * 80)

        print("\n💡 Key Features:")
        print("  ✅ NO API required - 100% local intelligence")
        print("  ✅ Plain English questions → Auto visualizations")
        print("  ✅ Intent detection (10+ types)")
        print("  ✅ Smart column matching")
        print("  ✅ Auto chart type selection")
        print("  ✅ Handles complex queries")
        print("  ✅ Fast - instant response")

        print("\n🎯 Supported Query Types:")
        print("  • Trend: 'Show X trend over time'")
        print("  • Comparison: 'Compare X vs Y'")
        print("  • Top N: 'Find top 10 by X'")
        print("  • Breakdown: 'Show X by region'")
        print("  • Correlation: 'What is correlation between X and Y'")
        print("  • Distribution: 'Show distribution of X'")

        print("\n📚 Usage:")
        print("  import vizforge as vz")
        print("  chart = vz.ask('Show sales trend', df)")
        print("  chart.show()")

        print("\n" + "=" * 80)
        print("🎉 VizForge v1.3.0 - Natural Language Query Engine!")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
