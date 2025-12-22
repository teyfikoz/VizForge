"""
VizForge v1.2.0 - Enhanced Auto-Insights v2 Demo

Demonstrates intelligent automatic insights generation WITHOUT any API!
Pure statistical analysis with natural language report generation.

Run this file to see auto-insights in action!
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

from vizforge.intelligence import EnhancedInsightsEngine, InsightReport

print("=" * 80)
print("VizForge v1.2.0 - Enhanced Auto-Insights Engine v2")
print("=" * 80)
print("\n🧠 NO API Required - Pure Statistical Intelligence + NLG!\n")


# ==================== Example 1: E-Commerce Sales Analysis ====================

def example_1_ecommerce():
    """Example 1: Complete e-commerce sales analysis."""
    print("\n" + "=" * 80)
    print("Example 1: E-Commerce Sales Analysis")
    print("=" * 80)

    # Create realistic e-commerce data
    np.random.seed(42)
    n = 365

    # Sales with trend + seasonality
    time = np.arange(n)
    trend = 0.4 * time + 1000  # Growing business
    seasonality = 200 * np.sin(2 * np.pi * time / 7)  # Weekly pattern
    noise = np.random.normal(0, 50, n)

    sales = trend + seasonality + noise

    # Add holiday spikes
    sales[50] += 500   # Valentine's Day
    sales[120] += 700  # Summer sale
    sales[300] += 900  # Black Friday
    sales[355] += 600  # Christmas

    # Marketing spend (correlated with sales)
    marketing = 0.3 * sales + np.random.normal(100, 30, n)

    # Customer acquisition cost
    cac = marketing / (sales / 100)  # Cost per customer

    # Profit margin
    revenue = sales
    costs = marketing + np.random.normal(400, 50, n)
    profit = revenue - costs

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'revenue': revenue,
        'marketing_spend': marketing,
        'costs': costs,
        'profit': profit,
        'cac': cac,
        'conversion_rate': np.random.normal(0.03, 0.005, n)
    })

    print("\n📊 Data: 1 year of e-commerce metrics")
    print(f"   - {n} days")
    print(f"   - {len(df.columns)} metrics")
    print(f"   - Revenue range: ${revenue.min():.0f} - ${revenue.max():.0f}")

    # Generate comprehensive insights
    print("\n🔍 Generating comprehensive insights report...\n")

    engine = EnhancedInsightsEngine(df)
    report = engine.generate_report(verbose=True)

    print("\n" + "=" * 80)
    print("📝 EXECUTIVE SUMMARY")
    print("=" * 80)
    print(report.summary)

    print("\n" + "=" * 80)
    print("🎯 KEY FINDINGS")
    print("=" * 80)
    for i, finding in enumerate(report.key_findings, 1):
        print(f"\n{i}. {finding}")

    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"\n{i}. {rec}")

    print("\n" + "=" * 80)
    print("📊 STATISTICS SUMMARY")
    print("=" * 80)
    for key, value in report.statistics.items():
        print(f"  {key}: {value}")

    # Export to Markdown
    print("\n💾 Exporting report to Markdown...")
    markdown = report.to_markdown()
    with open('/tmp/ecommerce_insights.md', 'w') as f:
        f.write(markdown)
    print("   ✅ Saved to: /tmp/ecommerce_insights.md")

    # Export to HTML
    print("\n💾 Exporting report to HTML...")
    html = report.to_html()
    with open('/tmp/ecommerce_insights.html', 'w') as f:
        f.write(html)
    print("   ✅ Saved to: /tmp/ecommerce_insights.html")


# ==================== Example 2: Financial Market Analysis ====================

def example_2_financial():
    """Example 2: Stock market analysis."""
    print("\n" + "=" * 80)
    print("Example 2: Financial Market Analysis")
    print("=" * 80)

    # Create stock market data
    np.random.seed(42)
    n = 250  # Trading days

    # Stock price with trend + volatility
    returns = np.random.normal(0.001, 0.02, n)
    price = 100 * np.exp(np.cumsum(returns))

    # Add market crash
    price[150:160] *= 0.85  # 15% crash

    # Volume (inverse relationship with price changes)
    volume = 1000000 + 500000 * np.abs(np.diff(price, prepend=100))

    # Market indicators
    rsi = 50 + 30 * np.sin(2 * np.pi * np.arange(n) / 14)  # RSI indicator

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'price': price,
        'volume': volume,
        'rsi': rsi,
        'returns': np.concatenate([[0], np.diff(price) / price[:-1]])
    })

    print("\n📊 Data: 250 trading days")
    print(f"   - Price range: ${price.min():.2f} - ${price.max():.2f}")
    print(f"   - Avg volume: {volume.mean():.0f}")

    # Generate insights
    print("\n🔍 Analyzing market patterns...\n")

    engine = EnhancedInsightsEngine(df)

    # Get specific insights
    print("📈 Trend Analysis:")
    trend_insight = engine.explain_trend('price')
    print(f"   {trend_insight}\n")

    print("🔗 Correlation Analysis:")
    corr_insight = engine.explain_correlation('price', 'volume')
    print(f"   {corr_insight}\n")

    # Full report
    report = engine.generate_report(verbose=True)

    print("\n" + "=" * 80)
    print("📝 MARKET INSIGHTS REPORT")
    print("=" * 80)
    print(report.summary)

    print("\n🎯 Key Findings:")
    for finding in report.key_findings[:3]:
        print(f"  • {finding}")


# ==================== Example 3: SaaS Business Metrics ====================

def example_3_saas():
    """Example 3: SaaS company metrics."""
    print("\n" + "=" * 80)
    print("Example 3: SaaS Business Metrics")
    print("=" * 80)

    # Create SaaS metrics
    np.random.seed(42)
    n = 12  # Monthly data

    months = pd.date_range('2024-01-01', periods=n, freq='MS')

    # MRR (Monthly Recurring Revenue) - growing
    mrr = 50000 + 5000 * np.arange(n) + np.random.normal(0, 2000, n)

    # Churn rate (decreasing - good sign)
    churn_rate = 0.08 - 0.004 * np.arange(n) + np.random.normal(0, 0.005, n)

    # Customer count
    customers = 500 + 40 * np.arange(n) + np.random.randint(-10, 20, n)

    # LTV (Lifetime Value) - increasing
    ltv = mrr / customers / churn_rate

    # CAC (Customer Acquisition Cost)
    cac = np.random.normal(300, 50, n)

    # LTV/CAC ratio (should be > 3)
    ltv_cac_ratio = ltv / cac

    df = pd.DataFrame({
        'month': months,
        'mrr': mrr,
        'churn_rate': churn_rate,
        'customers': customers,
        'ltv': ltv,
        'cac': cac,
        'ltv_cac_ratio': ltv_cac_ratio,
        'revenue_per_customer': mrr / customers
    })

    print("\n📊 Data: 12 months of SaaS metrics")
    print(f"   - MRR: ${mrr.min():.0f} - ${mrr.max():.0f}")
    print(f"   - Customers: {customers.min():.0f} - {customers.max():.0f}")

    # Generate insights
    print("\n🔍 Analyzing SaaS health...\n")

    engine = EnhancedInsightsEngine(df)
    report = engine.generate_report(verbose=True)

    print("\n" + "=" * 80)
    print("📝 SAAS HEALTH REPORT")
    print("=" * 80)
    print(report.summary)

    print("\n💡 Strategic Recommendations:")
    for rec in report.recommendations:
        print(f"  ✓ {rec}")

    # Detailed insights
    print("\n" + "=" * 80)
    print("🔬 DETAILED INSIGHTS")
    print("=" * 80)
    for category, insights in report.detailed_insights.items():
        print(f"\n{category.upper()}:")
        for insight in insights:
            print(f"  • {insight}")


# ==================== Example 4: Marketing Campaign Analysis ====================

def example_4_marketing():
    """Example 4: Marketing campaign effectiveness."""
    print("\n" + "=" * 80)
    print("Example 4: Marketing Campaign Analysis")
    print("=" * 80)

    # Create campaign data
    np.random.seed(42)
    n = 30  # Days

    # Multiple channels
    email_spend = np.random.normal(1000, 200, n)
    social_spend = np.random.normal(1500, 300, n)
    search_spend = np.random.normal(2000, 400, n)

    # Conversions (correlated with spend)
    email_conv = 0.05 * email_spend + np.random.normal(20, 5, n)
    social_conv = 0.04 * social_spend + np.random.normal(25, 6, n)
    search_conv = 0.06 * search_spend + np.random.normal(50, 10, n)

    total_spend = email_spend + social_spend + search_spend
    total_conv = email_conv + social_conv + search_conv

    # ROI
    revenue = total_conv * 150  # $150 per conversion
    roi = (revenue - total_spend) / total_spend * 100

    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'email_spend': email_spend,
        'social_spend': social_spend,
        'search_spend': search_spend,
        'email_conversions': email_conv,
        'social_conversions': social_conv,
        'search_conversions': search_conv,
        'total_spend': total_spend,
        'total_conversions': total_conv,
        'revenue': revenue,
        'roi': roi
    })

    print("\n📊 Data: 30 days of marketing campaigns")
    print(f"   - 3 channels: Email, Social, Search")
    print(f"   - Total spend: ${total_spend.sum():.0f}")
    print(f"   - Total revenue: ${revenue.sum():.0f}")

    # Generate insights
    print("\n🔍 Analyzing campaign performance...\n")

    engine = EnhancedInsightsEngine(df)
    report = engine.generate_report(verbose=True)

    print("\n" + "=" * 80)
    print("📝 CAMPAIGN PERFORMANCE REPORT")
    print("=" * 80)

    print("\n🎯 Top Insights:")
    for i, finding in enumerate(report.key_findings[:5], 1):
        print(f"{i}. {finding}")

    print("\n💰 ROI Analysis:")
    roi_insight = engine.explain_trend('roi')
    print(f"   {roi_insight}")

    # Export
    markdown = report.to_markdown()
    with open('/tmp/marketing_insights.md', 'w') as f:
        f.write(markdown)
    print("\n✅ Full report exported to: /tmp/marketing_insights.md")


# ==================== Example 5: Quick One-Liner Analysis ====================

def example_5_quick():
    """Example 5: Quick one-liner analysis."""
    print("\n" + "=" * 80)
    print("Example 5: Quick One-Liner Analysis")
    print("=" * 80)

    # Simple dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'product_views': np.random.poisson(1000, 100),
        'add_to_cart': np.random.poisson(100, 100),
        'purchases': np.random.poisson(20, 100)
    })

    # Calculate conversion funnel
    df['view_to_cart'] = df['add_to_cart'] / df['product_views'] * 100
    df['cart_to_purchase'] = df['purchases'] / df['add_to_cart'] * 100

    print("\n📊 Data: 100 days of conversion funnel")
    print(f"   - Avg views: {df['product_views'].mean():.0f}")
    print(f"   - Avg purchases: {df['purchases'].mean():.0f}")

    # One-liner analysis!
    print("\n🔍 Quick analysis (one line!):")
    print("\n>>> report = EnhancedInsightsEngine(df).generate_report()")

    report = EnhancedInsightsEngine(df).generate_report()

    print("\n✅ Instant insights generated!\n")
    print("Summary:", report.summary)
    print("\nTop Finding:", report.key_findings[0] if report.key_findings else "No findings")


# ==================== Example 6: Custom Insights Request ====================

def example_6_custom():
    """Example 6: Custom insight requests."""
    print("\n" + "=" * 80)
    print("Example 6: Custom Insights (Explain Specific Patterns)")
    print("=" * 80)

    # Create data
    np.random.seed(42)
    n = 200

    # Temperature and ice cream sales (strong correlation)
    temperature = np.random.normal(25, 8, n)
    ice_cream_sales = 100 + 15 * temperature + np.random.normal(0, 50, n)

    # Umbrella sales (negative correlation with temperature)
    umbrella_sales = 500 - 10 * temperature + np.random.normal(0, 30, n)

    df = pd.DataFrame({
        'temperature': temperature,
        'ice_cream_sales': ice_cream_sales,
        'umbrella_sales': umbrella_sales
    })

    print("\n📊 Data: 200 days of weather & sales")

    engine = EnhancedInsightsEngine(df)

    # Get specific explanations
    print("\n🔍 Specific Insights:\n")

    print("1. Temperature Trend:")
    print(f"   {engine.explain_trend('temperature')}\n")

    print("2. Temperature ↔ Ice Cream Correlation:")
    print(f"   {engine.explain_correlation('temperature', 'ice_cream_sales')}\n")

    print("3. Temperature ↔ Umbrella Correlation:")
    print(f"   {engine.explain_correlation('temperature', 'umbrella_sales')}\n")


# ==================== Main Runner ====================

def main():
    """Run all enhanced insights examples."""
    print("\n" + "=" * 80)
    print("🚀 VizForge Enhanced Auto-Insights v2 - Complete Demo")
    print("=" * 80)

    try:
        example_1_ecommerce()
        example_2_financial()
        example_3_saas()
        example_4_marketing()
        example_5_quick()
        example_6_custom()

        print("\n" + "=" * 80)
        print("✅ All Enhanced Auto-Insights Examples Completed!")
        print("=" * 80)

        print("\n💡 Key Takeaways:")
        print("  ✅ NO API required - 100% local processing")
        print("  ✅ Natural language reports - human-readable insights")
        print("  ✅ Fast - milliseconds for comprehensive analysis")
        print("  ✅ Comprehensive - patterns + trends + correlations + anomalies")
        print("  ✅ Actionable - specific recommendations")
        print("  ✅ Exportable - Markdown & HTML formats")

        print("\n🎯 Features Demonstrated:")
        print("  1. Executive summaries")
        print("  2. Key findings extraction")
        print("  3. Actionable recommendations")
        print("  4. Detailed insights by category")
        print("  5. Statistical summaries")
        print("  6. Trend explanations")
        print("  7. Correlation explanations")
        print("  8. Export to Markdown/HTML")

        print("\n📚 Usage:")
        print("  from vizforge.intelligence import EnhancedInsightsEngine")
        print("  engine = EnhancedInsightsEngine(df)")
        print("  report = engine.generate_report()")
        print("  print(report.summary)")
        print("  report.to_html()  # Export as HTML")

        print("\n📄 Exported Reports:")
        print("  • /tmp/ecommerce_insights.md")
        print("  • /tmp/ecommerce_insights.html")
        print("  • /tmp/marketing_insights.md")

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
