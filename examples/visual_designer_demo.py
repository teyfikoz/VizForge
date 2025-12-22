"""
VizForge Visual Designer Demo

Demonstrates how to launch the visual chart designer web interface.
"""

import vizforge as vz

def demo_launch_designer():
    """Demo 1: Launch the visual designer."""
    print("=" * 60)
    print("DEMO 1: Launch Visual Designer")
    print("=" * 60)

    print("\n✨ Launching Visual Chart Designer...")
    print("\nThis will open a web interface at http://localhost:5000")
    print("\nFeatures:")
    print("  - Upload data (CSV, Excel, JSON, Parquet)")
    print("  - Drag & drop chart builder")
    print("  - 30+ chart types organized by category")
    print("  - Live preview")
    print("  - Property editor")
    print("  - Export Python code")
    print("  - Export images")

    print("\n" + "=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # Launch the designer
    # Uncomment to actually launch (blocks execution):
    # vz.launch_designer(port=5000, debug=True)

    print("✅ Demo complete!")
    print("\nTo actually launch the designer, uncomment the line:")
    print("   vz.launch_designer(port=5000, debug=True)")


def demo_programmatic_usage():
    """Demo 2: Use the designer components programmatically."""
    print("\n" + "=" * 60)
    print("DEMO 2: Programmatic Code Generation")
    print("=" * 60)

    from vizforge.visual_designer import ChartConfig, ChartType, CodeGenerator

    # Create a chart configuration
    config = ChartConfig(
        chart_type=ChartType.LINE,
        title="Sales Trend Analysis",
        properties={
            'x': 'date',
            'y': 'sales',
            'color': 'region',
            'width': 900,
            'height': 600,
            'theme': 'professional',
            'show_legend': True
        },
        data_source='sales_data.csv'
    )

    print("\n📊 Chart Configuration:")
    print(f"  Type: {config.chart_type.value}")
    print(f"  Title: {config.title}")
    print(f"  Properties: {len(config.properties)} configured")

    # Validate configuration
    valid, error = config.validate()
    print(f"\n✅ Validation: {'PASSED' if valid else f'FAILED - {error}'}")

    # Generate Python code
    generator = CodeGenerator()
    code = generator.generate(config, include_imports=True, include_data_loading=True)

    print("\n📝 Generated Python Code:")
    print("-" * 60)
    print(code)
    print("-" * 60)

    print("\n✅ Demo complete!")


def demo_chart_properties():
    """Demo 3: Explore available chart properties."""
    print("\n" + "=" * 60)
    print("DEMO 3: Chart Properties Explorer")
    print("=" * 60)

    from vizforge.visual_designer import ChartConfig, ChartType

    # Test different chart types
    test_types = [
        ChartType.LINE,
        ChartType.BAR,
        ChartType.PIE,
        ChartType.HEATMAP,
        ChartType.SCATTER3D,
        ChartType.BUBBLE
    ]

    for chart_type in test_types:
        print(f"\n📊 {chart_type.value.upper()} Chart Properties:")
        print("-" * 40)

        properties = ChartConfig.get_available_properties(chart_type)

        # Group by required/optional
        required = [p for p in properties if p.required]
        optional = [p for p in properties if not p.required]

        print(f"\n  Required ({len(required)}):")
        for prop in required:
            print(f"    - {prop.label} ({prop.type.value})")

        print(f"\n  Optional ({len(optional)}):")
        for prop in optional[:5]:  # Show first 5
            print(f"    - {prop.label} ({prop.type.value}): {prop.description[:50]}...")

        if len(optional) > 5:
            print(f"    ... and {len(optional) - 5} more")

    print("\n✅ Demo complete!")


def demo_chart_categories():
    """Demo 4: Display all available chart categories."""
    print("\n" + "=" * 60)
    print("DEMO 4: Available Chart Categories")
    print("=" * 60)

    from vizforge.visual_designer.chart_config import CHART_CATEGORIES

    total_charts = sum(len(types) for types in CHART_CATEGORIES.values())

    print(f"\n📚 Total: {len(CHART_CATEGORIES)} categories, {total_charts} chart types\n")

    for category, types in CHART_CATEGORIES.items():
        print(f"\n{category} ({len(types)} types):")
        print("-" * 40)
        for chart_type in types:
            print(f"  • {chart_type.value.replace('_', ' ').title()}")

    print("\n✅ Demo complete!")


def demo_multi_chart_notebook():
    """Demo 5: Generate notebook code for multiple charts."""
    print("\n" + "=" * 60)
    print("DEMO 5: Multi-Chart Notebook Generation")
    print("=" * 60)

    from vizforge.visual_designer import ChartConfig, ChartType, CodeGenerator

    # Create multiple chart configurations
    configs = [
        ChartConfig(
            chart_type=ChartType.LINE,
            title="Sales Trend",
            properties={'x': 'date', 'y': 'sales', 'color': 'region'},
            data_source='data.csv'
        ),
        ChartConfig(
            chart_type=ChartType.BAR,
            title="Top Products",
            properties={'x': 'product', 'y': 'revenue', 'orientation': 'horizontal'},
            data_source='data.csv'
        ),
        ChartConfig(
            chart_type=ChartType.PIE,
            title="Market Share",
            properties={'labels': 'category', 'values': 'market_share'},
            data_source='data.csv'
        ),
    ]

    print(f"\n📊 Generating notebook with {len(configs)} charts...\n")

    # Generate notebook code
    generator = CodeGenerator()
    notebook_code = generator.generate_notebook(configs)

    print("📝 Generated Notebook Code:")
    print("=" * 60)
    print(notebook_code)
    print("=" * 60)

    print("\n✅ Demo complete!")
    print("\n💡 Tip: Copy this code into a Jupyter notebook and run it!")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("🎨 VizForge Visual Designer - Comprehensive Demo")
    print("=" * 70)

    print("\nThis demo showcases the Visual Chart Designer features:")
    print("  1. Launch web-based designer")
    print("  2. Programmatic code generation")
    print("  3. Chart properties exploration")
    print("  4. Available chart categories")
    print("  5. Multi-chart notebook generation")

    # Run demos
    demo_launch_designer()
    demo_programmatic_usage()
    demo_chart_properties()
    demo_chart_categories()
    demo_multi_chart_notebook()

    print("\n" + "=" * 70)
    print("✅ ALL DEMOS COMPLETE!")
    print("=" * 70)

    print("\n🚀 Next Steps:")
    print("  1. Launch the visual designer:")
    print("     >>> import vizforge as vz")
    print("     >>> vz.launch_designer()")
    print("\n  2. Upload your data and start creating charts!")
    print("\n  3. Export Python code and use in your projects!")


if __name__ == '__main__':
    main()
