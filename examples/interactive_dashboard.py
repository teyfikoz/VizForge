"""
VizForge v1.0.0 - Interactive Dashboard Demo

Demonstrates full interactive dashboard capabilities with:
- Streamlit-style widgets
- Tableau-style filters and actions
- Dash-style callbacks
- Live dashboard server

This shows VizForge's SUPERIOR interactivity compared to competitors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import VizForge v1.0.0 features
import sys
sys.path.insert(0, '/Users/teyfikoz/Projects/vizforge')

from vizforge.interactive.widgets import (
    Slider, SelectBox, MultiSelect, DateRangePicker,
    Button, WidgetFactory
)
from vizforge.interactive.filters import (
    FilterContext, RangeFilter, ListFilter, DateRangeFilter
)
from vizforge.interactive.actions import (
    FilterAction, DrillDownAction, ActionManager
)
from vizforge.interactive.state import SessionState, get_session_state
from vizforge.interactive.callbacks import CallbackManager
from vizforge.dashboard.dashboard import Dashboard
from vizforge.dashboard.builder import DashboardServer, QuickDashboard
from vizforge.dashboard.templates import create_template
from vizforge.analytics.hierarchies import Hierarchy, HierarchyManager
from vizforge.analytics.calculated_fields import CalculatedField


# ==================== Sample Data ====================

def create_sample_sales_dashboard_data():
    """Create comprehensive sales data for dashboard."""
    np.random.seed(42)

    # Generate 2 years of daily sales data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')

    data = pd.DataFrame({
        'date': dates,
        'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany'], len(dates)),
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], len(dates)),
        'product': np.random.choice([f'Product_{i}' for i in range(1, 11)], len(dates)),
        'sales': np.random.uniform(100, 5000, len(dates)),
        'quantity': np.random.randint(1, 100, len(dates)),
        'discount': np.random.uniform(0, 0.3, len(dates)),
    })

    # Add calculated columns
    data['revenue'] = data['sales'] * (1 - data['discount'])
    data['cost'] = data['sales'] * 0.6  # 40% margin
    data['profit'] = data['revenue'] - data['cost']

    # Add seasonal trend
    data['sales'] = data['sales'] + np.sin(np.arange(len(dates)) / 365 * 2 * np.pi) * 500

    return data


# ==================== Demo 1: Basic Widget Dashboard ====================

def demo_basic_widgets():
    """
    Demo: Basic dashboard with Streamlit-style widgets.

    Shows how easy it is to create interactive controls.
    """
    print("\n" + "="*60)
    print("DEMO 1: Basic Widgets (Streamlit Parity)")
    print("="*60)

    # Create widgets
    print("\n📊 Creating widgets...")

    year_slider = WidgetFactory.year_slider(
        widget_id='year',
        min_year=2023,
        max_year=2024,
        default=2024
    )

    category_select = SelectBox(
        'category',
        'Select Category',
        options=['All', 'Electronics', 'Clothing', 'Food', 'Books'],
        default='All'
    )

    region_multiselect = MultiSelect(
        'regions',
        'Select Regions',
        options=['North', 'South', 'East', 'West'],
        default=['North', 'South']
    )

    threshold_slider = Slider(
        'threshold',
        'Sales Threshold',
        min_value=0,
        max_value=10000,
        default=1000,
        step=100
    )

    print(f"✅ Year Slider: {year_slider.value}")
    print(f"✅ Category: {category_select.value}")
    print(f"✅ Regions: {region_multiselect.value}")
    print(f"✅ Threshold: ${threshold_slider.value:,.0f}")

    # Demonstrate widget callbacks
    print("\n🔗 Testing callbacks...")

    def on_year_change(new_year):
        print(f"   Year changed to: {new_year}")

    year_slider.on_change = on_year_change
    year_slider.value = 2023  # Triggers callback


# ==================== Demo 2: Filters and Cascading ====================

def demo_filters_and_cascading():
    """
    Demo: Tableau-style filters with cascading.

    Shows advanced filtering capabilities.
    """
    print("\n" + "="*60)
    print("DEMO 2: Filters and Cascading (Tableau Parity)")
    print("="*60)

    data = create_sample_sales_dashboard_data()
    print(f"\n📊 Original data: {len(data)} rows")

    # Create filter context
    context = FilterContext()

    # Add filters
    print("\n🔧 Adding filters...")

    # Filter 1: Date range
    date_filter = DateRangeFilter(
        'date_filter',
        'date',
        start_date=datetime(2024, 1, 1).date(),
        end_date=datetime(2024, 12, 31).date()
    )
    context.add_filter(date_filter)
    print("   ✅ Date range filter: 2024 only")

    # Filter 2: Category
    category_filter = ListFilter(
        'category_filter',
        'category',
        allowed_values=['Electronics', 'Clothing']
    )
    context.add_filter(category_filter)
    print("   ✅ Category filter: Electronics & Clothing")

    # Filter 3: Sales threshold
    sales_filter = RangeFilter(
        'sales_filter',
        'sales',
        min_value=1000,
        max_value=5000
    )
    context.add_filter(sales_filter)
    print("   ✅ Sales range filter: $1,000 - $5,000")

    # Apply cascading filters
    print("\n⚡ Applying cascading filters...")
    filtered_data = context.apply_all(data, cascade=True)

    print(f"   Original: {len(data)} rows")
    print(f"   Filtered: {len(filtered_data)} rows")
    print(f"   Reduction: {(1 - len(filtered_data)/len(data))*100:.1f}%")

    # Show filter summary
    print("\n📋 Filter Summary:")
    for filter_id, filter_obj in context.filters.items():
        print(f"   - {filter_id}: {filter_obj.__class__.__name__}")


# ==================== Demo 3: Actions and Drill-Down ====================

def demo_actions_and_drilldown():
    """
    Demo: Tableau-style actions and hierarchical drill-down.

    Shows interactive exploration capabilities.
    """
    print("\n" + "="*60)
    print("DEMO 3: Actions and Drill-Down (Tableau Parity)")
    print("="*60)

    data = create_sample_sales_dashboard_data()

    # Create geographic hierarchy
    print("\n🗺️ Creating geographic hierarchy...")
    geo_hierarchy = Hierarchy(
        'Geography',
        ['country', 'region'],
        description='Geographic drill-down'
    )

    manager = HierarchyManager()
    manager.add_hierarchy(geo_hierarchy)

    print(f"   Current level: {geo_hierarchy.get_current_level()}")
    print(f"   Can drill down: {geo_hierarchy.can_drill_down()}")

    # Drill down to USA
    print("\n⬇️ Drilling down to USA...")
    path = manager.drill_down('Geography', 'USA')

    print(f"   New level: {manager.get_current_level('Geography')}")
    print(f"   Breadcrumb: {[item['value'] for item in manager.get_breadcrumb('Geography')]}")

    # Apply hierarchy filters
    filtered = manager.apply_filters(data, 'Geography')
    print(f"   Filtered data: {len(filtered)} rows (USA only)")

    # Create filter action
    print("\n🎯 Creating filter action...")
    action_mgr = ActionManager()

    filter_action = FilterAction(
        'sales_filter_action',
        source_chart='map',
        target_charts=['sales_chart', 'trend_chart'],
        filter_column='country'
    )

    action_mgr.add_action(filter_action)

    # Trigger action
    print("   Triggering filter action (select USA)...")
    result = action_mgr.trigger_action('sales_filter_action', ['USA'], data)

    if result and 'filtered_data' in result:
        print(f"   Action result: {len(result['filtered_data'])} rows")


# ==================== Demo 4: Session State Management ====================

def demo_session_state():
    """
    Demo: Streamlit-style session state.

    Shows state management for dashboards.
    """
    print("\n" + "="*60)
    print("DEMO 4: Session State (Streamlit Parity)")
    print("="*60)

    # Get session state
    state = get_session_state()

    print(f"\n🔑 Session ID: {state.session_id}")

    # Store values
    print("\n💾 Storing values...")
    state.set('selected_year', 2024)
    state.set('selected_category', 'Electronics')
    state.set('sales_threshold', 1000)
    state.set('user_preferences', {'theme': 'dark', 'language': 'en'})

    print("   ✅ Stored: selected_year, selected_category, sales_threshold, user_preferences")

    # Retrieve values
    print("\n📥 Retrieving values...")
    print(f"   Year: {state.get('selected_year')}")
    print(f"   Category: {state.get('selected_category')}")
    print(f"   Threshold: ${state.get('sales_threshold'):,}")
    print(f"   Preferences: {state.get('user_preferences')}")

    # Check existence
    print("\n🔍 Checking keys...")
    print(f"   Has 'selected_year': {state.has('selected_year')}")
    print(f"   Has 'nonexistent': {state.has('nonexistent')}")

    # Get with default
    print(f"   Get 'nonexistent' (default='N/A'): {state.get('nonexistent', default='N/A')}")


# ==================== Demo 5: Complete Interactive Dashboard ====================

def demo_complete_dashboard():
    """
    Demo: Complete interactive dashboard with all features.

    This is the showcase example!
    """
    print("\n" + "="*60)
    print("DEMO 5: Complete Interactive Dashboard (Full Stack)")
    print("="*60)

    data = create_sample_sales_dashboard_data()

    # Create dashboard
    print("\n🏗️ Building dashboard...")
    dashboard = Dashboard(rows=3, cols=3, title="Sales Analytics Dashboard")

    # Add widgets
    print("\n🎛️ Adding widgets...")

    year_slider = WidgetFactory.year_slider('year', 2023, 2024, default=2024)
    dashboard.add_widget(year_slider, row=1, col=1)

    category_select = SelectBox(
        'category',
        'Category',
        options=['All'] + data['category'].unique().tolist(),
        default='All'
    )
    dashboard.add_widget(category_select, row=1, col=2)

    print("   ✅ Added year slider and category selector")

    # Add filters
    print("\n🔧 Adding filters...")

    sales_filter = RangeFilter('sales_range', 'sales', min_value=0, max_value=10000)
    dashboard.add_filter(sales_filter)

    print("   ✅ Added sales range filter")

    # Add actions
    print("\n🎯 Adding actions...")

    hierarchy = Hierarchy('Product', ['category', 'product'])
    drill_action = DrillDownAction('product_drilldown', 'main_chart', hierarchy.levels)

    dashboard.add_action(drill_action)

    print("   ✅ Added drill-down action")

    # Add calculated fields
    print("\n🧮 Adding calculated fields...")

    profit_field = CalculatedField('profit_margin', '([profit] / [revenue]) * 100')

    print("   ✅ Added profit margin calculation")

    # Setup callbacks
    print("\n🔗 Setting up callbacks...")

    @dashboard.callback(outputs='sales_chart', inputs='year')
    def update_sales(year):
        filtered = data[data['date'].dt.year == year]
        return f"Filtered to {len(filtered)} rows for {year}"

    print("   ✅ Added callback for year selection")

    # Get session state
    state = dashboard.get_session_state()
    state.set('dashboard_loaded', True)

    print("\n✅ Dashboard assembly complete!")
    print(f"   Widgets: {len(dashboard.widgets)}")
    print(f"   Filters: {len(dashboard.filters)}")
    print(f"   Actions: {len(dashboard.actions) if hasattr(dashboard, 'actions') else 0}")


# ==================== Demo 6: Dashboard Templates ====================

def demo_dashboard_templates():
    """
    Demo: Pre-built dashboard templates.

    Quick start with professional templates.
    """
    print("\n" + "="*60)
    print("DEMO 6: Dashboard Templates (Quick Start)")
    print("="*60)

    data = create_sample_sales_dashboard_data()

    # Test different templates
    templates = ['kpi', 'analytics', 'executive']

    for template_type in templates:
        print(f"\n📊 Creating {template_type.upper()} dashboard...")

        template = create_template(
            template_type,
            name=f'{template_type.title()} Dashboard',
            color_scheme='dark'
        )

        dashboard = template.build(data)
        result = dashboard.render()

        print(f"   ✅ Template: {result['template']}")
        print(f"   Theme: {template.theme['primary']}")
        print(f"   Layout: {template.config.layout_type.value}")


# ==================== Demo 7: Dashboard Server ====================

def demo_dashboard_server():
    """
    Demo: Launch interactive dashboard server.

    NOTE: This creates server but doesn't actually start it
    (to avoid blocking in demo mode).
    """
    print("\n" + "="*60)
    print("DEMO 7: Dashboard Server (Dash Integration)")
    print("="*60)

    data = create_sample_sales_dashboard_data()

    print("\n🚀 Creating dashboard server...")

    # Create quick dashboard
    server = QuickDashboard.from_dataframe(
        data,
        x='date',
        y='sales',
        title="Sales Analytics"
    )

    print("   ✅ Server created")
    print("   ℹ️ To run server: dashboard.serve(port=8050)")
    print("   ℹ️ Would open: http://localhost:8050")
    print("\n   NOTE: Server not started in demo mode")
    print("         In production, call: dashboard.serve()")


# ==================== Demo 8: Advanced Workflow ====================

def demo_advanced_workflow():
    """
    Demo: Advanced workflow combining all features.

    Real-world example with full interactivity.
    """
    print("\n" + "="*60)
    print("DEMO 8: Advanced Workflow (Production Example)")
    print("="*60)

    data = create_sample_sales_dashboard_data()

    print("\n📊 Creating production-ready dashboard...")

    # Step 1: Initialize dashboard
    dashboard = Dashboard(rows=4, cols=4, title="Executive Sales Dashboard")
    print("   ✅ Dashboard initialized")

    # Step 2: Add widgets
    widgets = {
        'year': WidgetFactory.year_slider('year', 2023, 2024, 2024),
        'category': MultiSelect('category', 'Categories',
                               options=data['category'].unique().tolist(),
                               default=data['category'].unique().tolist()),
        'threshold': WidgetFactory.percentage_slider('threshold', default=50.0),
    }

    for name, widget in widgets.items():
        dashboard.add_widget(widget, row=1, col=1)

    print(f"   ✅ Added {len(widgets)} widgets")

    # Step 3: Add filters with cascading
    filter_context = FilterContext()

    filter_context.add_filter(
        DateRangeFilter('date', 'date',
                       start_date=datetime(2024, 1, 1).date(),
                       end_date=datetime(2024, 12, 31).date())
    )

    filter_context.add_filter(
        ListFilter('category', 'category',
                  allowed_values=['Electronics', 'Clothing'])
    )

    print("   ✅ Added cascading filters")

    # Step 4: Add hierarchies
    hierarchies = HierarchyManager()
    hierarchies.add_hierarchy(
        Hierarchy('Geography', ['country', 'region'])
    )
    hierarchies.add_hierarchy(
        Hierarchy('Product', ['category', 'product'])
    )

    print("   ✅ Added 2 hierarchies")

    # Step 5: Add calculated fields
    calc_fields = [
        CalculatedField('margin', '([profit] / [revenue]) * 100'),
        CalculatedField('avg_order_value', '[revenue] / [quantity]'),
    ]

    print(f"   ✅ Added {len(calc_fields)} calculated fields")

    # Step 6: Setup session state
    state = dashboard.get_session_state()
    state.set('filters_applied', True)
    state.set('hierarchies_loaded', True)

    print("   ✅ Session state configured")

    # Step 7: Create summary
    print("\n📈 Dashboard Summary:")
    print(f"   Data: {len(data):,} rows")
    print(f"   Widgets: {len(widgets)}")
    print(f"   Filters: {len(filter_context.filters)}")
    print(f"   Hierarchies: {len(hierarchies.hierarchies)}")
    print(f"   Calculated Fields: {len(calc_fields)}")

    print("\n✅ Production dashboard ready to serve!")


# ==================== Main ====================

def main():
    """Run all interactive dashboard demos."""
    print("\n" + "="*60)
    print("🚀 VizForge v1.0.0 - Interactive Dashboard Demo")
    print("="*60)
    print("\nDemonstrating SUPERIOR interactivity:")
    print("  • Streamlit-style widgets (easier)")
    print("  • Tableau-style filters & actions (more powerful)")
    print("  • Dash-style callbacks (flexible)")
    print("  • All-in-one solution!")

    try:
        # Run all demos
        demo_basic_widgets()
        demo_filters_and_cascading()
        demo_actions_and_drilldown()
        demo_session_state()
        demo_complete_dashboard()
        demo_dashboard_templates()
        demo_dashboard_server()
        demo_advanced_workflow()

        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)

        print("\n💡 Key Features Demonstrated:")
        print("  1. Streamlit-style widgets - Easy to use")
        print("  2. Tableau-style filters - Powerful cascading")
        print("  3. Hierarchical drill-down - Explore data")
        print("  4. Session state - Persistent user data")
        print("  5. Dashboard templates - Quick start")
        print("  6. Dash server - Production deployment")
        print("  7. Complete integration - All features work together")

        print("\n🎯 Next Steps:")
        print("  - Run: python examples/v1_migration.py")
        print("  - Read: MIGRATION_GUIDE.md")
        print("  - Build: Your own interactive dashboard!")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
