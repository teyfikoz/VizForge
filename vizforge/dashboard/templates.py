"""
VizForge Dashboard Templates

Pre-built dashboard templates for common use cases.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go
from .layout import SmartLayoutEngine, LayoutType, create_grid_layout, create_sidebar_layout


@dataclass
class TemplateConfig:
    """
    Dashboard template configuration.

    Attributes:
        name: Template name
        description: Template description
        layout_type: Layout type
        color_scheme: Color scheme
        show_logo: Show company logo
        show_filters: Show filter panel
        show_export: Show export button
    """
    name: str
    description: str = ""
    layout_type: LayoutType = LayoutType.GRID
    color_scheme: str = "default"
    show_logo: bool = True
    show_filters: bool = True
    show_export: bool = True


class DashboardTemplate:
    """
    Base class for dashboard templates.

    Provides common functionality for all templates.
    """

    def __init__(self, config: TemplateConfig):
        """
        Initialize dashboard template.

        Args:
            config: Template configuration
        """
        self.config = config
        self.layout_engine: Optional[SmartLayoutEngine] = None
        self.components: List[Any] = []
        self.theme = self._get_color_scheme(config.color_scheme)

    def _get_color_scheme(self, scheme: str) -> Dict[str, str]:
        """Get color scheme configuration."""
        schemes = {
            'default': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ff9800',
                'info': '#17a2b8',
                'background': '#ffffff',
                'text': '#212529',
            },
            'dark': {
                'primary': '#375a7f',
                'secondary': '#444444',
                'success': '#00bc8c',
                'danger': '#e74c3c',
                'warning': '#f39c12',
                'info': '#3498db',
                'background': '#222222',
                'text': '#ffffff',
            },
            'corporate': {
                'primary': '#003f5c',
                'secondary': '#58508d',
                'success': '#2f4b7c',
                'danger': '#bc5090',
                'warning': '#ff6361',
                'info': '#ffa600',
                'background': '#f8f9fa',
                'text': '#212529',
            },
        }

        return schemes.get(scheme, schemes['default'])

    def build(self, data: pd.DataFrame) -> 'DashboardTemplate':
        """
        Build dashboard from data.

        Args:
            data: Input DataFrame

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement build()")

    def render(self) -> Any:
        """
        Render dashboard.

        Returns:
            Dashboard object
        """
        raise NotImplementedError("Subclasses must implement render()")


class KPIDashboard(DashboardTemplate):
    """
    KPI Dashboard Template.

    Displays key performance indicators with trends and comparisons.

    Example:
        >>> template = KPIDashboard(TemplateConfig(name='Sales KPIs'))
        >>> dashboard = template.build(sales_df).render()
    """

    def build(self, data: pd.DataFrame) -> 'KPIDashboard':
        """
        Build KPI dashboard.

        Args:
            data: DataFrame with KPI metrics

        Returns:
            Self for method chaining
        """
        # Create 3x3 grid layout
        self.layout_engine = create_grid_layout(rows=3, cols=3)

        # Top row: Main KPIs (spans 3 columns)
        kpi_metrics = self._create_kpi_cards(data)
        for i, kpi in enumerate(kpi_metrics[:3]):
            self.layout_engine.add_component(kpi, row=1, col=i+1)

        # Middle row: Trend charts
        trend_chart = self._create_trend_chart(data)
        self.layout_engine.add_component(trend_chart, row=2, col=1, col_span=2)

        comparison_chart = self._create_comparison_chart(data)
        self.layout_engine.add_component(comparison_chart, row=2, col=3)

        # Bottom row: Detailed metrics
        detail_table = self._create_detail_table(data)
        self.layout_engine.add_component(detail_table, row=3, col=1, col_span=3)

        return self

    def _create_kpi_cards(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create KPI cards."""
        # Example KPI extraction (customize based on data)
        kpis = []

        numeric_cols = data.select_dtypes(include=['number']).columns

        for col in numeric_cols[:3]:
            kpi = {
                'title': col.replace('_', ' ').title(),
                'value': data[col].sum(),
                'change': self._calculate_change(data, col),
                'color': self.theme['primary'],
            }
            kpis.append(kpi)

        return kpis

    def _calculate_change(self, data: pd.DataFrame, column: str) -> float:
        """Calculate percentage change."""
        if len(data) < 2:
            return 0.0

        # Compare first half vs second half
        mid = len(data) // 2
        first_half = data[column].iloc[:mid].mean()
        second_half = data[column].iloc[mid:].mean()

        if first_half == 0:
            return 0.0

        return ((second_half - first_half) / first_half) * 100

    def _create_trend_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create trend line chart."""
        fig = go.Figure()

        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig.add_trace(go.Scatter(
                y=data[numeric_cols[0]],
                mode='lines+markers',
                name=numeric_cols[0],
                line=dict(color=self.theme['primary'], width=3)
            ))

        fig.update_layout(
            title='Trend Over Time',
            showlegend=True,
            height=300
        )

        return fig

    def _create_comparison_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create comparison bar chart."""
        fig = go.Figure()

        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            fig.add_trace(go.Bar(
                y=numeric_cols[:5],
                x=[data[col].sum() for col in numeric_cols[:5]],
                orientation='h',
                marker=dict(color=self.theme['secondary'])
            ))

        fig.update_layout(
            title='Metric Comparison',
            showlegend=False,
            height=300
        )

        return fig

    def _create_detail_table(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create detailed metrics table."""
        return {
            'type': 'table',
            'data': data.head(10),
        }

    def render(self) -> Dict[str, Any]:
        """Render KPI dashboard."""
        return {
            'template': 'kpi_dashboard',
            'config': self.config,
            'layout': self.layout_engine.generate_layout() if self.layout_engine else {},
            'theme': self.theme,
        }


class AnalyticsDashboard(DashboardTemplate):
    """
    Analytics Dashboard Template.

    Comprehensive analytics with multiple chart types and filters.

    Example:
        >>> template = AnalyticsDashboard(TemplateConfig(name='Sales Analytics'))
        >>> dashboard = template.build(sales_df).render()
    """

    def build(self, data: pd.DataFrame) -> 'AnalyticsDashboard':
        """Build analytics dashboard."""
        # Create sidebar layout
        self.layout_engine = create_sidebar_layout()

        # Sidebar: Filters and controls
        filters = self._create_filter_panel(data)
        self.layout_engine.add_component(filters, row=1, col=1)

        # Main content: Charts grid
        main_layout = create_grid_layout(rows=3, cols=2)

        # Add various chart types
        charts = [
            self._create_time_series(data),
            self._create_distribution(data),
            self._create_correlation(data),
            self._create_breakdown(data),
            self._create_heatmap(data),
            self._create_scatter(data),
        ]

        row, col = 1, 1
        for chart in charts:
            main_layout.add_component(chart, row=row, col=col)
            col += 1
            if col > 2:
                col = 1
                row += 1

        self.layout_engine.add_component(main_layout, row=1, col=2)

        return self

    def _create_filter_panel(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create filter panel."""
        return {
            'type': 'filter_panel',
            'filters': [
                {'type': 'date_range', 'label': 'Date Range'},
                {'type': 'multi_select', 'label': 'Categories'},
                {'type': 'slider', 'label': 'Value Range'},
            ]
        }

    def _create_time_series(self, data: pd.DataFrame) -> go.Figure:
        """Create time series chart."""
        fig = go.Figure()
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig.add_trace(go.Scatter(y=data[numeric_cols[0]], mode='lines'))
        fig.update_layout(title='Time Series', height=250)
        return fig

    def _create_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Create distribution chart."""
        fig = go.Figure()
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig.add_trace(go.Histogram(x=data[numeric_cols[0]]))
        fig.update_layout(title='Distribution', height=250)
        return fig

    def _create_correlation(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation matrix."""
        fig = go.Figure()
        numeric_data = data.select_dtypes(include=['number'])
        if len(numeric_data.columns) > 1:
            corr = numeric_data.corr()
            fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
        fig.update_layout(title='Correlation', height=250)
        return fig

    def _create_breakdown(self, data: pd.DataFrame) -> go.Figure:
        """Create breakdown chart."""
        fig = go.Figure()
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig.add_trace(go.Bar(y=data[numeric_cols[0]].value_counts().values))
        fig.update_layout(title='Breakdown', height=250)
        return fig

    def _create_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create heatmap."""
        fig = go.Figure()
        numeric_data = data.select_dtypes(include=['number']).iloc[:20, :10]
        if not numeric_data.empty:
            fig.add_trace(go.Heatmap(z=numeric_data.values))
        fig.update_layout(title='Heatmap', height=250)
        return fig

    def _create_scatter(self, data: pd.DataFrame) -> go.Figure:
        """Create scatter plot."""
        fig = go.Figure()
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            fig.add_trace(go.Scatter(
                x=data[numeric_cols[0]],
                y=data[numeric_cols[1]],
                mode='markers'
            ))
        fig.update_layout(title='Scatter Plot', height=250)
        return fig

    def render(self) -> Dict[str, Any]:
        """Render analytics dashboard."""
        return {
            'template': 'analytics_dashboard',
            'config': self.config,
            'layout': self.layout_engine.generate_layout() if self.layout_engine else {},
            'theme': self.theme,
        }


class ExecutiveDashboard(DashboardTemplate):
    """
    Executive Dashboard Template.

    High-level overview for executives with KPIs and summaries.

    Example:
        >>> template = ExecutiveDashboard(TemplateConfig(name='Executive Summary'))
        >>> dashboard = template.build(data_df).render()
    """

    def build(self, data: pd.DataFrame) -> 'ExecutiveDashboard':
        """Build executive dashboard."""
        self.layout_engine = create_grid_layout(rows=4, cols=4)

        # Top row: 4 key KPIs
        kpis = self._create_executive_kpis(data)
        for i, kpi in enumerate(kpis[:4]):
            self.layout_engine.add_component(kpi, row=1, col=i+1)

        # Second row: Main chart (spans 3 cols) + summary
        main_chart = self._create_main_chart(data)
        self.layout_engine.add_component(main_chart, row=2, col=1, col_span=3)

        summary = self._create_summary_card(data)
        self.layout_engine.add_component(summary, row=2, col=4)

        # Third row: Regional breakdown
        regional_chart = self._create_regional_chart(data)
        self.layout_engine.add_component(regional_chart, row=3, col=1, col_span=4)

        # Fourth row: Top performers
        top_performers = self._create_top_performers(data)
        self.layout_engine.add_component(top_performers, row=4, col=1, col_span=4)

        return self

    def _create_executive_kpis(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create executive KPIs."""
        kpis = []
        numeric_cols = data.select_dtypes(include=['number']).columns

        for col in numeric_cols[:4]:
            kpi = {
                'title': col.replace('_', ' ').title(),
                'value': f"${data[col].sum():,.0f}",
                'change': f"+{self._calculate_change(data, col):.1f}%",
                'trend': 'up' if self._calculate_change(data, col) > 0 else 'down',
            }
            kpis.append(kpi)

        return kpis

    def _calculate_change(self, data: pd.DataFrame, column: str) -> float:
        """Calculate percentage change."""
        if len(data) < 2:
            return 0.0
        mid = len(data) // 2
        first = data[column].iloc[:mid].mean()
        second = data[column].iloc[mid:].mean()
        return ((second - first) / first * 100) if first != 0 else 0.0

    def _create_main_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create main executive chart."""
        fig = go.Figure()
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig.add_trace(go.Scatter(
                y=data[numeric_cols[0]],
                mode='lines',
                fill='tozeroy',
                line=dict(color=self.theme['primary'], width=3)
            ))
        fig.update_layout(title='Revenue Trend', height=300, showlegend=False)
        return fig

    def _create_summary_card(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create executive summary card."""
        return {
            'type': 'summary',
            'total_records': len(data),
            'highlights': [
                'Strong Q4 performance',
                'Revenue up 15%',
                'New markets opened',
            ]
        }

    def _create_regional_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create regional breakdown chart."""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Region 1', 'Region 2', 'Region 3', 'Region 4'],
            y=[100, 150, 120, 180],
            marker=dict(color=self.theme['secondary'])
        ))
        fig.update_layout(title='Regional Performance', height=250)
        return fig

    def _create_top_performers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create top performers table."""
        return {
            'type': 'table',
            'title': 'Top Performers',
            'data': data.head(5),
        }

    def render(self) -> Dict[str, Any]:
        """Render executive dashboard."""
        return {
            'template': 'executive_dashboard',
            'config': self.config,
            'layout': self.layout_engine.generate_layout() if self.layout_engine else {},
            'theme': self.theme,
        }


# ==================== Template Factory ====================

def create_template(
    template_type: str,
    name: str = "Dashboard",
    color_scheme: str = "default"
) -> DashboardTemplate:
    """
    Create dashboard template by type.

    Args:
        template_type: Template type ('kpi', 'analytics', 'executive')
        name: Dashboard name
        color_scheme: Color scheme ('default', 'dark', 'corporate')

    Returns:
        Dashboard template instance

    Example:
        >>> template = create_template('kpi', name='Sales KPIs', color_scheme='dark')
        >>> dashboard = template.build(df).render()
    """
    config = TemplateConfig(
        name=name,
        color_scheme=color_scheme
    )

    templates = {
        'kpi': KPIDashboard,
        'analytics': AnalyticsDashboard,
        'executive': ExecutiveDashboard,
    }

    template_class = templates.get(template_type.lower())
    if not template_class:
        available = ', '.join(templates.keys())
        raise ValueError(f"Unknown template type '{template_type}'. Available: {available}")

    return template_class(config)
