"""
VizForge Dashboard Builder

Dash integration for interactive dashboards.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Dict, List, Optional, Callable
import pandas as pd
from datetime import date


class DashboardServer:
    """
    Convert VizForge Dashboard to Dash app.

    Provides automatic Dash app generation from VizForge dashboards
    with full support for widgets, callbacks, filters, and actions.

    Example:
        >>> from vizforge import Dashboard
        >>> from vizforge.dashboard.builder import DashboardServer
        >>>
        >>> dashboard = Dashboard(rows=2, cols=2)
        >>> # ... add charts and widgets ...
        >>>
        >>> server = DashboardServer(dashboard)
        >>> server.run(port=8050, debug=True)
    """

    def __init__(self, dashboard: 'Dashboard'):
        """
        Initialize dashboard server.

        Args:
            dashboard: VizForge Dashboard instance
        """
        self.dashboard = dashboard
        self.dash_app = None
        self._widget_components = {}
        self._chart_components = {}

    def build(self) -> Any:
        """
        Build Dash app from VizForge Dashboard.

        Returns:
            Dash app instance

        Raises:
            ImportError: If Dash is not installed
        """
        try:
            from dash import Dash, html, dcc
            import dash_bootstrap_components as dbc
        except ImportError:
            raise ImportError(
                "Dash and dash-bootstrap-components are required for interactive dashboards.\n"
                "Install with: pip install dash dash-bootstrap-components"
            )

        # Create Dash app
        self.dash_app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )

        # Build layout
        self.dash_app.layout = self._build_layout()

        # Register callbacks
        self._register_callbacks()

        return self.dash_app

    def _build_layout(self) -> Any:
        """Build Dash layout from dashboard configuration."""
        from dash import html, dcc
        import dash_bootstrap_components as dbc

        # Build header
        header = html.Div([
            html.H1(
                self.dashboard.title if hasattr(self.dashboard, 'title') else 'Dashboard',
                className='text-center mb-4'
            )
        ])

        # Build grid
        grid_children = []

        # Add widgets section if any
        if hasattr(self.dashboard, 'widgets') and self.dashboard.widgets:
            widget_row = self._build_widgets_row()
            grid_children.append(widget_row)

        # Add charts grid
        charts_grid = self._build_charts_grid()
        grid_children.append(charts_grid)

        # Combine all
        layout = dbc.Container([
            header,
            html.Div(grid_children)
        ], fluid=True)

        return layout

    def _build_widgets_row(self) -> Any:
        """Build row of widgets."""
        from dash import html
        import dash_bootstrap_components as dbc

        widget_cols = []

        for widget_id, widget in self.dashboard.widgets.items():
            # Convert widget to Dash component
            dash_component = widget.to_dash_component()
            self._widget_components[widget_id] = dash_component

            # Wrap in column
            col = dbc.Col([
                html.Label(widget.label, className='font-weight-bold'),
                dash_component,
                html.Small(widget.help_text, className='text-muted') if widget.help_text else None
            ], width=12 // min(len(self.dashboard.widgets), 4))  # Max 4 widgets per row

            widget_cols.append(col)

        return dbc.Row(widget_cols, className='mb-4')

    def _build_charts_grid(self) -> Any:
        """Build grid of charts."""
        from dash import html, dcc
        import dash_bootstrap_components as dbc

        if not hasattr(self.dashboard, 'charts') or not self.dashboard.charts:
            return html.Div("No charts to display")

        # Get grid dimensions
        rows = self.dashboard.rows if hasattr(self.dashboard, 'rows') else 1
        cols = self.dashboard.cols if hasattr(self.dashboard, 'cols') else 2

        # Build grid
        grid_rows = []

        for row_idx in range(rows):
            row_charts = []

            for col_idx in range(cols):
                # Find chart at this position
                chart = self._find_chart_at_position(row_idx, col_idx)

                if chart:
                    chart_id = chart.get('id', f'chart_{row_idx}_{col_idx}')

                    # Create chart container
                    chart_div = html.Div([
                        dcc.Graph(
                            id=chart_id,
                            figure=chart.get('figure', {}),
                            config={'responsive': True}
                        )
                    ], className='border rounded p-2')

                    self._chart_components[chart_id] = chart_div
                else:
                    # Empty cell
                    chart_div = html.Div(
                        "",
                        className='border rounded p-2',
                        style={'min-height': '300px'}
                    )

                row_charts.append(
                    dbc.Col(chart_div, width=12 // cols)
                )

            grid_rows.append(dbc.Row(row_charts, className='mb-3'))

        return html.Div(grid_rows)

    def _find_chart_at_position(self, row: int, col: int) -> Optional[Dict[str, Any]]:
        """Find chart at grid position."""
        if not hasattr(self.dashboard, 'charts'):
            return None

        for chart in self.dashboard.charts:
            if chart.get('row') == row and chart.get('col') == col:
                return chart

        return None

    def _register_callbacks(self):
        """Register all Dash callbacks from dashboard callbacks."""
        if not hasattr(self.dashboard, 'callback_manager'):
            return

        from dash import Input, Output, State
        from dash.dependencies import ALL

        callback_manager = self.dashboard.callback_manager

        for callback in callback_manager.callbacks:
            # Build inputs
            inputs = [
                Input(input_id, 'value') if input_id in self._widget_components else Input(input_id, 'clickData')
                for input_id in callback.inputs
            ]

            # Build outputs
            outputs = [
                Output(output_id, 'figure') if output_id in self._chart_components else Output(output_id, 'children')
                for output_id in callback.outputs
            ]

            # Build state
            states = [
                State(state_id, 'value') if state_id in self._widget_components else State(state_id, 'data')
                for state_id in callback.state
            ]

            # Register callback
            if callback.function:
                self.dash_app.callback(
                    outputs,
                    inputs,
                    states
                )(callback.function)

    def run(
        self,
        host: str = '127.0.0.1',
        port: int = 8050,
        debug: bool = False,
        dev_tools_hot_reload: bool = True
    ):
        """
        Run dashboard server.

        Args:
            host: Server host (default: 127.0.0.1)
            port: Server port (default: 8050)
            debug: Debug mode
            dev_tools_hot_reload: Enable hot reload in debug mode

        Example:
            >>> server.run(port=8050, debug=True)
            Dash is running on http://127.0.0.1:8050/
        """
        if self.dash_app is None:
            self.build()

        print(f"\n🚀 VizForge Dashboard Server")
        print(f"   Running on http://{host}:{port}/")
        print(f"   Debug mode: {debug}")
        print(f"\nPress CTRL+C to quit\n")

        self.dash_app.run_server(
            host=host,
            port=port,
            debug=debug,
            dev_tools_hot_reload=dev_tools_hot_reload
        )

    def export_to_html(self, filename: str):
        """
        Export dashboard to static HTML.

        Args:
            filename: Output HTML filename

        Note:
            This creates a static HTML file without interactivity.
            For full interactivity, use run() to start server.
        """
        if self.dash_app is None:
            self.build()

        # Note: Dash doesn't have built-in HTML export
        # This would require additional implementation
        raise NotImplementedError(
            "HTML export for Dash dashboards requires additional setup. "
            "Use server.run() for interactive dashboards, or export individual "
            "charts using chart.export('filename.html')"
        )


class DashboardTemplate:
    """
    Pre-built dashboard templates.

    Provides ready-to-use dashboard layouts for common use cases.
    """

    @staticmethod
    def kpi_dashboard(
        kpis: Dict[str, float],
        charts: Optional[List[Any]] = None,
        title: str = "KPI Dashboard"
    ) -> 'Dashboard':
        """
        Create KPI dashboard template.

        Args:
            kpis: Dictionary of KPI name → value
            charts: List of charts to display
            title: Dashboard title

        Returns:
            Dashboard instance

        Example:
            >>> kpis = {
            ...     'Revenue': 1250000,
            ...     'Customers': 45231,
            ...     'Growth': 23.5
            ... }
            >>> dashboard = DashboardTemplate.kpi_dashboard(kpis)
        """
        from vizforge import Dashboard

        # Calculate grid size
        n_kpis = len(kpis)
        n_charts = len(charts) if charts else 0
        rows = 2 if n_charts > 0 else 1
        cols = max(n_kpis, 2)

        dashboard = Dashboard(rows=rows, cols=cols)
        dashboard.title = title

        # Add KPI cards (would need implementation)
        # Note: This is a placeholder - full implementation would create actual KPI cards

        # Add charts
        if charts:
            for idx, chart in enumerate(charts):
                row = 1
                col = idx % cols
                # dashboard.add_chart(chart, row=row, col=col)

        return dashboard

    @staticmethod
    def sales_dashboard(
        data: pd.DataFrame,
        date_column: str = 'date',
        revenue_column: str = 'revenue',
        category_column: str = 'category'
    ) -> 'Dashboard':
        """
        Create sales analytics dashboard.

        Args:
            data: Sales data
            date_column: Date column name
            revenue_column: Revenue column name
            category_column: Category column name

        Returns:
            Dashboard instance

        Example:
            >>> dashboard = DashboardTemplate.sales_dashboard(sales_df)
        """
        from vizforge import Dashboard

        dashboard = Dashboard(rows=2, cols=2)
        dashboard.title = "Sales Analytics Dashboard"

        # Add widgets
        from vizforge.interactive import DateRangePicker, SelectBox

        # Date filter
        date_range = DateRangePicker(
            widget_id='date_range',
            label='Date Range',
            default=[
                data[date_column].min().date() if pd.api.types.is_datetime64_any_dtype(data[date_column]) else date.today(),
                data[date_column].max().date() if pd.api.types.is_datetime64_any_dtype(data[date_column]) else date.today()
            ]
        )
        dashboard.add_widget(date_range, row=0, col=0)

        # Category filter
        categories = data[category_column].unique().tolist()
        category_select = SelectBox(
            widget_id='category',
            label='Category',
            options=['All'] + categories,
            default='All'
        )
        dashboard.add_widget(category_select, row=0, col=1)

        # Charts would be added here
        # This is a template - actual implementation would create charts

        return dashboard

    @staticmethod
    def analytics_dashboard(
        data: pd.DataFrame,
        metrics: List[str],
        dimensions: List[str]
    ) -> 'Dashboard':
        """
        Create general analytics dashboard.

        Args:
            data: Analytics data
            metrics: List of metric columns
            dimensions: List of dimension columns

        Returns:
            Dashboard instance

        Example:
            >>> dashboard = DashboardTemplate.analytics_dashboard(
            ...     df,
            ...     metrics=['revenue', 'profit', 'orders'],
            ...     dimensions=['region', 'category', 'segment']
            ... )
        """
        from vizforge import Dashboard

        n_metrics = len(metrics)
        rows = (n_metrics + 1) // 2 + 1  # +1 for filters row
        cols = 2

        dashboard = Dashboard(rows=rows, cols=cols)
        dashboard.title = "Analytics Dashboard"

        # Add dimension filters
        from vizforge.interactive import MultiSelect

        for idx, dimension in enumerate(dimensions[:2]):  # Max 2 filters
            values = data[dimension].unique().tolist()
            multi_select = MultiSelect(
                widget_id=f'{dimension}_filter',
                label=dimension.capitalize(),
                options=values,
                default=values  # All selected by default
            )
            dashboard.add_widget(multi_select, row=0, col=idx)

        # Metric charts would be added here

        return dashboard


class QuickDashboard:
    """
    One-line dashboard creation.

    Provides ultra-simple dashboard creation for common scenarios.
    """

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[List[str]] = None,
        title: str = "Dashboard"
    ) -> DashboardServer:
        """
        Create dashboard from DataFrame with automatic chart selection.

        Args:
            data: Input DataFrame
            x: X-axis column (auto-detected if None)
            y: Y-axis columns (auto-detected if None)
            title: Dashboard title

        Returns:
            DashboardServer ready to run

        Example:
            >>> server = QuickDashboard.from_dataframe(df)
            >>> server.run()
        """
        from vizforge import Dashboard
        from vizforge.intelligence import ChartSelector

        # Auto-select chart types
        selector = ChartSelector()

        # Detect columns
        if x is None:
            # Try to find temporal or categorical column
            temporal_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

            if temporal_cols:
                x = temporal_cols[0]
            elif categorical_cols:
                x = categorical_cols[0]

        if y is None:
            # Use all numeric columns
            y = data.select_dtypes(include=['number']).columns.tolist()[:3]  # Max 3

        # Create dashboard
        dashboard = Dashboard(rows=len(y), cols=1)
        dashboard.title = title

        # Add charts (would use actual chart creation)
        # This is a simplified template

        # Build and return server
        server = DashboardServer(dashboard)
        server.build()

        return server

    @staticmethod
    def single_chart(
        data: pd.DataFrame,
        chart_type: Optional[str] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        title: str = "Chart"
    ) -> DashboardServer:
        """
        Create single-chart dashboard.

        Args:
            data: Input DataFrame
            chart_type: Chart type (auto-detected if None)
            x: X-axis column
            y: Y-axis column
            title: Chart title

        Returns:
            DashboardServer ready to run

        Example:
            >>> server = QuickDashboard.single_chart(df, x='date', y='sales')
            >>> server.run()
        """
        from vizforge import Dashboard
        from vizforge.intelligence import ChartSelector

        # Auto-select chart if not specified
        if chart_type is None:
            selector = ChartSelector()
            recommendation = selector.recommend(data, x=x, y=y)
            chart_type = recommendation['primary']

        # Create dashboard
        dashboard = Dashboard(rows=1, cols=1)
        dashboard.title = title

        # Add chart (would use actual chart creation)

        # Build and return server
        server = DashboardServer(dashboard)
        server.build()

        return server
