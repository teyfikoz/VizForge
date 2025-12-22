"""Dashboard builder for VizForge."""

from typing import List, Dict, Optional, Union, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ..core.base import BaseChart
from ..core.theme import Theme, get_theme


class DashboardLayout:
    """Layout configurations for dashboards."""

    GRID = "grid"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    CUSTOM = "custom"


class Dashboard:
    """
    Interactive dashboard builder.

    Combine multiple charts, KPIs, and filters into a single dashboard.
    Perfect for executive reports, analytics dashboards, monitoring.

    Examples:
        >>> import vizforge as vz
        >>>
        >>> # Create dashboard
        >>> dashboard = vz.Dashboard(title="Sales Dashboard", rows=2, cols=2)
        >>>
        >>> # Add charts
        >>> dashboard.add_chart(chart1, row=1, col=1)
        >>> dashboard.add_chart(chart2, row=1, col=2)
        >>> dashboard.add_kpi("Revenue", "$1.2M", row=2, col=1)
        >>>
        >>> # Display
        >>> dashboard.show()
    """

    def __init__(
        self,
        title: str = "Dashboard",
        rows: int = 2,
        cols: int = 2,
        theme: Optional[str] = None,
        layout: str = DashboardLayout.GRID,
        height: int = 800,
        width: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Dashboard.

        Args:
            title: Dashboard title
            rows: Number of rows
            cols: Number of columns
            theme: Theme name
            layout: Layout type
            height: Dashboard height
            width: Dashboard width
            **kwargs: Additional arguments
        """
        self.title = title
        self.rows = rows
        self.cols = cols
        self.theme = get_theme(theme)
        self.layout_type = layout
        self.height = height
        self.width = width
        self.kwargs = kwargs

        # Components storage
        self.components = []
        self.kpis = []
        self.filters = []

        # NEW v1.0.0: Interactive features
        self.widgets = {}  # Widget storage
        self.charts = []  # Chart storage for callbacks
        self.callback_manager = None  # Initialized on first callback
        self._session_state = None  # Initialized on first use

        # Create subplot figure
        self._create_subplots()

    def _create_subplots(self):
        """Create subplot structure."""
        self.fig = make_subplots(
            rows=self.rows,
            cols=self.cols,
            subplot_titles=[None] * (self.rows * self.cols),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # Apply theme
        self._apply_theme()

    def _apply_theme(self):
        """Apply theme to dashboard."""
        self.fig.update_layout(
            title=self.title,
            showlegend=True,
            height=self.height,
            width=self.width,
            template=self.theme.plotly_template,
            font=dict(
                family=self.theme.font_family,
                size=self.theme.font_size,
                color=self.theme.text_color
            ),
            paper_bgcolor=self.theme.background_color,
            plot_bgcolor=self.theme.background_color
        )

    def add_chart(
        self,
        chart: Union[BaseChart, go.Figure],
        row: int,
        col: int,
        title: Optional[str] = None
    ):
        """
        Add chart to dashboard.

        Args:
            chart: VizForge chart or Plotly figure
            row: Row position (1-indexed)
            col: Column position (1-indexed)
            title: Subplot title
        """
        # Get figure from chart
        if isinstance(chart, BaseChart):
            fig = chart.create_figure()
        else:
            fig = chart

        # Add traces to subplot
        for trace in fig.data:
            self.fig.add_trace(trace, row=row, col=col)

        # Update subplot title
        if title:
            subplot_idx = (row - 1) * self.cols + (col - 1)
            self.fig.layout.annotations[subplot_idx].text = title

        self.components.append({
            'type': 'chart',
            'row': row,
            'col': col,
            'title': title
        })

    def add_kpi(
        self,
        label: str,
        value: Union[str, float, int],
        row: int,
        col: int,
        delta: Optional[Union[str, float]] = None,
        delta_color: str = "green"
    ):
        """
        Add KPI card to dashboard.

        Args:
            label: KPI label
            value: KPI value
            row: Row position
            col: Column position
            delta: Change indicator
            delta_color: Delta color
        """
        # Create KPI indicator
        indicator = go.Indicator(
            mode="number+delta" if delta else "number",
            value=float(value) if isinstance(value, (int, float)) else 0,
            title={'text': label},
            delta={'reference': delta} if delta else None,
            domain={'x': [0, 1], 'y': [0, 1]}
        )

        self.fig.add_trace(indicator, row=row, col=col)

        self.kpis.append({
            'label': label,
            'value': value,
            'row': row,
            'col': col
        })

    def add_text(
        self,
        text: str,
        row: int,
        col: int,
        font_size: int = 14
    ):
        """
        Add text annotation to dashboard.

        Args:
            text: Text content
            row: Row position
            col: Column position
            font_size: Font size
        """
        # Calculate position
        x_pos = (col - 0.5) / self.cols
        y_pos = 1 - (row - 0.5) / self.rows

        self.fig.add_annotation(
            text=text,
            x=x_pos,
            y=y_pos,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=font_size)
        )

    def set_column_widths(self, widths: List[float]):
        """
        Set custom column widths.

        Args:
            widths: List of width ratios (must sum to 1)
        """
        if len(widths) != self.cols:
            raise ValueError(f"Must provide {self.cols} width values")
        if abs(sum(widths) - 1.0) > 0.01:
            raise ValueError("Widths must sum to 1.0")

        self.fig.update_layout(
            grid={'columns': self.cols, 'rows': self.rows}
        )

    def set_row_heights(self, heights: List[float]):
        """
        Set custom row heights.

        Args:
            heights: List of height ratios (must sum to 1)
        """
        if len(heights) != self.rows:
            raise ValueError(f"Must provide {self.rows} height values")
        if abs(sum(heights) - 1.0) > 0.01:
            raise ValueError("Heights must sum to 1.0")

    def show(self):
        """Display dashboard."""
        self.fig.show()

    def export(self, filepath: str, **kwargs):
        """
        Export dashboard to file.

        Args:
            filepath: Output file path
            **kwargs: Additional export arguments
        """
        from ..core.export import export_chart
        export_chart(self.fig, filepath, **kwargs)

    def to_html(self, filepath: str):
        """Export dashboard as HTML."""
        self.fig.write_html(filepath)

    # ==================== VizForge v1.0.0 NEW METHODS ====================

    def add_widget(
        self,
        widget: 'Widget',
        row: int,
        col: int
    ) -> 'Dashboard':
        """
        Add interactive widget to dashboard.

        NEW in v1.0.0: Adds Streamlit-style widgets for interactivity.

        Args:
            widget: Widget instance (Slider, SelectBox, etc.)
            row: Row position
            col: Column position

        Returns:
            Self for method chaining

        Example:
            >>> from vizforge.interactive import Slider
            >>> dashboard = Dashboard(rows=2, cols=2)
            >>> slider = Slider('year', 'Select Year', 2020, 2024, default=2023)
            >>> dashboard.add_widget(slider, row=1, col=1)
        """
        # Store widget
        self.widgets[widget.id] = widget

        # Initialize callback manager if needed
        if self.callback_manager is None:
            from ..interactive.callbacks import CallbackManager
            self.callback_manager = CallbackManager()

        # Register widget with callback manager
        self.callback_manager.register_component(widget.id, widget)

        return self

    def callback(
        self,
        outputs: Union[str, List[str]],
        inputs: Union[str, List[str]],
        state: Union[str, List[str], None] = None
    ):
        """
        Dash-style callback decorator.

        NEW in v1.0.0: Register reactive callbacks between widgets and charts.

        Args:
            outputs: Output component ID(s)
            inputs: Input component ID(s) that trigger callback
            state: State component ID(s) (read-only)

        Returns:
            Decorated function

        Example:
            >>> @dashboard.callback(
            ...     outputs='sales_chart',
            ...     inputs=['year_slider', 'category_select']
            ... )
            ... def update_sales(year, category):
            ...     filtered_data = df[(df['year'] == year) & (df['category'] == category)]
            ...     return LineChart(filtered_data, x='month', y='sales')
        """
        # Initialize callback manager if needed
        if self.callback_manager is None:
            from ..interactive.callbacks import CallbackManager
            self.callback_manager = CallbackManager()

        # Use callback manager's decorator
        return self.callback_manager.callback(outputs, inputs, state)

    def serve(
        self,
        port: int = 8050,
        debug: bool = False,
        host: str = '127.0.0.1'
    ):
        """
        Launch interactive dashboard server.

        NEW in v1.0.0: Starts Dash server for full interactivity.

        Args:
            port: Server port (default: 8050)
            debug: Debug mode with hot reload
            host: Server host (default: 127.0.0.1)

        Example:
            >>> dashboard = Dashboard(rows=2, cols=2)
            >>> # ... add widgets and charts ...
            >>> dashboard.serve(port=8050, debug=True)
            Dash is running on http://127.0.0.1:8050/
        """
        from .builder import DashboardServer

        # Create and run server
        server = DashboardServer(self)
        server.run(host=host, port=port, debug=debug)

    def get_session_state(self):
        """
        Get session state for dashboard.

        NEW in v1.0.0: Access Streamlit-style session state.

        Returns:
            SessionState instance

        Example:
            >>> state = dashboard.get_session_state()
            >>> state['user_selection'] = 'Electronics'
            >>> print(state['user_selection'])
        """
        if self._session_state is None:
            from ..interactive.state import get_session_state
            self._session_state = get_session_state()

        return self._session_state

    def add_filter(
        self,
        filter: 'Filter',
        apply_to: Optional[List[str]] = None
    ) -> 'Dashboard':
        """
        Add filter to dashboard.

        NEW in v1.0.0: Tableau-style filtering with cascading support.

        Args:
            filter: Filter instance (RangeFilter, ListFilter, etc.)
            apply_to: Chart IDs to apply filter to (None = all charts)

        Returns:
            Self for method chaining

        Example:
            >>> from vizforge.interactive import RangeFilter
            >>> price_filter = RangeFilter('price', 'price', 100, 500)
            >>> dashboard.add_filter(price_filter)
        """
        self.filters.append({
            'filter': filter,
            'apply_to': apply_to
        })

        return self

    def add_action(
        self,
        action: 'Action'
    ) -> 'Dashboard':
        """
        Add action to dashboard.

        NEW in v1.0.0: Tableau-style actions (filter, highlight, drill-down).

        Args:
            action: Action instance (FilterAction, HighlightAction, etc.)

        Returns:
            Self for method chaining

        Example:
            >>> from vizforge.interactive import FilterAction
            >>> action = FilterAction(
            ...     action_id='filter_by_region',
            ...     source_chart='map',
            ...     target_charts=['sales', 'profit'],
            ...     filter_column='region'
            ... )
            >>> dashboard.add_action(action)
        """
        if not hasattr(self, 'actions'):
            self.actions = []

        self.actions.append(action)

        return self

    def enable_smart_mode(self) -> 'Dashboard':
        """
        Enable intelligent recommendations for dashboard.

        NEW in v1.0.0: Activates automatic chart selection,
        data quality warnings, and best practice suggestions.

        Returns:
            Self for method chaining

        Example:
            >>> dashboard = Dashboard(rows=2, cols=2)
            >>> dashboard.enable_smart_mode()
        """
        self._smart_mode = True
        return self


def create_dashboard(
    title: str = "Dashboard",
    rows: int = 2,
    cols: int = 2,
    theme: Optional[str] = None,
    **kwargs
) -> Dashboard:
    """
    Create a new dashboard.

    Args:
        title: Dashboard title
        rows: Number of rows
        cols: Number of columns
        theme: Theme name
        **kwargs: Additional arguments

    Returns:
        Dashboard instance

    Examples:
        >>> import vizforge as vz
        >>>
        >>> # Create 2x2 dashboard
        >>> dashboard = vz.create_dashboard("Sales Dashboard", rows=2, cols=2)
        >>>
        >>> # Add components
        >>> dashboard.add_chart(line_chart, row=1, col=1, title="Sales Trend")
        >>> dashboard.add_chart(bar_chart, row=1, col=2, title="By Category")
        >>> dashboard.add_kpi("Total Revenue", "$1.2M", row=2, col=1, delta="+15%")
        >>> dashboard.add_kpi("Total Orders", "5,432", row=2, col=2, delta="+8%")
        >>>
        >>> # Show
        >>> dashboard.show()
    """
    return Dashboard(title=title, rows=rows, cols=cols, theme=theme, **kwargs)
