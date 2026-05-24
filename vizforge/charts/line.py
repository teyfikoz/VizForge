"""Line chart implementation for VizForge."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.base import BaseChart
from ..core.theme import Theme


class LineChart(BaseChart):
    """
    Line chart visualization.

    Supports single and multi-line charts, area charts, and step charts.
    """

    def __init__(
        self,
        data: pd.DataFrame | dict | list | None = None,
        x: str | list | np.ndarray | None = None,
        y: str | list | np.ndarray | None = None,
        title: str | None = None,
        theme: str | Theme | None = None,
        mode: str = "lines",
        fill: str | None = None,
        **kwargs
    ):
        """
        Create a line chart.

        Args:
            data: Data source (DataFrame, dict, or list)
            x: X-axis data or column name
            y: Y-axis data or column name (can be list of columns for multi-line)
            title: Chart title
            theme: Theme name or Theme object
            mode: Display mode ('lines', 'markers', 'lines+markers')
            fill: Fill area ('none', 'tozeroy', 'tonexty')
            **kwargs: Additional arguments

        Example:
            >>> line = LineChart(data=df, x='date', y='sales', title='Sales Trend')
            >>> line.show()
        """
        super().__init__(title=title, theme=theme, **kwargs)
        self.mode = mode
        self.fill = fill

        if data is not None:
            self.plot(data, x, y)

    def plot(
        self,
        data: pd.DataFrame | dict | list,
        x: str | list | np.ndarray | None = None,
        y: str | list | np.ndarray | None = None,
        name: str | None = None,
        **kwargs
    ) -> 'LineChart':
        """
        Plot data on the chart.

        Args:
            data: Data source
            x: X-axis data or column name
            y: Y-axis data or column name
            name: Series name
            **kwargs: Additional trace arguments

        Returns:
            Self for method chaining
        """
        # Create figure if not exists
        if self.fig is None:
            self.fig = self._create_figure()

        # Parse data
        x_data, y_data_list, names = self._parse_data(data, x, y, name)

        # Add traces
        for y_data, trace_name in zip(y_data_list, names):
            trace_kwargs = {
                'mode': self.mode,
                'line': {'width': self._theme.line_width},
                'marker': {'size': self._theme.marker_size},
            }

            if self.fill:
                trace_kwargs['fill'] = self.fill

            trace_kwargs.update(kwargs)

            self.fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    name=trace_name,
                    **trace_kwargs
                )
            )

        return self

    def _parse_data(
        self,
        data: pd.DataFrame | dict | list,
        x: str | list | np.ndarray | None,
        y: str | list | np.ndarray | None,
        name: str | None
    ) -> tuple[Any, list[Any], list[str]]:
        """Parse different data formats into x, y, and names."""

        # Handle DataFrame
        if isinstance(data, pd.DataFrame):
            if isinstance(x, str):
                x_data = data[x].values
            elif x is None:
                x_data = data.index.values
            else:
                x_data = x

            if isinstance(y, str):
                y_data_list = [data[y].values]
                names = [name or y]
            elif isinstance(y, list) and all(isinstance(col, str) for col in y):
                # Multiple y columns
                y_data_list = [data[col].values for col in y]
                names = y if name is None else [f"{name} - {col}" for col in y]
            else:
                y_data_list = [y]
                names = [name or "trace"]

            return x_data, y_data_list, names

        # Handle dict
        elif isinstance(data, dict):
            if x is None:
                x_data = list(range(len(next(iter(data.values())))))
            elif isinstance(x, str):
                x_data = data[x]
            else:
                x_data = x

            if isinstance(y, str):
                y_data_list = [data[y]]
                names = [name or y]
            elif isinstance(y, list) and all(isinstance(col, str) for col in y):
                y_data_list = [data[col] for col in y]
                names = y if name is None else [f"{name} - {col}" for col in y]
            else:
                y_data_list = [y]
                names = [name or "trace"]

            return x_data, y_data_list, names

        # Handle arrays/lists
        else:
            x_data = x if x is not None else list(range(len(y)))
            y_data_list = [y]
            names = [name or "trace"]
            return x_data, y_data_list, names


def line(
    data: pd.DataFrame | dict | list,
    x: str | list | np.ndarray | None = None,
    y: str | list | np.ndarray | None = None,
    title: str | None = None,
    theme: str | Theme | None = None,
    mode: str = "lines",
    fill: str | None = None,
    show: bool = False,
    export: str | None = None,
    **kwargs
) -> LineChart:
    """
    Create and display a line chart (convenience function).

    Args:
        data: Data source (DataFrame, dict, or list)
        x: X-axis data or column name
        y: Y-axis data or column name
        title: Chart title
        theme: Theme name or Theme object
        mode: Display mode ('lines', 'markers', 'lines+markers')
        fill: Fill area ('none', 'tozeroy', 'tonexty')
        show: Whether to display the chart
        export: Export filename (optional)
        **kwargs: Additional arguments

    Returns:
        LineChart object

    Example:
        >>> line(df, x='date', y='sales', title='Sales Trend')
        >>> line(df, x='date', y=['sales', 'profit'], theme='dark')
    """
    chart = LineChart(
        data=data,
        x=x,
        y=y,
        title=title,
        theme=theme,
        mode=mode,
        fill=fill,
        **kwargs
    )

    if export:
        chart.export(export)

    if show:
        chart.show()

    return chart
