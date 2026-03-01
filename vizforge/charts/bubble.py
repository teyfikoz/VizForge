"""Bubble chart implementation for VizForge."""


import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.base import BaseChart
from ..core.theme import Theme


class BubbleChart(BaseChart):
    """
    Bubble chart visualization.

    Scatter plot with size encoding (3 variables: x, y, size).
    """

    def __init__(
        self,
        data: pd.DataFrame | dict | None = None,
        x: str | list | np.ndarray | None = None,
        y: str | list | np.ndarray | None = None,
        size: str | list | np.ndarray | None = None,
        color: str | list | np.ndarray | None = None,
        text: str | list | np.ndarray | None = None,
        title: str | None = None,
        theme: str | Theme | None = None,
        **kwargs
    ):
        """
        Create a bubble chart.

        Args:
            data: Data source
            x: X-axis data or column name
            y: Y-axis data or column name
            size: Bubble size data or column name
            color: Color data or column name
            text: Hover text data or column name
            title: Chart title
            theme: Theme
            **kwargs: Additional arguments
        """
        super().__init__(title=title, theme=theme, **kwargs)
        self.size_data = size
        self.color_data = color
        self.text_data = text

        if data is not None:
            self.plot(data, x, y)

    def plot(
        self,
        data: pd.DataFrame | dict,
        x: str | list | np.ndarray | None = None,
        y: str | list | np.ndarray | None = None,
        name: str | None = None,
        **kwargs
    ) -> 'BubbleChart':
        """Plot bubble chart data."""
        if self.fig is None:
            self.fig = self._create_figure()

        # Parse data
        x_data, y_data, size, color, text = self._parse_data(data, x, y)

        # Prepare marker configuration
        marker = {
            'size': size if size is not None else 20,
            'sizemode': 'diameter',
        }

        if size is not None and isinstance(size, (list, np.ndarray)):
            marker['sizeref'] = 2. * max(size) / (60. ** 2)
            marker['sizemin'] = 4

        if color is not None:
            marker['color'] = color
            if isinstance(color, (list, np.ndarray)) and not isinstance(color[0], str):
                marker['colorscale'] = 'Viridis'
                marker['showscale'] = True
                marker['colorbar'] = {'title': 'Value'}

        marker.update(kwargs.pop('marker', {}))

        self.fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name=name or "bubble",
                marker=marker,
                text=text,
                **kwargs
            )
        )

        return self

    def _parse_data(self, data, x, y):
        """Parse data into x, y, size, color, text."""
        if isinstance(data, pd.DataFrame):
            x_data = data[x].values if isinstance(x, str) else x
            y_data = data[y].values if isinstance(y, str) else y

            size = None
            if self.size_data:
                size = data[self.size_data].values if isinstance(self.size_data, str) else self.size_data

            color = None
            if self.color_data:
                color = data[self.color_data].values if isinstance(self.color_data, str) else self.color_data

            text = None
            if self.text_data:
                text = data[self.text_data].values if isinstance(self.text_data, str) else self.text_data

            return x_data, y_data, size, color, text

        elif isinstance(data, dict):
            x_data = data[x] if isinstance(x, str) else x
            y_data = data[y] if isinstance(y, str) else y

            size = data.get(self.size_data) if isinstance(self.size_data, str) else self.size_data
            color = data.get(self.color_data) if isinstance(self.color_data, str) else self.color_data
            text = data.get(self.text_data) if isinstance(self.text_data, str) else self.text_data

            return x_data, y_data, size, color, text

        else:
            return x, y, self.size_data, self.color_data, self.text_data


def bubble(
    data: pd.DataFrame | dict,
    x: str | list | np.ndarray | None = None,
    y: str | list | np.ndarray | None = None,
    size: str | list | np.ndarray | None = None,
    color: str | list | np.ndarray | None = None,
    text: str | list | np.ndarray | None = None,
    title: str | None = None,
    theme: str | Theme | None = None,
    show: bool = False,
    export: str | None = None,
    **kwargs
) -> BubbleChart:
    """
    Create bubble chart (convenience function).

    Example:
        >>> # GDP bubble chart
        >>> bubble(df, x='gdp_per_capita', y='life_expectancy',
        ...        size='population', color='continent',
        ...        title='World Development')
        >>>
        >>> # Sales bubble chart
        >>> bubble(df, x='units_sold', y='revenue',
        ...        size='market_share', color='region')
    """
    chart = BubbleChart(
        data=data,
        x=x,
        y=y,
        size=size,
        color=color,
        text=text,
        title=title,
        theme=theme,
        **kwargs
    )

    if export:
        chart.export(export)
    if show:
        chart.show()

    return chart
