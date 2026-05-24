"""Scatter plot implementation for VizForge."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.base import BaseChart
from ..core.theme import Theme


class ScatterPlot(BaseChart):
    """
    Scatter plot visualization.

    Supports 2D and 3D scatter plots, bubble charts.
    """

    def __init__(
        self,
        data: pd.DataFrame | dict | list | None = None,
        x: str | list | np.ndarray | None = None,
        y: str | list | np.ndarray | None = None,
        z: str | list | np.ndarray | None = None,
        title: str | None = None,
        theme: str | Theme | None = None,
        size: str | list | np.ndarray | int | None = None,
        color: str | list | np.ndarray | None = None,
        text: str | list | np.ndarray | None = None,
        **kwargs
    ):
        """
        Create a scatter plot.

        Args:
            data: Data source (DataFrame, dict, or list)
            x: X-axis data or column name
            y: Y-axis data or column name
            z: Z-axis data or column name (for 3D scatter)
            title: Chart title
            theme: Theme name or Theme object
            size: Marker size data or column name (for bubble charts)
            color: Color data or column name
            text: Hover text data or column name
            **kwargs: Additional arguments

        Example:
            >>> scatter = ScatterPlot(data=df, x='age', y='income', title='Age vs Income')
            >>> scatter.show()
        """
        super().__init__(title=title, theme=theme, **kwargs)
        self.size_data = size
        self.color_data = color
        self.text_data = text
        self.is_3d = z is not None

        if data is not None:
            if self.is_3d:
                self.plot_3d(data, x, y, z)
            else:
                self.plot(data, x, y)

    def plot(
        self,
        data: pd.DataFrame | dict | list,
        x: str | list | np.ndarray | None = None,
        y: str | list | np.ndarray | None = None,
        name: str | None = None,
        **kwargs
    ) -> 'ScatterPlot':
        """
        Plot 2D scatter data.

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
        x_data, y_data, size, color, text = self._parse_data_2d(data, x, y)

        # Prepare marker configuration
        marker = {'size': self._theme.marker_size}

        if size is not None:
            marker['size'] = size
            if isinstance(size, (list, np.ndarray)):
                marker['sizemode'] = 'diameter'
                marker['sizeref'] = 2. * max(size) / (40. ** 2)

        if color is not None:
            marker['color'] = color
            if isinstance(color, (list, np.ndarray)) and not isinstance(color[0], str):
                marker['colorscale'] = 'Viridis'
                marker['showscale'] = True

        marker.update(kwargs.pop('marker', {}))

        self.fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name=name or "scatter",
                marker=marker,
                text=text,
                **kwargs
            )
        )

        return self

    def plot_3d(
        self,
        data: pd.DataFrame | dict | list,
        x: str | list | np.ndarray | None = None,
        y: str | list | np.ndarray | None = None,
        z: str | list | np.ndarray | None = None,
        name: str | None = None,
        **kwargs
    ) -> 'ScatterPlot':
        """
        Plot 3D scatter data.

        Args:
            data: Data source
            x: X-axis data or column name
            y: Y-axis data or column name
            z: Z-axis data or column name
            name: Series name
            **kwargs: Additional trace arguments

        Returns:
            Self for method chaining
        """
        # Create figure if not exists
        if self.fig is None:
            self.fig = self._create_figure()

        # Parse data
        x_data, y_data, z_data, size, color, text = self._parse_data_3d(data, x, y, z)

        # Prepare marker configuration
        marker = {'size': self._theme.marker_size}

        if size is not None:
            marker['size'] = size

        if color is not None:
            marker['color'] = color
            if isinstance(color, (list, np.ndarray)) and not isinstance(color[0], str):
                marker['colorscale'] = 'Viridis'
                marker['showscale'] = True

        marker.update(kwargs.pop('marker', {}))

        self.fig.add_trace(
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                name=name or "scatter3d",
                marker=marker,
                text=text,
                **kwargs
            )
        )

        return self

    def _parse_data_2d(
        self,
        data: pd.DataFrame | dict | list,
        x: str | list | np.ndarray | None,
        y: str | list | np.ndarray | None
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Parse 2D data and extract x, y, size, color, text."""

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

    def _parse_data_3d(
        self,
        data: pd.DataFrame | dict | list,
        x: str | list | np.ndarray | None,
        y: str | list | np.ndarray | None,
        z: str | list | np.ndarray | None
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Parse 3D data and extract x, y, z, size, color, text."""

        if isinstance(data, pd.DataFrame):
            x_data = data[x].values if isinstance(x, str) else x
            y_data = data[y].values if isinstance(y, str) else y
            z_data = data[z].values if isinstance(z, str) else z

            size = None
            if self.size_data:
                size = data[self.size_data].values if isinstance(self.size_data, str) else self.size_data

            color = None
            if self.color_data:
                color = data[self.color_data].values if isinstance(self.color_data, str) else self.color_data

            text = None
            if self.text_data:
                text = data[self.text_data].values if isinstance(self.text_data, str) else self.text_data

            return x_data, y_data, z_data, size, color, text

        elif isinstance(data, dict):
            x_data = data[x] if isinstance(x, str) else x
            y_data = data[y] if isinstance(y, str) else y
            z_data = data[z] if isinstance(z, str) else z

            size = data.get(self.size_data) if isinstance(self.size_data, str) else self.size_data
            color = data.get(self.color_data) if isinstance(self.color_data, str) else self.color_data
            text = data.get(self.text_data) if isinstance(self.text_data, str) else self.text_data

            return x_data, y_data, z_data, size, color, text

        else:
            return x, y, z, self.size_data, self.color_data, self.text_data


def scatter(
    data: pd.DataFrame | dict | list,
    x: str | list | np.ndarray | None = None,
    y: str | list | np.ndarray | None = None,
    z: str | list | np.ndarray | None = None,
    title: str | None = None,
    theme: str | Theme | None = None,
    size: str | list | np.ndarray | int | None = None,
    color: str | list | np.ndarray | None = None,
    text: str | list | np.ndarray | None = None,
    show: bool = False,
    export: str | None = None,
    **kwargs
) -> ScatterPlot:
    """
    Create and display a scatter plot (convenience function).

    Args:
        data: Data source (DataFrame, dict, or list)
        x: X-axis data or column name
        y: Y-axis data or column name
        z: Z-axis data or column name (for 3D)
        title: Chart title
        theme: Theme name or Theme object
        size: Marker size data or column name (for bubble charts)
        color: Color data or column name
        text: Hover text data or column name
        show: Whether to display the chart
        export: Export filename (optional)
        **kwargs: Additional arguments

    Returns:
        ScatterPlot object

    Example:
        >>> scatter(df, x='age', y='income', title='Age vs Income')
        >>> scatter(df, x='x', y='y', size='population', color='category')
        >>> scatter(df, x='x', y='y', z='z', title='3D Scatter')
    """
    chart = ScatterPlot(
        data=data,
        x=x,
        y=y,
        z=z,
        title=title,
        theme=theme,
        size=size,
        color=color,
        text=text,
        **kwargs
    )

    if export:
        chart.export(export)

    if show:
        chart.show()

    return chart
