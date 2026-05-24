"""Funnel chart implementation for VizForge."""


import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...core.base import BaseChart
from ...core.theme import Theme


class FunnelChart(BaseChart):
    """
    Funnel chart visualization.

    Shows stages in a process (e.g., sales funnel, conversion funnel).
    """

    def __init__(
        self,
        data: pd.DataFrame | dict | None = None,
        x: str | list | np.ndarray | None = None,
        y: str | list | None = None,
        title: str | None = None,
        theme: str | Theme | None = None,
        **kwargs
    ):
        """
        Create a funnel chart.

        Args:
            data: Data source
            x: Values or column name
            y: Stage names or column name
            title: Chart title
            theme: Theme
            **kwargs: Additional arguments
        """
        super().__init__(title=title, theme=theme, **kwargs)

        if data is not None:
            self.plot(data, x, y)

    def plot(
        self,
        data: pd.DataFrame | dict,
        x: str | list | np.ndarray | None = None,
        y: str | list | None = None,
        **kwargs
    ) -> 'FunnelChart':
        """Plot funnel chart data."""
        if self.fig is None:
            self.fig = self._create_figure()

        # Parse data
        x_data, y_data = self._parse_data(data, x, y)

        trace_kwargs = {
            'marker': {
                'color': self._theme.color_palette[:len(x_data)]
            },
        }
        trace_kwargs.update(kwargs)

        self.fig.add_trace(
            go.Funnel(
                x=x_data,
                y=y_data,
                **trace_kwargs
            )
        )

        return self

    def _parse_data(self, data, x, y):
        """Parse data into x, y."""
        if isinstance(data, pd.DataFrame):
            x_data = data[x].values if isinstance(x, str) else x
            y_data = data[y].values if isinstance(y, str) else y
            return list(x_data), list(y_data)

        elif isinstance(data, dict):
            if x is None and y is None:
                # Dict format: {stage: value}
                y_data = list(data.keys())
                x_data = list(data.values())
            else:
                x_data = data[x] if isinstance(x, str) else x
                y_data = data[y] if isinstance(y, str) else y
            return x_data, y_data

        else:
            return x, y


def funnel(
    data: pd.DataFrame | dict,
    x: str | list | np.ndarray | None = None,
    y: str | list | None = None,
    title: str | None = None,
    theme: str | Theme | None = None,
    show: bool = False,
    export: str | None = None,
    **kwargs
) -> FunnelChart:
    """
    Create funnel chart (convenience function).

    Example:
        >>> # Sales funnel
        >>> data = {
        ...     'Visitors': 10000,
        ...     'Signed Up': 5000,
        ...     'Active': 2000,
        ...     'Purchased': 500
        ... }
        >>> funnel(data, title='Sales Funnel')
        >>>
        >>> # DataFrame format
        >>> funnel(df, x='count', y='stage', title='Conversion Funnel')
    """
    chart = FunnelChart(
        data=data,
        x=x,
        y=y,
        title=title,
        theme=theme,
        **kwargs
    )

    if export:
        chart.export(export)
    if show:
        chart.show()

    return chart
