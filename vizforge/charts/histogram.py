"""Histogram implementation for VizForge."""


import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.base import BaseChart
from ..core.theme import Theme


class Histogram(BaseChart):
    """
    Histogram visualization.

    Distribution of numerical data using bins.
    """

    def __init__(
        self,
        data: pd.DataFrame | list | np.ndarray | None = None,
        x: str | list | np.ndarray | None = None,
        title: str | None = None,
        theme: str | Theme | None = None,
        nbins: int | None = None,
        cumulative: bool = False,
        **kwargs
    ):
        """
        Create a histogram.

        Args:
            data: Data source
            x: Data column or array
            title: Chart title
            theme: Theme
            nbins: Number of bins
            cumulative: Cumulative histogram
            **kwargs: Additional arguments
        """
        super().__init__(title=title, theme=theme, **kwargs)
        self.nbins = nbins
        self.cumulative = cumulative

        if data is not None:
            self.plot(data, x)

    def plot(
        self,
        data: pd.DataFrame | list | np.ndarray,
        x: str | list | np.ndarray | None = None,
        name: str | None = None,
        **kwargs
    ) -> 'Histogram':
        """Plot histogram data."""
        if self.fig is None:
            self.fig = self._create_figure()

        # Parse data
        x_data = self._parse_data(data, x)

        trace_kwargs = {
            'marker': {'color': self._theme.primary_color},
        }

        if self.nbins:
            trace_kwargs['nbinsx'] = self.nbins

        if self.cumulative:
            trace_kwargs['cumulative'] = {'enabled': True}

        trace_kwargs.update(kwargs)

        self.fig.add_trace(
            go.Histogram(
                x=x_data,
                name=name or "histogram",
                **trace_kwargs
            )
        )

        return self

    def _parse_data(self, data, x):
        """Parse data into array."""
        if isinstance(data, pd.DataFrame):
            return data[x].values if isinstance(x, str) else x
        elif isinstance(data, (list, np.ndarray)):
            return data
        else:
            return x


def histogram(
    data: pd.DataFrame | list | np.ndarray,
    x: str | list | np.ndarray | None = None,
    title: str | None = None,
    theme: str | Theme | None = None,
    nbins: int | None = None,
    cumulative: bool = False,
    show: bool = False,
    export: str | None = None,
    **kwargs
) -> Histogram:
    """
    Create histogram (convenience function).

    Example:
        >>> histogram(df, x='age', title='Age Distribution', nbins=20)
        >>> histogram(df['price'], title='Price Distribution')
    """
    chart = Histogram(
        data=data,
        x=x,
        title=title,
        theme=theme,
        nbins=nbins,
        cumulative=cumulative,
        **kwargs
    )

    if export:
        chart.export(export)
    if show:
        chart.show()

    return chart
