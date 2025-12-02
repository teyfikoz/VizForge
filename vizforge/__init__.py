"""
VizForge - Production-grade data visualization library with zero AI dependencies.

Create beautiful, interactive visualizations with a single line of code.
"""

from .core import (
    BaseChart,
    Theme,
    get_theme,
    set_theme,
    register_theme,
    list_themes,
)

from .charts import (
    LineChart,
    line,
    BarChart,
    bar,
    ScatterPlot,
    scatter,
    PieChart,
    pie,
    donut,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "BaseChart",
    "Theme",
    "get_theme",
    "set_theme",
    "register_theme",
    "list_themes",
    # Charts
    "LineChart",
    "line",
    "BarChart",
    "bar",
    "ScatterPlot",
    "scatter",
    "PieChart",
    "pie",
    "donut",
]
