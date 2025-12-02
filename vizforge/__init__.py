"""
VizForge - Production-grade data visualization library with zero AI dependencies.

Create beautiful, interactive visualizations with a single line of code.

v0.2.0: Now with 12 chart types!
"""

from .version import __version__

from .core import (
    BaseChart,
    Theme,
    get_theme,
    set_theme,
    register_theme,
    list_themes,
)

# Import all 12 chart types
from .charts import (
    # Classes
    LineChart,
    BarChart,
    AreaChart,
    ScatterPlot,
    PieChart,
    Heatmap,
    Histogram,
    Boxplot,
    RadarChart,
    WaterfallChart,
    FunnelChart,
    BubbleChart,
    # Functions
    line,
    bar,
    area,
    scatter,
    pie,
    donut,
    heatmap,
    histogram,
    boxplot,
    radar,
    waterfall,
    funnel,
    bubble,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseChart",
    "Theme",
    "get_theme",
    "set_theme",
    "register_theme",
    "list_themes",
    # Chart Classes
    "LineChart",
    "BarChart",
    "AreaChart",
    "ScatterPlot",
    "PieChart",
    "Heatmap",
    "Histogram",
    "Boxplot",
    "RadarChart",
    "WaterfallChart",
    "FunnelChart",
    "BubbleChart",
    # Convenience Functions
    "line",
    "bar",
    "area",
    "scatter",
    "pie",
    "donut",
    "heatmap",
    "histogram",
    "boxplot",
    "radar",
    "waterfall",
    "funnel",
    "bubble",
]
