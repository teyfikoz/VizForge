"""
VizForge - Production-grade data visualization library with zero AI dependencies.

Create beautiful, interactive visualizations with a single line of code.

v0.3.0: Now with 23 chart types! (12 2D + 6 3D + 5 Geographic)
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

# Import all 23 chart types
from .charts import (
    # 2D Chart Classes
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
    # 3D Chart Classes
    SurfacePlot,
    Scatter3D,
    Mesh3D,
    VolumePlot,
    ConePlot,
    IsosurfacePlot,
    # Geographic Chart Classes
    ChoroplethMap,
    ScatterGeoMap,
    LineGeoMap,
    DensityGeoMap,
    FlowMap,
    # 2D Functions
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
    # 3D Functions
    surface,
    scatter3d,
    mesh3d,
    volume,
    cone,
    isosurface,
    # Geographic Functions
    choropleth,
    scattergeo,
    linegeo,
    densitygeo,
    flowmap,
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
    # 2D Chart Classes
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
    # 3D Chart Classes
    "SurfacePlot",
    "Scatter3D",
    "Mesh3D",
    "VolumePlot",
    "ConePlot",
    "IsosurfacePlot",
    # Geographic Chart Classes
    "ChoroplethMap",
    "ScatterGeoMap",
    "LineGeoMap",
    "DensityGeoMap",
    "FlowMap",
    # 2D Functions
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
    # 3D Functions
    "surface",
    "scatter3d",
    "mesh3d",
    "volume",
    "cone",
    "isosurface",
    # Geographic Functions
    "choropleth",
    "scattergeo",
    "linegeo",
    "densitygeo",
    "flowmap",
]
