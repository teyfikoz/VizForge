"""Chart types for VizForge (2D, 3D, Geographic)."""

# 2D Charts
from .line import LineChart, line
from .bar import BarChart, bar
from .area import AreaChart, area
from .scatter import ScatterPlot, scatter
from .pie import PieChart, pie, donut
from .heatmap import Heatmap, heatmap
from .histogram import Histogram, histogram
from .boxplot import Boxplot, boxplot
from .radar import RadarChart, radar
from .waterfall import WaterfallChart, waterfall
from .funnel import FunnelChart, funnel
from .bubble import BubbleChart, bubble

# 3D Charts
from .threed import (
    SurfacePlot, surface,
    Scatter3D, scatter3d,
    Mesh3D, mesh3d,
    VolumePlot, volume,
    ConePlot, cone,
    IsosurfacePlot, isosurface,
)

# Geographic Charts
from .geo import (
    ChoroplethMap, choropleth,
    ScatterGeoMap, scattergeo,
    LineGeoMap, linegeo,
    DensityGeoMap, densitygeo,
    FlowMap, flowmap,
)

__all__ = [
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
