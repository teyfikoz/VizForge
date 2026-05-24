"""Chart types for VizForge (2D, 3D, Geographic, Network, Real-time, Statistical, Advanced)."""

# 2D Charts
# Advanced Charts
from .advanced import (
    ContourPlot,
    FilledContour,
    ParallelCoordinates,
    Sunburst,
    Treemap,
    contour,
    filled_contour,
    parallel_coordinates,
    sunburst,
    treemap,
)
from .area import AreaChart, area
from .bar import BarChart, bar
from .boxplot import Boxplot, boxplot
from .bubble import BubbleChart, bubble
from .funnel import FunnelChart, funnel

# Geographic Charts
from .geo import (
    ChoroplethMap,
    DensityGeoMap,
    FlowMap,
    LineGeoMap,
    ScatterGeoMap,
    choropleth,
    densitygeo,
    flowmap,
    linegeo,
    scattergeo,
)
from .heatmap import Heatmap, heatmap
from .histogram import Histogram, histogram
from .line import LineChart, line

# Network Charts
from .network import (
    ClusterHeatmap,
    Dendrogram,
    IcicleDiagram,
    NetworkGraph,
    SankeyDiagram,
    TreeDiagram,
    cluster_heatmap,
    dendrogram,
    icicle,
    network_graph,
    sankey,
    tree,
)
from .pie import PieChart, donut, pie
from .radar import RadarChart, radar

# Real-time Charts
from .realtime import (
    AnimatedBar,
    AnimatedChoropleth,
    AnimatedScatter,
    LiveHeatmap,
    StreamingLine,
    animated_bar,
    animated_choropleth,
    animated_scatter,
    live_heatmap,
    streaming_line,
)
from .scatter import ScatterPlot, scatter

# Statistical Charts
from .stats import (
    KDE2D,
    CorrelationMatrix,
    FeatureImportance,
    KDEPlot,
    MultiROCCurve,
    PermutationImportance,
    RegressionPlot,
    ROCCurve,
    ViolinPlot,
    correlation_matrix,
    feature_importance,
    kde,
    kde2d,
    multi_roc_curve,
    permutation_importance,
    regression,
    roc_curve_plot,
    violin,
)

# 3D Charts
from .threed import (
    ConePlot,
    IsosurfacePlot,
    Mesh3D,
    Scatter3D,
    SurfacePlot,
    VolumePlot,
    cone,
    isosurface,
    mesh3d,
    scatter3d,
    surface,
    volume,
)
from .waterfall import WaterfallChart, waterfall

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
    # Network Chart Classes
    "NetworkGraph",
    "SankeyDiagram",
    "TreeDiagram",
    "IcicleDiagram",
    "Dendrogram",
    "ClusterHeatmap",
    # Real-time Chart Classes
    "StreamingLine",
    "LiveHeatmap",
    "AnimatedScatter",
    "AnimatedBar",
    "AnimatedChoropleth",
    # Statistical Chart Classes
    "ViolinPlot",
    "KDEPlot",
    "KDE2D",
    "RegressionPlot",
    "CorrelationMatrix",
    "ROCCurve",
    "MultiROCCurve",
    "FeatureImportance",
    "PermutationImportance",
    # Advanced Chart Classes
    "Treemap",
    "Sunburst",
    "ParallelCoordinates",
    "ContourPlot",
    "FilledContour",
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
    # Network Functions
    "network_graph",
    "sankey",
    "tree",
    "icicle",
    "dendrogram",
    "cluster_heatmap",
    # Real-time Functions
    "streaming_line",
    "live_heatmap",
    "animated_scatter",
    "animated_bar",
    "animated_choropleth",
    # Statistical Functions
    "violin",
    "kde",
    "kde2d",
    "regression",
    "correlation_matrix",
    "roc_curve_plot",
    "multi_roc_curve",
    "feature_importance",
    "permutation_importance",
    # Advanced Functions
    "treemap",
    "sunburst",
    "parallel_coordinates",
    "contour",
    "filled_contour",
]
