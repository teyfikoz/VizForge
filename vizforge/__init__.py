"""
VizForge - Visualization Intelligence Platform

Not just charts - Visual reasoning, explainability, and evidence-based insights.

v2.0.0 NEW: Visualization Intelligence Layer!
- Smart chart selection with reasoning
- Visual bias detection
- Evidence-based insights with provenance
- Synthetic data generation
- 100% offline, zero API costs
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

from .config import (
    Config,
    get_config,
    set_config,
    reset_config,
)

from .dashboard import (
    Dashboard,
    DashboardLayout,
    create_dashboard,
    ChartComponent,
    KPICard,
    FilterComponent,
    TextComponent,
)

from .utils import (
    clean_data,
    aggregate_data,
    resample_timeseries,
    detect_outliers,
    normalize_data,
    bin_data,
    generate_color_palette,
    color_scale,
    hex_to_rgb,
    rgb_to_hex,
)

# NEW v1.3.0: Natural Language Query (NLQ) Engine
from .nlq import (
    NLQEngine,
    ask,  # Main entry point: vz.ask("Show sales trend", df)
)

# NEW v1.3.0: Predictive Analytics Engine
from .predictive import (
    forecast,
    detect_trend,
    detect_anomalies,
    analyze_seasonality,
    TimeSeriesForecaster,
    TrendDetector,
    AnomalyDetector,
    SeasonalityAnalyzer,
)

# NEW v1.3.0: Auto Data Storytelling Engine
from .storytelling import (
    discover_insights,
    generate_story,
    generate_report,
    InsightDiscovery,
    NarrativeGenerator,
    ReportGenerator,
)

# NEW v1.3.0: Visual Chart Designer (Web-based UI)
from .visual_designer import (
    launch_designer,  # Main entry point: vz.launch_designer()
    DesignerApp,
    ChartConfig,
    ChartType,
    CodeGenerator,
)

# NEW v1.3.0: Universal Data Connectors (13+ sources)
from .connectors import (
    connect,  # Main entry point: vz.connect('postgresql', ...)
    list_connectors,
    BaseConnector,
    DataSource,
    # Database connectors
    PostgreSQLConnector,
    MySQLConnector,
    SQLiteConnector,
    MongoDBConnector,
    # Cloud connectors
    S3Connector,
    GCSConnector,
    AzureBlobConnector,
    # API connectors
    RESTConnector,
    GraphQLConnector,
    # File connectors
    ExcelConnector,
    ParquetConnector,
    HDF5Connector,
    # Web connectors
    HTMLTableConnector,
    WebScraperConnector,
)

# NEW v1.3.0: Video Export Engine (MP4/WebM/GIF)
from .video_export import (
    export_video,  # Main entry point: vz.export_video(chart, 'output.mp4', ...)
    VideoExporter,
    VideoConfig,
    VideoFormat,
    AnimationEngine,
    AnimationType,
    FrameGenerator,
)

# NEW v2.0.0: Visualization Intelligence Layer
from .intelligence import (
    ChartReasoningEngine,
    ChartDecision,
    VisualBiasDetector,
    BiasReport,
)

# NEW v2.0.0: Insight Provenance & Evidence Tracking
from .insights import (
    InsightProvenanceEngine,
    Insight,
)

# NEW v2.0.0: Synthetic Data Generation
from .synthetic import (
    SyntheticVisualizationEngine,
    SyntheticVizConfig,
)

# Import all 40+ chart types
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
    # Network Chart Classes
    NetworkGraph,
    SankeyDiagram,
    TreeDiagram,
    IcicleDiagram,
    Dendrogram,
    ClusterHeatmap,
    # Real-time Chart Classes
    StreamingLine,
    LiveHeatmap,
    AnimatedScatter,
    AnimatedBar,
    AnimatedChoropleth,
    # Statistical Chart Classes
    ViolinPlot,
    KDEPlot,
    KDE2D,
    RegressionPlot,
    CorrelationMatrix,
    ROCCurve,
    MultiROCCurve,
    FeatureImportance,
    PermutationImportance,
    # Advanced Chart Classes
    Treemap,
    Sunburst,
    ParallelCoordinates,
    ContourPlot,
    FilledContour,
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
    # Network Functions
    network_graph,
    sankey,
    tree,
    icicle,
    dendrogram,
    cluster_heatmap,
    # Real-time Functions
    streaming_line,
    live_heatmap,
    animated_scatter,
    animated_bar,
    animated_choropleth,
    # Statistical Functions
    violin,
    kde,
    kde2d,
    regression,
    correlation_matrix,
    roc_curve_plot,
    multi_roc_curve,
    feature_importance,
    permutation_importance,
    # Advanced Functions
    treemap,
    sunburst,
    parallel_coordinates,
    contour,
    filled_contour,
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
    # Config
    "Config",
    "get_config",
    "set_config",
    "reset_config",
    # Dashboard
    "Dashboard",
    "DashboardLayout",
    "create_dashboard",
    "ChartComponent",
    "KPICard",
    "FilterComponent",
    "TextComponent",
    # Utils
    "clean_data",
    "aggregate_data",
    "resample_timeseries",
    "detect_outliers",
    "normalize_data",
    "bin_data",
    "generate_color_palette",
    "color_scale",
    "hex_to_rgb",
    "rgb_to_hex",
    # NLQ Engine
    "NLQEngine",
    "ask",
    # Predictive Analytics
    "forecast",
    "detect_trend",
    "detect_anomalies",
    "analyze_seasonality",
    "TimeSeriesForecaster",
    "TrendDetector",
    "AnomalyDetector",
    "SeasonalityAnalyzer",
    # Storytelling
    "discover_insights",
    "generate_story",
    "generate_report",
    "InsightDiscovery",
    "NarrativeGenerator",
    "ReportGenerator",
    # Visual Designer
    "launch_designer",
    "DesignerApp",
    "ChartConfig",
    "ChartType",
    "CodeGenerator",
    # Data Connectors
    "connect",
    "list_connectors",
    "BaseConnector",
    "DataSource",
    "PostgreSQLConnector",
    "MySQLConnector",
    "SQLiteConnector",
    "MongoDBConnector",
    "S3Connector",
    "GCSConnector",
    "AzureBlobConnector",
    "RESTConnector",
    "GraphQLConnector",
    "ExcelConnector",
    "ParquetConnector",
    "HDF5Connector",
    "HTMLTableConnector",
    "WebScraperConnector",
    # Video Export
    "export_video",
    "VideoExporter",
    "VideoConfig",
    "VideoFormat",
    "AnimationEngine",
    "AnimationType",
    "FrameGenerator",
    # v2.0.0: Intelligence Layer
    "ChartReasoningEngine",
    "ChartDecision",
    "VisualBiasDetector",
    "BiasReport",
    # v2.0.0: Insights & Provenance
    "InsightProvenanceEngine",
    "Insight",
    # v2.0.0: Synthetic Data
    "SyntheticVisualizationEngine",
    "SyntheticVizConfig",
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
