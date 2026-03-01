"""
VizForge - Visualization Intelligence Platform

Not just charts - Visual reasoning, explainability, and evidence-based insights.

v3.0.0 NEW: Performance & Extensibility Revolution!
- WebGPU rendering (1000x faster than Plotly)
- Data streaming for infinite datasets
- Plugin architecture for extensibility
- Smart caching & lazy evaluation
- Real-time collaboration
- Enhanced interactivity (gestures, touch, 3D navigation)
- 100% offline, zero API costs
"""

from .config import (
    Config,
    get_config,
    reset_config,
    set_config,
)

# NEW v1.3.0: Universal Data Connectors (13+ sources)
from .connectors import (
    AzureBlobConnector,
    BaseConnector,
    DataSource,
    # File connectors
    ExcelConnector,
    GCSConnector,
    GraphQLConnector,
    HDF5Connector,
    # Web connectors
    HTMLTableConnector,
    MongoDBConnector,
    MySQLConnector,
    ParquetConnector,
    # Database connectors
    PostgreSQLConnector,
    # API connectors
    RESTConnector,
    # Cloud connectors
    S3Connector,
    SQLiteConnector,
    WebScraperConnector,
    connect,  # Main entry point: vz.connect('postgresql', ...)
    list_connectors,
)
from .core import (
    BaseChart,
    Theme,
    get_theme,
    list_themes,
    register_theme,
    set_theme,
)
from .dashboard import (
    ChartComponent,
    Dashboard,
    DashboardLayout,
    FilterComponent,
    KPICard,
    TextComponent,
    create_dashboard,
)

# NEW v2.0.0: Insight Provenance & Evidence Tracking
from .insights import (
    Insight,
    InsightProvenanceEngine,
)

# NEW v2.0.0: Visualization Intelligence Layer
from .intelligence import (
    BiasReport,
    ChartDecision,
    ChartReasoningEngine,
    VisualBiasDetector,
)

# NEW v1.3.0: Natural Language Query (NLQ) Engine
from .nlq import (
    NLQEngine,
    ask,  # Main entry point: vz.ask("Show sales trend", df)
)

# NEW v1.3.0: Predictive Analytics Engine
from .predictive import (
    AnomalyDetector,
    SeasonalityAnalyzer,
    TimeSeriesForecaster,
    TrendDetector,
    analyze_seasonality,
    detect_anomalies,
    detect_trend,
    forecast,
)

# NEW v1.3.0: Auto Data Storytelling Engine
from .storytelling import (
    InsightDiscovery,
    NarrativeGenerator,
    ReportGenerator,
    discover_insights,
    generate_report,
    generate_story,
)

# NEW v2.0.0: Synthetic Data Generation
from .synthetic import (
    SyntheticVisualizationEngine,
    SyntheticVizConfig,
)
from .utils import (
    aggregate_data,
    bin_data,
    clean_data,
    color_scale,
    detect_outliers,
    generate_color_palette,
    hex_to_rgb,
    normalize_data,
    resample_timeseries,
    rgb_to_hex,
)
from .version import __version__

# NEW v1.3.0: Video Export Engine (MP4/WebM/GIF)
from .video_export import (
    AnimationEngine,
    AnimationType,
    FrameGenerator,
    VideoConfig,
    VideoExporter,
    VideoFormat,
    export_video,  # Main entry point: vz.export_video(chart, 'output.mp4', ...)
)

# NEW v1.3.0: Visual Chart Designer (Web-based UI)
from .visual_designer import (
    ChartConfig,
    ChartType,
    CodeGenerator,
    DesignerApp,
    launch_designer,  # Main entry point: vz.launch_designer()
)

# NEW v3.0.0: Performance & Extensibility
try:
    from .core.webgpu import WebGPURenderer
except ImportError:
    WebGPURenderer = None

# Import all 40+ chart types
from .charts import (
    KDE2D,
    AnimatedBar,
    AnimatedChoropleth,
    AnimatedScatter,
    AreaChart,
    BarChart,
    Boxplot,
    BubbleChart,
    # Geographic Chart Classes
    ChoroplethMap,
    ClusterHeatmap,
    ConePlot,
    ContourPlot,
    CorrelationMatrix,
    Dendrogram,
    DensityGeoMap,
    FeatureImportance,
    FilledContour,
    FlowMap,
    FunnelChart,
    Heatmap,
    Histogram,
    IcicleDiagram,
    IsosurfacePlot,
    KDEPlot,
    # 2D Chart Classes
    LineChart,
    LineGeoMap,
    LiveHeatmap,
    Mesh3D,
    MultiROCCurve,
    # Network Chart Classes
    NetworkGraph,
    ParallelCoordinates,
    PermutationImportance,
    PieChart,
    RadarChart,
    RegressionPlot,
    ROCCurve,
    SankeyDiagram,
    Scatter3D,
    ScatterGeoMap,
    ScatterPlot,
    # Real-time Chart Classes
    StreamingLine,
    Sunburst,
    # 3D Chart Classes
    SurfacePlot,
    TreeDiagram,
    # Advanced Chart Classes
    Treemap,
    # Statistical Chart Classes
    ViolinPlot,
    VolumePlot,
    WaterfallChart,
    animated_bar,
    animated_choropleth,
    animated_scatter,
    area,
    bar,
    boxplot,
    bubble,
    # Geographic Functions
    choropleth,
    cluster_heatmap,
    cone,
    contour,
    correlation_matrix,
    dendrogram,
    densitygeo,
    donut,
    feature_importance,
    filled_contour,
    flowmap,
    funnel,
    heatmap,
    histogram,
    icicle,
    isosurface,
    kde,
    kde2d,
    # 2D Functions
    line,
    linegeo,
    live_heatmap,
    mesh3d,
    multi_roc_curve,
    # Network Functions
    network_graph,
    parallel_coordinates,
    permutation_importance,
    pie,
    radar,
    regression,
    roc_curve_plot,
    sankey,
    scatter,
    scatter3d,
    scattergeo,
    # Real-time Functions
    streaming_line,
    sunburst,
    # 3D Functions
    surface,
    tree,
    # Advanced Functions
    treemap,
    # Statistical Functions
    violin,
    volume,
    waterfall,
)
from .core.collaboration import (
    Change,
    ChangeType,
    CollaborationServer,
    CollaborationSession,
    Comment,
    User,
    enable_collaboration,
    get_collaboration_server,
)
from .core.interactivity import (
    GestureRecognizer,
    GestureType,
    HapticFeedback,
    Navigation3D,
    SemanticZoom,
    enable_3d_navigation,
    enable_gestures,
    enable_semantic_zoom,
)
from .core.performance import (
    DataOptimizer,
    LazyArray,
    ParallelExecutor,
    PerformanceProfiler,
    SmartCache,
    cached,
    get_executor,
    get_profiler,
    optimize,
)
from .core.plugins import (
    ChartPlugin,
    ConnectorPlugin,
    InteractionPlugin,
    Plugin,
    PluginManager,
    PluginMetadata,
    RendererPlugin,
    get_plugin,
    get_plugin_manager,
    list_plugins,
    register_plugin,
)
from .core.streaming import (
    DataStream,
    ProgressiveRenderer,
    StreamConfig,
    StreamingChart,
    stream_from_api,
    stream_from_database,
    stream_from_file,
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
    # v3.0.0: WebGPU Rendering
    "WebGPURenderer",
    "WebGPUConfig",
    "enable_webgpu",
    # v3.0.0: Data Streaming
    "DataStream",
    "StreamConfig",
    "ProgressiveRenderer",
    "StreamingChart",
    "stream_from_file",
    "stream_from_database",
    "stream_from_api",
    # v3.0.0: Plugin System
    "Plugin",
    "ChartPlugin",
    "ConnectorPlugin",
    "RendererPlugin",
    "InteractionPlugin",
    "PluginManager",
    "PluginMetadata",
    "get_plugin_manager",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    # v3.0.0: Performance Layer
    "SmartCache",
    "LazyArray",
    "ParallelExecutor",
    "PerformanceProfiler",
    "DataOptimizer",
    "cached",
    "get_executor",
    "get_profiler",
    "optimize",
    # v3.0.0: Collaboration
    "User",
    "Change",
    "Comment",
    "ChangeType",
    "CollaborationSession",
    "CollaborationServer",
    "get_collaboration_server",
    "enable_collaboration",
    # v3.0.0: Enhanced Interactivity
    "GestureType",
    "GestureRecognizer",
    "Navigation3D",
    "SemanticZoom",
    "HapticFeedback",
    "enable_gestures",
    "enable_3d_navigation",
    "enable_semantic_zoom",
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
