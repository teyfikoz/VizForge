"""
Chart configuration classes for Visual Designer.

Defines available chart types, their properties, and validation rules.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ChartType(Enum):
    """Supported chart types in Visual Designer."""
    # 2D Charts
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    BOXPLOT = "boxplot"
    HEATMAP = "heatmap"
    BUBBLE = "bubble"
    WATERFALL = "waterfall"
    FUNNEL = "funnel"
    RADAR = "radar"

    # 3D Charts
    SURFACE = "surface"
    SCATTER3D = "scatter3d"
    MESH3D = "mesh3d"

    # Geographic
    CHOROPLETH = "choropleth"
    SCATTERGEO = "scattergeo"

    # Network
    NETWORK = "network_graph"
    SANKEY = "sankey"
    TREE = "tree"

    # Statistical
    VIOLIN = "violin"
    KDE = "kde"
    REGRESSION = "regression"
    CORRELATION_MATRIX = "correlation_matrix"

    # Advanced
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    PARALLEL_COORDINATES = "parallel_coordinates"


class PropertyType(Enum):
    """Property types for chart configuration."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    COLOR = "color"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    COLUMN = "column"
    COLUMNS = "columns"
    DICT = "dict"


@dataclass
class PropertyConfig:
    """Configuration for a chart property."""
    name: str
    type: PropertyType
    label: str
    default: Any = None
    required: bool = False
    options: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""

    def validate(self, value: Any) -> bool:
        """Validate property value."""
        if self.required and value is None:
            return False

        if value is None:
            return True

        if self.type == PropertyType.NUMBER:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.type == PropertyType.BOOLEAN:
            if not isinstance(value, bool):
                return False

        elif self.type == PropertyType.SELECT:
            if self.options and value not in self.options:
                return False

        elif self.type == PropertyType.MULTI_SELECT:
            if self.options and not all(v in self.options for v in value):
                return False

        return True


@dataclass
class ChartConfig:
    """Complete configuration for a chart."""
    chart_type: ChartType
    title: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None
    filters: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def get_available_properties(cls, chart_type: ChartType) -> List[PropertyConfig]:
        """Get available properties for a chart type."""
        # Common properties for all charts
        common_props = [
            PropertyConfig(
                name="title",
                type=PropertyType.STRING,
                label="Chart Title",
                default="",
                description="Title displayed above the chart"
            ),
            PropertyConfig(
                name="width",
                type=PropertyType.NUMBER,
                label="Width (px)",
                default=800,
                min_value=200,
                max_value=2000,
                description="Chart width in pixels"
            ),
            PropertyConfig(
                name="height",
                type=PropertyType.NUMBER,
                label="Height (px)",
                default=600,
                min_value=200,
                max_value=1500,
                description="Chart height in pixels"
            ),
            PropertyConfig(
                name="theme",
                type=PropertyType.SELECT,
                label="Theme",
                default="default",
                options=["default", "dark", "minimal", "scientific", "colorful", "professional"],
                description="Visual theme for the chart"
            ),
        ]

        # Chart-specific properties
        if chart_type in [ChartType.LINE, ChartType.AREA, ChartType.SCATTER]:
            specific_props = [
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="X Axis Column",
                    required=True,
                    description="Column to use for X axis"
                ),
                PropertyConfig(
                    name="y",
                    type=PropertyType.COLUMN,
                    label="Y Axis Column",
                    required=True,
                    description="Column to use for Y axis"
                ),
                PropertyConfig(
                    name="color",
                    type=PropertyType.COLUMN,
                    label="Color By (Optional)",
                    description="Column to use for color grouping"
                ),
                PropertyConfig(
                    name="show_legend",
                    type=PropertyType.BOOLEAN,
                    label="Show Legend",
                    default=True,
                    description="Display legend on chart"
                ),
            ]

        elif chart_type == ChartType.BAR:
            specific_props = [
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="X Axis Column",
                    required=True,
                    description="Category column"
                ),
                PropertyConfig(
                    name="y",
                    type=PropertyType.COLUMN,
                    label="Y Axis Column",
                    required=True,
                    description="Value column"
                ),
                PropertyConfig(
                    name="orientation",
                    type=PropertyType.SELECT,
                    label="Orientation",
                    default="vertical",
                    options=["vertical", "horizontal"],
                    description="Bar orientation"
                ),
                PropertyConfig(
                    name="color",
                    type=PropertyType.COLUMN,
                    label="Color By (Optional)",
                    description="Column to use for color grouping"
                ),
            ]

        elif chart_type in [ChartType.PIE, ChartType.DONUT]:
            specific_props = [
                PropertyConfig(
                    name="labels",
                    type=PropertyType.COLUMN,
                    label="Labels Column",
                    required=True,
                    description="Column for pie slice labels"
                ),
                PropertyConfig(
                    name="values",
                    type=PropertyType.COLUMN,
                    label="Values Column",
                    required=True,
                    description="Column for pie slice values"
                ),
                PropertyConfig(
                    name="show_percentage",
                    type=PropertyType.BOOLEAN,
                    label="Show Percentage",
                    default=True,
                    description="Display percentages on slices"
                ),
            ]

        elif chart_type == ChartType.HISTOGRAM:
            specific_props = [
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="Data Column",
                    required=True,
                    description="Column to create histogram from"
                ),
                PropertyConfig(
                    name="bins",
                    type=PropertyType.NUMBER,
                    label="Number of Bins",
                    default=20,
                    min_value=5,
                    max_value=100,
                    description="Number of histogram bins"
                ),
            ]

        elif chart_type == ChartType.BOXPLOT:
            specific_props = [
                PropertyConfig(
                    name="y",
                    type=PropertyType.COLUMN,
                    label="Value Column",
                    required=True,
                    description="Column for boxplot values"
                ),
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="Group By (Optional)",
                    description="Column to group boxplots by"
                ),
            ]

        elif chart_type == ChartType.HEATMAP:
            specific_props = [
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="X Axis Column",
                    required=True,
                    description="Column for X axis"
                ),
                PropertyConfig(
                    name="y",
                    type=PropertyType.COLUMN,
                    label="Y Axis Column",
                    required=True,
                    description="Column for Y axis"
                ),
                PropertyConfig(
                    name="values",
                    type=PropertyType.COLUMN,
                    label="Values Column",
                    required=True,
                    description="Column for heatmap values"
                ),
                PropertyConfig(
                    name="colorscale",
                    type=PropertyType.SELECT,
                    label="Color Scale",
                    default="Viridis",
                    options=["Viridis", "RdBu", "Blues", "Greens", "Reds", "YlOrRd"],
                    description="Color scale for heatmap"
                ),
            ]

        elif chart_type == ChartType.SCATTER3D:
            specific_props = [
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="X Axis Column",
                    required=True
                ),
                PropertyConfig(
                    name="y",
                    type=PropertyType.COLUMN,
                    label="Y Axis Column",
                    required=True
                ),
                PropertyConfig(
                    name="z",
                    type=PropertyType.COLUMN,
                    label="Z Axis Column",
                    required=True
                ),
                PropertyConfig(
                    name="color",
                    type=PropertyType.COLUMN,
                    label="Color By (Optional)"
                ),
            ]

        elif chart_type == ChartType.BUBBLE:
            specific_props = [
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="X Axis Column",
                    required=True
                ),
                PropertyConfig(
                    name="y",
                    type=PropertyType.COLUMN,
                    label="Y Axis Column",
                    required=True
                ),
                PropertyConfig(
                    name="size",
                    type=PropertyType.COLUMN,
                    label="Bubble Size Column",
                    required=True,
                    description="Column to determine bubble size"
                ),
                PropertyConfig(
                    name="color",
                    type=PropertyType.COLUMN,
                    label="Color By (Optional)"
                ),
            ]

        elif chart_type == ChartType.CORRELATION_MATRIX:
            specific_props = [
                PropertyConfig(
                    name="columns",
                    type=PropertyType.COLUMNS,
                    label="Columns to Correlate",
                    required=True,
                    description="Select numeric columns for correlation matrix"
                ),
                PropertyConfig(
                    name="method",
                    type=PropertyType.SELECT,
                    label="Correlation Method",
                    default="pearson",
                    options=["pearson", "spearman", "kendall"],
                    description="Statistical method for correlation"
                ),
            ]

        else:
            # Generic properties for other chart types
            specific_props = [
                PropertyConfig(
                    name="x",
                    type=PropertyType.COLUMN,
                    label="X Column",
                    required=True
                ),
                PropertyConfig(
                    name="y",
                    type=PropertyType.COLUMN,
                    label="Y Column",
                    required=True
                ),
            ]

        return common_props + specific_props

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate chart configuration."""
        property_configs = self.get_available_properties(self.chart_type)

        for prop_config in property_configs:
            value = self.properties.get(prop_config.name)

            if not prop_config.validate(value):
                return False, f"Invalid value for {prop_config.label}"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'chart_type': self.chart_type.value,
            'title': self.title,
            'properties': self.properties,
            'data_source': self.data_source,
            'filters': self.filters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartConfig':
        """Create config from dictionary."""
        return cls(
            chart_type=ChartType(data['chart_type']),
            title=data.get('title', ''),
            properties=data.get('properties', {}),
            data_source=data.get('data_source'),
            filters=data.get('filters', []),
        )


# Chart type categories for UI organization
CHART_CATEGORIES = {
    '2D Charts': [
        ChartType.LINE, ChartType.BAR, ChartType.AREA, ChartType.SCATTER,
        ChartType.PIE, ChartType.DONUT, ChartType.HISTOGRAM, ChartType.BOXPLOT,
        ChartType.HEATMAP, ChartType.BUBBLE, ChartType.WATERFALL, ChartType.FUNNEL,
        ChartType.RADAR
    ],
    '3D Charts': [
        ChartType.SURFACE, ChartType.SCATTER3D, ChartType.MESH3D
    ],
    'Geographic': [
        ChartType.CHOROPLETH, ChartType.SCATTERGEO
    ],
    'Network': [
        ChartType.NETWORK, ChartType.SANKEY, ChartType.TREE
    ],
    'Statistical': [
        ChartType.VIOLIN, ChartType.KDE, ChartType.REGRESSION,
        ChartType.CORRELATION_MATRIX
    ],
    'Advanced': [
        ChartType.TREEMAP, ChartType.SUNBURST, ChartType.PARALLEL_COORDINATES
    ],
}
