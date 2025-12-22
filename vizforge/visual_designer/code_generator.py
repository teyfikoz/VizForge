"""
Python code generator for Visual Designer.

Converts chart configurations into executable VizForge Python code.
"""

from typing import Dict, Any, List
from .chart_config import ChartConfig, ChartType


class CodeGenerator:
    """Generates Python code from chart configurations."""

    def __init__(self):
        """Initialize code generator."""
        pass

    def generate(self, config: ChartConfig, include_imports: bool = True,
                 include_data_loading: bool = True) -> str:
        """
        Generate Python code for a chart configuration.

        Args:
            config: Chart configuration
            include_imports: Include import statements
            include_data_loading: Include data loading code

        Returns:
            Python code as string
        """
        code_lines = []

        # Add imports
        if include_imports:
            code_lines.extend(self._generate_imports(config))
            code_lines.append("")

        # Add data loading
        if include_data_loading and config.data_source:
            code_lines.extend(self._generate_data_loading(config))
            code_lines.append("")

        # Add filters if any
        if config.filters:
            code_lines.extend(self._generate_filters(config.filters))
            code_lines.append("")

        # Generate chart creation
        code_lines.extend(self._generate_chart_creation(config))
        code_lines.append("")

        # Generate display
        code_lines.append("# Display the chart")
        code_lines.append("chart.show()")

        return "\n".join(code_lines)

    def _generate_imports(self, config: ChartConfig) -> List[str]:
        """Generate import statements."""
        imports = [
            "import vizforge as vz",
            "import pandas as pd",
        ]

        # Add numpy if needed for certain chart types
        if config.chart_type in [ChartType.SURFACE, ChartType.MESH3D, ChartType.SCATTER3D]:
            imports.append("import numpy as np")

        return imports

    def _generate_data_loading(self, config: ChartConfig) -> List[str]:
        """Generate data loading code."""
        data_source = config.data_source

        lines = ["# Load data"]

        if data_source.endswith('.csv'):
            lines.append(f"df = pd.read_csv('{data_source}')")
        elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
            lines.append(f"df = pd.read_excel('{data_source}')")
        elif data_source.endswith('.json'):
            lines.append(f"df = pd.read_json('{data_source}')")
        elif data_source.endswith('.parquet'):
            lines.append(f"df = pd.read_parquet('{data_source}')")
        else:
            # Assume CSV
            lines.append(f"df = pd.read_csv('{data_source}')")

        return lines

    def _generate_filters(self, filters: List[Dict[str, Any]]) -> List[str]:
        """Generate data filtering code."""
        lines = ["# Apply filters"]

        for filt in filters:
            column = filt.get('column')
            operator = filt.get('operator')
            value = filt.get('value')

            if operator == 'equals':
                if isinstance(value, str):
                    lines.append(f"df = df[df['{column}'] == '{value}']")
                else:
                    lines.append(f"df = df[df['{column}'] == {value}]")

            elif operator == 'not_equals':
                if isinstance(value, str):
                    lines.append(f"df = df[df['{column}'] != '{value}']")
                else:
                    lines.append(f"df = df[df['{column}'] != {value}]")

            elif operator == 'greater_than':
                lines.append(f"df = df[df['{column}'] > {value}]")

            elif operator == 'less_than':
                lines.append(f"df = df[df['{column}'] < {value}]")

            elif operator == 'contains':
                lines.append(f"df = df[df['{column}'].str.contains('{value}', na=False)]")

            elif operator == 'in':
                if isinstance(value, list):
                    values_str = str(value)
                    lines.append(f"df = df[df['{column}'].isin({values_str})]")

        return lines

    def _generate_chart_creation(self, config: ChartConfig) -> List[str]:
        """Generate chart creation code."""
        lines = ["# Create chart"]

        chart_type = config.chart_type
        props = config.properties

        # Get function name
        function_name = self._get_function_name(chart_type)

        # Build parameters
        params = []

        # Add data parameter
        params.append("data=df")

        # Add chart-specific parameters
        if chart_type in [ChartType.LINE, ChartType.BAR, ChartType.AREA, ChartType.SCATTER]:
            if 'x' in props and props['x']:
                params.append(f"x='{props['x']}'")
            if 'y' in props and props['y']:
                params.append(f"y='{props['y']}'")
            if 'color' in props and props['color']:
                params.append(f"color='{props['color']}'")

        elif chart_type in [ChartType.PIE, ChartType.DONUT]:
            if 'labels' in props and props['labels']:
                params.append(f"labels='{props['labels']}'")
            if 'values' in props and props['values']:
                params.append(f"values='{props['values']}'")

        elif chart_type == ChartType.HISTOGRAM:
            if 'x' in props and props['x']:
                params.append(f"x='{props['x']}'")
            if 'bins' in props:
                params.append(f"bins={props['bins']}")

        elif chart_type == ChartType.BOXPLOT:
            if 'y' in props and props['y']:
                params.append(f"y='{props['y']}'")
            if 'x' in props and props['x']:
                params.append(f"x='{props['x']}'")

        elif chart_type == ChartType.HEATMAP:
            if 'x' in props and props['x']:
                params.append(f"x='{props['x']}'")
            if 'y' in props and props['y']:
                params.append(f"y='{props['y']}'")
            if 'values' in props and props['values']:
                params.append(f"values='{props['values']}'")
            if 'colorscale' in props:
                params.append(f"colorscale='{props['colorscale']}'")

        elif chart_type == ChartType.SCATTER3D:
            if 'x' in props and props['x']:
                params.append(f"x='{props['x']}'")
            if 'y' in props and props['y']:
                params.append(f"y='{props['y']}'")
            if 'z' in props and props['z']:
                params.append(f"z='{props['z']}'")
            if 'color' in props and props['color']:
                params.append(f"color='{props['color']}'")

        elif chart_type == ChartType.BUBBLE:
            if 'x' in props and props['x']:
                params.append(f"x='{props['x']}'")
            if 'y' in props and props['y']:
                params.append(f"y='{props['y']}'")
            if 'size' in props and props['size']:
                params.append(f"size='{props['size']}'")
            if 'color' in props and props['color']:
                params.append(f"color='{props['color']}'")

        elif chart_type == ChartType.CORRELATION_MATRIX:
            if 'columns' in props and props['columns']:
                cols = props['columns']
                params.append(f"columns={cols}")
            if 'method' in props:
                params.append(f"method='{props['method']}'")

        # Add common parameters
        if 'title' in props and props['title']:
            params.append(f"title='{props['title']}'")

        if 'width' in props:
            params.append(f"width={props['width']}")

        if 'height' in props:
            params.append(f"height={props['height']}")

        # Construct function call
        params_str = ",\n    ".join(params)
        lines.append(f"chart = vz.{function_name}(")
        lines.append(f"    {params_str}")
        lines.append(")")

        # Apply theme if specified
        if 'theme' in props and props['theme'] and props['theme'] != 'default':
            lines.append("")
            lines.append(f"# Apply theme")
            lines.append(f"vz.set_theme('{props['theme']}')")

        return lines

    def _get_function_name(self, chart_type: ChartType) -> str:
        """Get VizForge function name for chart type."""
        mapping = {
            ChartType.LINE: 'line',
            ChartType.BAR: 'bar',
            ChartType.AREA: 'area',
            ChartType.SCATTER: 'scatter',
            ChartType.PIE: 'pie',
            ChartType.DONUT: 'donut',
            ChartType.HISTOGRAM: 'histogram',
            ChartType.BOXPLOT: 'boxplot',
            ChartType.HEATMAP: 'heatmap',
            ChartType.BUBBLE: 'bubble',
            ChartType.WATERFALL: 'waterfall',
            ChartType.FUNNEL: 'funnel',
            ChartType.RADAR: 'radar',
            ChartType.SURFACE: 'surface',
            ChartType.SCATTER3D: 'scatter3d',
            ChartType.MESH3D: 'mesh3d',
            ChartType.CHOROPLETH: 'choropleth',
            ChartType.SCATTERGEO: 'scattergeo',
            ChartType.NETWORK: 'network_graph',
            ChartType.SANKEY: 'sankey',
            ChartType.TREE: 'tree',
            ChartType.VIOLIN: 'violin',
            ChartType.KDE: 'kde',
            ChartType.REGRESSION: 'regression',
            ChartType.CORRELATION_MATRIX: 'correlation_matrix',
            ChartType.TREEMAP: 'treemap',
            ChartType.SUNBURST: 'sunburst',
            ChartType.PARALLEL_COORDINATES: 'parallel_coordinates',
        }

        return mapping.get(chart_type, 'line')

    def generate_notebook(self, configs: List[ChartConfig]) -> str:
        """
        Generate Jupyter notebook code for multiple charts.

        Args:
            configs: List of chart configurations

        Returns:
            Notebook code as string
        """
        code_lines = []

        # Add imports once
        if configs:
            code_lines.extend(self._generate_imports(configs[0]))
            code_lines.append("")

        # Add data loading once (assuming same data source)
        if configs and configs[0].data_source:
            code_lines.extend(self._generate_data_loading(configs[0]))
            code_lines.append("")

        # Generate each chart
        for i, config in enumerate(configs, 1):
            code_lines.append(f"# Chart {i}: {config.title or config.chart_type.value}")
            code_lines.append("")

            # Filters for this chart
            if config.filters:
                code_lines.append(f"# Filter data for Chart {i}")
                code_lines.append(f"df_chart{i} = df.copy()")
                for filt in config.filters:
                    column = filt.get('column')
                    operator = filt.get('operator')
                    value = filt.get('value')
                    if operator == 'equals':
                        if isinstance(value, str):
                            code_lines.append(f"df_chart{i} = df_chart{i}[df_chart{i}['{column}'] == '{value}']")
                        else:
                            code_lines.append(f"df_chart{i} = df_chart{i}[df_chart{i}['{column}'] == {value}]")
                code_lines.append("")
                data_var = f"df_chart{i}"
            else:
                data_var = "df"

            # Generate chart
            temp_config = ChartConfig(
                chart_type=config.chart_type,
                title=config.title,
                properties=config.properties.copy(),
                data_source=None,
                filters=[]
            )

            # Replace data=df with data=df_chartN
            chart_code = self._generate_chart_creation(temp_config)
            chart_code = [line.replace("data=df", f"data={data_var}") for line in chart_code]

            code_lines.extend(chart_code)
            code_lines.append(f"chart{i}.show()")
            code_lines.append("")
            code_lines.append("")

        return "\n".join(code_lines)
