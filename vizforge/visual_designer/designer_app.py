"""
Flask web application for Visual Chart Designer.

Provides a web-based UI for building charts visually.
"""

import json
import os
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
import plotly

from .chart_config import ChartConfig, ChartType, CHART_CATEGORIES
from .code_generator import CodeGenerator


class DesignerApp:
    """Visual Chart Designer web application."""

    def __init__(self, host: str = 'localhost', port: int = 5000):
        """
        Initialize designer app.

        Args:
            host: Host address
            port: Port number
        """
        self.host = host
        self.port = port
        self.app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static')
        )

        self.code_generator = CodeGenerator()
        self.current_data: Optional[pd.DataFrame] = None
        self.current_data_source: Optional[str] = None

        self._setup_routes()

    def _setup_routes(self):
        """Set up Flask routes."""

        @self.app.route('/')
        def index():
            """Main designer page."""
            return render_template('designer.html')

        @self.app.route('/api/chart_types', methods=['GET'])
        def get_chart_types():
            """Get available chart types by category."""
            categories = {}
            for category, types in CHART_CATEGORIES.items():
                categories[category] = [
                    {
                        'value': ct.value,
                        'label': ct.value.replace('_', ' ').title()
                    }
                    for ct in types
                ]
            return jsonify(categories)

        @self.app.route('/api/upload_data', methods=['POST'])
        def upload_data():
            """Upload data file."""
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            try:
                # Read file based on extension
                filename = file.filename.lower()

                if filename.endswith('.csv'):
                    self.current_data = pd.read_csv(file)
                elif filename.endswith(('.xlsx', '.xls')):
                    self.current_data = pd.read_excel(file)
                elif filename.endswith('.json'):
                    self.current_data = pd.read_json(file)
                elif filename.endswith('.parquet'):
                    self.current_data = pd.read_parquet(file)
                else:
                    return jsonify({'error': 'Unsupported file type'}), 400

                self.current_data_source = file.filename

                # Return data info
                return jsonify({
                    'success': True,
                    'filename': file.filename,
                    'rows': len(self.current_data),
                    'columns': list(self.current_data.columns),
                    'column_types': {
                        col: str(self.current_data[col].dtype)
                        for col in self.current_data.columns
                    },
                    'preview': self.current_data.head(5).to_dict('records')
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/data_info', methods=['GET'])
        def get_data_info():
            """Get current data information."""
            if self.current_data is None:
                return jsonify({'error': 'No data loaded'}), 400

            return jsonify({
                'filename': self.current_data_source,
                'rows': len(self.current_data),
                'columns': list(self.current_data.columns),
                'column_types': {
                    col: str(self.current_data[col].dtype)
                    for col in self.current_data.columns
                },
                'numeric_columns': list(self.current_data.select_dtypes(include=['number']).columns),
                'categorical_columns': list(self.current_data.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(self.current_data.select_dtypes(include=['datetime']).columns),
            })

        @self.app.route('/api/chart_properties', methods=['POST'])
        def get_chart_properties():
            """Get properties for a chart type."""
            data = request.json
            chart_type_str = data.get('chart_type')

            if not chart_type_str:
                return jsonify({'error': 'No chart_type provided'}), 400

            try:
                chart_type = ChartType(chart_type_str)
                properties = ChartConfig.get_available_properties(chart_type)

                return jsonify({
                    'properties': [
                        {
                            'name': p.name,
                            'type': p.type.value,
                            'label': p.label,
                            'default': p.default,
                            'required': p.required,
                            'options': p.options,
                            'min_value': p.min_value,
                            'max_value': p.max_value,
                            'description': p.description
                        }
                        for p in properties
                    ]
                })

            except ValueError:
                return jsonify({'error': 'Invalid chart type'}), 400

        @self.app.route('/api/preview_chart', methods=['POST'])
        def preview_chart():
            """Preview chart with current configuration."""
            if self.current_data is None:
                return jsonify({'error': 'No data loaded'}), 400

            try:
                data = request.json
                config = ChartConfig.from_dict(data)

                # Validate config
                valid, error = config.validate()
                if not valid:
                    return jsonify({'error': error}), 400

                # Generate chart
                chart_html = self._create_chart(config)

                return jsonify({
                    'success': True,
                    'html': chart_html
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/generate_code', methods=['POST'])
        def generate_code():
            """Generate Python code for chart configuration."""
            try:
                data = request.json
                config = ChartConfig.from_dict(data)

                # Generate code
                code = self.code_generator.generate(
                    config,
                    include_imports=data.get('include_imports', True),
                    include_data_loading=data.get('include_data_loading', True)
                )

                return jsonify({
                    'success': True,
                    'code': code
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/export_chart', methods=['POST'])
        def export_chart():
            """Export chart as image."""
            if self.current_data is None:
                return jsonify({'error': 'No data loaded'}), 400

            try:
                data = request.json
                config = ChartConfig.from_dict(data)
                export_format = data.get('format', 'png')

                # Create chart
                fig = self._create_chart_figure(config)

                # Export to image
                img_bytes = io.BytesIO()
                fig.write_image(img_bytes, format=export_format)
                img_bytes.seek(0)

                # Encode to base64
                img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

                return jsonify({
                    'success': True,
                    'image': f"data:image/{export_format};base64,{img_base64}"
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def _create_chart(self, config: ChartConfig) -> str:
        """Create chart HTML from configuration."""
        fig = self._create_chart_figure(config)
        return plotly.io.to_html(fig, include_plotlyjs='cdn', div_id='chart-preview')

    def _create_chart_figure(self, config: ChartConfig):
        """Create Plotly figure from configuration."""
        import vizforge as vz

        # Apply theme if specified
        theme = config.properties.get('theme', 'default')
        if theme and theme != 'default':
            vz.set_theme(theme)

        # Apply filters
        data = self.current_data.copy()
        for filt in config.filters:
            column = filt.get('column')
            operator = filt.get('operator')
            value = filt.get('value')

            if operator == 'equals':
                data = data[data[column] == value]
            elif operator == 'not_equals':
                data = data[data[column] != value]
            elif operator == 'greater_than':
                data = data[data[column] > value]
            elif operator == 'less_than':
                data = data[data[column] < value]
            elif operator == 'contains':
                data = data[data[column].str.contains(value, na=False)]
            elif operator == 'in':
                data = data[data[column].isin(value)]

        # Build chart parameters
        params = {'data': data}

        # Add chart-specific parameters
        props = config.properties
        chart_type = config.chart_type

        if chart_type in [ChartType.LINE, ChartType.BAR, ChartType.AREA, ChartType.SCATTER]:
            if 'x' in props and props['x']:
                params['x'] = props['x']
            if 'y' in props and props['y']:
                params['y'] = props['y']
            if 'color' in props and props['color']:
                params['color'] = props['color']

        elif chart_type in [ChartType.PIE, ChartType.DONUT]:
            if 'labels' in props and props['labels']:
                params['labels'] = props['labels']
            if 'values' in props and props['values']:
                params['values'] = props['values']

        elif chart_type == ChartType.HISTOGRAM:
            if 'x' in props and props['x']:
                params['x'] = props['x']
            if 'bins' in props:
                params['bins'] = props['bins']

        elif chart_type == ChartType.BOXPLOT:
            if 'y' in props and props['y']:
                params['y'] = props['y']
            if 'x' in props and props['x']:
                params['x'] = props['x']

        elif chart_type == ChartType.HEATMAP:
            if 'x' in props and props['x']:
                params['x'] = props['x']
            if 'y' in props and props['y']:
                params['y'] = props['y']
            if 'values' in props and props['values']:
                params['values'] = props['values']

        elif chart_type == ChartType.SCATTER3D:
            if 'x' in props and props['x']:
                params['x'] = props['x']
            if 'y' in props and props['y']:
                params['y'] = props['y']
            if 'z' in props and props['z']:
                params['z'] = props['z']
            if 'color' in props and props['color']:
                params['color'] = props['color']

        elif chart_type == ChartType.BUBBLE:
            if 'x' in props and props['x']:
                params['x'] = props['x']
            if 'y' in props and props['y']:
                params['y'] = props['y']
            if 'size' in props and props['size']:
                params['size'] = props['size']
            if 'color' in props and props['color']:
                params['color'] = props['color']

        # Add common parameters
        if 'title' in props and props['title']:
            params['title'] = props['title']
        if 'width' in props:
            params['width'] = props['width']
        if 'height' in props:
            params['height'] = props['height']

        # Create chart using VizForge
        function_map = {
            ChartType.LINE: vz.line,
            ChartType.BAR: vz.bar,
            ChartType.AREA: vz.area,
            ChartType.SCATTER: vz.scatter,
            ChartType.PIE: vz.pie,
            ChartType.DONUT: vz.donut,
            ChartType.HISTOGRAM: vz.histogram,
            ChartType.BOXPLOT: vz.boxplot,
            ChartType.HEATMAP: vz.heatmap,
            ChartType.BUBBLE: vz.bubble,
            ChartType.WATERFALL: vz.waterfall,
            ChartType.FUNNEL: vz.funnel,
            ChartType.RADAR: vz.radar,
            ChartType.SCATTER3D: vz.scatter3d,
            ChartType.VIOLIN: vz.violin,
            ChartType.KDE: vz.kde,
            ChartType.REGRESSION: vz.regression,
            ChartType.CORRELATION_MATRIX: vz.correlation_matrix,
            ChartType.TREEMAP: vz.treemap,
            ChartType.SUNBURST: vz.sunburst,
        }

        chart_func = function_map.get(chart_type, vz.line)
        chart = chart_func(**params)

        return chart.fig

    def run(self, debug: bool = True):
        """
        Run the designer app.

        Args:
            debug: Enable debug mode
        """
        print(f"\n{'='*60}")
        print("🎨 VizForge Visual Chart Designer")
        print(f"{'='*60}")
        print(f"\n✅ Server starting at: http://{self.host}:{self.port}")
        print("\n📖 Usage:")
        print("  1. Upload your data (CSV, Excel, JSON, Parquet)")
        print("  2. Drag a chart type from the library")
        print("  3. Configure properties in the panel")
        print("  4. Preview in real-time")
        print("  5. Export Python code or image")
        print(f"\n{'='*60}\n")

        self.app.run(host=self.host, port=self.port, debug=debug)


def launch_designer(host: str = 'localhost', port: int = 5000, debug: bool = True):
    """
    Launch Visual Chart Designer.

    Args:
        host: Host address (default: localhost)
        port: Port number (default: 5000)
        debug: Enable debug mode (default: True)

    Example:
        >>> import vizforge as vz
        >>> vz.launch_designer()
        # Opens http://localhost:5000 in browser
    """
    app = DesignerApp(host=host, port=port)
    app.run(debug=debug)
