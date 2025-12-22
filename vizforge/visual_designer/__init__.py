"""
VizForge Visual Chart Designer

A web-based drag-and-drop interface for creating charts visually.
No coding required - build professional charts through an intuitive UI.

Features:
- Drag-and-drop chart builder
- Live preview
- Property editor
- Code generation (exports to Python)
- Chart templates library
- Data upload/preview
"""

from .designer_app import DesignerApp, launch_designer
from .chart_config import ChartConfig, ChartType, PropertyType
from .code_generator import CodeGenerator

__all__ = [
    # Main entry point
    'launch_designer',

    # Core classes
    'DesignerApp',
    'ChartConfig',
    'ChartType',
    'PropertyType',
    'CodeGenerator',
]
