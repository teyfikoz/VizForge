"""
VizForge Analytics Module

Tableau-style analytics features for professional data analysis.
Part of VizForge v1.0.0 - Super AGI features.
"""

from .calculated_fields import CalculatedField, Expression, ExpressionParser
from .hierarchies import Hierarchy, DrillPath, HierarchyManager
from .aggregations import Aggregation, WindowFunction, AggregationEngine
from .parameters import Parameter, ParameterType, ParameterManager

__all__ = [
    # Calculated Fields
    'CalculatedField',
    'Expression',
    'ExpressionParser',

    # Hierarchies
    'Hierarchy',
    'DrillPath',
    'HierarchyManager',

    # Aggregations
    'Aggregation',
    'WindowFunction',
    'AggregationEngine',

    # Parameters
    'Parameter',
    'ParameterType',
    'ParameterManager',
]
