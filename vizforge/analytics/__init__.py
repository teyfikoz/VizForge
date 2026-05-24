"""
VizForge Analytics Module

Tableau-style analytics features for professional data analysis.
Part of VizForge v1.0.0 - Super AGI features.
"""

from .aggregations import Aggregation, AggregationEngine, WindowFunction
from .calculated_fields import CalculatedField, Expression, ExpressionParser
from .hierarchies import DrillPath, Hierarchy, HierarchyManager
from .parameters import Parameter, ParameterManager, ParameterType

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
