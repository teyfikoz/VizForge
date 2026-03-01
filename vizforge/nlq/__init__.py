"""
VizForge Natural Language Query (NLQ) Engine

Ask questions in plain English, get automatic visualizations!
NO API required - pure rule-based NLP + pattern matching.

Examples:
    >>> chart = vz.ask("Show me sales trend by month")
    >>> chart = vz.ask("Compare revenue vs profit")
    >>> chart = vz.ask("Find top 10 products by sales")
"""

from .engine import NLQEngine, ask
from .entity_extractor import Entity, EntityExtractor
from .query_parser import Intent, QueryParser

__all__ = [
    'NLQEngine',
    'ask',
    'QueryParser',
    'Intent',
    'EntityExtractor',
    'Entity',
]
