"""
VizForge Intelligence Module v2.0

Visual intelligence for smart chart selection and bias detection.
"""

from vizforge.intelligence.bias_detector import BiasReport, VisualBiasDetector
from vizforge.intelligence.reasoning import ChartDecision, ChartReasoningEngine

__all__ = [
    "ChartReasoningEngine",
    "ChartDecision",
    "VisualBiasDetector",
    "BiasReport"
]
