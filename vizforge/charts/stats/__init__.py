"""Statistical chart types for VizForge."""

from .correlation import CorrelationMatrix, correlation_matrix
from .feature_importance import (
    FeatureImportance,
    PermutationImportance,
    feature_importance,
    permutation_importance,
)
from .kde import KDE2D, KDEPlot, kde, kde2d
from .regression import RegressionPlot, regression
from .roc import MultiROCCurve, ROCCurve, multi_roc_curve, roc_curve_plot
from .violin import ViolinPlot, violin

__all__ = [
    # Classes
    "ViolinPlot",
    "KDEPlot",
    "KDE2D",
    "RegressionPlot",
    "CorrelationMatrix",
    "ROCCurve",
    "MultiROCCurve",
    "FeatureImportance",
    "PermutationImportance",
    # Functions
    "violin",
    "kde",
    "kde2d",
    "regression",
    "correlation_matrix",
    "roc_curve_plot",
    "multi_roc_curve",
    "feature_importance",
    "permutation_importance",
]
