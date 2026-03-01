"""
VizForge Predictive Analytics Module

Time series forecasting, trend detection, and anomaly detection.
NO API required - pure statistical models!

Examples:
    >>> import vizforge as vz
    >>> from vizforge.predictive import forecast, detect_anomalies
    >>>
    >>> # Forecast future values
    >>> predictions = forecast(df['sales'], periods=30)
    >>> vz.line(predictions, title='Sales Forecast')
    >>>
    >>> # Detect anomalies
    >>> anomalies = detect_anomalies(df['revenue'])
    >>> print(f"Found {len(anomalies)} anomalies")
"""

from .anomaly_detector import (
    Anomaly,
    AnomalyDetector,
    detect_anomalies,
)
from .forecaster import (
    ForecastResult,
    TimeSeriesForecaster,
    forecast,
)
from .seasonality import (
    SeasonalityAnalyzer,
    SeasonalPattern,
    analyze_seasonality,
)
from .trend_detector import (
    TrendDetector,
    TrendType,
    detect_trend,
)

__all__ = [
    # Forecasting
    'TimeSeriesForecaster',
    'forecast',
    'ForecastResult',

    # Trend Detection
    'TrendDetector',
    'detect_trend',
    'TrendType',

    # Anomaly Detection
    'AnomalyDetector',
    'detect_anomalies',
    'Anomaly',

    # Seasonality
    'SeasonalityAnalyzer',
    'analyze_seasonality',
    'SeasonalPattern',
]
