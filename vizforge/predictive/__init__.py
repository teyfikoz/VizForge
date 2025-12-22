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

from .forecaster import (
    TimeSeriesForecaster,
    forecast,
    ForecastResult,
)

from .trend_detector import (
    TrendDetector,
    detect_trend,
    TrendType,
)

from .anomaly_detector import (
    AnomalyDetector,
    detect_anomalies,
    Anomaly,
)

from .seasonality import (
    SeasonalityAnalyzer,
    analyze_seasonality,
    SeasonalPattern,
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
