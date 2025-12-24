"""
VizForge Trend Detector

Detect and analyze trends in time series data.
NO API required - Statistical trend analysis!
"""

from dataclasses import dataclass
from typing import Union, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class TrendType(Enum):
    """Types of trends."""
    STRONG_UPWARD = "strong_upward"  # Strong increasing trend
    MODERATE_UPWARD = "moderate_upward"  # Moderate increasing trend
    WEAK_UPWARD = "weak_upward"  # Weak increasing trend
    FLAT = "flat"  # No trend
    WEAK_DOWNWARD = "weak_downward"  # Weak decreasing trend
    MODERATE_DOWNWARD = "moderate_downward"  # Moderate decreasing trend
    STRONG_DOWNWARD = "strong_downward"  # Strong decreasing trend
    VOLATILE = "volatile"  # High volatility, unclear trend
    CYCLICAL = "cyclical"  # Cyclical pattern detected


@dataclass
class TrendResult:
    """
    Trend detection result.

    Attributes:
        trend_type: Detected trend type
        slope: Trend slope (rate of change)
        strength: Trend strength (0.0 to 1.0)
        confidence: Confidence level (0.0 to 1.0)
        r_squared: R-squared value
        volatility: Data volatility
        turning_points: Indices of turning points
    """
    trend_type: TrendType
    slope: float
    strength: float
    confidence: float
    r_squared: float
    volatility: float
    turning_points: list


class TrendDetector:
    """
    Trend detection and analysis.

    Detects trends, calculates trend strength, and identifies
    turning points in time series data.

    Examples:
        >>> detector = TrendDetector(df['sales'])
        >>> result = detector.detect()
        >>> print(f"Trend: {result.trend_type.value}")
        >>> print(f"Slope: {result.slope:.2f}")
        >>> print(f"Strength: {result.strength:.2%}")
    """

    def __init__(self, data: Union[pd.Series, np.ndarray, list]):
        """
        Initialize trend detector.

        Args:
            data: Time series data
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            self.data = data.values
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data

        # Validate
        if len(self.data) < 3:
            raise ValueError("Need at least 3 data points for trend detection")

        # Handle missing values
        self.data = pd.Series(self.data).ffill().bfill().values

    def detect(self) -> TrendResult:
        """
        Detect trend in time series.

        Returns:
            TrendResult with trend type, slope, strength, etc.
        """
        # Calculate trend metrics
        slope, r_squared = self._calculate_linear_trend()
        volatility = self._calculate_volatility()
        turning_points = self._find_turning_points()

        # Calculate trend strength (based on R-squared and slope magnitude)
        strength = self._calculate_strength(slope, r_squared, volatility)

        # Determine trend type
        trend_type = self._classify_trend(slope, r_squared, volatility)

        # Calculate confidence (based on R-squared and sample size)
        confidence = self._calculate_confidence(r_squared, len(self.data))

        return TrendResult(
            trend_type=trend_type,
            slope=slope,
            strength=strength,
            confidence=confidence,
            r_squared=r_squared,
            volatility=volatility,
            turning_points=turning_points
        )

    def _calculate_linear_trend(self) -> Tuple[float, float]:
        """Calculate linear trend slope and R-squared."""
        x = np.arange(len(self.data))
        y = self.data

        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Calculate R-squared
        fitted = slope * x + intercept
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Normalize slope by data mean to make it scale-independent
        mean_val = np.mean(y)
        normalized_slope = (slope / mean_val) if mean_val != 0 else slope

        return normalized_slope, r_squared

    def _calculate_volatility(self) -> float:
        """Calculate data volatility (coefficient of variation)."""
        std = np.std(self.data)
        mean = np.mean(self.data)

        if mean == 0:
            return 0.0

        cv = std / abs(mean)  # Coefficient of variation
        return cv

    def _find_turning_points(self) -> list:
        """Find turning points (local maxima and minima)."""
        turning_points = []

        for i in range(1, len(self.data) - 1):
            # Local maximum
            if self.data[i] > self.data[i-1] and self.data[i] > self.data[i+1]:
                turning_points.append(i)
            # Local minimum
            elif self.data[i] < self.data[i-1] and self.data[i] < self.data[i+1]:
                turning_points.append(i)

        return turning_points

    def _calculate_strength(self, slope: float, r_squared: float, volatility: float) -> float:
        """Calculate trend strength (0.0 to 1.0)."""
        # Strength is combination of:
        # 1. R-squared (how well linear model fits)
        # 2. Slope magnitude (how steep the trend)
        # 3. Low volatility (stable trend)

        slope_component = min(abs(slope) * 10, 1.0)  # Normalize slope contribution
        r_squared_component = r_squared
        volatility_component = max(0, 1 - volatility)  # Lower volatility = stronger trend

        # Weighted average
        strength = (
            0.4 * r_squared_component +
            0.3 * slope_component +
            0.3 * volatility_component
        )

        return min(strength, 1.0)

    def _classify_trend(self, slope: float, r_squared: float, volatility: float) -> TrendType:
        """Classify trend type."""
        # High volatility = volatile trend
        if volatility > 0.5:
            return TrendType.VOLATILE

        # Check if cyclical (many turning points)
        turning_points_ratio = len(self._find_turning_points()) / len(self.data)
        if turning_points_ratio > 0.3:
            return TrendType.CYCLICAL

        # Weak trend if R-squared is low
        if r_squared < 0.3:
            return TrendType.FLAT

        # Classify based on slope and R-squared
        abs_slope = abs(slope)

        # Strong trend thresholds
        if r_squared > 0.7:
            if slope > 0.05:
                return TrendType.STRONG_UPWARD
            elif slope < -0.05:
                return TrendType.STRONG_DOWNWARD

        # Moderate trend thresholds
        if r_squared > 0.5:
            if slope > 0.02:
                return TrendType.MODERATE_UPWARD
            elif slope < -0.02:
                return TrendType.MODERATE_DOWNWARD

        # Weak trend thresholds
        if r_squared > 0.3:
            if slope > 0.01:
                return TrendType.WEAK_UPWARD
            elif slope < -0.01:
                return TrendType.WEAK_DOWNWARD

        # Default to flat
        return TrendType.FLAT

    def _calculate_confidence(self, r_squared: float, n: int) -> float:
        """Calculate confidence level."""
        # Confidence increases with:
        # 1. Higher R-squared
        # 2. More data points

        r_squared_component = r_squared

        # Sample size component (diminishing returns after 100 points)
        sample_component = min(n / 100, 1.0)

        # Weighted average
        confidence = 0.7 * r_squared_component + 0.3 * sample_component

        return min(confidence, 1.0)


# ==================== Convenience Function ====================

def detect_trend(data: Union[pd.Series, np.ndarray, list]) -> TrendResult:
    """
    Detect trend in time series (one-liner!).

    Args:
        data: Time series data

    Returns:
        TrendResult with trend analysis

    Examples:
        >>> from vizforge.predictive import detect_trend
        >>>
        >>> result = detect_trend(df['sales'])
        >>> print(f"Trend: {result.trend_type.value}")
        >>> print(f"Slope: {result.slope:.3f} per period")
        >>> print(f"Strength: {result.strength:.0%}")
        >>> print(f"Confidence: {result.confidence:.0%}")
        >>>
        >>> if result.trend_type.value.endswith('upward'):
        >>>     print("📈 Growing!")
        >>> elif result.trend_type.value.endswith('downward'):
        >>>     print("📉 Declining!")
    """
    detector = TrendDetector(data)
    return detector.detect()
