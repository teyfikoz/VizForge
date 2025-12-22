"""
VizForge Seasonality Analyzer

Detect and analyze seasonal patterns in time series data.
NO API required - Statistical seasonality detection!
"""

from dataclasses import dataclass
from typing import Union, Optional, List, Dict
from enum import Enum
import numpy as np
import pandas as pd


class SeasonalityType(Enum):
    """Types of seasonality."""
    NONE = "none"  # No seasonality detected
    DAILY = "daily"  # Daily pattern (24-hour cycle)
    WEEKLY = "weekly"  # Weekly pattern (7-day cycle)
    MONTHLY = "monthly"  # Monthly pattern (30-day cycle)
    QUARTERLY = "quarterly"  # Quarterly pattern (90-day cycle)
    YEARLY = "yearly"  # Yearly pattern (365-day cycle)
    CUSTOM = "custom"  # Custom period detected


@dataclass
class SeasonalPattern:
    """
    Detected seasonal pattern.

    Attributes:
        type: Seasonality type
        period: Seasonal period (number of time units)
        strength: Seasonality strength (0.0 to 1.0)
        confidence: Confidence level (0.0 to 1.0)
        peak_indices: Indices of seasonal peaks
        trough_indices: Indices of seasonal troughs
        decomposition: Trend, seasonal, and residual components
    """
    type: SeasonalityType
    period: int
    strength: float
    confidence: float
    peak_indices: List[int]
    trough_indices: List[int]
    decomposition: Dict[str, np.ndarray]


class SeasonalityAnalyzer:
    """
    Seasonality detection and analysis.

    Detects seasonal patterns, calculates seasonality strength,
    and decomposes time series into trend + seasonal + residual.

    Examples:
        >>> analyzer = SeasonalityAnalyzer(df['sales'])
        >>> pattern = analyzer.analyze()
        >>> print(f"Seasonality: {pattern.type.value}")
        >>> print(f"Period: {pattern.period} units")
        >>> print(f"Strength: {pattern.strength:.2%}")
    """

    def __init__(
        self,
        data: Union[pd.Series, np.ndarray, list],
        max_period: Optional[int] = None,
    ):
        """
        Initialize seasonality analyzer.

        Args:
            data: Time series data
            max_period: Maximum period to test (default: len(data) // 3)
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            self.data = data.values
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data

        # Set max period
        if max_period is None:
            self.max_period = len(self.data) // 3
        else:
            self.max_period = min(max_period, len(self.data) // 3)

        # Validate
        if len(self.data) < 6:
            raise ValueError("Need at least 6 data points for seasonality analysis")

        # Handle missing values
        self.data = pd.Series(self.data).fillna(method='ffill').fillna(method='bfill').values

    def analyze(self) -> SeasonalPattern:
        """
        Analyze seasonality in time series.

        Returns:
            SeasonalPattern with detected seasonality info
        """
        # Detect seasonal period
        period, strength, confidence = self._detect_period()

        # Classify seasonality type
        seasonality_type = self._classify_seasonality(period)

        # Decompose time series
        decomposition = self._decompose(period)

        # Find peaks and troughs in seasonal component
        peaks = self._find_peaks(decomposition['seasonal'])
        troughs = self._find_troughs(decomposition['seasonal'])

        return SeasonalPattern(
            type=seasonality_type,
            period=period,
            strength=strength,
            confidence=confidence,
            peak_indices=peaks,
            trough_indices=troughs,
            decomposition=decomposition
        )

    def _detect_period(self) -> tuple:
        """Detect seasonal period using autocorrelation."""
        # Test common periods first
        common_periods = [7, 12, 24, 30, 90, 365]
        common_periods = [p for p in common_periods if p <= self.max_period]

        best_period = None
        best_score = 0
        best_confidence = 0

        # Test common periods
        for period in common_periods:
            if len(self.data) < 2 * period:
                continue

            score = self._autocorrelation(period)

            if score > best_score:
                best_score = score
                best_period = period
                best_confidence = score

        # If no strong common period, scan all periods
        if best_score < 0.5:
            for period in range(2, min(self.max_period + 1, len(self.data) // 2)):
                score = self._autocorrelation(period)

                if score > best_score:
                    best_score = score
                    best_period = period
                    best_confidence = score

        # If still no period found, return no seasonality
        if best_period is None or best_score < 0.3:
            return 1, 0.0, 0.0

        return best_period, best_score, best_confidence

    def _autocorrelation(self, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        n = len(self.data)

        if lag >= n or lag < 1:
            return 0.0

        # Calculate autocorrelation
        y = self.data - np.mean(self.data)
        c0 = np.dot(y, y) / n

        if c0 == 0:
            return 0.0

        c_lag = np.dot(y[:-lag], y[lag:]) / n
        acf = c_lag / c0

        return abs(acf)

    def _classify_seasonality(self, period: int) -> SeasonalityType:
        """Classify seasonality type based on period."""
        if period == 1:
            return SeasonalityType.NONE
        elif period == 7:
            return SeasonalityType.WEEKLY
        elif 10 <= period <= 14:
            return SeasonalityType.WEEKLY  # Approximately weekly
        elif period == 24:
            return SeasonalityType.DAILY
        elif 20 <= period <= 35:
            return SeasonalityType.MONTHLY
        elif 85 <= period <= 95:
            return SeasonalityType.QUARTERLY
        elif 360 <= period <= 370:
            return SeasonalityType.YEARLY
        else:
            return SeasonalityType.CUSTOM

    def _decompose(self, period: int) -> Dict[str, np.ndarray]:
        """Decompose time series into trend + seasonal + residual."""
        if period <= 1:
            # No seasonality, just trend + residual
            trend = self._calculate_trend()
            seasonal = np.zeros_like(self.data)
            residual = self.data - trend
        else:
            # Classical decomposition
            trend = self._calculate_trend()
            detrended = self.data - trend
            seasonal = self._extract_seasonal(detrended, period)
            residual = self.data - trend - seasonal

        return {
            'original': self.data.copy(),
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }

    def _calculate_trend(self) -> np.ndarray:
        """Calculate trend using moving average."""
        # Use 7-period moving average
        window = min(7, len(self.data) // 3)
        if window < 2:
            window = 2

        trend = pd.Series(self.data).rolling(window=window, center=True).mean()
        trend = trend.fillna(method='bfill').fillna(method='ffill').values

        return trend

    def _extract_seasonal(self, detrended: np.ndarray, period: int) -> np.ndarray:
        """Extract seasonal component."""
        n = len(detrended)

        # Calculate average for each phase in the cycle
        seasonal_avg = np.zeros(period)

        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                seasonal_avg[i] = np.mean(detrended[indices])

        # Normalize seasonal component (center around 0)
        seasonal_avg = seasonal_avg - np.mean(seasonal_avg)

        # Repeat seasonal pattern to match data length
        seasonal = np.tile(seasonal_avg, n // period + 1)[:n]

        return seasonal

    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """Find peak indices in data."""
        peaks = []

        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)

        return peaks

    def _find_troughs(self, data: np.ndarray) -> List[int]:
        """Find trough indices in data."""
        troughs = []

        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                troughs.append(i)

        return troughs


# ==================== Convenience Function ====================

def analyze_seasonality(
    data: Union[pd.Series, np.ndarray, list],
    max_period: Optional[int] = None
) -> SeasonalPattern:
    """
    Analyze seasonality in time series (one-liner!).

    Args:
        data: Time series data
        max_period: Maximum period to test (optional)

    Returns:
        SeasonalPattern with detected seasonality info

    Examples:
        >>> from vizforge.predictive import analyze_seasonality
        >>>
        >>> pattern = analyze_seasonality(df['sales'])
        >>> print(f"Seasonality: {pattern.type.value}")
        >>> print(f"Period: {pattern.period} days")
        >>> print(f"Strength: {pattern.strength:.0%}")
        >>>
        >>> if pattern.type == SeasonalityType.WEEKLY:
        >>>     print("Weekly pattern detected!")
        >>>
        >>> # Visualize decomposition
        >>> import vizforge as vz
        >>> vz.line(pattern.decomposition['trend'], title='Trend Component')
        >>> vz.line(pattern.decomposition['seasonal'], title='Seasonal Component')
    """
    analyzer = SeasonalityAnalyzer(data, max_period=max_period)
    return analyzer.analyze()
