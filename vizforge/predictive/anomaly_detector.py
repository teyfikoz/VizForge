"""
VizForge Anomaly Detector

Detect outliers and anomalies in time series data.
NO API required - Statistical anomaly detection!

Methods:
- Z-Score (standard deviation)
- IQR (Interquartile Range)
- Isolation Forest (optional, requires scikit-learn)
- Moving Average deviation
"""

from dataclasses import dataclass
from typing import Union, Optional, List, Tuple
from enum import Enum
import numpy as np
import pandas as pd


class AnomalyMethod(Enum):
    """Anomaly detection methods."""
    AUTO = "auto"  # Auto-select best method
    ZSCORE = "zscore"  # Z-score method
    IQR = "iqr"  # Interquartile range
    MAD = "mad"  # Median Absolute Deviation
    MOVING_AVERAGE = "moving_average"  # Moving average deviation


@dataclass
class Anomaly:
    """
    Detected anomaly.

    Attributes:
        index: Index of anomaly in original data
        value: Anomalous value
        expected: Expected value (baseline)
        score: Anomaly score (higher = more anomalous)
        severity: Severity level ('low', 'medium', 'high')
    """
    index: int
    value: float
    expected: float
    score: float
    severity: str


class AnomalyDetector:
    """
    Anomaly detection engine.

    Detects outliers and anomalies in time series data
    using statistical methods.

    Examples:
        >>> detector = AnomalyDetector(df['sales'])
        >>> anomalies = detector.detect()
        >>> print(f"Found {len(anomalies)} anomalies")
        >>> for a in anomalies:
        >>>     print(f"Index {a.index}: {a.value} (expected {a.expected})")
    """

    def __init__(
        self,
        data: Union[pd.Series, np.ndarray, list],
        method: AnomalyMethod = AnomalyMethod.AUTO,
        sensitivity: float = 2.0,
    ):
        """
        Initialize anomaly detector.

        Args:
            data: Time series data
            method: Detection method (default: AUTO)
            sensitivity: Sensitivity threshold (lower = more sensitive)
                       For Z-score: 2.0 = ~95% confidence, 3.0 = ~99.7% confidence
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            self.data = data.values
            self.index = data.index
        elif isinstance(data, list):
            self.data = np.array(data)
            self.index = np.arange(len(data))
        else:
            self.data = data
            self.index = np.arange(len(data))

        self.method = method
        self.sensitivity = sensitivity

        # Validate
        if len(self.data) < 3:
            raise ValueError("Need at least 3 data points for anomaly detection")

        # Handle missing values
        self.data = pd.Series(self.data).fillna(method='ffill').fillna(method='bfill').values

    def detect(self) -> List[Anomaly]:
        """
        Detect anomalies in time series.

        Returns:
            List of detected anomalies
        """
        if self.method == AnomalyMethod.AUTO:
            method = self._auto_select_method()
        else:
            method = self.method

        # Detect anomalies based on method
        if method == AnomalyMethod.ZSCORE:
            return self._detect_zscore()
        elif method == AnomalyMethod.IQR:
            return self._detect_iqr()
        elif method == AnomalyMethod.MAD:
            return self._detect_mad()
        elif method == AnomalyMethod.MOVING_AVERAGE:
            return self._detect_moving_average()
        else:
            # Default to Z-score
            return self._detect_zscore()

    def _auto_select_method(self) -> AnomalyMethod:
        """Auto-select best anomaly detection method."""
        n = len(self.data)

        # For small datasets, use IQR (more robust)
        if n < 20:
            return AnomalyMethod.IQR

        # For larger datasets with potential drift, use moving average
        if n > 100:
            return AnomalyMethod.MOVING_AVERAGE

        # Default to Z-score for medium datasets
        return AnomalyMethod.ZSCORE

    def _detect_zscore(self) -> List[Anomaly]:
        """Z-score anomaly detection."""
        mean = np.mean(self.data)
        std = np.std(self.data)

        if std == 0:
            return []  # No variation, no anomalies

        # Calculate Z-scores
        z_scores = np.abs((self.data - mean) / std)

        # Find anomalies (beyond sensitivity threshold)
        anomaly_indices = np.where(z_scores > self.sensitivity)[0]

        # Create anomaly objects
        anomalies = []
        for idx in anomaly_indices:
            score = z_scores[idx]
            severity = self._classify_severity(score, [2, 3, 4])

            anomalies.append(Anomaly(
                index=int(idx),
                value=self.data[idx],
                expected=mean,
                score=score,
                severity=severity
            ))

        return anomalies

    def _detect_iqr(self) -> List[Anomaly]:
        """IQR (Interquartile Range) anomaly detection."""
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1

        if iqr == 0:
            return []  # No variation, no anomalies

        # Calculate bounds (using sensitivity as multiplier)
        lower_bound = q1 - self.sensitivity * iqr
        upper_bound = q3 + self.sensitivity * iqr

        # Find anomalies
        anomaly_indices = np.where((self.data < lower_bound) | (self.data > upper_bound))[0]

        # Calculate median as expected value
        median = np.median(self.data)

        # Create anomaly objects
        anomalies = []
        for idx in anomaly_indices:
            # Score based on distance from bounds
            if self.data[idx] < lower_bound:
                score = (lower_bound - self.data[idx]) / iqr
            else:
                score = (self.data[idx] - upper_bound) / iqr

            severity = self._classify_severity(score, [1, 2, 3])

            anomalies.append(Anomaly(
                index=int(idx),
                value=self.data[idx],
                expected=median,
                score=score,
                severity=severity
            ))

        return anomalies

    def _detect_mad(self) -> List[Anomaly]:
        """MAD (Median Absolute Deviation) anomaly detection."""
        median = np.median(self.data)
        mad = np.median(np.abs(self.data - median))

        if mad == 0:
            return []  # No variation, no anomalies

        # Modified Z-score using MAD
        modified_z = 0.6745 * (self.data - median) / mad

        # Find anomalies
        anomaly_indices = np.where(np.abs(modified_z) > self.sensitivity)[0]

        # Create anomaly objects
        anomalies = []
        for idx in anomaly_indices:
            score = np.abs(modified_z[idx])
            severity = self._classify_severity(score, [2, 3, 4])

            anomalies.append(Anomaly(
                index=int(idx),
                value=self.data[idx],
                expected=median,
                score=score,
                severity=severity
            ))

        return anomalies

    def _detect_moving_average(self) -> List[Anomaly]:
        """Moving average deviation anomaly detection."""
        # Use 7-period moving average
        window = min(7, len(self.data) // 3)
        if window < 2:
            window = 2

        # Calculate moving average and std
        ma = pd.Series(self.data).rolling(window=window, center=True).mean()
        ma_std = pd.Series(self.data).rolling(window=window, center=True).std()

        # Fill NaN values
        ma = ma.fillna(method='bfill').fillna(method='ffill')
        ma_std = ma_std.fillna(method='bfill').fillna(method='ffill')

        # Calculate deviation from moving average
        deviation = np.abs(self.data - ma.values)
        threshold = self.sensitivity * ma_std.values

        # Find anomalies
        anomaly_indices = np.where(deviation > threshold)[0]

        # Create anomaly objects
        anomalies = []
        for idx in anomaly_indices:
            if ma_std.values[idx] > 0:
                score = deviation[idx] / ma_std.values[idx]
            else:
                score = deviation[idx]

            severity = self._classify_severity(score, [2, 3, 4])

            anomalies.append(Anomaly(
                index=int(idx),
                value=self.data[idx],
                expected=ma.values[idx],
                score=score,
                severity=severity
            ))

        return anomalies

    def _classify_severity(self, score: float, thresholds: List[float]) -> str:
        """Classify anomaly severity based on score."""
        if score < thresholds[0]:
            return 'low'
        elif score < thresholds[1]:
            return 'medium'
        elif score < thresholds[2]:
            return 'high'
        else:
            return 'critical'


# ==================== Convenience Function ====================

def detect_anomalies(
    data: Union[pd.Series, np.ndarray, list],
    method: Union[str, AnomalyMethod] = AnomalyMethod.AUTO,
    sensitivity: float = 2.0
) -> List[Anomaly]:
    """
    Detect anomalies in time series (one-liner!).

    Args:
        data: Time series data
        method: Detection method ('auto', 'zscore', 'iqr', 'mad', 'moving_average')
        sensitivity: Sensitivity threshold (lower = more sensitive)

    Returns:
        List of detected anomalies

    Examples:
        >>> from vizforge.predictive import detect_anomalies
        >>>
        >>> anomalies = detect_anomalies(df['sales'])
        >>> print(f"Found {len(anomalies)} anomalies")
        >>>
        >>> for a in anomalies:
        >>>     print(f"Day {a.index}: ${a.value} (expected ${a.expected})")
        >>>     print(f"  Severity: {a.severity}")
        >>>
        >>> # More sensitive detection
        >>> anomalies = detect_anomalies(df['sales'], sensitivity=1.5)
        >>>
        >>> # Specific method
        >>> anomalies = detect_anomalies(df['sales'], method='iqr')
    """
    # Convert method string to enum
    if isinstance(method, str):
        method = AnomalyMethod(method)

    detector = AnomalyDetector(data, method=method, sensitivity=sensitivity)
    return detector.detect()
