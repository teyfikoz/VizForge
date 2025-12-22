"""
VizForge Pattern Detector - Intelligent Pattern Recognition

NO API required! Pure statistical and mathematical pattern detection.
Detects time series patterns, correlations, clusters, anomalies, and distributions.

Part of VizForge v1.2.0 - ULTRA Intelligence Features
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class PatternType(Enum):
    """Types of patterns that can be detected."""
    TREND_INCREASING = "trend_increasing"
    TREND_DECREASING = "trend_decreasing"
    TREND_STABLE = "trend_stable"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    SPIKE = "spike"
    DIP = "dip"
    PLATEAU = "plateau"
    VOLATILITY_HIGH = "volatility_high"
    VOLATILITY_LOW = "volatility_low"
    CORRELATION_STRONG = "correlation_strong"
    CORRELATION_WEAK = "correlation_weak"
    CLUSTER = "cluster"
    ANOMALY = "anomaly"
    NORMAL_DISTRIBUTION = "normal_distribution"
    SKEWED_DISTRIBUTION = "skewed_distribution"


@dataclass
class Pattern:
    """A detected pattern with metadata."""
    pattern_type: PatternType
    confidence: float  # 0-1
    description: str
    location: Optional[Union[int, Tuple[int, int]]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PatternDetector:
    """
    Intelligent Pattern Detection Engine.

    Detects patterns in data using statistical and mathematical methods.
    NO API required - all processing is local and fast!

    Features:
    - Time series pattern detection
    - Correlation analysis
    - Cluster identification
    - Anomaly detection
    - Distribution analysis

    Example:
        >>> detector = PatternDetector(df)
        >>> patterns = detector.detect_all_patterns()
        >>> for p in patterns:
        ...     print(f"{p.pattern_type}: {p.description} (confidence: {p.confidence:.2f})")
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        confidence_threshold: float = 0.7
    ):
        """
        Initialize pattern detector.

        Args:
            data: Input data (DataFrame, Series, or numpy array)
            confidence_threshold: Minimum confidence for pattern detection (0-1)
        """
        self.data = self._prepare_data(data)
        self.confidence_threshold = confidence_threshold
        self.patterns: List[Pattern] = []

    def _prepare_data(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            return data.to_frame()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({'value': data})
            else:
                return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    # ==================== TIME SERIES PATTERNS ====================

    def detect_trend(self, column: str = None, method: str = 'linear') -> List[Pattern]:
        """
        Detect trends in time series data.

        Args:
            column: Column name (if None, uses first numeric column)
            method: 'linear', 'polynomial', or 'exponential'

        Returns:
            List of detected trend patterns
        """
        if column is None:
            column = self._get_numeric_columns()[0]

        values = self.data[column].dropna().values

        if len(values) < 3:
            return []

        patterns = []

        # Linear trend using least squares
        x = np.arange(len(values))

        if method == 'linear':
            # Fit linear regression
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]

            # Calculate R-squared
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Detect trend type
            if r_squared > 0.7:  # Strong fit
                if slope > 0:
                    pattern = Pattern(
                        pattern_type=PatternType.TREND_INCREASING,
                        confidence=r_squared,
                        description=f"Strong upward trend detected (slope: {slope:.4f})",
                        metadata={'slope': slope, 'r_squared': r_squared, 'method': 'linear'}
                    )
                elif slope < 0:
                    pattern = Pattern(
                        pattern_type=PatternType.TREND_DECREASING,
                        confidence=r_squared,
                        description=f"Strong downward trend detected (slope: {slope:.4f})",
                        metadata={'slope': slope, 'r_squared': r_squared, 'method': 'linear'}
                    )
                else:
                    pattern = Pattern(
                        pattern_type=PatternType.TREND_STABLE,
                        confidence=r_squared,
                        description="Stable trend detected (no significant change)",
                        metadata={'slope': slope, 'r_squared': r_squared, 'method': 'linear'}
                    )

                if pattern.confidence >= self.confidence_threshold:
                    patterns.append(pattern)

        return patterns

    def detect_seasonality(self, column: str = None, periods: List[int] = None) -> List[Pattern]:
        """
        Detect seasonal patterns using autocorrelation.

        Args:
            column: Column name
            periods: List of periods to check (default: [7, 30, 365] for daily data)

        Returns:
            List of detected seasonal patterns
        """
        if column is None:
            column = self._get_numeric_columns()[0]

        values = self.data[column].dropna().values

        if len(values) < 14:
            return []

        if periods is None:
            # Default periods for different data frequencies
            if len(values) > 365:
                periods = [7, 30, 90, 365]  # Daily data
            elif len(values) > 52:
                periods = [4, 12, 52]  # Weekly data
            else:
                periods = [int(len(values) / 4)]  # Quarterly

        patterns = []

        for period in periods:
            if period >= len(values):
                continue

            # Calculate autocorrelation at lag=period
            autocorr = self._autocorrelation(values, period)

            # Strong autocorrelation indicates seasonality
            if autocorr > 0.6:
                pattern = Pattern(
                    pattern_type=PatternType.SEASONAL,
                    confidence=autocorr,
                    description=f"Seasonal pattern detected (period: {period}, autocorr: {autocorr:.2f})",
                    metadata={'period': period, 'autocorrelation': autocorr}
                )

                if pattern.confidence >= self.confidence_threshold:
                    patterns.append(pattern)

        return patterns

    def detect_spikes_and_dips(self, column: str = None, threshold: float = 3.0) -> List[Pattern]:
        """
        Detect sudden spikes and dips in data.

        Args:
            column: Column name
            threshold: Z-score threshold for spike/dip detection

        Returns:
            List of detected spike/dip patterns
        """
        if column is None:
            column = self._get_numeric_columns()[0]

        values = self.data[column].dropna().values

        if len(values) < 3:
            return []

        # Calculate Z-scores
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return []

        z_scores = (values - mean) / std

        patterns = []

        # Detect spikes (positive outliers)
        spike_indices = np.where(z_scores > threshold)[0]
        for idx in spike_indices:
            confidence = min((z_scores[idx] - threshold) / threshold, 1.0)
            pattern = Pattern(
                pattern_type=PatternType.SPIKE,
                confidence=confidence,
                description=f"Spike detected at index {idx} (z-score: {z_scores[idx]:.2f})",
                location=int(idx),
                metadata={'z_score': float(z_scores[idx]), 'value': float(values[idx])}
            )

            if pattern.confidence >= self.confidence_threshold:
                patterns.append(pattern)

        # Detect dips (negative outliers)
        dip_indices = np.where(z_scores < -threshold)[0]
        for idx in dip_indices:
            confidence = min((-z_scores[idx] - threshold) / threshold, 1.0)
            pattern = Pattern(
                pattern_type=PatternType.DIP,
                confidence=confidence,
                description=f"Dip detected at index {idx} (z-score: {z_scores[idx]:.2f})",
                location=int(idx),
                metadata={'z_score': float(z_scores[idx]), 'value': float(values[idx])}
            )

            if pattern.confidence >= self.confidence_threshold:
                patterns.append(pattern)

        return patterns

    def detect_volatility(self, column: str = None, window: int = 20) -> List[Pattern]:
        """
        Detect high/low volatility periods.

        Args:
            column: Column name
            window: Rolling window size for volatility calculation

        Returns:
            List of detected volatility patterns
        """
        if column is None:
            column = self._get_numeric_columns()[0]

        values = self.data[column].dropna().values

        if len(values) < window * 2:
            return []

        # Calculate rolling standard deviation
        rolling_std = pd.Series(values).rolling(window=window).std().dropna().values

        # Overall volatility
        overall_std = np.std(values)

        # High volatility threshold
        high_threshold = overall_std * 1.5
        low_threshold = overall_std * 0.5

        patterns = []

        # Check current volatility
        current_volatility = rolling_std[-1]

        if current_volatility > high_threshold:
            confidence = min((current_volatility - high_threshold) / high_threshold, 1.0)
            pattern = Pattern(
                pattern_type=PatternType.VOLATILITY_HIGH,
                confidence=confidence,
                description=f"High volatility detected (std: {current_volatility:.4f})",
                metadata={'volatility': float(current_volatility), 'threshold': float(high_threshold)}
            )

            if pattern.confidence >= self.confidence_threshold:
                patterns.append(pattern)

        elif current_volatility < low_threshold:
            confidence = min((low_threshold - current_volatility) / low_threshold, 1.0)
            pattern = Pattern(
                pattern_type=PatternType.VOLATILITY_LOW,
                confidence=confidence,
                description=f"Low volatility detected (std: {current_volatility:.4f})",
                metadata={'volatility': float(current_volatility), 'threshold': float(low_threshold)}
            )

            if pattern.confidence >= self.confidence_threshold:
                patterns.append(pattern)

        return patterns

    # ==================== CORRELATION PATTERNS ====================

    def detect_correlations(self, method: str = 'pearson', threshold: float = 0.7) -> List[Pattern]:
        """
        Detect strong correlations between variables.

        Args:
            method: 'pearson', 'spearman', or 'kendall'
            threshold: Correlation threshold (0-1)

        Returns:
            List of detected correlation patterns
        """
        numeric_cols = self._get_numeric_columns()

        if len(numeric_cols) < 2:
            return []

        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr(method=method)

        patterns = []

        # Find strong correlations
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]

                if abs(corr) >= threshold:
                    confidence = abs(corr)

                    if corr > 0:
                        pattern_type = PatternType.CORRELATION_STRONG
                        direction = "positive"
                    else:
                        pattern_type = PatternType.CORRELATION_STRONG
                        direction = "negative"

                    pattern = Pattern(
                        pattern_type=pattern_type,
                        confidence=confidence,
                        description=f"Strong {direction} correlation between '{numeric_cols[i]}' and '{numeric_cols[j]}' ({corr:.2f})",
                        metadata={
                            'var1': numeric_cols[i],
                            'var2': numeric_cols[j],
                            'correlation': float(corr),
                            'method': method
                        }
                    )

                    if pattern.confidence >= self.confidence_threshold:
                        patterns.append(pattern)

        return patterns

    # ==================== CLUSTER PATTERNS ====================

    def detect_clusters(self, columns: List[str] = None, max_clusters: int = 10) -> List[Pattern]:
        """
        Detect natural clusters in data.

        Args:
            columns: Columns to use for clustering (default: all numeric)
            max_clusters: Maximum number of clusters to detect

        Returns:
            List of detected cluster patterns
        """
        if columns is None:
            columns = self._get_numeric_columns()

        if len(columns) < 2:
            return []

        data = self.data[columns].dropna()

        if len(data) < 10:
            return []

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            return []  # Skip if sklearn not available

        patterns = []

        # Try different cluster counts
        best_score = -1
        best_k = 2

        for k in range(2, min(max_clusters + 1, len(data) // 2)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            # Calculate silhouette score
            score = silhouette_score(data, labels)

            if score > best_score:
                best_score = score
                best_k = k

        # If good clustering found
        if best_score > 0.5:
            pattern = Pattern(
                pattern_type=PatternType.CLUSTER,
                confidence=best_score,
                description=f"Natural clustering detected ({best_k} clusters, silhouette score: {best_score:.2f})",
                metadata={
                    'n_clusters': best_k,
                    'silhouette_score': float(best_score),
                    'columns': columns
                }
            )

            if pattern.confidence >= self.confidence_threshold:
                patterns.append(pattern)

        return patterns

    # ==================== ANOMALY PATTERNS ====================

    def detect_anomalies(self, column: str = None, method: str = 'zscore') -> List[Pattern]:
        """
        Detect anomalies/outliers in data.

        Args:
            column: Column name
            method: 'zscore', 'iqr', or 'isolation_forest'

        Returns:
            List of detected anomaly patterns
        """
        if column is None:
            column = self._get_numeric_columns()[0]

        values = self.data[column].dropna().values

        if len(values) < 3:
            return []

        patterns = []

        if method == 'zscore':
            # Z-score method
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                return []

            z_scores = np.abs((values - mean) / std)
            anomaly_indices = np.where(z_scores > 3)[0]

            for idx in anomaly_indices:
                confidence = min((z_scores[idx] - 3) / 3, 1.0)
                pattern = Pattern(
                    pattern_type=PatternType.ANOMALY,
                    confidence=confidence,
                    description=f"Anomaly detected at index {idx} (z-score: {z_scores[idx]:.2f})",
                    location=int(idx),
                    metadata={'z_score': float(z_scores[idx]), 'value': float(values[idx]), 'method': 'zscore'}
                )

                if pattern.confidence >= self.confidence_threshold:
                    patterns.append(pattern)

        elif method == 'iqr':
            # Interquartile range method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]

            for idx in anomaly_indices:
                distance = max(abs(values[idx] - lower_bound), abs(values[idx] - upper_bound))
                confidence = min(distance / iqr, 1.0) if iqr > 0 else 0.5

                pattern = Pattern(
                    pattern_type=PatternType.ANOMALY,
                    confidence=confidence,
                    description=f"Anomaly detected at index {idx} (IQR outlier)",
                    location=int(idx),
                    metadata={'value': float(values[idx]), 'method': 'iqr', 'iqr': float(iqr)}
                )

                if pattern.confidence >= self.confidence_threshold:
                    patterns.append(pattern)

        return patterns

    # ==================== DISTRIBUTION PATTERNS ====================

    def detect_distribution(self, column: str = None) -> List[Pattern]:
        """
        Detect distribution characteristics.

        Args:
            column: Column name

        Returns:
            List of detected distribution patterns
        """
        if column is None:
            column = self._get_numeric_columns()[0]

        values = self.data[column].dropna().values

        if len(values) < 3:
            return []

        patterns = []

        # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
        from scipy import stats

        if len(values) < 5000:
            stat, p_value = stats.shapiro(values)
        else:
            stat, p_value = stats.normaltest(values)

        # If p-value > 0.05, data is likely normal
        if p_value > 0.05:
            pattern = Pattern(
                pattern_type=PatternType.NORMAL_DISTRIBUTION,
                confidence=p_value,
                description=f"Normal distribution detected (p-value: {p_value:.4f})",
                metadata={'p_value': float(p_value), 'test': 'shapiro' if len(values) < 5000 else 'dagostino'}
            )

            if pattern.confidence >= self.confidence_threshold:
                patterns.append(pattern)
        else:
            # Check skewness
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)

            if abs(skewness) > 1:
                confidence = min(abs(skewness) / 3, 1.0)
                direction = "right" if skewness > 0 else "left"

                pattern = Pattern(
                    pattern_type=PatternType.SKEWED_DISTRIBUTION,
                    confidence=confidence,
                    description=f"Skewed distribution detected ({direction}-skewed, skewness: {skewness:.2f})",
                    metadata={'skewness': float(skewness), 'kurtosis': float(kurtosis)}
                )

                if pattern.confidence >= self.confidence_threshold:
                    patterns.append(pattern)

        return patterns

    # ==================== HELPER METHODS ====================

    def detect_all_patterns(self, verbose: bool = False) -> List[Pattern]:
        """
        Detect all patterns in data.

        Args:
            verbose: Print progress messages

        Returns:
            List of all detected patterns
        """
        all_patterns = []

        if verbose:
            print("🔍 Detecting patterns...")

        # Time series patterns
        if verbose:
            print("  - Trends...")
        all_patterns.extend(self.detect_trend())

        if verbose:
            print("  - Seasonality...")
        all_patterns.extend(self.detect_seasonality())

        if verbose:
            print("  - Spikes & Dips...")
        all_patterns.extend(self.detect_spikes_and_dips())

        if verbose:
            print("  - Volatility...")
        all_patterns.extend(self.detect_volatility())

        # Correlation patterns
        if len(self._get_numeric_columns()) >= 2:
            if verbose:
                print("  - Correlations...")
            all_patterns.extend(self.detect_correlations())

        # Cluster patterns
        if len(self._get_numeric_columns()) >= 2:
            if verbose:
                print("  - Clusters...")
            all_patterns.extend(self.detect_clusters())

        # Anomaly patterns
        if verbose:
            print("  - Anomalies...")
        all_patterns.extend(self.detect_anomalies())

        # Distribution patterns
        if verbose:
            print("  - Distribution...")
        all_patterns.extend(self.detect_distribution())

        # Sort by confidence
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)

        if verbose:
            print(f"\n✅ Found {len(all_patterns)} patterns!")

        self.patterns = all_patterns
        return all_patterns

    def get_summary(self) -> str:
        """
        Get human-readable summary of detected patterns.

        Returns:
            Summary string
        """
        if not self.patterns:
            self.detect_all_patterns()

        if not self.patterns:
            return "No significant patterns detected."

        summary = f"Pattern Detection Summary ({len(self.patterns)} patterns found):\n\n"

        # Group by pattern type
        by_type = {}
        for pattern in self.patterns:
            type_name = pattern.pattern_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(pattern)

        for pattern_type, patterns in by_type.items():
            summary += f"📊 {pattern_type.upper().replace('_', ' ')}:\n"
            for p in patterns[:3]:  # Top 3 per type
                summary += f"   • {p.description}\n"
            if len(patterns) > 3:
                summary += f"   ... and {len(patterns) - 3} more\n"
            summary += "\n"

        return summary

    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        n = len(series)
        if lag >= n:
            return 0.0

        mean = np.mean(series)
        c0 = np.sum((series - mean) ** 2) / n

        if c0 == 0:
            return 0.0

        c_lag = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n

        return c_lag / c0


# ==================== CONVENIENCE FUNCTIONS ====================

def detect_patterns(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    confidence_threshold: float = 0.7,
    verbose: bool = False
) -> List[Pattern]:
    """
    Quick pattern detection function.

    Args:
        data: Input data
        confidence_threshold: Minimum confidence (0-1)
        verbose: Print progress

    Returns:
        List of detected patterns

    Example:
        >>> patterns = detect_patterns(df, confidence_threshold=0.8, verbose=True)
        >>> for p in patterns:
        ...     print(p.description)
    """
    detector = PatternDetector(data, confidence_threshold=confidence_threshold)
    return detector.detect_all_patterns(verbose=verbose)


def get_pattern_summary(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    confidence_threshold: float = 0.7
) -> str:
    """
    Get human-readable pattern summary.

    Args:
        data: Input data
        confidence_threshold: Minimum confidence

    Returns:
        Summary string

    Example:
        >>> summary = get_pattern_summary(df)
        >>> print(summary)
    """
    detector = PatternDetector(data, confidence_threshold=confidence_threshold)
    detector.detect_all_patterns()
    return detector.get_summary()
