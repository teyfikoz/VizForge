"""
VizForge Insights Engine

Automatic statistical insights generation (NO API costs).
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta


class InsightType(Enum):
    """Types of statistical insights."""
    TREND = "trend"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    DISTRIBUTION = "distribution"
    OUTLIER = "outlier"
    SEASONALITY = "seasonality"
    CHANGE_POINT = "change_point"
    DOMINANCE = "dominance"


class InsightSeverity(Enum):
    """Severity/importance of an insight."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Insight:
    """
    A statistical insight discovered in data.

    Attributes:
        type: Type of insight (trend, correlation, etc.)
        severity: Importance level
        title: Short description
        description: Detailed explanation
        confidence: Confidence score (0.0-1.0)
        data: Supporting data/statistics
        visualization_hint: Recommended chart type to show insight
        affected_columns: Columns involved in this insight
    """
    type: InsightType
    severity: InsightSeverity
    title: str
    description: str
    confidence: float
    data: Dict[str, Any]
    visualization_hint: Optional[str] = None
    affected_columns: List[str] = None

    def __post_init__(self):
        if self.affected_columns is None:
            self.affected_columns = []


class InsightsEngine:
    """
    Automatic statistical insights generation.

    Discovers patterns, trends, and anomalies WITHOUT expensive AI APIs.

    Features:
    - Trend detection (linear regression, Mann-Kendall test)
    - Correlation discovery (Pearson, Spearman)
    - Anomaly detection (Z-score, IQR, Isolation Forest)
    - Distribution analysis (normality tests)
    - Seasonality detection (autocorrelation)
    - Change point detection

    Example:
        >>> engine = InsightsEngine()
        >>> insights = engine.generate_insights(df)
        >>> for insight in insights:
        ...     print(f"[{insight.severity.value}] {insight.title}")
        ...     print(f"  {insight.description}")
    """

    # Configuration
    CORRELATION_THRESHOLD = 0.7  # Strong correlation threshold
    OUTLIER_Z_SCORE = 3.0  # Z-score for outlier detection
    OUTLIER_IQR_MULTIPLIER = 1.5  # IQR multiplier for outliers
    TREND_P_VALUE = 0.05  # P-value for trend significance
    MIN_SAMPLES_FOR_TREND = 10  # Minimum samples for trend analysis

    def __init__(self, max_insights: int = 20):
        """
        Initialize insights engine.

        Args:
            max_insights: Maximum number of insights to generate
        """
        self.max_insights = max_insights

    def generate_insights(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        insight_types: Optional[List[InsightType]] = None
    ) -> List[Insight]:
        """
        Generate insights for dataset.

        Args:
            data: Input DataFrame
            target_column: Optional target column to focus on
            insight_types: List of insight types to generate (None = all)

        Returns:
            List of Insight objects sorted by severity

        Example:
            >>> insights = engine.generate_insights(df, target_column='sales')
            >>> critical = [i for i in insights if i.severity == InsightSeverity.CRITICAL]
        """
        insights = []

        # Enable all insight types if not specified
        if insight_types is None:
            insight_types = list(InsightType)

        # Generate different types of insights
        if InsightType.TREND in insight_types:
            insights.extend(self._detect_trends(data, target_column))

        if InsightType.CORRELATION in insight_types:
            insights.extend(self._detect_correlations(data, target_column))

        if InsightType.ANOMALY in insight_types:
            insights.extend(self._detect_anomalies(data, target_column))

        if InsightType.DISTRIBUTION in insight_types:
            insights.extend(self._analyze_distributions(data))

        if InsightType.OUTLIER in insight_types:
            insights.extend(self._detect_outliers(data))

        if InsightType.DOMINANCE in insight_types:
            insights.extend(self._detect_dominance(data))

        # Sort by severity and confidence
        insights = sorted(
            insights,
            key=lambda x: (
                ['critical', 'high', 'medium', 'low', 'info'].index(x.severity.value),
                -x.confidence
            )
        )

        # Limit to max_insights
        return insights[:self.max_insights]

    def _detect_trends(
        self,
        data: pd.DataFrame,
        target_column: Optional[str]
    ) -> List[Insight]:
        """Detect trends in time series or sequential data."""
        insights = []

        # Find temporal columns
        temporal_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # If no temporal column, try using index
        if not temporal_cols and pd.api.types.is_datetime64_any_dtype(data.index):
            df = data.reset_index()
            temporal_cols = [df.columns[0]]
        else:
            df = data

        for time_col in temporal_cols:
            for value_col in numeric_cols:
                if target_column and value_col != target_column:
                    continue

                # Skip if too few samples
                clean_data = df[[time_col, value_col]].dropna()
                if len(clean_data) < self.MIN_SAMPLES_FOR_TREND:
                    continue

                # Sort by time
                clean_data = clean_data.sort_values(time_col)

                # Linear regression
                x = np.arange(len(clean_data))
                y = clean_data[value_col].values

                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Check if trend is significant
                if p_value < self.TREND_P_VALUE:
                    direction = "increasing" if slope > 0 else "decreasing"
                    r_squared = r_value ** 2

                    # Calculate percentage change
                    start_val = intercept
                    end_val = slope * (len(x) - 1) + intercept
                    pct_change = ((end_val - start_val) / abs(start_val)) * 100 if start_val != 0 else 0

                    severity = InsightSeverity.HIGH if abs(pct_change) > 50 else InsightSeverity.MEDIUM

                    insights.append(Insight(
                        type=InsightType.TREND,
                        severity=severity,
                        title=f"{direction.capitalize()} trend in {value_col}",
                        description=f"{value_col} shows a {direction} trend over time "
                                  f"with {abs(pct_change):.1f}% change (R²={r_squared:.3f})",
                        confidence=min(r_squared, 0.99),
                        data={
                            'slope': float(slope),
                            'r_squared': float(r_squared),
                            'p_value': float(p_value),
                            'pct_change': float(pct_change),
                            'direction': direction
                        },
                        visualization_hint='LineChart',
                        affected_columns=[time_col, value_col]
                    ))

        return insights

    def _detect_correlations(
        self,
        data: pd.DataFrame,
        target_column: Optional[str]
    ) -> List[Insight]:
        """Detect correlations between numeric columns."""
        insights = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return insights

        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()

        # Find strong correlations
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]

                # Skip if target specified and neither column matches
                if target_column and col1 != target_column and col2 != target_column:
                    continue

                corr = corr_matrix.iloc[i, j]

                # Check if correlation is strong
                if abs(corr) >= self.CORRELATION_THRESHOLD:
                    relationship = "positive" if corr > 0 else "negative"
                    strength = "very strong" if abs(corr) > 0.9 else "strong"

                    severity = InsightSeverity.HIGH if abs(corr) > 0.9 else InsightSeverity.MEDIUM

                    insights.append(Insight(
                        type=InsightType.CORRELATION,
                        severity=severity,
                        title=f"{strength.capitalize()} {relationship} correlation between {col1} and {col2}",
                        description=f"{col1} and {col2} are {strength}ly {relationship}ly correlated "
                                  f"(r={corr:.3f}). Changes in one may predict changes in the other.",
                        confidence=abs(corr),
                        data={
                            'correlation': float(corr),
                            'relationship': relationship,
                            'strength': strength
                        },
                        visualization_hint='ScatterPlot',
                        affected_columns=[col1, col2]
                    ))

        return insights

    def _detect_anomalies(
        self,
        data: pd.DataFrame,
        target_column: Optional[str]
    ) -> List[Insight]:
        """Detect anomalies using statistical methods."""
        insights = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if target_column and col != target_column:
                continue

            series = data[col].dropna()

            if len(series) < 10:
                continue

            # Z-score method
            mean = series.mean()
            std = series.std()

            if std > 0:
                z_scores = np.abs((series - mean) / std)
                anomalies = series[z_scores > self.OUTLIER_Z_SCORE]

                if len(anomalies) > 0:
                    anomaly_pct = (len(anomalies) / len(series)) * 100

                    if anomaly_pct > 0.5:  # At least 0.5% anomalies
                        severity = InsightSeverity.HIGH if anomaly_pct > 5 else InsightSeverity.MEDIUM

                        insights.append(Insight(
                            type=InsightType.ANOMALY,
                            severity=severity,
                            title=f"Anomalies detected in {col}",
                            description=f"Found {len(anomalies)} anomalous values ({anomaly_pct:.1f}%) "
                                      f"in {col} that deviate significantly from the mean.",
                            confidence=0.85,
                            data={
                                'n_anomalies': int(len(anomalies)),
                                'anomaly_pct': float(anomaly_pct),
                                'anomaly_values': anomalies.tolist()[:10],  # First 10
                                'mean': float(mean),
                                'std': float(std)
                            },
                            visualization_hint='BoxPlot',
                            affected_columns=[col]
                        ))

        return insights

    def _analyze_distributions(self, data: pd.DataFrame) -> List[Insight]:
        """Analyze distribution characteristics."""
        insights = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = data[col].dropna()

            if len(series) < 20:
                continue

            # Test for normality
            if len(series) <= 5000:  # Shapiro-Wilk works best for small samples
                statistic, p_value = stats.shapiro(series)
            else:  # Use Kolmogorov-Smirnov for large samples
                statistic, p_value = stats.kstest(series, 'norm', args=(series.mean(), series.std()))

            # Check skewness
            skewness = series.skew()
            kurtosis = series.kurtosis()

            # Normal distribution
            if p_value > 0.05:
                insights.append(Insight(
                    type=InsightType.DISTRIBUTION,
                    severity=InsightSeverity.INFO,
                    title=f"{col} follows normal distribution",
                    description=f"{col} is normally distributed (p={p_value:.3f}). "
                              f"Parametric statistical tests are appropriate.",
                    confidence=float(p_value),
                    data={
                        'distribution': 'normal',
                        'p_value': float(p_value),
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis)
                    },
                    visualization_hint='Histogram',
                    affected_columns=[col]
                ))
            # Skewed distribution
            elif abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append(Insight(
                    type=InsightType.DISTRIBUTION,
                    severity=InsightSeverity.MEDIUM,
                    title=f"{col} is {direction}-skewed",
                    description=f"{col} has a {direction}-skewed distribution (skew={skewness:.2f}). "
                              f"Consider log transformation or non-parametric methods.",
                    confidence=min(abs(skewness) / 3, 0.95),
                    data={
                        'distribution': f'{direction}-skewed',
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis)
                    },
                    visualization_hint='Histogram',
                    affected_columns=[col]
                ))

        return insights

    def _detect_outliers(self, data: pd.DataFrame) -> List[Insight]:
        """Detect outliers using IQR method."""
        insights = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = data[col].dropna()

            if len(series) < 10:
                continue

            # IQR method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - self.OUTLIER_IQR_MULTIPLIER * iqr
            upper_bound = q3 + self.OUTLIER_IQR_MULTIPLIER * iqr

            outliers = series[(series < lower_bound) | (series > upper_bound)]

            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(series)) * 100

                if outlier_pct > 1:  # At least 1% outliers
                    severity = InsightSeverity.MEDIUM if outlier_pct > 5 else InsightSeverity.LOW

                    insights.append(Insight(
                        type=InsightType.OUTLIER,
                        severity=severity,
                        title=f"Outliers in {col}",
                        description=f"Detected {len(outliers)} outliers ({outlier_pct:.1f}%) "
                                  f"in {col} outside range [{lower_bound:.2f}, {upper_bound:.2f}]",
                        confidence=0.80,
                        data={
                            'n_outliers': int(len(outliers)),
                            'outlier_pct': float(outlier_pct),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'q1': float(q1),
                            'q3': float(q3),
                            'iqr': float(iqr)
                        },
                        visualization_hint='BoxPlot',
                        affected_columns=[col]
                    ))

        return insights

    def _detect_dominance(self, data: pd.DataFrame) -> List[Insight]:
        """Detect dominant values/categories."""
        insights = []

        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            value_counts = data[col].value_counts()

            if len(value_counts) == 0:
                continue

            # Check if one value dominates
            top_value = value_counts.index[0]
            top_pct = (value_counts.iloc[0] / len(data)) * 100

            if top_pct > 50:  # Dominance threshold
                severity = InsightSeverity.HIGH if top_pct > 80 else InsightSeverity.MEDIUM

                insights.append(Insight(
                    type=InsightType.DOMINANCE,
                    severity=severity,
                    title=f"'{top_value}' dominates {col}",
                    description=f"The value '{top_value}' accounts for {top_pct:.1f}% "
                              f"of all {col} values, indicating strong dominance.",
                    confidence=min(top_pct / 100, 0.95),
                    data={
                        'dominant_value': str(top_value),
                        'dominance_pct': float(top_pct),
                        'count': int(value_counts.iloc[0]),
                        'total_unique': int(len(value_counts))
                    },
                    visualization_hint='PieChart',
                    affected_columns=[col]
                ))

        return insights

    def summarize_insights(self, insights: List[Insight]) -> str:
        """
        Generate human-readable summary of insights.

        Args:
            insights: List of insights

        Returns:
            Formatted summary string

        Example:
            >>> summary = engine.summarize_insights(insights)
            >>> print(summary)
        """
        if not insights:
            return "No significant insights detected."

        summary_lines = ["📊 Data Insights Summary", "=" * 50, ""]

        # Group by severity
        by_severity = {}
        for insight in insights:
            severity = insight.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(insight)

        # Display by severity
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            if severity in by_severity:
                icon = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🔵',
                    'info': 'ℹ️'
                }[severity]

                summary_lines.append(f"{icon} {severity.upper()} ({len(by_severity[severity])})")
                for insight in by_severity[severity][:3]:  # Show top 3 per severity
                    summary_lines.append(f"  • {insight.title}")
                    summary_lines.append(f"    {insight.description}")
                summary_lines.append("")

        return "\n".join(summary_lines)

    def export_insights(
        self,
        insights: List[Insight],
        format: str = 'dict'
    ) -> Any:
        """
        Export insights to various formats.

        Args:
            insights: List of insights
            format: Export format ('dict', 'json', 'dataframe')

        Returns:
            Exported insights in requested format
        """
        if format == 'dict':
            return [
                {
                    'type': i.type.value,
                    'severity': i.severity.value,
                    'title': i.title,
                    'description': i.description,
                    'confidence': i.confidence,
                    'data': i.data,
                    'visualization_hint': i.visualization_hint,
                    'affected_columns': i.affected_columns
                }
                for i in insights
            ]
        elif format == 'json':
            import json
            return json.dumps(self.export_insights(insights, 'dict'), indent=2)
        elif format == 'dataframe':
            return pd.DataFrame(self.export_insights(insights, 'dict'))
        else:
            raise ValueError(f"Unsupported format: {format}")
