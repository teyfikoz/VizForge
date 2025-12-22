"""
VizForge Insight Discovery Engine

Automatically discover key insights, patterns, and findings in data.
NO API required - intelligent statistical analysis!
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import numpy as np
import pandas as pd


class InsightType(Enum):
    """Types of insights that can be discovered."""
    TREND = "trend"  # Growing/declining trend
    CORRELATION = "correlation"  # Strong correlation between variables
    ANOMALY = "anomaly"  # Unusual data points
    EXTREME = "extreme"  # Extreme values (max/min)
    DISTRIBUTION = "distribution"  # Distribution characteristics
    SEASONALITY = "seasonality"  # Seasonal patterns
    COMPARISON = "comparison"  # Comparison between groups
    CHANGE = "change"  # Significant change over time
    SUMMARY = "summary"  # Summary statistics


@dataclass
class Insight:
    """
    A discovered insight from data.

    Attributes:
        type: Type of insight
        title: Short title
        description: Detailed description
        importance: Importance score (0.0 to 1.0)
        data: Supporting data/metrics
        recommendation: Optional recommendation
    """
    type: InsightType
    title: str
    description: str
    importance: float
    data: Dict[str, Any]
    recommendation: Optional[str] = None


class InsightDiscovery:
    """
    Automatic insight discovery engine.

    Analyzes data and discovers key insights automatically.

    Examples:
        >>> discovery = InsightDiscovery(df)
        >>> insights = discovery.discover()
        >>> for insight in insights:
        >>>     print(f"{insight.title}: {insight.description}")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        time_column: Optional[str] = None,
        min_importance: float = 0.5,
    ):
        """
        Initialize insight discovery.

        Args:
            data: DataFrame to analyze
            target_column: Optional target column for focused analysis
            time_column: Optional time column for temporal analysis
            min_importance: Minimum importance threshold (0.0 to 1.0)
        """
        self.data = data.copy()
        self.target_column = target_column
        self.time_column = time_column
        self.min_importance = min_importance

        # Analyze column types
        self.numeric_cols = self._get_numeric_columns()
        self.categorical_cols = self._get_categorical_columns()
        self.datetime_cols = self._get_datetime_columns()

    def discover(self, max_insights: int = 10) -> List[Insight]:
        """
        Discover key insights from data.

        Args:
            max_insights: Maximum number of insights to return

        Returns:
            List of discovered insights, sorted by importance
        """
        insights = []

        # Summary insights
        insights.extend(self._discover_summary())

        # Trend insights
        insights.extend(self._discover_trends())

        # Correlation insights
        insights.extend(self._discover_correlations())

        # Anomaly insights
        insights.extend(self._discover_anomalies())

        # Extreme value insights
        insights.extend(self._discover_extremes())

        # Distribution insights
        insights.extend(self._discover_distributions())

        # Seasonality insights
        insights.extend(self._discover_seasonality())

        # Comparison insights
        insights.extend(self._discover_comparisons())

        # Change insights
        insights.extend(self._discover_changes())

        # Filter by importance
        insights = [i for i in insights if i.importance >= self.min_importance]

        # Sort by importance and limit
        insights.sort(key=lambda x: x.importance, reverse=True)
        return insights[:max_insights]

    def _discover_summary(self) -> List[Insight]:
        """Discover summary insights."""
        insights = []

        # Overall data summary
        n_rows = len(self.data)
        n_cols = len(self.data.columns)

        description = (
            f"The dataset contains {n_rows:,} rows and {n_cols} columns. "
            f"It includes {len(self.numeric_cols)} numeric columns, "
            f"{len(self.categorical_cols)} categorical columns"
        )

        if self.datetime_cols:
            description += f", and {len(self.datetime_cols)} datetime columns"

        description += "."

        insights.append(Insight(
            type=InsightType.SUMMARY,
            title="Dataset Overview",
            description=description,
            importance=0.8,
            data={
                'rows': n_rows,
                'columns': n_cols,
                'numeric': len(self.numeric_cols),
                'categorical': len(self.categorical_cols),
                'datetime': len(self.datetime_cols),
            }
        ))

        # Missing data insight
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).round(1)
        cols_with_missing = missing[missing > 0]

        if len(cols_with_missing) > 0:
            worst_col = missing_pct.idxmax()
            worst_pct = missing_pct.max()

            description = (
                f"{len(cols_with_missing)} columns have missing values. "
                f"'{worst_col}' has the most missing data ({worst_pct:.1f}%)."
            )

            insights.append(Insight(
                type=InsightType.SUMMARY,
                title="Missing Data Detected",
                description=description,
                importance=min(0.9, worst_pct / 100),
                data={
                    'columns_with_missing': len(cols_with_missing),
                    'worst_column': worst_col,
                    'worst_percentage': worst_pct,
                },
                recommendation="Consider handling missing values before analysis."
            ))

        return insights

    def _discover_trends(self) -> List[Insight]:
        """Discover trend insights."""
        insights = []

        # Use predictive module if available
        try:
            from ..predictive import detect_trend

            for col in self.numeric_cols[:5]:  # Top 5 numeric columns
                try:
                    result = detect_trend(self.data[col])

                    if result.strength > 0.5:  # Strong trend
                        trend_emoji = "📈" if result.slope > 0 else "📉"
                        trend_direction = "increasing" if result.slope > 0 else "decreasing"

                        description = (
                            f"{trend_emoji} '{col}' shows a {result.trend_type.value} trend "
                            f"({trend_direction} by {abs(result.slope)*100:.1f}% per period). "
                            f"Trend strength: {result.strength:.0%}."
                        )

                        insights.append(Insight(
                            type=InsightType.TREND,
                            title=f"{col.title()} {trend_direction.title()}",
                            description=description,
                            importance=result.strength * 0.9,
                            data={
                                'column': col,
                                'trend_type': result.trend_type.value,
                                'slope': result.slope,
                                'strength': result.strength,
                                'confidence': result.confidence,
                            }
                        ))
                except:
                    continue
        except ImportError:
            pass

        return insights

    def _discover_correlations(self) -> List[Insight]:
        """Discover correlation insights."""
        insights = []

        if len(self.numeric_cols) < 2:
            return insights

        # Calculate correlation matrix
        corr_matrix = self.data[self.numeric_cols].corr()

        # Find strong correlations
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, len(self.numeric_cols)):
                col1 = self.numeric_cols[i]
                col2 = self.numeric_cols[j]
                corr = corr_matrix.loc[col1, col2]

                if abs(corr) > 0.7:  # Strong correlation
                    direction = "positively" if corr > 0 else "negatively"
                    strength = "very strong" if abs(corr) > 0.9 else "strong"

                    description = (
                        f"'{col1}' and '{col2}' are {strength} {direction} correlated "
                        f"(r = {corr:.2f}). "
                    )

                    if corr > 0:
                        description += f"As {col1} increases, {col2} tends to increase."
                    else:
                        description += f"As {col1} increases, {col2} tends to decrease."

                    insights.append(Insight(
                        type=InsightType.CORRELATION,
                        title=f"{col1.title()} ↔ {col2.title()}",
                        description=description,
                        importance=abs(corr) * 0.85,
                        data={
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr,
                        }
                    ))

        return insights

    def _discover_anomalies(self) -> List[Insight]:
        """Discover anomaly insights."""
        insights = []

        try:
            from ..predictive import detect_anomalies

            for col in self.numeric_cols[:3]:  # Top 3 numeric columns
                try:
                    anomalies = detect_anomalies(self.data[col], sensitivity=2.5)

                    if anomalies:
                        critical_anomalies = [a for a in anomalies if a.severity in ['high', 'critical']]

                        if critical_anomalies:
                            description = (
                                f"⚠️ Detected {len(anomalies)} anomalies in '{col}', "
                                f"including {len(critical_anomalies)} critical outliers. "
                            )

                            worst = max(critical_anomalies, key=lambda x: x.score)
                            description += (
                                f"Most extreme: {worst.value:.2f} at index {worst.index} "
                                f"(expected {worst.expected:.2f})."
                            )

                            insights.append(Insight(
                                type=InsightType.ANOMALY,
                                title=f"Anomalies in {col.title()}",
                                description=description,
                                importance=min(0.95, len(critical_anomalies) / len(self.data) * 10),
                                data={
                                    'column': col,
                                    'total_anomalies': len(anomalies),
                                    'critical_anomalies': len(critical_anomalies),
                                    'worst_index': worst.index,
                                    'worst_value': worst.value,
                                },
                                recommendation="Investigate these outliers for data quality or special events."
                            ))
                except:
                    continue
        except ImportError:
            pass

        return insights

    def _discover_extremes(self) -> List[Insight]:
        """Discover extreme value insights."""
        insights = []

        for col in self.numeric_cols[:5]:
            max_val = self.data[col].max()
            min_val = self.data[col].min()
            max_idx = self.data[col].idxmax()
            min_idx = self.data[col].idxmin()

            mean_val = self.data[col].mean()
            range_val = max_val - min_val

            # Check if extreme values are significant
            if max_val > mean_val * 1.5 or min_val < mean_val * 0.5:
                description = (
                    f"'{col}' ranges from {min_val:.2f} to {max_val:.2f} (range: {range_val:.2f}). "
                    f"Maximum value occurs at index {max_idx}."
                )

                insights.append(Insight(
                    type=InsightType.EXTREME,
                    title=f"{col.title()} Range",
                    description=description,
                    importance=0.6,
                    data={
                        'column': col,
                        'max': max_val,
                        'min': min_val,
                        'range': range_val,
                        'max_index': max_idx,
                        'min_index': min_idx,
                    }
                ))

        return insights

    def _discover_distributions(self) -> List[Insight]:
        """Discover distribution insights."""
        insights = []

        for col in self.numeric_cols[:3]:
            skewness = self.data[col].skew()
            kurtosis = self.data[col].kurtosis()

            if abs(skewness) > 1:  # Significant skewness
                direction = "right-skewed (positive)" if skewness > 0 else "left-skewed (negative)"

                description = (
                    f"'{col}' has a {direction} distribution (skewness: {skewness:.2f}). "
                )

                if skewness > 0:
                    description += "Most values are concentrated on the lower end."
                else:
                    description += "Most values are concentrated on the higher end."

                insights.append(Insight(
                    type=InsightType.DISTRIBUTION,
                    title=f"{col.title()} Distribution",
                    description=description,
                    importance=min(0.75, abs(skewness) / 3),
                    data={
                        'column': col,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                    }
                ))

        return insights

    def _discover_seasonality(self) -> List[Insight]:
        """Discover seasonality insights."""
        insights = []

        try:
            from ..predictive import analyze_seasonality

            for col in self.numeric_cols[:3]:
                if len(self.data) < 14:  # Need enough data
                    continue

                try:
                    pattern = analyze_seasonality(self.data[col])

                    if pattern.strength > 0.6 and pattern.type.value != 'none':
                        description = (
                            f"📊 '{col}' exhibits {pattern.type.value} seasonality "
                            f"with a period of {pattern.period} units. "
                            f"Pattern strength: {pattern.strength:.0%}."
                        )

                        insights.append(Insight(
                            type=InsightType.SEASONALITY,
                            title=f"{col.title()} Seasonal Pattern",
                            description=description,
                            importance=pattern.strength * 0.85,
                            data={
                                'column': col,
                                'pattern_type': pattern.type.value,
                                'period': pattern.period,
                                'strength': pattern.strength,
                            }
                        ))
                except:
                    continue
        except ImportError:
            pass

        return insights

    def _discover_comparisons(self) -> List[Insight]:
        """Discover comparison insights."""
        insights = []

        # Compare groups if categorical columns exist
        for cat_col in self.categorical_cols[:2]:
            unique_values = self.data[cat_col].nunique()

            if 2 <= unique_values <= 10:  # Reasonable number of categories
                for num_col in self.numeric_cols[:2]:
                    grouped = self.data.groupby(cat_col)[num_col].mean()
                    max_group = grouped.idxmax()
                    min_group = grouped.idxmin()
                    max_val = grouped.max()
                    min_val = grouped.min()

                    ratio = max_val / min_val if min_val != 0 else float('inf')

                    if ratio > 1.5:  # Significant difference
                        description = (
                            f"'{max_group}' has {ratio:.1f}x higher '{num_col}' "
                            f"than '{min_group}' (${max_val:.2f} vs ${min_val:.2f})."
                        )

                        insights.append(Insight(
                            type=InsightType.COMPARISON,
                            title=f"{num_col.title()} by {cat_col.title()}",
                            description=description,
                            importance=min(0.85, (ratio - 1) / 10),
                            data={
                                'category_column': cat_col,
                                'numeric_column': num_col,
                                'max_group': max_group,
                                'min_group': min_group,
                                'max_value': max_val,
                                'min_value': min_val,
                                'ratio': ratio,
                            }
                        ))

        return insights

    def _discover_changes(self) -> List[Insight]:
        """Discover change insights."""
        insights = []

        # Detect significant changes over time
        if self.time_column or self.datetime_cols:
            time_col = self.time_column or self.datetime_cols[0]

            for num_col in self.numeric_cols[:3]:
                try:
                    # Sort by time
                    sorted_data = self.data.sort_values(time_col)
                    values = sorted_data[num_col].values

                    # Calculate period-over-period change
                    first_val = values[0]
                    last_val = values[-1]
                    change = last_val - first_val
                    pct_change = (change / first_val * 100) if first_val != 0 else 0

                    if abs(pct_change) > 20:  # Significant change
                        direction = "increased" if change > 0 else "decreased"
                        emoji = "📈" if change > 0 else "📉"

                        description = (
                            f"{emoji} '{num_col}' {direction} by {abs(pct_change):.1f}% "
                            f"from {first_val:.2f} to {last_val:.2f}."
                        )

                        insights.append(Insight(
                            type=InsightType.CHANGE,
                            title=f"{num_col.title()} Change",
                            description=description,
                            importance=min(0.9, abs(pct_change) / 100),
                            data={
                                'column': num_col,
                                'first_value': first_val,
                                'last_value': last_val,
                                'change': change,
                                'percent_change': pct_change,
                            }
                        ))
                except:
                    continue

        return insights

    def _get_numeric_columns(self) -> List[str]:
        """Get numeric column names."""
        return [col for col in self.data.columns if pd.api.types.is_numeric_dtype(self.data[col])]

    def _get_categorical_columns(self) -> List[str]:
        """Get categorical column names."""
        return [col for col in self.data.columns
                if pd.api.types.is_categorical_dtype(self.data[col]) or
                pd.api.types.is_object_dtype(self.data[col])]

    def _get_datetime_columns(self) -> List[str]:
        """Get datetime column names."""
        return [col for col in self.data.columns if pd.api.types.is_datetime64_any_dtype(self.data[col])]


# ==================== Convenience Function ====================

def discover_insights(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    max_insights: int = 10,
    min_importance: float = 0.5,
) -> List[Insight]:
    """
    Discover key insights from data (one-liner!).

    Args:
        data: DataFrame to analyze
        target_column: Optional target column for focused analysis
        time_column: Optional time column for temporal analysis
        max_insights: Maximum number of insights to return
        min_importance: Minimum importance threshold (0.0 to 1.0)

    Returns:
        List of discovered insights, sorted by importance

    Examples:
        >>> from vizforge.storytelling import discover_insights
        >>>
        >>> insights = discover_insights(df, max_insights=5)
        >>> for insight in insights:
        >>>     print(f"[{insight.type.value}] {insight.title}")
        >>>     print(f"   {insight.description}")
        >>>     if insight.recommendation:
        >>>         print(f"   💡 {insight.recommendation}")
    """
    discovery = InsightDiscovery(
        data=data,
        target_column=target_column,
        time_column=time_column,
        min_importance=min_importance
    )
    return discovery.discover(max_insights=max_insights)
