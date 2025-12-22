"""
VizForge v1.2.0 - Smart Chart Recommender v2

Enhanced chart recommendation engine with multi-criteria scoring.
NO API required - uses statistical analysis + visualization best practices.

Author: VizForge Team
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime


class ChartCategory(Enum):
    """Chart categories."""
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    RELATIONSHIP = "relationship"
    COMPOSITION = "composition"
    TIME_SERIES = "time_series"
    GEOGRAPHIC = "geographic"
    HIERARCHICAL = "hierarchical"
    NETWORK = "network"
    STATISTICAL = "statistical"
    SCIENTIFIC = "scientific"


class ChartType(Enum):
    """Supported chart types."""
    # 2D Charts
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"

    # Statistical
    CORRELATION = "correlation"
    REGRESSION = "regression"
    DISTRIBUTION = "distribution"
    QQ_PLOT = "qq_plot"

    # Advanced
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    SANKEY = "sankey"
    NETWORK_GRAPH = "network"
    CHORD = "chord"

    # Geographic
    CHOROPLETH = "choropleth"
    BUBBLE_MAP = "bubble_map"

    # 3D
    SURFACE_3D = "surface_3d"
    SCATTER_3D = "scatter_3d"
    LINE_3D = "line_3d"


@dataclass
class ChartRecommendation:
    """A chart recommendation with confidence and reasoning."""

    chart_type: ChartType
    category: ChartCategory
    confidence: float  # 0.0 to 1.0
    score: float  # Total score from all criteria

    # Detailed scoring
    data_fit_score: float  # How well data fits this chart
    best_practice_score: float  # Follows visualization best practices
    performance_score: float  # Performance considerations
    accessibility_score: float  # Accessibility & readability
    aesthetic_score: float  # Visual appeal

    # Reasoning
    reasoning: str
    pros: List[str]
    cons: List[str]

    # Requirements & recommendations
    required_columns: Dict[str, str]  # role -> column type
    optional_features: List[str]

    # Usage example
    example_code: str

    def __str__(self) -> str:
        """String representation."""
        return f"{self.chart_type.value} ({self.confidence:.0%} confidence)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"ChartRecommendation(type={self.chart_type.value}, confidence={self.confidence:.2%}, score={self.score:.2f})"


class SmartChartRecommender:
    """
    Enhanced chart recommendation engine v2.

    Uses multi-criteria scoring to recommend optimal chart types:
    - Data characteristics analysis
    - Visualization best practices
    - Performance considerations
    - Accessibility guidelines
    - Aesthetic principles

    NO API required - pure statistical + rule-based analysis.
    """

    def __init__(self, data: pd.DataFrame, verbose: bool = False):
        """
        Initialize recommender.

        Args:
            data: Input DataFrame to analyze
            verbose: Print detailed analysis steps
        """
        self.data = data
        self.verbose = verbose

        # Analyze data characteristics
        self._analyze_data()

    def _analyze_data(self):
        """Analyze data characteristics."""
        if self.verbose:
            print("🔍 Analyzing data characteristics...")

        self.n_rows = len(self.data)
        self.n_cols = len(self.data.columns)

        # Column type analysis
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()

        # Detect datetime-like string columns
        for col in self.categorical_cols:
            if self._is_datetime_like(col):
                self.datetime_cols.append(col)
                self.categorical_cols.remove(col)

        self.n_numeric = len(self.numeric_cols)
        self.n_categorical = len(self.categorical_cols)
        self.n_datetime = len(self.datetime_cols)

        # Cardinality analysis
        self.cardinalities = {}
        for col in self.data.columns:
            self.cardinalities[col] = self.data[col].nunique()

        # Data size assessment
        self.is_large_dataset = self.n_rows > 10000
        self.is_very_large_dataset = self.n_rows > 100000

        if self.verbose:
            print(f"  - Rows: {self.n_rows:,}")
            print(f"  - Columns: {self.n_cols} ({self.n_numeric} numeric, {self.n_categorical} categorical, {self.n_datetime} datetime)")
            print(f"  - Large dataset: {self.is_large_dataset}")

    def _is_datetime_like(self, col: str) -> bool:
        """Check if column contains datetime-like strings."""
        sample = self.data[col].dropna().head(10)
        if len(sample) == 0:
            return False

        # Check for common date patterns
        date_keywords = ['date', 'time', 'year', 'month', 'day']
        if any(keyword in col.lower() for keyword in date_keywords):
            return True

        return False

    def recommend(self, top_n: int = 3, min_confidence: float = 0.5) -> List[ChartRecommendation]:
        """
        Get top N chart recommendations.

        Args:
            top_n: Number of recommendations to return
            min_confidence: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            List of ChartRecommendation objects, sorted by confidence
        """
        if self.verbose:
            print(f"\n🎨 Generating chart recommendations (top {top_n})...")

        recommendations = []

        # Generate recommendations for each chart type
        recommendations.extend(self._recommend_time_series())
        recommendations.extend(self._recommend_comparison())
        recommendations.extend(self._recommend_distribution())
        recommendations.extend(self._recommend_relationship())
        recommendations.extend(self._recommend_composition())
        recommendations.extend(self._recommend_statistical())

        # Filter by confidence
        recommendations = [r for r in recommendations if r.confidence >= min_confidence]

        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        if self.verbose:
            print(f"  ✅ Generated {len(recommendations)} recommendations")
            print(f"  📊 Top recommendation: {recommendations[0] if recommendations else 'None'}")

        return recommendations[:top_n]

    def _recommend_time_series(self) -> List[ChartRecommendation]:
        """Recommend time series charts."""
        recommendations = []

        # Need at least 1 datetime and 1 numeric column
        if self.n_datetime == 0 or self.n_numeric == 0:
            return recommendations

        # LINE CHART - Best for time series
        data_fit = 0.9 if self.n_datetime >= 1 else 0.5
        best_practice = 0.95  # Industry standard for time series
        performance = 0.8 if not self.is_very_large_dataset else 0.6
        accessibility = 0.9
        aesthetic = 0.85

        score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
        confidence = score * 0.95  # Slight adjustment

        recommendations.append(ChartRecommendation(
            chart_type=ChartType.LINE,
            category=ChartCategory.TIME_SERIES,
            confidence=confidence,
            score=score,
            data_fit_score=data_fit,
            best_practice_score=best_practice,
            performance_score=performance,
            accessibility_score=accessibility,
            aesthetic_score=aesthetic,
            reasoning=f"Line charts are optimal for time series data. Your data has {self.n_datetime} datetime column(s) and {self.n_numeric} numeric column(s), perfect for tracking changes over time.",
            pros=[
                "Clear visualization of trends over time",
                "Easy to compare multiple time series",
                "Widely understood by all audiences",
                "Excellent for showing patterns and seasonality"
            ],
            cons=[
                "Can become cluttered with many series" if self.n_numeric > 5 else "Clean visualization",
                "Requires sequential time data"
            ],
            required_columns={"x": "datetime", "y": "numeric"},
            optional_features=["Multiple series", "Trend lines", "Annotations"],
            example_code=f"import vizforge as vz\nchart = vz.line(df, x='{self.datetime_cols[0]}', y='{self.numeric_cols[0]}')\nchart.show()"
        ))

        # AREA CHART - Good for cumulative data
        if self.n_numeric >= 1:
            data_fit = 0.75
            best_practice = 0.8
            performance = 0.75 if not self.is_very_large_dataset else 0.5
            accessibility = 0.8
            aesthetic = 0.85

            score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
            confidence = score * 0.9

            recommendations.append(ChartRecommendation(
                chart_type=ChartType.AREA,
                category=ChartCategory.TIME_SERIES,
                confidence=confidence,
                score=score,
                data_fit_score=data_fit,
                best_practice_score=best_practice,
                performance_score=performance,
                accessibility_score=accessibility,
                aesthetic_score=aesthetic,
                reasoning="Area charts work well for showing volume or magnitude over time, especially for cumulative data.",
                pros=[
                    "Emphasizes magnitude of change",
                    "Good for showing cumulative totals",
                    "Visually appealing"
                ],
                cons=[
                    "Can obscure data with multiple series",
                    "Harder to read exact values"
                ],
                required_columns={"x": "datetime", "y": "numeric"},
                optional_features=["Stacked areas", "Normalized", "Fill opacity"],
                example_code=f"import vizforge as vz\nchart = vz.area(df, x='{self.datetime_cols[0]}', y='{self.numeric_cols[0]}')\nchart.show()"
            ))

        return recommendations

    def _recommend_comparison(self) -> List[ChartRecommendation]:
        """Recommend comparison charts."""
        recommendations = []

        # Need at least 1 categorical and 1 numeric column
        if self.n_categorical == 0 or self.n_numeric == 0:
            return recommendations

        # Check if categorical column has reasonable cardinality for bar chart
        suitable_categorical = [col for col in self.categorical_cols
                               if self.cardinalities[col] <= 20]

        if not suitable_categorical:
            return recommendations

        # BAR CHART - Best for comparisons
        cardinality = self.cardinalities[suitable_categorical[0]]

        data_fit = 0.95 if cardinality <= 10 else 0.8
        best_practice = 0.95  # Industry standard for comparisons
        performance = 0.9 if not self.is_very_large_dataset else 0.7
        accessibility = 0.95  # Excellent for screen readers
        aesthetic = 0.85

        score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
        confidence = score * 0.96

        recommendations.append(ChartRecommendation(
            chart_type=ChartType.BAR,
            category=ChartCategory.COMPARISON,
            confidence=confidence,
            score=score,
            data_fit_score=data_fit,
            best_practice_score=best_practice,
            performance_score=performance,
            accessibility_score=accessibility,
            aesthetic_score=aesthetic,
            reasoning=f"Bar charts excel at comparing categorical data. Your '{suitable_categorical[0]}' column has {cardinality} categories, ideal for bar visualization.",
            pros=[
                "Easy to compare values across categories",
                "Highly accessible and intuitive",
                "Works well for both horizontal and vertical layouts",
                "Supports grouping and stacking"
            ],
            cons=[
                "Limited to categorical data",
                f"Becomes cluttered with >{cardinality} categories" if cardinality > 15 else "Clear visualization"
            ],
            required_columns={"x": "categorical", "y": "numeric"},
            optional_features=["Grouped bars", "Stacked bars", "Error bars"],
            example_code=f"import vizforge as vz\nchart = vz.bar(df, x='{suitable_categorical[0]}', y='{self.numeric_cols[0]}')\nchart.show()"
        ))

        return recommendations

    def _recommend_distribution(self) -> List[ChartRecommendation]:
        """Recommend distribution charts."""
        recommendations = []

        # Need at least 1 numeric column
        if self.n_numeric == 0:
            return recommendations

        # HISTOGRAM - Best for single variable distribution
        data_fit = 0.9
        best_practice = 0.9
        performance = 0.85 if not self.is_very_large_dataset else 0.7
        accessibility = 0.85
        aesthetic = 0.8

        score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
        confidence = score * 0.92

        recommendations.append(ChartRecommendation(
            chart_type=ChartType.HISTOGRAM,
            category=ChartCategory.DISTRIBUTION,
            confidence=confidence,
            score=score,
            data_fit_score=data_fit,
            best_practice_score=best_practice,
            performance_score=performance,
            accessibility_score=accessibility,
            aesthetic_score=aesthetic,
            reasoning=f"Histogram is ideal for understanding the distribution of numeric data. Perfect for exploring your {self.n_numeric} numeric column(s).",
            pros=[
                "Shows distribution shape clearly",
                "Identifies skewness and outliers",
                "Easy to interpret",
                "Statistical foundation"
            ],
            cons=[
                "Bin size affects interpretation",
                "Not suitable for categorical data"
            ],
            required_columns={"x": "numeric"},
            optional_features=["KDE overlay", "Custom bins", "Cumulative"],
            example_code=f"import vizforge as vz\nchart = vz.histogram(df, x='{self.numeric_cols[0]}')\nchart.show()"
        ))

        # BOX PLOT - Good for comparing distributions
        if self.n_numeric >= 1:
            data_fit = 0.85
            best_practice = 0.9
            performance = 0.9
            accessibility = 0.75  # Requires statistical knowledge
            aesthetic = 0.8

            score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
            confidence = score * 0.88

            recommendations.append(ChartRecommendation(
                chart_type=ChartType.BOX,
                category=ChartCategory.DISTRIBUTION,
                confidence=confidence,
                score=score,
                data_fit_score=data_fit,
                best_practice_score=best_practice,
                performance_score=performance,
                accessibility_score=accessibility,
                aesthetic_score=aesthetic,
                reasoning="Box plots provide statistical summary and identify outliers effectively.",
                pros=[
                    "Shows quartiles and median",
                    "Identifies outliers clearly",
                    "Compact visualization",
                    "Good for comparing groups"
                ],
                cons=[
                    "Requires statistical understanding",
                    "Hides distribution shape"
                ],
                required_columns={"y": "numeric"},
                optional_features=["Grouped by category", "Violin overlay"],
                example_code=f"import vizforge as vz\nchart = vz.box(df, y='{self.numeric_cols[0]}')\nchart.show()"
            ))

        return recommendations

    def _recommend_relationship(self) -> List[ChartRecommendation]:
        """Recommend relationship/correlation charts."""
        recommendations = []

        # Need at least 2 numeric columns
        if self.n_numeric < 2:
            return recommendations

        # SCATTER PLOT - Best for relationships
        data_fit = 0.95
        best_practice = 0.95
        performance = 0.7 if not self.is_very_large_dataset else 0.5
        accessibility = 0.8
        aesthetic = 0.9

        score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
        confidence = score * 0.94

        recommendations.append(ChartRecommendation(
            chart_type=ChartType.SCATTER,
            category=ChartCategory.RELATIONSHIP,
            confidence=confidence,
            score=score,
            data_fit_score=data_fit,
            best_practice_score=best_practice,
            performance_score=performance,
            accessibility_score=accessibility,
            aesthetic_score=aesthetic,
            reasoning=f"Scatter plots reveal relationships between variables. Your {self.n_numeric} numeric columns are perfect for correlation analysis.",
            pros=[
                "Shows correlation patterns clearly",
                "Identifies clusters and outliers",
                "Supports additional dimensions (color, size)",
                "Interactive exploration"
            ],
            cons=[
                "Can be cluttered with large datasets" if self.is_large_dataset else "Clean visualization",
                "Requires numeric data"
            ],
            required_columns={"x": "numeric", "y": "numeric"},
            optional_features=["Color by category", "Size by value", "Trend line"],
            example_code=f"import vizforge as vz\nchart = vz.scatter(df, x='{self.numeric_cols[0]}', y='{self.numeric_cols[1] if self.n_numeric > 1 else self.numeric_cols[0]}')\nchart.show()"
        ))

        # HEATMAP - Good for correlation matrix
        if self.n_numeric >= 3:
            data_fit = 0.85
            best_practice = 0.9
            performance = 0.9
            accessibility = 0.7  # Color dependency
            aesthetic = 0.95

            score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
            confidence = score * 0.87

            recommendations.append(ChartRecommendation(
                chart_type=ChartType.HEATMAP,
                category=ChartCategory.RELATIONSHIP,
                confidence=confidence,
                score=score,
                data_fit_score=data_fit,
                best_practice_score=best_practice,
                performance_score=performance,
                accessibility_score=accessibility,
                aesthetic_score=aesthetic,
                reasoning="Heatmaps visualize correlation matrices effectively for multiple variables.",
                pros=[
                    "Shows all correlations at once",
                    "Color-coded for quick insights",
                    "Handles many variables",
                    "Visually appealing"
                ],
                cons=[
                    "Relies on color perception",
                    "Can be overwhelming with too many variables"
                ],
                required_columns={"values": "numeric"},
                optional_features=["Correlation coefficients", "Clustering", "Custom colormap"],
                example_code="import vizforge as vz\nchart = vz.heatmap(df.corr())\nchart.show()"
            ))

        return recommendations

    def _recommend_composition(self) -> List[ChartRecommendation]:
        """Recommend composition/part-to-whole charts."""
        recommendations = []

        # Need categorical with reasonable cardinality
        suitable_categorical = [col for col in self.categorical_cols
                               if 2 <= self.cardinalities[col] <= 7]

        if not suitable_categorical or self.n_numeric == 0:
            return recommendations

        # PIE CHART - For part-to-whole (limited use)
        cardinality = self.cardinalities[suitable_categorical[0]]

        data_fit = 0.7 if cardinality <= 5 else 0.5
        best_practice = 0.6  # Often overused
        performance = 0.95
        accessibility = 0.6  # Hard for precise comparison
        aesthetic = 0.8

        score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
        confidence = score * 0.75

        recommendations.append(ChartRecommendation(
            chart_type=ChartType.PIE,
            category=ChartCategory.COMPOSITION,
            confidence=confidence,
            score=score,
            data_fit_score=data_fit,
            best_practice_score=best_practice,
            performance_score=performance,
            accessibility_score=accessibility,
            aesthetic_score=aesthetic,
            reasoning=f"Pie chart shows part-to-whole relationships. Best with {cardinality} categories or fewer.",
            pros=[
                "Intuitive part-to-whole representation",
                "Visually appealing",
                "Works well for percentages"
            ],
            cons=[
                "Hard to compare slice sizes accurately",
                "Limited to few categories (ideally ≤5)",
                "Often misused - bar charts usually better"
            ],
            required_columns={"labels": "categorical", "values": "numeric"},
            optional_features=["Donut chart", "Exploded slices"],
            example_code=f"import vizforge as vz\nchart = vz.pie(df, labels='{suitable_categorical[0]}', values='{self.numeric_cols[0]}')\nchart.show()"
        ))

        return recommendations

    def _recommend_statistical(self) -> List[ChartRecommendation]:
        """Recommend statistical analysis charts."""
        recommendations = []

        # Need at least 2 numeric columns for correlation
        if self.n_numeric < 2:
            return recommendations

        # CORRELATION MATRIX
        data_fit = 0.9
        best_practice = 0.85
        performance = 0.9
        accessibility = 0.75
        aesthetic = 0.85

        score = (data_fit + best_practice + performance + accessibility + aesthetic) / 5
        confidence = score * 0.86

        recommendations.append(ChartRecommendation(
            chart_type=ChartType.CORRELATION,
            category=ChartCategory.STATISTICAL,
            confidence=confidence,
            score=score,
            data_fit_score=data_fit,
            best_practice_score=best_practice,
            performance_score=performance,
            accessibility_score=accessibility,
            aesthetic_score=aesthetic,
            reasoning=f"With {self.n_numeric} numeric variables, correlation analysis reveals relationships.",
            pros=[
                "Statistical rigor",
                "Identifies multicollinearity",
                "Foundation for feature selection"
            ],
            cons=[
                "Assumes linear relationships",
                "Requires statistical knowledge"
            ],
            required_columns={"variables": "numeric"},
            optional_features=["P-values", "Confidence intervals"],
            example_code="import vizforge as vz\nfrom vizforge.charts.stats import CorrelationMatrix\nchart = CorrelationMatrix(df)\nchart.show()"
        ))

        return recommendations

    def explain_recommendation(self, recommendation: ChartRecommendation) -> str:
        """
        Generate detailed explanation for a recommendation.

        Args:
            recommendation: ChartRecommendation to explain

        Returns:
            Detailed explanation string
        """
        explanation = f"""
╔════════════════════════════════════════════════════════════════════╗
║  {recommendation.chart_type.value.upper()} CHART RECOMMENDATION
╚════════════════════════════════════════════════════════════════════╝

📊 CONFIDENCE: {recommendation.confidence:.1%}
📈 OVERALL SCORE: {recommendation.score:.2f}/1.00

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETAILED SCORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Data Fit:          {"█" * int(recommendation.data_fit_score * 10)} {recommendation.data_fit_score:.0%}
  Best Practices:    {"█" * int(recommendation.best_practice_score * 10)} {recommendation.best_practice_score:.0%}
  Performance:       {"█" * int(recommendation.performance_score * 10)} {recommendation.performance_score:.0%}
  Accessibility:     {"█" * int(recommendation.accessibility_score * 10)} {recommendation.accessibility_score:.0%}
  Aesthetics:        {"█" * int(recommendation.aesthetic_score * 10)} {recommendation.aesthetic_score:.0%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{recommendation.reasoning}

✅ PROS:
{''.join(f'  • {pro}' + chr(10) for pro in recommendation.pros)}
⚠️  CONS:
{''.join(f'  • {con}' + chr(10) for con in recommendation.cons)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{recommendation.example_code}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return explanation

    def generate_report(self, top_n: int = 5) -> str:
        """
        Generate comprehensive recommendation report.

        Args:
            top_n: Number of recommendations to include

        Returns:
            Formatted report string
        """
        recommendations = self.recommend(top_n=top_n, min_confidence=0.3)

        report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          VIZFORGE SMART CHART RECOMMENDER v2 - ANALYSIS REPORT           ║
╚══════════════════════════════════════════════════════════════════════════╝

📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total Rows:        {self.n_rows:,}
  Total Columns:     {self.n_cols}

  Column Types:
    • Numeric:       {self.n_numeric} columns
    • Categorical:   {self.n_categorical} columns
    • Datetime:      {self.n_datetime} columns

  Dataset Size:      {"Very Large (>100k)" if self.is_very_large_dataset else "Large (>10k)" if self.is_large_dataset else "Medium/Small"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP {len(recommendations)} CHART RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

        for i, rec in enumerate(recommendations, 1):
            confidence_bar = "█" * int(rec.confidence * 20)
            report += f"""
{i}. {rec.chart_type.value.upper()} ({rec.category.value})
   Confidence: {confidence_bar} {rec.confidence:.0%}
   Score: {rec.score:.2f}/1.00

   {rec.reasoning}

   Quick Start:
   {rec.example_code}

{'─' * 78}
"""

        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 For detailed explanation of any recommendation:
   recommender.explain_recommendation(recommendations[0])

📚 For best practices:
   - Start with the highest confidence recommendation
   - Consider your audience's visualization literacy
   - Test multiple chart types for complex data
   - Prioritize accessibility and clarity over aesthetics

╚══════════════════════════════════════════════════════════════════════════╝
"""

        return report


# ==================== Convenience Functions ====================

def recommend_chart(data: pd.DataFrame, top_n: int = 3, verbose: bool = False) -> List[ChartRecommendation]:
    """
    Quick chart recommendation (one-liner).

    Args:
        data: DataFrame to analyze
        top_n: Number of recommendations
        verbose: Print analysis steps

    Returns:
        List of ChartRecommendation objects

    Example:
        >>> recommendations = recommend_chart(df, top_n=3)
        >>> print(recommendations[0])
    """
    recommender = SmartChartRecommender(data, verbose=verbose)
    return recommender.recommend(top_n=top_n)


def get_recommendation_report(data: pd.DataFrame, top_n: int = 5) -> str:
    """
    Get comprehensive recommendation report (one-liner).

    Args:
        data: DataFrame to analyze
        top_n: Number of recommendations

    Returns:
        Formatted report string

    Example:
        >>> report = get_recommendation_report(df)
        >>> print(report)
    """
    recommender = SmartChartRecommender(data)
    return recommender.generate_report(top_n=top_n)
