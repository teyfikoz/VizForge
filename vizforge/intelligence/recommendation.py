"""
VizForge Recommendation Engine

Best practices and improvement recommendations (NO API costs).
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class RecommendationType(Enum):
    """Types of recommendations."""
    CHART_TYPE = "chart_type"
    DATA_QUALITY = "data_quality"
    VISUAL_DESIGN = "visual_design"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    STORYTELLING = "storytelling"


@dataclass
class Recommendation:
    """
    A visualization improvement recommendation.

    Attributes:
        type: Type of recommendation
        priority: Priority level (1=highest, 5=lowest)
        title: Short recommendation title
        description: Detailed explanation
        action: Suggested action to take
        rationale: Why this is important
    """
    type: RecommendationType
    priority: int
    title: str
    description: str
    action: str
    rationale: str


class RecommendationEngine:
    """
    Generate best practice recommendations for visualizations.

    Provides expert guidance WITHOUT expensive AI APIs - based on
    data visualization research and industry standards.

    Features:
    - Chart type recommendations
    - Data quality warnings
    - Visual design tips
    - Accessibility guidelines
    - Performance optimization
    - Storytelling advice

    Example:
        >>> engine = RecommendationEngine()
        >>> recommendations = engine.analyze(df, chart_type='BarChart')
        >>> for rec in recommendations:
        ...     print(f"[P{rec.priority}] {rec.title}")
        ...     print(f"    Action: {rec.action}")
    """

    # Best practice rules
    MAX_CATEGORIES_PIE = 7  # Max categories for pie chart
    MAX_CATEGORIES_BAR = 20  # Max categories for bar chart
    MAX_POINTS_SCATTER = 10000  # Max points before WebGL needed
    MIN_SAMPLES_STATS = 30  # Minimum samples for statistical charts

    def __init__(self):
        """Initialize recommendation engine."""
        pass

    def analyze(
        self,
        data: pd.DataFrame,
        chart_type: Optional[str] = None,
        x: Optional[str] = None,
        y: Optional[str] = None
    ) -> List[Recommendation]:
        """
        Analyze data and chart setup, generate recommendations.

        Args:
            data: Input DataFrame
            chart_type: Chosen chart type (e.g., 'BarChart', 'LineChart')
            x: X-axis column
            y: Y-axis column

        Returns:
            List of Recommendation objects sorted by priority

        Example:
            >>> recs = engine.analyze(df, chart_type='PieChart', x='category', y='value')
            >>> print(f"Found {len(recs)} recommendations")
        """
        recommendations = []

        # Chart type recommendations
        if chart_type:
            recommendations.extend(
                self._recommend_chart_improvements(data, chart_type, x, y)
            )

        # Data quality recommendations
        recommendations.extend(self._recommend_data_quality(data))

        # Visual design recommendations
        recommendations.extend(self._recommend_visual_design(data, chart_type))

        # Accessibility recommendations
        recommendations.extend(self._recommend_accessibility(chart_type))

        # Performance recommendations
        recommendations.extend(self._recommend_performance(data, chart_type))

        # Sort by priority
        return sorted(recommendations, key=lambda r: r.priority)

    def _recommend_chart_improvements(
        self,
        data: pd.DataFrame,
        chart_type: str,
        x: Optional[str],
        y: Optional[str]
    ) -> List[Recommendation]:
        """Recommend chart-specific improvements."""
        recommendations = []

        # Pie chart - too many categories
        if chart_type in ['PieChart', 'DonutChart']:
            if x and data[x].nunique() > self.MAX_CATEGORIES_PIE:
                recommendations.append(Recommendation(
                    type=RecommendationType.CHART_TYPE,
                    priority=1,
                    title="Too many categories for pie chart",
                    description=f"Pie charts work best with ≤{self.MAX_CATEGORIES_PIE} categories. "
                              f"Your data has {data[x].nunique()} categories.",
                    action="Consider using a BarChart or Treemap instead",
                    rationale="Pie charts become unreadable with many slices. "
                             "Humans struggle to compare angles beyond 5-7 slices."
                ))

        # Bar chart - too many categories
        elif chart_type in ['BarChart', 'ColumnChart']:
            if x and data[x].nunique() > self.MAX_CATEGORIES_BAR:
                recommendations.append(Recommendation(
                    type=RecommendationType.CHART_TYPE,
                    priority=2,
                    title="Too many categories for bar chart",
                    description=f"Bar charts work best with ≤{self.MAX_CATEGORIES_BAR} categories. "
                              f"Your data has {data[x].nunique()} categories.",
                    action="Consider using a Treemap, filtering top N, or grouping categories",
                    rationale="Too many bars create clutter and make labels unreadable."
                ))

        # Scatter plot - too many points
        elif chart_type in ['ScatterPlot', 'Scatter']:
            if len(data) > self.MAX_POINTS_SCATTER:
                recommendations.append(Recommendation(
                    type=RecommendationType.PERFORMANCE,
                    priority=2,
                    title="Large dataset - enable WebGL rendering",
                    description=f"Your dataset has {len(data)} points. "
                              f"Standard rendering may be slow.",
                    action="Enable WebGL with chart.enable_webgl() or use HexbinPlot",
                    rationale="WebGL rendering is 2-10x faster for large datasets."
                ))

        # Line chart - check temporal data
        elif chart_type in ['LineChart', 'Line']:
            if x and not pd.api.types.is_datetime64_any_dtype(data[x]):
                recommendations.append(Recommendation(
                    type=RecommendationType.CHART_TYPE,
                    priority=3,
                    title="X-axis may not be temporal",
                    description=f"Line charts typically show trends over time, "
                              f"but '{x}' doesn't appear to be a date/time column.",
                    action="Verify that X-axis represents ordered sequential data",
                    rationale="Line charts imply continuity. Use BarChart for categorical X-axis."
                ))

        return recommendations

    def _recommend_data_quality(self, data: pd.DataFrame) -> List[Recommendation]:
        """Recommend data quality improvements."""
        recommendations = []

        # Missing values warning
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        if missing_pct > 10:
            recommendations.append(Recommendation(
                type=RecommendationType.DATA_QUALITY,
                priority=1,
                title="High percentage of missing values",
                description=f"{missing_pct:.1f}% of your data is missing",
                action="Handle missing values with imputation, removal, or explicit visualization",
                rationale="Missing data can mislead viewers if not properly addressed."
            ))

        # Duplicate rows warning
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(data)) * 100
            if dup_pct > 5:
                recommendations.append(Recommendation(
                    type=RecommendationType.DATA_QUALITY,
                    priority=2,
                    title="Duplicate rows detected",
                    description=f"{duplicates} duplicate rows ({dup_pct:.1f}%) found in dataset",
                    action="Remove duplicates with df.drop_duplicates() or aggregate",
                    rationale="Duplicates can skew aggregations and mislead analysis."
                ))

        # Small sample size warning
        if len(data) < self.MIN_SAMPLES_STATS:
            recommendations.append(Recommendation(
                type=RecommendationType.DATA_QUALITY,
                priority=3,
                title="Small sample size",
                description=f"Only {len(data)} rows in dataset",
                action="Be cautious with statistical conclusions. Consider gathering more data.",
                rationale="Small samples have high variance and may not be representative."
            ))

        return recommendations

    def _recommend_visual_design(
        self,
        data: pd.DataFrame,
        chart_type: Optional[str]
    ) -> List[Recommendation]:
        """Recommend visual design improvements."""
        recommendations = []

        # Always recommend clear titles
        recommendations.append(Recommendation(
            type=RecommendationType.VISUAL_DESIGN,
            priority=4,
            title="Add descriptive title",
            description="Clear titles help viewers understand the chart's purpose",
            action="Set chart title that describes what is being shown and why",
            rationale="Research shows titled charts are understood 3x faster than untitled ones."
        ))

        # Recommend axis labels
        recommendations.append(Recommendation(
            type=RecommendationType.VISUAL_DESIGN,
            priority=4,
            title="Label axes with units",
            description="Axis labels should include units (e.g., 'Revenue ($M)', 'Time (hours)')",
            action="Use chart.update_xaxis(title='...') and chart.update_yaxis(title='...')",
            rationale="Units prevent misinterpretation of scale and magnitude."
        ))

        return recommendations

    def _recommend_accessibility(
        self,
        chart_type: Optional[str]
    ) -> List[Recommendation]:
        """Recommend accessibility improvements."""
        recommendations = []

        # Color-blind friendly palette
        recommendations.append(Recommendation(
            type=RecommendationType.ACCESSIBILITY,
            priority=2,
            title="Use color-blind friendly palette",
            description="8% of men have color vision deficiency",
            action="Apply chart.make_accessible('AA') for WCAG 2.1 AA compliance",
            rationale="Accessible charts reach wider audience and meet legal requirements."
        ))

        # Sufficient contrast
        recommendations.append(Recommendation(
            type=RecommendationType.ACCESSIBILITY,
            priority=3,
            title="Ensure sufficient color contrast",
            description="Text and chart elements need ≥4.5:1 contrast ratio (WCAG AA)",
            action="Use high-contrast theme or chart.make_accessible('AAA') for 7:1 ratio",
            rationale="Low contrast makes charts unusable for people with vision impairments."
        ))

        return recommendations

    def _recommend_performance(
        self,
        data: pd.DataFrame,
        chart_type: Optional[str]
    ) -> List[Recommendation]:
        """Recommend performance optimizations."""
        recommendations = []

        # Large dataset - sampling
        if len(data) > 100000:
            recommendations.append(Recommendation(
                type=RecommendationType.PERFORMANCE,
                priority=2,
                title="Consider data sampling for preview",
                description=f"Dataset has {len(data):,} rows - may slow down rendering",
                action="Use data.sample(n=10000) for preview, full data for export",
                rationale="Interactive performance degrades significantly beyond 100k points."
            ))

        # Wide dataset - select columns
        if data.shape[1] > 50:
            recommendations.append(Recommendation(
                type=RecommendationType.PERFORMANCE,
                priority=3,
                title="Too many columns",
                description=f"Dataset has {data.shape[1]} columns",
                action="Select only relevant columns for visualization",
                rationale="Fewer columns improve load time and reduce memory usage."
            ))

        return recommendations

    def summarize_recommendations(self, recommendations: List[Recommendation]) -> str:
        """
        Generate human-readable summary of recommendations.

        Args:
            recommendations: List of recommendations

        Returns:
            Formatted summary string
        """
        if not recommendations:
            return "No recommendations - chart setup looks good!"

        summary_lines = ["💡 Recommendations", "=" * 50, ""]

        # Group by priority
        by_priority = {}
        for rec in recommendations:
            if rec.priority not in by_priority:
                by_priority[rec.priority] = []
            by_priority[rec.priority].append(rec)

        # Display by priority
        priority_labels = {
            1: "🔴 CRITICAL",
            2: "🟠 HIGH",
            3: "🟡 MEDIUM",
            4: "🔵 LOW",
            5: "ℹ️ INFO"
        }

        for priority in sorted(by_priority.keys()):
            label = priority_labels.get(priority, f"Priority {priority}")
            summary_lines.append(f"{label} ({len(by_priority[priority])})")

            for rec in by_priority[priority]:
                summary_lines.append(f"  • {rec.title}")
                summary_lines.append(f"    → {rec.action}")
            summary_lines.append("")

        return "\n".join(summary_lines)
