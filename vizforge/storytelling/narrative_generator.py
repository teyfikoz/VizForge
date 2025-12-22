"""
VizForge Narrative Generator

Convert data insights into compelling narratives.
NO API required - template-based story generation!
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

from .insight_discovery import Insight, InsightType, discover_insights


@dataclass
class DataStory:
    """
    A data-driven story/narrative.

    Attributes:
        title: Story title
        summary: Executive summary
        narrative: Full narrative text
        insights: List of insights used
        recommendations: List of recommendations
        metadata: Additional metadata
    """
    title: str
    summary: str
    narrative: str
    insights: List[Insight]
    recommendations: List[str]
    metadata: Dict


class NarrativeGenerator:
    """
    Automatic narrative generation from data insights.

    Converts discovered insights into compelling stories.

    Examples:
        >>> generator = NarrativeGenerator(df)
        >>> story = generator.generate()
        >>> print(story.title)
        >>> print(story.narrative)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        tone: str = 'professional',  # 'professional', 'casual', 'technical'
    ):
        """
        Initialize narrative generator.

        Args:
            data: DataFrame to analyze
            title: Optional custom title
            tone: Narrative tone ('professional', 'casual', 'technical')
        """
        self.data = data
        self.custom_title = title
        self.tone = tone

    def generate(
        self,
        max_insights: int = 10,
        include_recommendations: bool = True
    ) -> DataStory:
        """
        Generate narrative story from data.

        Args:
            max_insights: Maximum insights to include
            include_recommendations: Include recommendations

        Returns:
            DataStory object with narrative and insights
        """
        # Discover insights
        insights = discover_insights(self.data, max_insights=max_insights)

        # Generate title
        title = self._generate_title(insights)

        # Generate summary
        summary = self._generate_summary(insights)

        # Generate full narrative
        narrative = self._generate_narrative(insights)

        # Extract recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._extract_recommendations(insights)

        # Metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'insights_count': len(insights),
            'tone': self.tone,
        }

        return DataStory(
            title=title,
            summary=summary,
            narrative=narrative,
            insights=insights,
            recommendations=recommendations,
            metadata=metadata
        )

    def _generate_title(self, insights: List[Insight]) -> str:
        """Generate story title."""
        if self.custom_title:
            return self.custom_title

        # Use most important insight for title
        if insights:
            top_insight = insights[0]

            if top_insight.type == InsightType.TREND:
                return f"Data Analysis: {top_insight.title} Detected"
            elif top_insight.type == InsightType.ANOMALY:
                return f"Alert: {top_insight.title}"
            else:
                return f"Data Insights: {top_insight.title}"

        return "Data Analysis Report"

    def _generate_summary(self, insights: List[Insight]) -> str:
        """Generate executive summary."""
        if not insights:
            return "No significant insights discovered in the data."

        # Count insights by type
        type_counts = {}
        for insight in insights:
            type_name = insight.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        summary_parts = []

        # Opening
        summary_parts.append(
            f"Analysis of the dataset revealed {len(insights)} key insights. "
        )

        # Highlight top finding
        top = insights[0]
        summary_parts.append(
            f"Most notably, {top.description.lower()} "
        )

        # Mention other insight types
        if len(type_counts) > 1:
            types = ", ".join([f"{count} {type_name}" for type_name, count in list(type_counts.items())[:3]])
            summary_parts.append(
                f"The analysis includes {types} insights. "
            )

        return "".join(summary_parts)

    def _generate_narrative(self, insights: List[Insight]) -> str:
        """Generate full narrative."""
        if not insights:
            return "No significant patterns or insights were discovered in the data."

        narrative_parts = []

        # Introduction
        narrative_parts.append(self._generate_introduction())

        # Group insights by type
        insights_by_type = {}
        for insight in insights:
            type_name = insight.type.value
            if type_name not in insights_by_type:
                insights_by_type[type_name] = []
            insights_by_type[type_name].append(insight)

        # Generate sections for each type
        section_order = [
            InsightType.SUMMARY,
            InsightType.TREND,
            InsightType.CHANGE,
            InsightType.CORRELATION,
            InsightType.SEASONALITY,
            InsightType.ANOMALY,
            InsightType.COMPARISON,
            InsightType.EXTREME,
            InsightType.DISTRIBUTION,
        ]

        for insight_type in section_order:
            type_name = insight_type.value
            if type_name in insights_by_type:
                section = self._generate_section(insight_type, insights_by_type[type_name])
                if section:
                    narrative_parts.append(f"\n\n## {self._get_section_title(insight_type)}\n\n")
                    narrative_parts.append(section)

        # Conclusion
        narrative_parts.append(self._generate_conclusion(insights))

        return "".join(narrative_parts)

    def _generate_introduction(self) -> str:
        """Generate introduction paragraph."""
        n_rows = len(self.data)
        n_cols = len(self.data.columns)

        intro = (
            f"# Data Analysis Report\n\n"
            f"This report presents a comprehensive analysis of a dataset containing "
            f"{n_rows:,} records across {n_cols} variables. "
            f"The analysis employed statistical methods to identify key patterns, "
            f"trends, and anomalies within the data."
        )

        return intro

    def _generate_section(self, insight_type: InsightType, insights: List[Insight]) -> str:
        """Generate section for specific insight type."""
        if not insights:
            return ""

        section_parts = []

        for i, insight in enumerate(insights, 1):
            # Add insight description
            section_parts.append(f"{i}. **{insight.title}**: {insight.description}\n\n")

            # Add recommendation if available
            if insight.recommendation:
                section_parts.append(f"   💡 *Recommendation: {insight.recommendation}*\n\n")

        return "".join(section_parts)

    def _get_section_title(self, insight_type: InsightType) -> str:
        """Get section title for insight type."""
        titles = {
            InsightType.SUMMARY: "Overview",
            InsightType.TREND: "Trends",
            InsightType.CORRELATION: "Correlations",
            InsightType.ANOMALY: "Anomalies",
            InsightType.EXTREME: "Extreme Values",
            InsightType.DISTRIBUTION: "Distributions",
            InsightType.SEASONALITY: "Seasonal Patterns",
            InsightType.COMPARISON: "Comparisons",
            InsightType.CHANGE: "Changes Over Time",
        }
        return titles.get(insight_type, insight_type.value.title())

    def _generate_conclusion(self, insights: List[Insight]) -> str:
        """Generate conclusion paragraph."""
        if not insights:
            return ""

        # Count high-importance insights
        important_insights = [i for i in insights if i.importance > 0.8]

        conclusion = (
            f"\n\n## Conclusion\n\n"
            f"This analysis identified {len(insights)} significant insights, "
            f"with {len(important_insights)} rated as high importance. "
        )

        # Highlight actionable items
        insights_with_recs = [i for i in insights if i.recommendation]
        if insights_with_recs:
            conclusion += (
                f"{len(insights_with_recs)} insights include specific recommendations for action. "
            )

        conclusion += (
            "These findings can inform data-driven decision making and strategic planning."
        )

        return conclusion

    def _extract_recommendations(self, insights: List[Insight]) -> List[str]:
        """Extract recommendations from insights."""
        recommendations = []

        for insight in insights:
            if insight.recommendation:
                recommendations.append(insight.recommendation)

        # Add general recommendations
        if not recommendations:
            recommendations.append("Continue monitoring key metrics for changes.")
            recommendations.append("Investigate any anomalies or unexpected patterns.")

        return recommendations


# ==================== Convenience Function ====================

def generate_story(
    data: pd.DataFrame,
    title: Optional[str] = None,
    max_insights: int = 10,
    tone: str = 'professional',
) -> DataStory:
    """
    Generate data-driven story (one-liner!).

    Args:
        data: DataFrame to analyze
        title: Optional custom title
        max_insights: Maximum insights to include
        tone: Narrative tone ('professional', 'casual', 'technical')

    Returns:
        DataStory object with narrative and insights

    Examples:
        >>> from vizforge.storytelling import generate_story
        >>>
        >>> story = generate_story(df, max_insights=5)
        >>> print(story.title)
        >>> print(story.summary)
        >>> print(story.narrative)
        >>>
        >>> # Save as markdown
        >>> with open('report.md', 'w') as f:
        >>>     f.write(story.narrative)
    """
    generator = NarrativeGenerator(data=data, title=title, tone=tone)
    return generator.generate(max_insights=max_insights)
