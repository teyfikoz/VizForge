"""
VizForge Auto Data Storytelling Module

Automatically discover insights and generate narratives from data.
NO API required - intelligent pattern recognition!

Examples:
    >>> import vizforge as vz
    >>> from vizforge.storytelling import generate_story, discover_insights
    >>>
    >>> # Generate automatic story
    >>> story = vz.generate_story(df)
    >>> print(story.narrative)
    >>>
    >>> # Discover key insights
    >>> insights = vz.discover_insights(df)
    >>> for insight in insights:
    >>>     print(f"- {insight.description}")
"""

from .insight_discovery import (
    InsightDiscovery,
    discover_insights,
    Insight,
    InsightType,
)

from .narrative_generator import (
    NarrativeGenerator,
    generate_story,
    DataStory,
)

from .report_generator import (
    ReportGenerator,
    generate_report,
    ReportFormat,
)

__all__ = [
    # Insight Discovery
    'InsightDiscovery',
    'discover_insights',
    'Insight',
    'InsightType',

    # Narrative Generation
    'NarrativeGenerator',
    'generate_story',
    'DataStory',

    # Report Generation
    'ReportGenerator',
    'generate_report',
    'ReportFormat',
]
