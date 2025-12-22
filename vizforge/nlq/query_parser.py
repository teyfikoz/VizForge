"""
VizForge NLQ Query Parser

Parses natural language queries and detects intent.
NO API required - rule-based pattern matching.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


class Intent(Enum):
    """Query intent types."""
    TREND = "trend"  # "show trend", "over time"
    COMPARISON = "comparison"  # "compare", "vs", "versus"
    DISTRIBUTION = "distribution"  # "distribution of", "histogram"
    CORRELATION = "correlation"  # "correlation", "relationship"
    TOP_N = "top_n"  # "top 10", "bottom 5"
    AGGREGATION = "aggregation"  # "total", "average", "sum"
    FILTER = "filter"  # "where", "for", "only"
    BREAKDOWN = "breakdown"  # "by region", "group by"
    ANOMALY = "anomaly"  # "anomalies", "outliers", "unusual"
    FORECAST = "forecast"  # "predict", "forecast", "future"
    UNKNOWN = "unknown"


class Aggregation(Enum):
    """Aggregation types."""
    SUM = "sum"
    AVG = "average"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STD = "std"


@dataclass
class ParsedQuery:
    """Parsed query result."""
    original_query: str
    intent: Intent
    confidence: float  # 0.0 to 1.0

    # Extracted components
    metrics: List[str]  # Numeric columns to analyze
    dimensions: List[str]  # Categorical columns for grouping
    time_column: Optional[str]  # Time-based column
    filters: List[Dict[str, Any]]  # Filter conditions
    aggregation: Optional[Aggregation]
    limit: Optional[int]  # Top N

    # Additional metadata
    keywords: List[str]
    chart_suggestion: Optional[str]


class QueryParser:
    """
    Natural Language Query Parser.

    Detects intent and extracts entities from queries like:
    - "Show me sales trend by month"
    - "Compare revenue vs profit by region"
    - "Find top 10 products by sales"
    - "What is the correlation between price and demand?"
    """

    def __init__(self):
        """Initialize parser with pattern rules."""
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for intent detection."""

        # TREND patterns
        self.trend_patterns = [
            r'\b(trend|trends|trending|over time|time series|change|changes|changing)\b',
            r'\b(by (month|quarter|year|day|week|date|time))\b',
            r'\b(historical|history|evolution|progress)\b',
        ]

        # COMPARISON patterns
        self.comparison_patterns = [
            r'\b(compare|comparison|vs|versus|against|between)\b',
            r'\b(difference|differences|gap|gaps)\b',
            r'\b(better|worse|higher|lower)\b',
        ]

        # DISTRIBUTION patterns
        self.distribution_patterns = [
            r'\b(distribution|histogram|spread|range)\b',
            r'\b(how (many|much)|frequency)\b',
            r'\b(breakdown|split)\b',
        ]

        # CORRELATION patterns
        self.correlation_patterns = [
            r'\b(correlation|correlate|relationship|related|association)\b',
            r'\b(impact|effect|influence|affect)\b',
            r'\b(depends on|dependent|independent)\b',
        ]

        # TOP_N patterns
        self.topn_patterns = [
            r'\b(top|bottom|best|worst|highest|lowest)\s+(\d+)\b',
            r'\b(first|last)\s+(\d+)\b',
            r'\b(most|least)\b',
        ]

        # AGGREGATION patterns
        self.aggregation_patterns = [
            (r'\b(total|sum|add up)\b', Aggregation.SUM),
            (r'\b(average|avg|mean)\b', Aggregation.AVG),
            (r'\b(count|number of|how many)\b', Aggregation.COUNT),
            (r'\b(minimum|min|smallest)\b', Aggregation.MIN),
            (r'\b(maximum|max|largest|biggest)\b', Aggregation.MAX),
            (r'\b(median|middle)\b', Aggregation.MEDIAN),
        ]

        # FILTER patterns
        self.filter_patterns = [
            r'\b(where|when|for|only|just|specifically)\b',
            r'\b(in|during|from|between)\b',
        ]

        # BREAKDOWN patterns
        self.breakdown_patterns = [
            r'\b(by|per|each|every|grouped by|group by)\b',
            r'\b(across|among)\b',
        ]

        # ANOMALY patterns
        self.anomaly_patterns = [
            r'\b(anomaly|anomalies|outlier|outliers|unusual|abnormal)\b',
            r'\b(spike|spikes|dip|dips|sudden)\b',
            r'\b(unexpected|strange|odd)\b',
        ]

        # FORECAST patterns
        self.forecast_patterns = [
            r'\b(forecast|predict|prediction|future|next|upcoming)\b',
            r'\b(will|expected|projected|projection)\b',
        ]

        # TIME column keywords
        self.time_keywords = [
            'date', 'time', 'timestamp', 'year', 'month', 'day',
            'week', 'quarter', 'hour', 'minute', 'second'
        ]

    def parse(self, query: str, available_columns: List[str] = None) -> ParsedQuery:
        """
        Parse natural language query.

        Args:
            query: Natural language question
            available_columns: List of available column names in data

        Returns:
            ParsedQuery object with detected intent and extracted entities
        """
        query_lower = query.lower()

        # Detect intent
        intent, confidence = self._detect_intent(query_lower)

        # Extract components
        metrics = self._extract_metrics(query_lower, available_columns)
        dimensions = self._extract_dimensions(query_lower, available_columns)
        time_column = self._extract_time_column(query_lower, available_columns)
        filters = self._extract_filters(query_lower)
        aggregation = self._extract_aggregation(query_lower)
        limit = self._extract_limit(query_lower)
        keywords = self._extract_keywords(query_lower)

        # Suggest chart type
        chart_suggestion = self._suggest_chart(intent, metrics, dimensions, time_column)

        return ParsedQuery(
            original_query=query,
            intent=intent,
            confidence=confidence,
            metrics=metrics,
            dimensions=dimensions,
            time_column=time_column,
            filters=filters,
            aggregation=aggregation,
            limit=limit,
            keywords=keywords,
            chart_suggestion=chart_suggestion
        )

    def _detect_intent(self, query: str) -> tuple[Intent, float]:
        """Detect primary intent of query."""
        scores = {}

        # Score each intent
        scores[Intent.TREND] = self._pattern_score(query, self.trend_patterns)
        scores[Intent.COMPARISON] = self._pattern_score(query, self.comparison_patterns)
        scores[Intent.DISTRIBUTION] = self._pattern_score(query, self.distribution_patterns)
        scores[Intent.CORRELATION] = self._pattern_score(query, self.correlation_patterns)
        scores[Intent.TOP_N] = self._pattern_score(query, self.topn_patterns)
        scores[Intent.ANOMALY] = self._pattern_score(query, self.anomaly_patterns)
        scores[Intent.FORECAST] = self._pattern_score(query, self.forecast_patterns)

        # Get highest score
        if scores:
            max_intent = max(scores.items(), key=lambda x: x[1])
            if max_intent[1] > 0:
                return max_intent[0], min(max_intent[1], 1.0)

        return Intent.UNKNOWN, 0.0

    def _pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate pattern match score."""
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3  # Each pattern match adds 0.3
        return min(score, 1.0)

    def _extract_metrics(self, query: str, columns: List[str] = None) -> List[str]:
        """Extract metric (numeric) columns mentioned in query."""
        if not columns:
            # Try to extract from query
            return self._extract_column_names(query)

        metrics = []
        for col in columns:
            # Look for column name in query
            if col.lower() in query:
                # Check if it's likely a numeric metric
                if any(kw in col.lower() for kw in [
                    'sales', 'revenue', 'profit', 'cost', 'price', 'amount',
                    'count', 'quantity', 'total', 'value', 'number'
                ]):
                    metrics.append(col)

        return metrics

    def _extract_dimensions(self, query: str, columns: List[str] = None) -> List[str]:
        """Extract dimension (categorical) columns mentioned in query."""
        if not columns:
            return self._extract_column_names(query)

        dimensions = []
        for col in columns:
            if col.lower() in query:
                # Check if it's likely a categorical dimension
                if any(kw in col.lower() for kw in [
                    'region', 'category', 'type', 'name', 'product', 'customer',
                    'city', 'country', 'state', 'department', 'status'
                ]):
                    dimensions.append(col)

        return dimensions

    def _extract_time_column(self, query: str, columns: List[str] = None) -> Optional[str]:
        """Extract time-based column."""
        if not columns:
            # Look for time keywords in query
            for kw in self.time_keywords:
                if kw in query:
                    return kw
            return None

        for col in columns:
            if col.lower() in query:
                # Check if it's likely a time column
                if any(kw in col.lower() for kw in self.time_keywords):
                    return col

        return None

    def _extract_filters(self, query: str) -> List[Dict[str, Any]]:
        """Extract filter conditions."""
        filters = []

        # Look for "where X = Y" patterns
        where_pattern = r'where\s+(\w+)\s*(=|>|<|>=|<=)\s*([^\s,]+)'
        matches = re.findall(where_pattern, query, re.IGNORECASE)
        for match in matches:
            filters.append({
                'column': match[0],
                'operator': match[1],
                'value': match[2]
            })

        # Look for "for X" patterns
        for_pattern = r'for\s+(\w+)'
        matches = re.findall(for_pattern, query, re.IGNORECASE)
        for match in matches:
            filters.append({
                'column': 'category',
                'operator': '=',
                'value': match
            })

        return filters

    def _extract_aggregation(self, query: str) -> Optional[Aggregation]:
        """Extract aggregation type."""
        for pattern, agg_type in self.aggregation_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return agg_type
        return None

    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract TOP N limit."""
        # Look for "top 10", "first 5", etc.
        top_pattern = r'\b(top|bottom|first|last)\s+(\d+)\b'
        match = re.search(top_pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(2))

        return None

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was',
                    'were', 'be', 'been', 'show', 'me', 'get', 'find'}

        words = re.findall(r'\b\w+\b', query)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        return keywords

    def _extract_column_names(self, query: str) -> List[str]:
        """Extract potential column names from query."""
        # Look for capitalized words or quoted strings
        column_pattern = r'[A-Z][a-z]+|"([^"]+)"|\'([^\']+)\''
        matches = re.findall(column_pattern, query)
        columns = []
        for match in matches:
            if isinstance(match, tuple):
                columns.extend([m for m in match if m])
            else:
                columns.append(match)
        return columns

    def _suggest_chart(self, intent: Intent, metrics: List[str],
                      dimensions: List[str], time_column: Optional[str]) -> Optional[str]:
        """Suggest appropriate chart type based on intent and data."""

        if intent == Intent.TREND and time_column:
            return "line"

        elif intent == Intent.COMPARISON:
            if len(metrics) == 2:
                return "scatter"
            elif dimensions:
                return "bar"
            return "bar"

        elif intent == Intent.DISTRIBUTION:
            return "histogram"

        elif intent == Intent.CORRELATION:
            if len(metrics) >= 2:
                return "scatter"
            return "heatmap"

        elif intent == Intent.TOP_N:
            return "bar"

        elif intent == Intent.BREAKDOWN:
            if dimensions:
                return "pie" if len(dimensions) == 1 else "treemap"
            return "bar"

        elif intent == Intent.ANOMALY:
            return "scatter"

        elif intent == Intent.FORECAST:
            return "line"

        return None
