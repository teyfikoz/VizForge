"""
VizForge NLQ Engine - Main Entry Point

The magic happens here! Ask questions, get visualizations.
NO API required - pure intelligence!

Examples:
    >>> import vizforge as vz
    >>> chart = vz.ask("Show me sales trend by month", df)
    >>> chart = vz.ask("Compare revenue vs profit", df)
    >>> chart = vz.ask("Find top 10 products", df)
"""

from typing import Any, Optional, Union
import pandas as pd
import numpy as np

from .query_parser import QueryParser, Intent, ParsedQuery
from .entity_extractor import EntityExtractor


class NLQEngine:
    """
    Natural Language Query Engine.

    The core intelligence that converts English questions
    into beautiful visualizations.

    Examples:
        >>> engine = NLQEngine(df)
        >>> chart = engine.ask("Show sales trend by month")
        >>> chart.show()

        >>> # With more context
        >>> chart = engine.ask("Compare revenue vs profit by region")
        >>> chart.show()
    """

    def __init__(self, dataframe: pd.DataFrame, verbose: bool = False):
        """
        Initialize NLQ Engine.

        Args:
            dataframe: Data to visualize
            verbose: Print processing steps
        """
        self.df = dataframe
        self.verbose = verbose

        # Initialize components
        self.parser = QueryParser()
        self.extractor = EntityExtractor(dataframe)

        if verbose:
            print(f"🧠 NLQ Engine initialized")
            print(f"   - Rows: {len(dataframe):,}")
            print(f"   - Columns: {len(dataframe.columns)}")
            print(f"   - Numeric: {len(self.extractor.get_numeric_columns())}")
            print(f"   - Categorical: {len(self.extractor.get_categorical_columns())}")
            print(f"   - Datetime: {len(self.extractor.get_datetime_columns())}")

    def ask(self, query: str) -> Any:
        """
        Ask a question in natural language.

        Args:
            query: Natural language question

        Returns:
            Visualization chart object

        Examples:
            >>> chart = engine.ask("Show me sales trend")
            >>> chart = engine.ask("Compare revenue vs profit")
            >>> chart = engine.ask("Find top 10 products by sales")
        """
        if self.verbose:
            print(f"\n🎤 Query: '{query}'")
            print("━" * 60)

        # Parse query
        parsed = self.parser.parse(query, list(self.df.columns))

        if self.verbose:
            print(f"📊 Intent: {parsed.intent.value} ({parsed.confidence:.0%} confidence)")
            if parsed.metrics:
                print(f"   Metrics: {parsed.metrics}")
            if parsed.dimensions:
                print(f"   Dimensions: {parsed.dimensions}")
            if parsed.time_column:
                print(f"   Time: {parsed.time_column}")
            print(f"   Suggested chart: {parsed.chart_suggestion}")

        # Extract entities
        entities = self.extractor.extract_entities(query)

        # Generate visualization
        chart = self._generate_chart(parsed, entities)

        if self.verbose:
            print(f"✅ Chart generated: {type(chart).__name__}")
            print("━" * 60)

        return chart

    def _generate_chart(self, parsed: ParsedQuery, entities: list) -> Any:
        """Generate appropriate chart based on parsed query."""

        # Import here to avoid circular imports
        from ..charts import (
            LineChart, BarChart, ScatterPlot, Histogram,
            PieChart, Heatmap, Boxplot
        )

        # Get column suggestions
        suggestions = self.extractor.suggest_columns_for_intent(parsed.intent.value)

        # Determine columns to use
        x_col, y_col = self._determine_columns(parsed, suggestions, entities)

        # Apply filters if any
        df_filtered = self._apply_filters(self.df, parsed.filters)

        # Apply aggregation if specified
        if parsed.aggregation:
            df_filtered = self._apply_aggregation(df_filtered, parsed.aggregation, x_col, y_col)

        # Apply limit if specified (TOP N)
        if parsed.limit:
            df_filtered = self._apply_limit(df_filtered, parsed.limit, y_col)

        # Generate chart based on intent and suggestion
        chart_type = parsed.chart_suggestion or 'line'

        try:
            if chart_type == 'line' and x_col and y_col:
                return LineChart(data=df_filtered, x=x_col, y=y_col)

            elif chart_type == 'bar' and x_col and y_col:
                return BarChart(data=df_filtered, x=x_col, y=y_col)

            elif chart_type == 'scatter' and x_col and y_col:
                return ScatterPlot(data=df_filtered, x=x_col, y=y_col)

            elif chart_type == 'histogram' and y_col:
                return Histogram(data=df_filtered, x=y_col)

            elif chart_type == 'pie' and x_col and y_col:
                return PieChart(data=df_filtered, labels=x_col, values=y_col)

            elif chart_type == 'heatmap':
                # Correlation heatmap
                numeric_cols = self.extractor.get_numeric_columns()
                if len(numeric_cols) >= 2:
                    corr_data = df_filtered[numeric_cols].corr()
                    return Heatmap(data=corr_data)

            elif chart_type == 'box' and y_col:
                return Boxplot(data=df_filtered, y=y_col, x=x_col if x_col else None)

            else:
                # Fallback: auto-select
                return self._auto_select_chart(df_filtered, x_col, y_col)

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Chart generation failed: {e}")
                print(f"   Falling back to auto-selection...")

            return self._auto_select_chart(df_filtered, x_col, y_col)

    def _determine_columns(self, parsed: ParsedQuery,
                          suggestions: dict, entities: list) -> tuple[Optional[str], Optional[str]]:
        """Determine which columns to use for x and y axes."""

        # Priority 1: Use columns from parsed query
        if parsed.metrics:
            y_col = parsed.metrics[0]
        else:
            y_col = None

        if parsed.dimensions:
            x_col = parsed.dimensions[0]
        elif parsed.time_column:
            x_col = parsed.time_column
        else:
            x_col = None

        # Priority 2: Use extracted entities
        if not y_col:
            numeric_entities = [e for e in entities if e.type == 'column'
                              and e.value in self.extractor.get_numeric_columns()]
            if numeric_entities:
                y_col = numeric_entities[0].value

        if not x_col:
            cat_entities = [e for e in entities if e.type == 'column'
                          and e.value in self.extractor.get_categorical_columns()]
            if cat_entities:
                x_col = cat_entities[0].value

            # Try datetime
            if not x_col:
                dt_entities = [e for e in entities if e.type == 'column'
                             and e.value in self.extractor.get_datetime_columns()]
                if dt_entities:
                    x_col = dt_entities[0].value

        # Priority 3: Use suggestions based on intent
        if not y_col and 'y' in suggestions and suggestions['y']:
            y_col = suggestions['y'][0]

        if not x_col and 'x' in suggestions and suggestions['x']:
            x_col = suggestions['x'][0]

        # Priority 4: Fallback to first available columns
        if not y_col:
            numeric_cols = self.extractor.get_numeric_columns()
            if numeric_cols:
                y_col = numeric_cols[0]

        if not x_col:
            # Try categorical first
            cat_cols = self.extractor.get_categorical_columns()
            if cat_cols:
                x_col = cat_cols[0]
            else:
                # Try datetime
                dt_cols = self.extractor.get_datetime_columns()
                if dt_cols:
                    x_col = dt_cols[0]

        return x_col, y_col

    def _apply_filters(self, df: pd.DataFrame, filters: list) -> pd.DataFrame:
        """Apply filter conditions."""
        if not filters:
            return df

        filtered = df.copy()
        for f in filters:
            col = f.get('column')
            op = f.get('operator')
            val = f.get('value')

            if col not in filtered.columns:
                continue

            try:
                if op == '=':
                    filtered = filtered[filtered[col] == val]
                elif op == '>':
                    filtered = filtered[filtered[col] > float(val)]
                elif op == '<':
                    filtered = filtered[filtered[col] < float(val)]
                elif op == '>=':
                    filtered = filtered[filtered[col] >= float(val)]
                elif op == '<=':
                    filtered = filtered[filtered[col] <= float(val)]
            except:
                continue

        return filtered

    def _apply_aggregation(self, df: pd.DataFrame, agg_type: Any,
                          x_col: Optional[str], y_col: Optional[str]) -> pd.DataFrame:
        """Apply aggregation."""
        if not x_col or not y_col:
            return df

        agg_map = {
            'sum': 'sum',
            'average': 'mean',
            'count': 'count',
            'min': 'min',
            'max': 'max',
            'median': 'median',
        }

        agg_func = agg_map.get(str(agg_type).lower(), 'sum')

        try:
            if x_col in df.columns and y_col in df.columns:
                aggregated = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                return aggregated
        except:
            pass

        return df

    def _apply_limit(self, df: pd.DataFrame, limit: int, sort_col: Optional[str]) -> pd.DataFrame:
        """Apply TOP N limit."""
        if not sort_col or sort_col not in df.columns:
            return df.head(limit)

        try:
            # Sort descending and take top N
            return df.nlargest(limit, sort_col)
        except:
            return df.head(limit)

    def _auto_select_chart(self, df: pd.DataFrame,
                          x_col: Optional[str], y_col: Optional[str]) -> Any:
        """Auto-select appropriate chart type."""
        from ..charts import LineChart, BarChart, ScatterPlot, Histogram

        # If we have both x and y
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            # Check if x is datetime
            if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                return LineChart(data=df, x=x_col, y=y_col)

            # Check if x is categorical
            elif df[x_col].nunique() < 20:
                return BarChart(data=df, x=x_col, y=y_col)

            # Otherwise scatter
            else:
                return ScatterPlot(data=df, x=x_col, y=y_col)

        # If we only have y (numeric)
        elif y_col and y_col in df.columns:
            return Histogram(data=df, x=y_col)

        # Fallback: bar chart with first two columns
        else:
            cols = list(df.columns)[:2]
            if len(cols) >= 2:
                return BarChart(data=df, x=cols[0], y=cols[1])

            # Last resort: line chart
            return LineChart(data=df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0])


# ==================== Convenience Function ====================

def ask(query: str, dataframe: pd.DataFrame, verbose: bool = False) -> Any:
    """
    Ask a question in natural language (one-liner!).

    Args:
        query: Natural language question
        dataframe: Data to visualize
        verbose: Print processing steps

    Returns:
        Visualization chart object

    Examples:
        >>> import vizforge as vz
        >>> chart = vz.ask("Show me sales trend by month", df)
        >>> chart.show()

        >>> chart = vz.ask("Compare revenue vs profit", df)
        >>> chart.show()

        >>> chart = vz.ask("Find top 10 products by sales", df)
        >>> chart.show()
    """
    engine = NLQEngine(dataframe, verbose=verbose)
    return engine.ask(query)
