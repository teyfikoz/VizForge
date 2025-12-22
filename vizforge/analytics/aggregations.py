"""
VizForge Advanced Aggregations

Advanced aggregation functions and window operations.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, List, Optional, Callable, Union
from enum import Enum
import pandas as pd
import numpy as np


class AggregationType(Enum):
    """Types of aggregations."""
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STDEV = "stdev"
    VAR = "var"
    FIRST = "first"
    LAST = "last"
    DISTINCT_COUNT = "distinct_count"


class WindowType(Enum):
    """Types of window functions."""
    RUNNING_TOTAL = "running_total"
    RUNNING_AVG = "running_avg"
    MOVING_AVG = "moving_avg"
    RANK = "rank"
    PERCENT_RANK = "percent_rank"
    ROW_NUMBER = "row_number"
    LAG = "lag"
    LEAD = "lead"
    PERCENT_OF_TOTAL = "percent_of_total"


class Aggregation:
    """
    Aggregation function with grouping support.

    Tableau equivalent: Aggregated field with dimensions.

    Example:
        >>> # Total sales by category
        >>> agg = Aggregation(
        ...     agg_type=AggregationType.SUM,
        ...     field='sales',
        ...     group_by=['category']
        ... )
        >>> result = agg.apply(df)
    """

    def __init__(
        self,
        agg_type: AggregationType,
        field: str,
        group_by: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize aggregation.

        Args:
            agg_type: Type of aggregation
            field: Field to aggregate
            group_by: Dimensions to group by
            name: Result column name (auto-generated if None)
        """
        self.agg_type = agg_type
        self.field = field
        self.group_by = group_by or []
        self.name = name or f"{agg_type.value}_{field}"

    def apply(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Apply aggregation to data.

        Args:
            data: Input DataFrame

        Returns:
            Aggregated result (DataFrame if group_by, Series otherwise)
        """
        if self.field not in data.columns:
            raise ValueError(f"Field '{self.field}' not found in data")

        if not self.group_by:
            # No grouping - return single value
            return self._aggregate_series(data[self.field])

        # Group and aggregate
        grouped = data.groupby(self.group_by)[self.field]
        return self._aggregate_grouped(grouped)

    def _aggregate_series(self, series: pd.Series) -> Any:
        """Aggregate a series to single value."""
        if self.agg_type == AggregationType.SUM:
            return series.sum()
        elif self.agg_type == AggregationType.AVG:
            return series.mean()
        elif self.agg_type == AggregationType.COUNT:
            return len(series)
        elif self.agg_type == AggregationType.MIN:
            return series.min()
        elif self.agg_type == AggregationType.MAX:
            return series.max()
        elif self.agg_type == AggregationType.MEDIAN:
            return series.median()
        elif self.agg_type == AggregationType.STDEV:
            return series.std()
        elif self.agg_type == AggregationType.VAR:
            return series.var()
        elif self.agg_type == AggregationType.FIRST:
            return series.iloc[0] if len(series) > 0 else None
        elif self.agg_type == AggregationType.LAST:
            return series.iloc[-1] if len(series) > 0 else None
        elif self.agg_type == AggregationType.DISTINCT_COUNT:
            return series.nunique()
        else:
            raise ValueError(f"Unknown aggregation type: {self.agg_type}")

    def _aggregate_grouped(self, grouped: pd.core.groupby.SeriesGroupBy) -> pd.Series:
        """Aggregate grouped data."""
        if self.agg_type == AggregationType.SUM:
            return grouped.sum()
        elif self.agg_type == AggregationType.AVG:
            return grouped.mean()
        elif self.agg_type == AggregationType.COUNT:
            return grouped.count()
        elif self.agg_type == AggregationType.MIN:
            return grouped.min()
        elif self.agg_type == AggregationType.MAX:
            return grouped.max()
        elif self.agg_type == AggregationType.MEDIAN:
            return grouped.median()
        elif self.agg_type == AggregationType.STDEV:
            return grouped.std()
        elif self.agg_type == AggregationType.VAR:
            return grouped.var()
        elif self.agg_type == AggregationType.FIRST:
            return grouped.first()
        elif self.agg_type == AggregationType.LAST:
            return grouped.last()
        elif self.agg_type == AggregationType.DISTINCT_COUNT:
            return grouped.nunique()
        else:
            raise ValueError(f"Unknown aggregation type: {self.agg_type}")


class WindowFunction:
    """
    Window function for running calculations.

    Tableau equivalent: Table calculations (Running Total, Moving Average, etc.).

    Example:
        >>> # Running total of sales
        >>> window = WindowFunction(
        ...     window_type=WindowType.RUNNING_TOTAL,
        ...     field='sales',
        ...     partition_by=['category']
        ... )
        >>> df['running_total'] = window.apply(df)
    """

    def __init__(
        self,
        window_type: WindowType,
        field: str,
        partition_by: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        window_size: Optional[int] = None,
        offset: int = 1
    ):
        """
        Initialize window function.

        Args:
            window_type: Type of window function
            field: Field to calculate on
            partition_by: Partition dimensions
            order_by: Ordering column
            window_size: Window size (for moving averages)
            offset: Offset for LAG/LEAD
        """
        self.window_type = window_type
        self.field = field
        self.partition_by = partition_by or []
        self.order_by = order_by
        self.window_size = window_size
        self.offset = offset

    def apply(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply window function to data.

        Args:
            data: Input DataFrame

        Returns:
            Series with calculated values
        """
        if self.field not in data.columns:
            raise ValueError(f"Field '{self.field}' not found in data")

        # Sort data if order_by specified
        if self.order_by and self.order_by in data.columns:
            data = data.sort_values(self.order_by)

        if not self.partition_by:
            # No partitioning - apply to entire dataset
            return self._apply_window(data)

        # Apply window function per partition
        result = data.groupby(self.partition_by).apply(
            lambda group: self._apply_window(group)
        )

        # Reset index to match original data
        if isinstance(result, pd.Series):
            result = result.droplevel(list(range(len(self.partition_by))))

        return result

    def _apply_window(self, data: pd.DataFrame) -> pd.Series:
        """Apply window function to data partition."""
        series = data[self.field]

        if self.window_type == WindowType.RUNNING_TOTAL:
            return series.cumsum()

        elif self.window_type == WindowType.RUNNING_AVG:
            return series.expanding().mean()

        elif self.window_type == WindowType.MOVING_AVG:
            window_size = self.window_size or 3
            return series.rolling(window=window_size, min_periods=1).mean()

        elif self.window_type == WindowType.RANK:
            return series.rank(method='min', ascending=False)

        elif self.window_type == WindowType.PERCENT_RANK:
            return series.rank(method='min', ascending=False, pct=True)

        elif self.window_type == WindowType.ROW_NUMBER:
            return pd.Series(range(1, len(series) + 1), index=series.index)

        elif self.window_type == WindowType.LAG:
            return series.shift(self.offset)

        elif self.window_type == WindowType.LEAD:
            return series.shift(-self.offset)

        elif self.window_type == WindowType.PERCENT_OF_TOTAL:
            total = series.sum()
            return (series / total * 100) if total != 0 else pd.Series([0] * len(series))

        else:
            raise ValueError(f"Unknown window type: {self.window_type}")


class AggregationEngine:
    """
    Engine for managing multiple aggregations and window functions.

    Example:
        >>> engine = AggregationEngine()
        >>>
        >>> # Add aggregations
        >>> engine.add_aggregation(Aggregation(
        ...     AggregationType.SUM, 'sales', group_by=['category']
        ... ))
        >>>
        >>> # Add window functions
        >>> engine.add_window_function(WindowFunction(
        ...     WindowType.RUNNING_TOTAL, 'sales', order_by='date'
        ... ))
        >>>
        >>> # Apply all
        >>> result = engine.apply_all(df)
    """

    def __init__(self):
        """Initialize aggregation engine."""
        self.aggregations: List[Aggregation] = []
        self.window_functions: List[WindowFunction] = []

    def add_aggregation(self, aggregation: Aggregation) -> 'AggregationEngine':
        """
        Add aggregation.

        Args:
            aggregation: Aggregation instance

        Returns:
            Self for method chaining
        """
        self.aggregations.append(aggregation)
        return self

    def add_window_function(
        self,
        window_function: WindowFunction
    ) -> 'AggregationEngine':
        """
        Add window function.

        Args:
            window_function: WindowFunction instance

        Returns:
            Self for method chaining
        """
        self.window_functions.append(window_function)
        return self

    def apply_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all aggregations and window functions.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with calculated columns
        """
        result = data.copy()

        # Apply window functions
        for window_func in self.window_functions:
            col_name = f"{window_func.window_type.value}_{window_func.field}"
            result[col_name] = window_func.apply(result)

        return result

    def clear(self):
        """Clear all aggregations and window functions."""
        self.aggregations.clear()
        self.window_functions.clear()


# ==================== Helper Functions ====================

def quick_aggregation(
    data: pd.DataFrame,
    agg_type: str,
    field: str,
    group_by: Optional[List[str]] = None
) -> Union[pd.DataFrame, pd.Series]:
    """
    Quick aggregation helper.

    Args:
        data: Input DataFrame
        agg_type: Aggregation type ('sum', 'avg', 'count', etc.)
        field: Field to aggregate
        group_by: Grouping dimensions

    Returns:
        Aggregated result

    Example:
        >>> total_sales = quick_aggregation(df, 'sum', 'sales', ['category'])
    """
    agg_enum = AggregationType(agg_type.lower())
    agg = Aggregation(agg_enum, field, group_by)
    return agg.apply(data)


def quick_window(
    data: pd.DataFrame,
    window_type: str,
    field: str,
    **kwargs
) -> pd.Series:
    """
    Quick window function helper.

    Args:
        data: Input DataFrame
        window_type: Window type ('running_total', 'moving_avg', etc.)
        field: Field to calculate on
        **kwargs: Additional window function arguments

    Returns:
        Calculated series

    Example:
        >>> running_total = quick_window(df, 'running_total', 'sales', order_by='date')
    """
    window_enum = WindowType(window_type.lower())
    window = WindowFunction(window_enum, field, **kwargs)
    return window.apply(data)
