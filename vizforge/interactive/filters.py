"""
VizForge Filtering System

Multi-level data filtering with cascading support.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, List, Optional, Callable, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import date, datetime
import pandas as pd
from abc import ABC, abstractmethod


class FilterType(Enum):
    """Types of filters."""
    RANGE = "range"  # Numeric range filter
    LIST = "list"  # Categorical list filter
    SEARCH = "search"  # Text search filter
    DATE_RANGE = "date_range"  # Date range filter
    TOP_N = "top_n"  # Top N by value
    CUSTOM = "custom"  # Custom filter function


@dataclass
class FilterConfig:
    """
    Filter configuration.

    Attributes:
        id: Unique filter identifier
        column: Column name to filter
        type: Filter type
        values: Filter values (depends on type)
        operator: Operator ('and', 'or', 'not')
        enabled: Whether filter is active
    """
    id: str
    column: str
    type: FilterType
    values: Any
    operator: str = 'and'
    enabled: bool = True


class Filter(ABC):
    """
    Base class for all filters.

    Provides abstract interface for data filtering.
    """

    def __init__(
        self,
        filter_id: str,
        column: str,
        enabled: bool = True
    ):
        """
        Initialize filter.

        Args:
            filter_id: Unique identifier
            column: Column name to filter
            enabled: Whether filter is active
        """
        self.id = filter_id
        self.column = column
        self.enabled = enabled

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filter to data.

        Args:
            data: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        pass

    @abstractmethod
    def to_config(self) -> FilterConfig:
        """Export filter configuration."""
        pass

    def enable(self):
        """Enable filter."""
        self.enabled = True

    def disable(self):
        """Disable filter."""
        self.enabled = False

    def toggle(self):
        """Toggle filter enabled state."""
        self.enabled = not self.enabled


class RangeFilter(Filter):
    """
    Numeric range filter.

    Example:
        >>> filter = RangeFilter('price', 'price', min_value=100, max_value=500)
        >>> filtered_df = filter.apply(df)
    """

    def __init__(
        self,
        filter_id: str,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        enabled: bool = True
    ):
        """
        Initialize range filter.

        Args:
            filter_id: Unique identifier
            column: Column name
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            enabled: Whether active
        """
        super().__init__(filter_id, column, enabled)
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply range filter."""
        if not self.enabled:
            return data

        if self.column not in data.columns:
            return data

        filtered = data.copy()

        if self.min_value is not None:
            filtered = filtered[filtered[self.column] >= self.min_value]

        if self.max_value is not None:
            filtered = filtered[filtered[self.column] <= self.max_value]

        return filtered

    def to_config(self) -> FilterConfig:
        """Export configuration."""
        return FilterConfig(
            id=self.id,
            column=self.column,
            type=FilterType.RANGE,
            values={'min': self.min_value, 'max': self.max_value},
            enabled=self.enabled
        )


class ListFilter(Filter):
    """
    Categorical list filter.

    Example:
        >>> filter = ListFilter('category', 'category', ['Electronics', 'Clothing'])
        >>> filtered_df = filter.apply(df)
    """

    def __init__(
        self,
        filter_id: str,
        column: str,
        values: List[Any],
        operator: str = 'in',
        enabled: bool = True
    ):
        """
        Initialize list filter.

        Args:
            filter_id: Unique identifier
            column: Column name
            values: List of values to include/exclude
            operator: 'in' (include) or 'not_in' (exclude)
            enabled: Whether active
        """
        super().__init__(filter_id, column, enabled)
        self.values = values
        self.operator = operator

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply list filter."""
        if not self.enabled:
            return data

        if self.column not in data.columns:
            return data

        if self.operator == 'in':
            return data[data[self.column].isin(self.values)]
        elif self.operator == 'not_in':
            return data[~data[self.column].isin(self.values)]
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def to_config(self) -> FilterConfig:
        """Export configuration."""
        return FilterConfig(
            id=self.id,
            column=self.column,
            type=FilterType.LIST,
            values=self.values,
            operator=self.operator,
            enabled=self.enabled
        )


class SearchFilter(Filter):
    """
    Text search filter.

    Example:
        >>> filter = SearchFilter('name', 'product_name', 'laptop', case_sensitive=False)
        >>> filtered_df = filter.apply(df)
    """

    def __init__(
        self,
        filter_id: str,
        column: str,
        search_term: str,
        case_sensitive: bool = False,
        regex: bool = False,
        enabled: bool = True
    ):
        """
        Initialize search filter.

        Args:
            filter_id: Unique identifier
            column: Column name
            search_term: Search term
            case_sensitive: Whether case-sensitive
            regex: Whether to use regex
            enabled: Whether active
        """
        super().__init__(filter_id, column, enabled)
        self.search_term = search_term
        self.case_sensitive = case_sensitive
        self.regex = regex

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply search filter."""
        if not self.enabled:
            return data

        if self.column not in data.columns:
            return data

        if not self.search_term:
            return data

        return data[
            data[self.column].astype(str).str.contains(
                self.search_term,
                case=self.case_sensitive,
                regex=self.regex,
                na=False
            )
        ]

    def to_config(self) -> FilterConfig:
        """Export configuration."""
        return FilterConfig(
            id=self.id,
            column=self.column,
            type=FilterType.SEARCH,
            values={
                'term': self.search_term,
                'case_sensitive': self.case_sensitive,
                'regex': self.regex
            },
            enabled=self.enabled
        )


class DateRangeFilter(Filter):
    """
    Date range filter.

    Example:
        >>> filter = DateRangeFilter(
        ...     'date_range',
        ...     'order_date',
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 12, 31)
        ... )
        >>> filtered_df = filter.apply(df)
    """

    def __init__(
        self,
        filter_id: str,
        column: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        enabled: bool = True
    ):
        """
        Initialize date range filter.

        Args:
            filter_id: Unique identifier
            column: Column name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            enabled: Whether active
        """
        super().__init__(filter_id, column, enabled)
        self.start_date = start_date
        self.end_date = end_date

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply date range filter."""
        if not self.enabled:
            return data

        if self.column not in data.columns:
            return data

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[self.column]):
            data = data.copy()
            data[self.column] = pd.to_datetime(data[self.column])

        filtered = data.copy()

        if self.start_date is not None:
            start_datetime = pd.to_datetime(self.start_date)
            filtered = filtered[filtered[self.column] >= start_datetime]

        if self.end_date is not None:
            end_datetime = pd.to_datetime(self.end_date)
            filtered = filtered[filtered[self.column] <= end_datetime]

        return filtered

    def to_config(self) -> FilterConfig:
        """Export configuration."""
        return FilterConfig(
            id=self.id,
            column=self.column,
            type=FilterType.DATE_RANGE,
            values={
                'start': self.start_date.isoformat() if self.start_date else None,
                'end': self.end_date.isoformat() if self.end_date else None
            },
            enabled=self.enabled
        )


class TopNFilter(Filter):
    """
    Top N by value filter.

    Example:
        >>> filter = TopNFilter('top_sales', 'revenue', n=10, ascending=False)
        >>> filtered_df = filter.apply(df)  # Top 10 by revenue
    """

    def __init__(
        self,
        filter_id: str,
        column: str,
        n: int = 10,
        ascending: bool = False,
        enabled: bool = True
    ):
        """
        Initialize top N filter.

        Args:
            filter_id: Unique identifier
            column: Column name to sort by
            n: Number of top rows
            ascending: Sort ascending (bottom N) or descending (top N)
            enabled: Whether active
        """
        super().__init__(filter_id, column, enabled)
        self.n = n
        self.ascending = ascending

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply top N filter."""
        if not self.enabled:
            return data

        if self.column not in data.columns:
            return data

        return data.nlargest(self.n, self.column) if not self.ascending else data.nsmallest(self.n, self.column)

    def to_config(self) -> FilterConfig:
        """Export configuration."""
        return FilterConfig(
            id=self.id,
            column=self.column,
            type=FilterType.TOP_N,
            values={'n': self.n, 'ascending': self.ascending},
            enabled=self.enabled
        )


class CustomFilter(Filter):
    """
    Custom filter with user-defined function.

    Example:
        >>> def is_profitable(df):
        ...     return df[df['profit'] > 0]
        >>> filter = CustomFilter('profitable', is_profitable)
        >>> filtered_df = filter.apply(df)
    """

    def __init__(
        self,
        filter_id: str,
        filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        description: str = "",
        enabled: bool = True
    ):
        """
        Initialize custom filter.

        Args:
            filter_id: Unique identifier
            filter_func: Function that takes DataFrame and returns filtered DataFrame
            description: Human-readable description
            enabled: Whether active
        """
        super().__init__(filter_id, "", enabled)
        self.filter_func = filter_func
        self.description = description

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply custom filter."""
        if not self.enabled:
            return data

        return self.filter_func(data)

    def to_config(self) -> FilterConfig:
        """Export configuration."""
        return FilterConfig(
            id=self.id,
            column='custom',
            type=FilterType.CUSTOM,
            values={'description': self.description},
            enabled=self.enabled
        )


class FilterContext:
    """
    Manages multiple filters with cascading support.

    Provides Tableau-style filter cascading where filters
    are applied in sequence, affecting subsequent filters.

    Example:
        >>> context = FilterContext()
        >>> context.add_filter(RangeFilter('price', 'price', 100, 500))
        >>> context.add_filter(ListFilter('category', 'category', ['Electronics']))
        >>> filtered_df = context.apply_all(df)
    """

    def __init__(self):
        """Initialize filter context."""
        self.filters: Dict[str, Filter] = {}
        self.filter_order: List[str] = []

    def add_filter(self, filter: Filter) -> 'FilterContext':
        """
        Add filter to context.

        Args:
            filter: Filter to add

        Returns:
            Self for method chaining
        """
        self.filters[filter.id] = filter
        if filter.id not in self.filter_order:
            self.filter_order.append(filter.id)
        return self

    def remove_filter(self, filter_id: str) -> 'FilterContext':
        """Remove filter from context."""
        if filter_id in self.filters:
            del self.filters[filter_id]
            self.filter_order.remove(filter_id)
        return self

    def get_filter(self, filter_id: str) -> Optional[Filter]:
        """Get filter by ID."""
        return self.filters.get(filter_id)

    def enable_filter(self, filter_id: str):
        """Enable specific filter."""
        if filter_id in self.filters:
            self.filters[filter_id].enable()

    def disable_filter(self, filter_id: str):
        """Disable specific filter."""
        if filter_id in self.filters:
            self.filters[filter_id].disable()

    def enable_all(self):
        """Enable all filters."""
        for filter in self.filters.values():
            filter.enable()

    def disable_all(self):
        """Disable all filters."""
        for filter in self.filters.values():
            filter.disable()

    def apply_all(
        self,
        data: pd.DataFrame,
        cascade: bool = True
    ) -> pd.DataFrame:
        """
        Apply all filters to data.

        Args:
            data: Input DataFrame
            cascade: Whether to apply filters in sequence (cascading)
                    True: Filter1 → Filter2 → Filter3 (Tableau-style)
                    False: All filters applied to original data independently

        Returns:
            Filtered DataFrame

        Example:
            >>> # Cascade: First filter price, then filter category on price-filtered data
            >>> df1 = context.apply_all(df, cascade=True)
            >>>
            >>> # Independent: Both filters applied to original data, then combined
            >>> df2 = context.apply_all(df, cascade=False)
        """
        if cascade:
            # Apply filters sequentially (Tableau-style cascading)
            result = data.copy()
            for filter_id in self.filter_order:
                filter = self.filters[filter_id]
                result = filter.apply(result)
            return result
        else:
            # Apply all filters to original data independently
            masks = []
            for filter_id in self.filter_order:
                filter = self.filters[filter_id]
                if filter.enabled:
                    filtered = filter.apply(data)
                    mask = data.index.isin(filtered.index)
                    masks.append(mask)

            if not masks:
                return data

            # Combine all masks with AND
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask &= mask

            return data[combined_mask]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all filters.

        Returns:
            Dictionary with filter statistics
        """
        return {
            'total_filters': len(self.filters),
            'enabled_filters': sum(1 for f in self.filters.values() if f.enabled),
            'disabled_filters': sum(1 for f in self.filters.values() if not f.enabled),
            'filter_types': {
                filter_type.value: sum(
                    1 for f in self.filters.values()
                    if f.to_config().type == filter_type
                )
                for filter_type in FilterType
            }
        }

    def clear(self):
        """Clear all filters."""
        self.filters.clear()
        self.filter_order.clear()

    def export_configs(self) -> List[FilterConfig]:
        """Export all filter configurations."""
        return [
            self.filters[filter_id].to_config()
            for filter_id in self.filter_order
        ]


class CrossFilter:
    """
    Cross-filtering between multiple charts.

    Enables Tableau-style filter actions where selecting data
    in one chart filters other charts.

    Example:
        >>> cross_filter = CrossFilter()
        >>> cross_filter.add_chart('sales', sales_chart)
        >>> cross_filter.add_chart('regions', region_chart)
        >>> cross_filter.link('sales', 'regions', on='region')
        >>> # Clicking on a region in region_chart will filter sales_chart
    """

    def __init__(self):
        """Initialize cross-filter."""
        self.charts: Dict[str, Any] = {}
        self.links: List[Dict[str, Any]] = []
        self.active_filters: Dict[str, FilterContext] = {}

    def add_chart(self, chart_id: str, chart: Any):
        """
        Add chart to cross-filter network.

        Args:
            chart_id: Unique chart identifier
            chart: Chart object
        """
        self.charts[chart_id] = chart
        self.active_filters[chart_id] = FilterContext()

    def link(
        self,
        source_chart: str,
        target_chart: str,
        on: str,
        link_type: str = 'filter'
    ):
        """
        Create cross-filter link between charts.

        Args:
            source_chart: Source chart ID (selection source)
            target_chart: Target chart ID (will be filtered)
            on: Column to filter on
            link_type: 'filter' or 'highlight'
        """
        self.links.append({
            'source': source_chart,
            'target': target_chart,
            'on': on,
            'type': link_type
        })

    def apply_selection(
        self,
        source_chart: str,
        selected_values: List[Any]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply selection from source chart to linked charts.

        Args:
            source_chart: Chart where selection occurred
            selected_values: Selected values

        Returns:
            Dictionary mapping chart IDs to filtered data
        """
        results = {}

        # Find all links from this source chart
        for link in self.links:
            if link['source'] == source_chart:
                target_id = link['target']
                column = link['on']

                # Create filter for target chart
                filter = ListFilter(
                    filter_id=f'{source_chart}_to_{target_id}',
                    column=column,
                    values=selected_values
                )

                # Apply filter to target chart
                self.active_filters[target_id].add_filter(filter)

                # Get filtered data
                # Note: In real implementation, would get data from chart
                # This is a placeholder
                results[target_id] = None

        return results

    def clear_filters(self, chart_id: Optional[str] = None):
        """
        Clear filters for specific chart or all charts.

        Args:
            chart_id: Chart ID to clear (None = clear all)
        """
        if chart_id:
            self.active_filters[chart_id].clear()
        else:
            for context in self.active_filters.values():
                context.clear()
