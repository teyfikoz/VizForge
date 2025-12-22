"""
VizForge Hierarchies

Tableau-style hierarchical drill-down paths.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DrillPath:
    """
    Drill path within a hierarchy.

    Represents current position in hierarchy with applied filters.

    Attributes:
        hierarchy_name: Name of parent hierarchy
        current_level: Current drill level index
        level_filters: Filters applied at each level
        breadcrumb: Path taken to current position
    """
    hierarchy_name: str
    current_level: int = 0
    level_filters: Dict[int, Any] = field(default_factory=dict)
    breadcrumb: List[Dict[str, Any]] = field(default_factory=list)


class Hierarchy:
    """
    Hierarchical dimension for drill-down analysis.

    Tableau equivalent: Hierarchy with drill-down levels.

    Example:
        >>> # Geographic hierarchy
        >>> geo_hierarchy = Hierarchy(
        ...     name='Geography',
        ...     levels=['Country', 'State', 'City']
        ... )
        >>>
        >>> # Time hierarchy
        >>> time_hierarchy = Hierarchy(
        ...     name='Time',
        ...     levels=['Year', 'Quarter', 'Month', 'Day']
        ... )
    """

    def __init__(
        self,
        name: str,
        levels: List[str],
        description: str = ""
    ):
        """
        Initialize hierarchy.

        Args:
            name: Hierarchy name
            levels: List of dimension levels (high to low granularity)
            description: Hierarchy description

        Example:
            >>> product_hierarchy = Hierarchy(
            ...     name='Product',
            ...     levels=['Category', 'Subcategory', 'Product Name'],
            ...     description='Product classification hierarchy'
            ... )
        """
        if len(levels) < 2:
            raise ValueError("Hierarchy must have at least 2 levels")

        self.name = name
        self.levels = levels
        self.description = description
        self.current_level = 0

    def get_current_level(self) -> str:
        """Get current drill level name."""
        return self.levels[self.current_level]

    def get_next_level(self) -> Optional[str]:
        """Get next drill level name."""
        if self.current_level < len(self.levels) - 1:
            return self.levels[self.current_level + 1]
        return None

    def get_previous_level(self) -> Optional[str]:
        """Get previous drill level name."""
        if self.current_level > 0:
            return self.levels[self.current_level - 1]
        return None

    def can_drill_down(self) -> bool:
        """Check if can drill down further."""
        return self.current_level < len(self.levels) - 1

    def can_drill_up(self) -> bool:
        """Check if can drill up."""
        return self.current_level > 0

    def drill_down(
        self,
        selected_value: Any,
        filters: Optional[Dict[str, Any]] = None
    ) -> DrillPath:
        """
        Drill down one level.

        Args:
            selected_value: Value selected at current level
            filters: Additional filters to apply

        Returns:
            DrillPath representing new position

        Example:
            >>> # Start at Country level
            >>> hierarchy.current_level = 0  # Country
            >>> path = hierarchy.drill_down('USA')
            >>> # Now at State level, filtered to USA
        """
        if not self.can_drill_down():
            raise ValueError(f"Cannot drill down from {self.get_current_level()}")

        # Create drill path
        path = DrillPath(
            hierarchy_name=self.name,
            current_level=self.current_level + 1
        )

        # Add filter for current level
        path.level_filters[self.current_level] = selected_value

        # Add to breadcrumb
        path.breadcrumb.append({
            'level': self.levels[self.current_level],
            'value': selected_value,
            'level_index': self.current_level
        })

        # Update current level
        self.current_level += 1

        return path

    def drill_up(self) -> DrillPath:
        """
        Drill up one level.

        Returns:
            DrillPath representing new position

        Example:
            >>> # At State level
            >>> path = hierarchy.drill_up()
            >>> # Back to Country level
        """
        if not self.can_drill_up():
            raise ValueError(f"Cannot drill up from {self.get_current_level()}")

        # Move up
        self.current_level -= 1

        path = DrillPath(
            hierarchy_name=self.name,
            current_level=self.current_level
        )

        return path

    def reset(self):
        """Reset to top level."""
        self.current_level = 0

    def get_level_index(self, level_name: str) -> int:
        """
        Get index of level by name.

        Args:
            level_name: Level name

        Returns:
            Level index
        """
        try:
            return self.levels.index(level_name)
        except ValueError:
            raise ValueError(f"Level '{level_name}' not found in hierarchy")

    def get_depth(self) -> int:
        """Get total hierarchy depth."""
        return len(self.levels)

    def __repr__(self) -> str:
        """String representation."""
        return f"Hierarchy('{self.name}', {self.levels})"


class HierarchyManager:
    """
    Manage multiple hierarchies for a dashboard.

    Handles drill-down navigation across multiple hierarchical dimensions.

    Example:
        >>> manager = HierarchyManager()
        >>> manager.add_hierarchy(geo_hierarchy)
        >>> manager.add_hierarchy(time_hierarchy)
        >>> manager.add_hierarchy(product_hierarchy)
        >>>
        >>> # Drill down geography
        >>> manager.drill_down('Geography', 'USA')
        >>> # Now showing State-level data for USA
    """

    def __init__(self):
        """Initialize hierarchy manager."""
        self.hierarchies: Dict[str, Hierarchy] = {}
        self.drill_paths: Dict[str, DrillPath] = {}

    def add_hierarchy(self, hierarchy: Hierarchy) -> 'HierarchyManager':
        """
        Add hierarchy to manager.

        Args:
            hierarchy: Hierarchy instance

        Returns:
            Self for method chaining
        """
        self.hierarchies[hierarchy.name] = hierarchy
        self.drill_paths[hierarchy.name] = DrillPath(
            hierarchy_name=hierarchy.name,
            current_level=0
        )
        return self

    def remove_hierarchy(self, name: str):
        """Remove hierarchy."""
        if name in self.hierarchies:
            del self.hierarchies[name]
            del self.drill_paths[name]

    def get_hierarchy(self, name: str) -> Optional[Hierarchy]:
        """Get hierarchy by name."""
        return self.hierarchies.get(name)

    def drill_down(
        self,
        hierarchy_name: str,
        selected_value: Any
    ) -> DrillPath:
        """
        Drill down in hierarchy.

        Args:
            hierarchy_name: Hierarchy name
            selected_value: Value selected at current level

        Returns:
            Updated drill path

        Example:
            >>> manager.drill_down('Geography', 'USA')
            >>> manager.drill_down('Geography', 'California')
            >>> manager.drill_down('Geography', 'San Francisco')
        """
        if hierarchy_name not in self.hierarchies:
            raise ValueError(f"Hierarchy '{hierarchy_name}' not found")

        hierarchy = self.hierarchies[hierarchy_name]
        path = hierarchy.drill_down(selected_value)

        # Update stored path
        self.drill_paths[hierarchy_name] = path

        return path

    def drill_up(self, hierarchy_name: str) -> DrillPath:
        """
        Drill up in hierarchy.

        Args:
            hierarchy_name: Hierarchy name

        Returns:
            Updated drill path
        """
        if hierarchy_name not in self.hierarchies:
            raise ValueError(f"Hierarchy '{hierarchy_name}' not found")

        hierarchy = self.hierarchies[hierarchy_name]
        path = hierarchy.drill_up()

        # Update stored path
        self.drill_paths[hierarchy_name] = path

        return path

    def reset(self, hierarchy_name: Optional[str] = None):
        """
        Reset hierarchy to top level.

        Args:
            hierarchy_name: Hierarchy to reset (None = reset all)
        """
        if hierarchy_name:
            if hierarchy_name in self.hierarchies:
                self.hierarchies[hierarchy_name].reset()
                self.drill_paths[hierarchy_name] = DrillPath(
                    hierarchy_name=hierarchy_name,
                    current_level=0
                )
        else:
            # Reset all
            for name, hierarchy in self.hierarchies.items():
                hierarchy.reset()
                self.drill_paths[name] = DrillPath(
                    hierarchy_name=name,
                    current_level=0
                )

    def get_current_level(self, hierarchy_name: str) -> str:
        """Get current level for hierarchy."""
        if hierarchy_name not in self.hierarchies:
            raise ValueError(f"Hierarchy '{hierarchy_name}' not found")

        return self.hierarchies[hierarchy_name].get_current_level()

    def get_breadcrumb(self, hierarchy_name: str) -> List[Dict[str, Any]]:
        """
        Get drill-down breadcrumb for hierarchy.

        Args:
            hierarchy_name: Hierarchy name

        Returns:
            List of breadcrumb items

        Example:
            >>> breadcrumb = manager.get_breadcrumb('Geography')
            >>> # [{'level': 'Country', 'value': 'USA'},
            >>> #  {'level': 'State', 'value': 'California'}]
        """
        if hierarchy_name not in self.drill_paths:
            return []

        return self.drill_paths[hierarchy_name].breadcrumb

    def apply_filters(
        self,
        data: pd.DataFrame,
        hierarchy_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply hierarchy filters to data.

        Args:
            data: Input DataFrame
            hierarchy_name: Specific hierarchy to apply (None = all)

        Returns:
            Filtered DataFrame

        Example:
            >>> # After drilling: USA > California > San Francisco
            >>> filtered_df = manager.apply_filters(df, 'Geography')
            >>> # Returns only San Francisco data
        """
        result = data.copy()

        hierarchies_to_apply = (
            [hierarchy_name] if hierarchy_name
            else list(self.hierarchies.keys())
        )

        for name in hierarchies_to_apply:
            if name not in self.drill_paths:
                continue

            path = self.drill_paths[name]
            hierarchy = self.hierarchies[name]

            # Apply filters for each level in path
            for level_idx, filter_value in path.level_filters.items():
                level_name = hierarchy.levels[level_idx]

                if level_name in result.columns:
                    result = result[result[level_name] == filter_value]

        return result

    def get_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all hierarchies.

        Returns:
            List of hierarchy summaries
        """
        return [
            {
                'name': hierarchy.name,
                'levels': hierarchy.levels,
                'current_level': hierarchy.get_current_level(),
                'current_level_index': hierarchy.current_level,
                'depth': hierarchy.get_depth(),
                'can_drill_down': hierarchy.can_drill_down(),
                'can_drill_up': hierarchy.can_drill_up(),
                'breadcrumb': self.get_breadcrumb(hierarchy.name)
            }
            for hierarchy in self.hierarchies.values()
        ]
