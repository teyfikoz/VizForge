"""
VizForge Actions System

Set actions and interactive behaviors for dashboards.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, List, Optional, Callable, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd


class ActionType(Enum):
    """Types of interactive actions."""
    FILTER = "filter"  # Filter action
    HIGHLIGHT = "highlight"  # Highlight action
    URL = "url"  # URL navigation
    PARAMETER = "parameter"  # Set parameter value
    DRILL_DOWN = "drill_down"  # Hierarchical drill-down
    DRILL_UP = "drill_up"  # Hierarchical drill-up
    CUSTOM = "custom"  # Custom callback


class TriggerEvent(Enum):
    """Events that trigger actions."""
    CLICK = "click"  # Mouse click
    HOVER = "hover"  # Mouse hover
    SELECT = "select"  # Selection
    DOUBLE_CLICK = "double_click"  # Double click
    RIGHT_CLICK = "right_click"  # Right click (context menu)


@dataclass
class ActionConfig:
    """
    Action configuration.

    Attributes:
        id: Unique action identifier
        type: Action type
        trigger: Trigger event
        source_chart: Source chart ID
        target_charts: Target chart IDs (empty = all charts)
        enabled: Whether action is active
        options: Action-specific options
    """
    id: str
    type: ActionType
    trigger: TriggerEvent
    source_chart: str
    target_charts: List[str] = field(default_factory=list)
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)


class Action:
    """
    Base class for interactive actions.

    Actions define what happens when users interact with charts.
    """

    def __init__(
        self,
        action_id: str,
        action_type: ActionType,
        trigger: TriggerEvent,
        source_chart: str,
        target_charts: Optional[List[str]] = None,
        enabled: bool = True
    ):
        """
        Initialize action.

        Args:
            action_id: Unique identifier
            action_type: Type of action
            trigger: Trigger event
            source_chart: Source chart ID
            target_charts: Target chart IDs (None = all charts)
            enabled: Whether active
        """
        self.id = action_id
        self.type = action_type
        self.trigger = trigger
        self.source_chart = source_chart
        self.target_charts = target_charts or []
        self.enabled = enabled

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action.

        Args:
            context: Execution context with event data

        Returns:
            Result dictionary
        """
        if not self.enabled:
            return {'executed': False, 'reason': 'Action disabled'}

        return self._execute_impl(context)

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation-specific execution logic."""
        raise NotImplementedError

    def enable(self):
        """Enable action."""
        self.enabled = True

    def disable(self):
        """Disable action."""
        self.enabled = False


class FilterAction(Action):
    """
    Filter action - filters target charts based on selection.

    Tableau equivalent: Filter action.

    Example:
        >>> action = FilterAction(
        ...     action_id='filter_by_region',
        ...     source_chart='map',
        ...     target_charts=['sales_chart', 'profit_chart'],
        ...     filter_column='region'
        ... )
    """

    def __init__(
        self,
        action_id: str,
        source_chart: str,
        target_charts: List[str],
        filter_column: str,
        trigger: TriggerEvent = TriggerEvent.CLICK,
        clear_on_deselect: bool = True,
        enabled: bool = True
    ):
        """
        Initialize filter action.

        Args:
            action_id: Unique identifier
            source_chart: Source chart ID
            target_charts: Target chart IDs
            filter_column: Column to filter on
            trigger: Trigger event
            clear_on_deselect: Clear filter when deselected
            enabled: Whether active
        """
        super().__init__(action_id, ActionType.FILTER, trigger, source_chart, target_charts, enabled)
        self.filter_column = filter_column
        self.clear_on_deselect = clear_on_deselect

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute filter action."""
        selected_values = context.get('selected_values', [])

        if not selected_values and self.clear_on_deselect:
            return {
                'executed': True,
                'action': 'clear_filter',
                'targets': self.target_charts
            }

        return {
            'executed': True,
            'action': 'apply_filter',
            'targets': self.target_charts,
            'column': self.filter_column,
            'values': selected_values
        }


class HighlightAction(Action):
    """
    Highlight action - highlights related data without filtering.

    Tableau equivalent: Highlight action.

    Example:
        >>> action = HighlightAction(
        ...     action_id='highlight_category',
        ...     source_chart='category_chart',
        ...     target_charts=['detail_chart'],
        ...     highlight_column='category'
        ... )
    """

    def __init__(
        self,
        action_id: str,
        source_chart: str,
        target_charts: List[str],
        highlight_column: str,
        trigger: TriggerEvent = TriggerEvent.HOVER,
        highlight_color: str = '#FFD700',
        enabled: bool = True
    ):
        """
        Initialize highlight action.

        Args:
            action_id: Unique identifier
            source_chart: Source chart ID
            target_charts: Target chart IDs
            highlight_column: Column to highlight on
            trigger: Trigger event
            highlight_color: Highlight color
            enabled: Whether active
        """
        super().__init__(action_id, ActionType.HIGHLIGHT, trigger, source_chart, target_charts, enabled)
        self.highlight_column = highlight_column
        self.highlight_color = highlight_color

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute highlight action."""
        selected_values = context.get('selected_values', [])

        return {
            'executed': True,
            'action': 'highlight',
            'targets': self.target_charts,
            'column': self.highlight_column,
            'values': selected_values,
            'color': self.highlight_color
        }


class URLAction(Action):
    """
    URL action - navigates to URL with dynamic parameters.

    Tableau equivalent: URL action.

    Example:
        >>> action = URLAction(
        ...     action_id='open_details',
        ...     source_chart='product_list',
        ...     url_template='https://example.com/product/{product_id}',
        ...     parameters={'product_id': 'id'}
        ... )
    """

    def __init__(
        self,
        action_id: str,
        source_chart: str,
        url_template: str,
        parameters: Dict[str, str],
        trigger: TriggerEvent = TriggerEvent.CLICK,
        open_in_new_tab: bool = True,
        enabled: bool = True
    ):
        """
        Initialize URL action.

        Args:
            action_id: Unique identifier
            source_chart: Source chart ID
            url_template: URL template with {placeholders}
            parameters: Mapping of placeholder → column name
            trigger: Trigger event
            open_in_new_tab: Open in new tab
            enabled: Whether active
        """
        super().__init__(action_id, ActionType.URL, trigger, source_chart, [], enabled)
        self.url_template = url_template
        self.parameters = parameters
        self.open_in_new_tab = open_in_new_tab

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute URL action."""
        selected_data = context.get('selected_data', {})

        # Build URL with parameters
        url = self.url_template
        for placeholder, column in self.parameters.items():
            value = selected_data.get(column, '')
            url = url.replace(f'{{{placeholder}}}', str(value))

        return {
            'executed': True,
            'action': 'navigate',
            'url': url,
            'new_tab': self.open_in_new_tab
        }


class DrillDownAction(Action):
    """
    Drill-down action - navigate down hierarchical dimensions.

    Tableau equivalent: Drill-down with hierarchy.

    Example:
        >>> action = DrillDownAction(
        ...     action_id='geo_drilldown',
        ...     source_chart='geo_map',
        ...     hierarchy=['Country', 'State', 'City']
        ... )
    """

    def __init__(
        self,
        action_id: str,
        source_chart: str,
        hierarchy: List[str],
        trigger: TriggerEvent = TriggerEvent.DOUBLE_CLICK,
        enabled: bool = True
    ):
        """
        Initialize drill-down action.

        Args:
            action_id: Unique identifier
            source_chart: Source chart ID
            hierarchy: Hierarchical levels (high to low)
            trigger: Trigger event
            enabled: Whether active
        """
        super().__init__(action_id, ActionType.DRILL_DOWN, trigger, source_chart, [source_chart], enabled)
        self.hierarchy = hierarchy
        self.current_level = 0

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute drill-down action."""
        if self.current_level >= len(self.hierarchy) - 1:
            return {
                'executed': False,
                'reason': 'Already at deepest level'
            }

        selected_value = context.get('selected_value')

        self.current_level += 1
        current_dimension = self.hierarchy[self.current_level]

        return {
            'executed': True,
            'action': 'drill_down',
            'level': self.current_level,
            'dimension': current_dimension,
            'filter_by': {
                self.hierarchy[self.current_level - 1]: selected_value
            }
        }

    def drill_up(self) -> Dict[str, Any]:
        """Navigate up one level in hierarchy."""
        if self.current_level == 0:
            return {
                'executed': False,
                'reason': 'Already at highest level'
            }

        self.current_level -= 1
        current_dimension = self.hierarchy[self.current_level]

        return {
            'executed': True,
            'action': 'drill_up',
            'level': self.current_level,
            'dimension': current_dimension
        }

    def reset(self):
        """Reset to top level."""
        self.current_level = 0

    def get_current_level(self) -> str:
        """Get current hierarchy level."""
        return self.hierarchy[self.current_level]


class ParameterAction(Action):
    """
    Parameter action - sets dashboard parameter based on selection.

    Tableau equivalent: Set parameter action.

    Example:
        >>> action = ParameterAction(
        ...     action_id='set_year',
        ...     source_chart='year_selector',
        ...     parameter_name='selected_year',
        ...     value_column='year'
        ... )
    """

    def __init__(
        self,
        action_id: str,
        source_chart: str,
        parameter_name: str,
        value_column: str,
        trigger: TriggerEvent = TriggerEvent.SELECT,
        enabled: bool = True
    ):
        """
        Initialize parameter action.

        Args:
            action_id: Unique identifier
            source_chart: Source chart ID
            parameter_name: Parameter name to set
            value_column: Column to get value from
            trigger: Trigger event
            enabled: Whether active
        """
        super().__init__(action_id, ActionType.PARAMETER, trigger, source_chart, [], enabled)
        self.parameter_name = parameter_name
        self.value_column = value_column

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parameter action."""
        selected_data = context.get('selected_data', {})
        value = selected_data.get(self.value_column)

        return {
            'executed': True,
            'action': 'set_parameter',
            'parameter': self.parameter_name,
            'value': value
        }


class CustomAction(Action):
    """
    Custom action with user-defined callback.

    Example:
        >>> def my_action(context):
        ...     print(f"Selected: {context['selected_value']}")
        ...     return {'executed': True}
        >>>
        >>> action = CustomAction(
        ...     action_id='custom',
        ...     source_chart='my_chart',
        ...     callback=my_action
        ... )
    """

    def __init__(
        self,
        action_id: str,
        source_chart: str,
        callback: Callable[[Dict[str, Any]], Dict[str, Any]],
        trigger: TriggerEvent = TriggerEvent.CLICK,
        enabled: bool = True
    ):
        """
        Initialize custom action.

        Args:
            action_id: Unique identifier
            source_chart: Source chart ID
            callback: Custom callback function
            trigger: Trigger event
            enabled: Whether active
        """
        super().__init__(action_id, ActionType.CUSTOM, trigger, source_chart, [], enabled)
        self.callback = callback

    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom action."""
        return self.callback(context)


class ActionManager:
    """
    Manages all actions for a dashboard.

    Provides centralized action registration, execution, and tracking.

    Example:
        >>> manager = ActionManager()
        >>> manager.register_action(filter_action)
        >>> manager.register_action(drill_down_action)
        >>> manager.trigger('map', TriggerEvent.CLICK, {'selected_values': ['USA']})
    """

    def __init__(self):
        """Initialize action manager."""
        self.actions: Dict[str, Action] = {}
        self.action_history: List[Dict[str, Any]] = []

    def register_action(self, action: Action) -> 'ActionManager':
        """
        Register action.

        Args:
            action: Action to register

        Returns:
            Self for method chaining
        """
        self.actions[action.id] = action
        return self

    def unregister_action(self, action_id: str):
        """Unregister action."""
        if action_id in self.actions:
            del self.actions[action_id]

    def get_action(self, action_id: str) -> Optional[Action]:
        """Get action by ID."""
        return self.actions.get(action_id)

    def trigger(
        self,
        source_chart: str,
        event: TriggerEvent,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Trigger actions for a chart event.

        Args:
            source_chart: Chart where event occurred
            event: Trigger event
            context: Event context

        Returns:
            List of action results
        """
        results = []

        # Find matching actions
        for action in self.actions.values():
            if action.source_chart == source_chart and action.trigger == event:
                result = action.execute(context)
                results.append({
                    'action_id': action.id,
                    'action_type': action.type.value,
                    'result': result
                })

                # Record in history
                self.action_history.append({
                    'action_id': action.id,
                    'source': source_chart,
                    'event': event.value,
                    'context': context,
                    'result': result
                })

        return results

    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()

    def get_actions_for_chart(self, chart_id: str) -> List[Action]:
        """Get all actions for a chart."""
        return [
            action for action in self.actions.values()
            if action.source_chart == chart_id
        ]

    def enable_all(self):
        """Enable all actions."""
        for action in self.actions.values():
            action.enable()

    def disable_all(self):
        """Disable all actions."""
        for action in self.actions.values():
            action.disable()
