"""
VizForge Callback System

Dash-style reactive callbacks for interactive dashboards.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import inspect


@dataclass
class Callback:
    """
    Callback specification (Dash-style).

    Defines reactive relationships between components:
    - Outputs: Components that will be updated
    - Inputs: Components that trigger the callback
    - State: Components whose values are read but don't trigger
    """

    outputs: List[str]  # Component IDs to update
    inputs: List[str]   # Component IDs that trigger callback
    state: List[str] = field(default_factory=list)  # Read-only components
    function: Optional[Callable] = None

    def __post_init__(self):
        """Validate callback specification."""
        if not self.outputs:
            raise ValueError("Callback must have at least one output")
        if not self.inputs:
            raise ValueError("Callback must have at least one input")


class CallbackManager:
    """
    Manage reactive callbacks between dashboard components.

    Provides Dash-style callback system for VizForge dashboards.

    Example:
        >>> manager = CallbackManager()
        >>> @manager.callback(outputs='chart1', inputs='filter_date')
        ... def update_chart(date_value):
        ...     return LineChart(filtered_data, x='date', y='sales')
    """

    def __init__(self):
        """Initialize callback manager."""
        self.callbacks: List[Callback] = []
        self.components: Dict[str, Any] = {}
        self._callback_graph: Dict[str, List[Callback]] = {}

    def register_component(self, component_id: str, component: Any):
        """
        Register a component for callback use.

        Args:
            component_id: Unique component identifier
            component: Component object (widget, chart, etc.)
        """
        self.components[component_id] = component

    def callback(
        self,
        outputs: Union[str, List[str]],
        inputs: Union[str, List[str]],
        state: Union[str, List[str], None] = None
    ):
        """
        Decorator for creating callbacks.

        Args:
            outputs: Output component ID(s)
            inputs: Input component ID(s) that trigger callback
            state: State component ID(s) (read-only)

        Returns:
            Decorated function

        Example:
            @callback_manager.callback(
                outputs='sales_chart',
                inputs=['filter_date', 'filter_category']
            )
            def update_sales(date_value, category_value):
                filtered_data = df[(df['date'] == date_value) &
                                  (df['category'] == category_value)]
                return LineChart(filtered_data, x='date', y='sales')
        """
        def decorator(func: Callable):
            # Normalize to lists
            outputs_list = [outputs] if isinstance(outputs, str) else outputs
            inputs_list = [inputs] if isinstance(inputs, str) else inputs
            state_list = [] if state is None else ([state] if isinstance(state, str) else state)

            # Create callback
            cb = Callback(
                outputs=outputs_list,
                inputs=inputs_list,
                state=state_list,
                function=func
            )

            self.callbacks.append(cb)

            # Build callback graph for efficient lookup
            for input_id in inputs_list:
                if input_id not in self._callback_graph:
                    self._callback_graph[input_id] = []
                self._callback_graph[input_id].append(cb)

            return func

        return decorator

    def trigger(self, component_id: str, value: Any) -> Dict[str, Any]:
        """
        Trigger callbacks when a component value changes.

        Args:
            component_id: ID of component that changed
            value: New value of the component

        Returns:
            Dictionary mapping output IDs to their new values
        """
        results = {}

        # Find all callbacks that depend on this input
        if component_id not in self._callback_graph:
            return results

        for callback in self._callback_graph[component_id]:
            try:
                # Gather input values
                input_values = []
                for input_id in callback.inputs:
                    if input_id == component_id:
                        input_values.append(value)
                    elif input_id in self.components:
                        input_values.append(self._get_component_value(input_id))
                    else:
                        input_values.append(None)

                # Gather state values
                state_values = []
                for state_id in callback.state:
                    if state_id in self.components:
                        state_values.append(self._get_component_value(state_id))
                    else:
                        state_values.append(None)

                # Execute callback
                callback_results = callback.function(*input_values, *state_values)

                # Handle results
                if not isinstance(callback_results, tuple):
                    callback_results = (callback_results,)

                # Map results to outputs
                for output_id, result in zip(callback.outputs, callback_results):
                    results[output_id] = result

                    # Update component value
                    if output_id in self.components:
                        self._set_component_value(output_id, result)

            except Exception as e:
                print(f"Error executing callback for {component_id}: {e}")
                # Continue with other callbacks even if one fails

        return results

    def _get_component_value(self, component_id: str) -> Any:
        """Get value from a component."""
        component = self.components.get(component_id)
        if component is None:
            return None

        # Try different value access patterns
        if hasattr(component, 'value'):
            return component.value
        elif hasattr(component, 'get_value'):
            return component.get_value()
        elif isinstance(component, dict) and 'value' in component:
            return component['value']
        else:
            return component

    def _set_component_value(self, component_id: str, value: Any):
        """Set value on a component."""
        component = self.components.get(component_id)
        if component is None:
            return

        # Try different value setting patterns
        if hasattr(component, 'update'):
            component.update(value)
        elif hasattr(component, 'value'):
            component.value = value
        elif hasattr(component, 'set_value'):
            component.set_value(value)
        elif isinstance(component, dict):
            component['value'] = value

    def get_dependencies(self, component_id: str) -> List[str]:
        """
        Get all components that depend on this component.

        Args:
            component_id: Component ID

        Returns:
            List of dependent component IDs
        """
        dependent_ids = []

        if component_id in self._callback_graph:
            for callback in self._callback_graph[component_id]:
                dependent_ids.extend(callback.outputs)

        return list(set(dependent_ids))  # Remove duplicates

    def clear_callbacks(self):
        """Clear all registered callbacks."""
        self.callbacks.clear()
        self._callback_graph.clear()

    def get_callback_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered callbacks.

        Returns:
            List of callback information dictionaries
        """
        info = []
        for cb in self.callbacks:
            info.append({
                'outputs': cb.outputs,
                'inputs': cb.inputs,
                'state': cb.state,
                'function_name': cb.function.__name__ if cb.function else None
            })
        return info
