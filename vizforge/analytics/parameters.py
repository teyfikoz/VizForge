"""
VizForge Dashboard Parameters

Tableau-style parameters for interactive dashboards.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import date, datetime


class ParameterType(Enum):
    """Types of parameters."""
    NUMBER = "number"
    STRING = "string"
    DATE = "date"
    BOOLEAN = "boolean"
    LIST = "list"


@dataclass
class ParameterConstraint:
    """
    Parameter value constraints.

    Attributes:
        min_value: Minimum value (for numbers/dates)
        max_value: Maximum value (for numbers/dates)
        allowed_values: List of allowed values
        pattern: Regex pattern (for strings)
    """
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None


class Parameter:
    """
    Dashboard parameter for dynamic values.

    Tableau equivalent: Parameter.

    Example:
        >>> # Numeric parameter
        >>> threshold = Parameter(
        ...     name='Sales Threshold',
        ...     param_type=ParameterType.NUMBER,
        ...     default_value=1000,
        ...     min_value=0,
        ...     max_value=10000
        ... )
        >>>
        >>> # List parameter
        >>> region = Parameter(
        ...     name='Region',
        ...     param_type=ParameterType.LIST,
        ...     allowed_values=['North', 'South', 'East', 'West'],
        ...     default_value='North'
        ... )
    """

    def __init__(
        self,
        name: str,
        param_type: ParameterType,
        default_value: Any,
        description: str = "",
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        allowed_values: Optional[List[Any]] = None,
        on_change: Optional[Callable] = None
    ):
        """
        Initialize parameter.

        Args:
            name: Parameter name
            param_type: Parameter type
            default_value: Default value
            description: Parameter description
            min_value: Minimum value (for numbers/dates)
            max_value: Maximum value (for numbers/dates)
            allowed_values: List of allowed values
            on_change: Callback when value changes
        """
        self.name = name
        self.param_type = param_type
        self.default_value = default_value
        self.description = description
        self._value = default_value
        self.on_change = on_change

        # Create constraints
        self.constraints = ParameterConstraint(
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values
        )

        # Validate default value
        self._validate(default_value)

    @property
    def value(self) -> Any:
        """Get current parameter value."""
        return self._value

    @value.setter
    def value(self, new_value: Any):
        """Set parameter value with validation."""
        self._validate(new_value)
        old_value = self._value
        self._value = new_value

        # Trigger callback if value changed
        if self.on_change and old_value != new_value:
            self.on_change(new_value)

    def _validate(self, value: Any):
        """
        Validate parameter value.

        Raises:
            ValueError: If value is invalid
        """
        # Type validation
        if self.param_type == ParameterType.NUMBER:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Expected number, got {type(value)}")

            # Range validation
            if self.constraints.min_value is not None and value < self.constraints.min_value:
                raise ValueError(f"Value {value} below minimum {self.constraints.min_value}")
            if self.constraints.max_value is not None and value > self.constraints.max_value:
                raise ValueError(f"Value {value} above maximum {self.constraints.max_value}")

        elif self.param_type == ParameterType.STRING:
            if not isinstance(value, str):
                raise ValueError(f"Expected string, got {type(value)}")

        elif self.param_type == ParameterType.DATE:
            if not isinstance(value, (date, datetime)):
                raise ValueError(f"Expected date, got {type(value)}")

            # Date range validation
            if self.constraints.min_value and value < self.constraints.min_value:
                raise ValueError(f"Date {value} before minimum {self.constraints.min_value}")
            if self.constraints.max_value and value > self.constraints.max_value:
                raise ValueError(f"Date {value} after maximum {self.constraints.max_value}")

        elif self.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"Expected boolean, got {type(value)}")

        elif self.param_type == ParameterType.LIST:
            # Check if value in allowed values
            if self.constraints.allowed_values and value not in self.constraints.allowed_values:
                raise ValueError(f"Value {value} not in allowed values: {self.constraints.allowed_values}")

    def reset(self):
        """Reset parameter to default value."""
        self.value = self.default_value

    def __repr__(self) -> str:
        """String representation."""
        return f"Parameter('{self.name}', {self.param_type.value}, value={self.value})"


class ParameterManager:
    """
    Manage dashboard parameters.

    Handles parameter creation, validation, and value updates.

    Example:
        >>> manager = ParameterManager()
        >>>
        >>> # Add parameters
        >>> manager.add_parameter(Parameter(
        ...     'Threshold', ParameterType.NUMBER, default_value=1000
        ... ))
        >>> manager.add_parameter(Parameter(
        ...     'Region', ParameterType.LIST,
        ...     allowed_values=['North', 'South'],
        ...     default_value='North'
        ... ))
        >>>
        >>> # Update values
        >>> manager.set_value('Threshold', 2000)
        >>> manager.set_value('Region', 'South')
    """

    def __init__(self):
        """Initialize parameter manager."""
        self.parameters: dict[str, Parameter] = {}
        self._change_listeners: List[Callable] = []

    def add_parameter(self, parameter: Parameter) -> 'ParameterManager':
        """
        Add parameter to manager.

        Args:
            parameter: Parameter instance

        Returns:
            Self for method chaining
        """
        self.parameters[parameter.name] = parameter
        return self

    def remove_parameter(self, name: str):
        """Remove parameter."""
        if name in self.parameters:
            del self.parameters[name]

    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Get parameter by name."""
        return self.parameters.get(name)

    def get_value(self, name: str) -> Any:
        """
        Get parameter value.

        Args:
            name: Parameter name

        Returns:
            Parameter value

        Example:
            >>> threshold = manager.get_value('Threshold')
        """
        if name not in self.parameters:
            raise ValueError(f"Parameter '{name}' not found")

        return self.parameters[name].value

    def set_value(self, name: str, value: Any):
        """
        Set parameter value.

        Args:
            name: Parameter name
            value: New value

        Example:
            >>> manager.set_value('Threshold', 2000)
        """
        if name not in self.parameters:
            raise ValueError(f"Parameter '{name}' not found")

        self.parameters[name].value = value

        # Notify listeners
        self._notify_listeners(name, value)

    def set_values(self, values: dict[str, Any]):
        """
        Set multiple parameter values.

        Args:
            values: Dictionary of name → value

        Example:
            >>> manager.set_values({
            ...     'Threshold': 2000,
            ...     'Region': 'South'
            ... })
        """
        for name, value in values.items():
            self.set_value(name, value)

    def get_all_values(self) -> dict[str, Any]:
        """
        Get all parameter values.

        Returns:
            Dictionary of name → value

        Example:
            >>> values = manager.get_all_values()
            >>> print(values)  # {'Threshold': 2000, 'Region': 'South'}
        """
        return {
            name: param.value
            for name, param in self.parameters.items()
        }

    def reset_all(self):
        """Reset all parameters to default values."""
        for param in self.parameters.values():
            param.reset()

    def add_change_listener(self, callback: Callable[[str, Any], None]):
        """
        Add listener for parameter changes.

        Args:
            callback: Function(parameter_name, new_value)

        Example:
            >>> def on_change(name, value):
            ...     print(f"{name} changed to {value}")
            >>> manager.add_change_listener(on_change)
        """
        self._change_listeners.append(callback)

    def _notify_listeners(self, parameter_name: str, value: Any):
        """Notify all change listeners."""
        for listener in self._change_listeners:
            try:
                listener(parameter_name, value)
            except Exception as e:
                print(f"Error in change listener: {e}")

    def get_summary(self) -> List[dict[str, Any]]:
        """
        Get summary of all parameters.

        Returns:
            List of parameter summaries
        """
        return [
            {
                'name': param.name,
                'type': param.param_type.value,
                'value': param.value,
                'default': param.default_value,
                'description': param.description,
                'constraints': {
                    'min_value': param.constraints.min_value,
                    'max_value': param.constraints.max_value,
                    'allowed_values': param.constraints.allowed_values
                }
            }
            for param in self.parameters.values()
        ]


# ==================== Helper Functions ====================

def create_numeric_parameter(
    name: str,
    default_value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    description: str = ""
) -> Parameter:
    """
    Create numeric parameter.

    Example:
        >>> threshold = create_numeric_parameter(
        ...     'Threshold', 1000, min_value=0, max_value=10000
        ... )
    """
    return Parameter(
        name=name,
        param_type=ParameterType.NUMBER,
        default_value=default_value,
        min_value=min_value,
        max_value=max_value,
        description=description
    )


def create_list_parameter(
    name: str,
    allowed_values: List[Any],
    default_value: Any,
    description: str = ""
) -> Parameter:
    """
    Create list parameter.

    Example:
        >>> region = create_list_parameter(
        ...     'Region', ['North', 'South', 'East', 'West'], 'North'
        ... )
    """
    return Parameter(
        name=name,
        param_type=ParameterType.LIST,
        default_value=default_value,
        allowed_values=allowed_values,
        description=description
    )


def create_date_parameter(
    name: str,
    default_value: date,
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
    description: str = ""
) -> Parameter:
    """
    Create date parameter.

    Example:
        >>> start_date = create_date_parameter(
        ...     'Start Date', date(2024, 1, 1),
        ...     min_date=date(2020, 1, 1)
        ... )
    """
    return Parameter(
        name=name,
        param_type=ParameterType.DATE,
        default_value=default_value,
        min_value=min_date,
        max_value=max_date,
        description=description
    )
