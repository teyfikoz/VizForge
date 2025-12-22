"""
VizForge Interactive Module

Provides Streamlit/Dash-style interactivity for dashboards.
Part of VizForge v1.0.0 - Super AGI features.
"""

# Session State
from .state import SessionState, get_session_state

# Callbacks
from .callbacks import CallbackManager, Callback

# Widgets (NEW v1.0.0)
from .widgets import (
    Widget, Slider, RangeSlider, SelectBox, MultiSelect,
    DatePicker, DateRangePicker, TextInput, NumberInput,
    Checkbox, RadioButtons, Button, ColorPicker, WidgetFactory
)

# Filters (NEW v1.0.0)
from .filters import (
    Filter, RangeFilter, ListFilter, SearchFilter,
    DateRangeFilter, TopNFilter, CustomFilter,
    FilterContext, CrossFilter, FilterType
)

# Actions (NEW v1.0.0)
from .actions import (
    Action, FilterAction, HighlightAction, URLAction,
    DrillDownAction, ParameterAction, CustomAction,
    ActionManager, ActionType, TriggerEvent
)

__all__ = [
    # Session State
    'SessionState',
    'get_session_state',

    # Callbacks
    'CallbackManager',
    'Callback',

    # Widgets
    'Widget', 'Slider', 'RangeSlider', 'SelectBox', 'MultiSelect',
    'DatePicker', 'DateRangePicker', 'TextInput', 'NumberInput',
    'Checkbox', 'RadioButtons', 'Button', 'ColorPicker', 'WidgetFactory',

    # Filters
    'Filter', 'RangeFilter', 'ListFilter', 'SearchFilter',
    'DateRangeFilter', 'TopNFilter', 'CustomFilter',
    'FilterContext', 'CrossFilter', 'FilterType',

    # Actions
    'Action', 'FilterAction', 'HighlightAction', 'URLAction',
    'DrillDownAction', 'ParameterAction', 'CustomAction',
    'ActionManager', 'ActionType', 'TriggerEvent',
]
