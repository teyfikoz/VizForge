"""
VizForge Interactive Module

Provides Streamlit/Dash-style interactivity for dashboards.
Part of VizForge v1.0.0 - Super AGI features.
"""

# Session State
# Actions (NEW v1.0.0)
from .actions import (
    Action,
    ActionManager,
    ActionType,
    CustomAction,
    DrillDownAction,
    FilterAction,
    HighlightAction,
    ParameterAction,
    TriggerEvent,
    URLAction,
)

# Callbacks
from .callbacks import Callback, CallbackManager

# Filters (NEW v1.0.0)
from .filters import (
    CrossFilter,
    CustomFilter,
    DateRangeFilter,
    Filter,
    FilterContext,
    FilterType,
    ListFilter,
    RangeFilter,
    SearchFilter,
    TopNFilter,
)
from .state import SessionState, get_session_state

# Widgets (NEW v1.0.0)
from .widgets import (
    Button,
    Checkbox,
    ColorPicker,
    DatePicker,
    DateRangePicker,
    MultiSelect,
    NumberInput,
    RadioButtons,
    RangeSlider,
    SelectBox,
    Slider,
    TextInput,
    Widget,
    WidgetFactory,
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
