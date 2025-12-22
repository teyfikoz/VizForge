"""
VizForge Interactive Widgets

Streamlit-style widgets for interactive dashboards.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Optional, List, Callable, Union, Dict
from dataclasses import dataclass, field
from datetime import datetime, date
from abc import ABC, abstractmethod
import uuid


@dataclass
class WidgetConfig:
    """
    Widget configuration for rendering.

    Attributes:
        id: Unique widget identifier
        label: Display label
        value: Current value
        default: Default value
        disabled: Whether widget is disabled
        help_text: Help text/tooltip
        on_change: Callback function when value changes
    """
    id: str
    label: str
    value: Any
    default: Any
    disabled: bool = False
    help_text: Optional[str] = None
    on_change: Optional[Callable] = None


class Widget(ABC):
    """
    Base class for all VizForge widgets.

    Provides Streamlit-style API for interactive controls.
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        default: Any = None,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        """
        Initialize widget.

        Args:
            widget_id: Unique identifier (auto-generated if None)
            label: Display label
            default: Default value
            disabled: Whether widget is disabled
            help_text: Help text/tooltip
            on_change: Callback function when value changes
        """
        self.id = widget_id or f"widget_{uuid.uuid4().hex[:8]}"
        self.label = label
        self.value = default
        self.default = default
        self.disabled = disabled
        self.help_text = help_text
        self.on_change = on_change

    @abstractmethod
    def to_dash_component(self) -> Any:
        """
        Convert to Dash component.

        Returns:
            Dash component (e.g., dcc.Slider, dcc.Dropdown)
        """
        pass

    def get_value(self) -> Any:
        """Get current widget value."""
        return self.value

    def set_value(self, value: Any):
        """Set widget value and trigger callback if defined."""
        old_value = self.value
        self.value = value

        if self.on_change and old_value != value:
            self.on_change(value)

    def reset(self):
        """Reset widget to default value."""
        self.set_value(self.default)

    def to_config(self) -> WidgetConfig:
        """Export widget configuration."""
        return WidgetConfig(
            id=self.id,
            label=self.label,
            value=self.value,
            default=self.default,
            disabled=self.disabled,
            help_text=self.help_text,
            on_change=self.on_change
        )


class Slider(Widget):
    """
    Slider widget for numeric input.

    Streamlit equivalent: st.slider()

    Example:
        >>> slider = Slider(
        ...     widget_id='year_slider',
        ...     label='Select Year',
        ...     min_value=2020,
        ...     max_value=2024,
        ...     default=2023,
        ...     step=1
        ... )
        >>> print(slider.value)  # 2023
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        min_value: float = 0.0,
        max_value: float = 100.0,
        default: Optional[float] = None,
        step: float = 1.0,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        """
        Initialize slider widget.

        Args:
            widget_id: Unique identifier
            label: Display label
            min_value: Minimum value
            max_value: Maximum value
            default: Default value (defaults to min_value)
            step: Step size
            disabled: Whether disabled
            help_text: Help text
            on_change: Callback function
        """
        if default is None:
            default = min_value

        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def to_dash_component(self) -> Any:
        """Convert to Dash Slider component."""
        try:
            from dash import dcc
            return dcc.Slider(
                id=self.id,
                min=self.min_value,
                max=self.max_value,
                step=self.step,
                value=self.value,
                marks={i: str(i) for i in range(int(self.min_value), int(self.max_value) + 1, int(self.step))},
                disabled=self.disabled,
                tooltip={"placement": "bottom", "always_visible": False}
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class RangeSlider(Widget):
    """
    Range slider widget for selecting a range.

    Streamlit equivalent: st.slider() with range

    Example:
        >>> range_slider = RangeSlider(
        ...     widget_id='price_range',
        ...     label='Price Range',
        ...     min_value=0,
        ...     max_value=1000,
        ...     default=[100, 500]
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        min_value: float = 0.0,
        max_value: float = 100.0,
        default: Optional[List[float]] = None,
        step: float = 1.0,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        if default is None:
            default = [min_value, max_value]

        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def to_dash_component(self) -> Any:
        """Convert to Dash RangeSlider component."""
        try:
            from dash import dcc
            return dcc.RangeSlider(
                id=self.id,
                min=self.min_value,
                max=self.max_value,
                step=self.step,
                value=self.value,
                marks={i: str(i) for i in range(int(self.min_value), int(self.max_value) + 1, int(self.step * 5))},
                disabled=self.disabled,
                tooltip={"placement": "bottom", "always_visible": False}
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class SelectBox(Widget):
    """
    Dropdown select widget.

    Streamlit equivalent: st.selectbox()

    Example:
        >>> select = SelectBox(
        ...     widget_id='category',
        ...     label='Select Category',
        ...     options=['Electronics', 'Clothing', 'Food'],
        ...     default='Electronics'
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        options: List[Any] = None,
        default: Optional[Any] = None,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        if options is None:
            options = []

        if default is None and options:
            default = options[0]

        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.options = options

    def to_dash_component(self) -> Any:
        """Convert to Dash Dropdown component."""
        try:
            from dash import dcc
            return dcc.Dropdown(
                id=self.id,
                options=[{'label': str(opt), 'value': opt} for opt in self.options],
                value=self.value,
                disabled=self.disabled,
                clearable=False
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class MultiSelect(Widget):
    """
    Multi-select dropdown widget.

    Streamlit equivalent: st.multiselect()

    Example:
        >>> multi = MultiSelect(
        ...     widget_id='regions',
        ...     label='Select Regions',
        ...     options=['North', 'South', 'East', 'West'],
        ...     default=['North', 'South']
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        options: List[Any] = None,
        default: Optional[List[Any]] = None,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        if options is None:
            options = []

        if default is None:
            default = []

        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.options = options

    def to_dash_component(self) -> Any:
        """Convert to Dash multi-select Dropdown component."""
        try:
            from dash import dcc
            return dcc.Dropdown(
                id=self.id,
                options=[{'label': str(opt), 'value': opt} for opt in self.options],
                value=self.value,
                multi=True,
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class DatePicker(Widget):
    """
    Date picker widget.

    Streamlit equivalent: st.date_input()

    Example:
        >>> date_picker = DatePicker(
        ...     widget_id='start_date',
        ...     label='Start Date',
        ...     default=date(2024, 1, 1)
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        default: Optional[date] = None,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        if default is None:
            default = date.today()

        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.min_date = min_date
        self.max_date = max_date

    def to_dash_component(self) -> Any:
        """Convert to Dash DatePickerSingle component."""
        try:
            from dash import dcc
            return dcc.DatePickerSingle(
                id=self.id,
                date=self.value,
                min_date_allowed=self.min_date,
                max_date_allowed=self.max_date,
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class DateRangePicker(Widget):
    """
    Date range picker widget.

    Streamlit equivalent: st.date_input() with range

    Example:
        >>> date_range = DateRangePicker(
        ...     widget_id='date_range',
        ...     label='Date Range',
        ...     default=[date(2024, 1, 1), date(2024, 12, 31)]
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        default: Optional[List[date]] = None,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        if default is None:
            default = [date.today(), date.today()]

        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.min_date = min_date
        self.max_date = max_date

    def to_dash_component(self) -> Any:
        """Convert to Dash DatePickerRange component."""
        try:
            from dash import dcc
            return dcc.DatePickerRange(
                id=self.id,
                start_date=self.value[0],
                end_date=self.value[1],
                min_date_allowed=self.min_date,
                max_date_allowed=self.max_date,
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class TextInput(Widget):
    """
    Text input widget.

    Streamlit equivalent: st.text_input()

    Example:
        >>> text = TextInput(
        ...     widget_id='search',
        ...     label='Search',
        ...     default='',
        ...     placeholder='Enter search term...'
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        default: str = "",
        placeholder: str = "",
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.placeholder = placeholder

    def to_dash_component(self) -> Any:
        """Convert to Dash Input component."""
        try:
            from dash import dcc
            return dcc.Input(
                id=self.id,
                type='text',
                value=self.value,
                placeholder=self.placeholder,
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class NumberInput(Widget):
    """
    Number input widget.

    Streamlit equivalent: st.number_input()

    Example:
        >>> number = NumberInput(
        ...     widget_id='quantity',
        ...     label='Quantity',
        ...     default=1,
        ...     min_value=1,
        ...     max_value=100
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        default: float = 0.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 1.0,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def to_dash_component(self) -> Any:
        """Convert to Dash Input component with number type."""
        try:
            from dash import dcc
            return dcc.Input(
                id=self.id,
                type='number',
                value=self.value,
                min=self.min_value,
                max=self.max_value,
                step=self.step,
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class Checkbox(Widget):
    """
    Checkbox widget.

    Streamlit equivalent: st.checkbox()

    Example:
        >>> checkbox = Checkbox(
        ...     widget_id='show_grid',
        ...     label='Show Grid',
        ...     default=True
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        default: bool = False,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        super().__init__(widget_id, label, default, disabled, help_text, on_change)

    def to_dash_component(self) -> Any:
        """Convert to Dash Checklist component."""
        try:
            from dash import dcc
            return dcc.Checklist(
                id=self.id,
                options=[{'label': self.label, 'value': 'checked'}],
                value=['checked'] if self.value else [],
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class RadioButtons(Widget):
    """
    Radio buttons widget.

    Streamlit equivalent: st.radio()

    Example:
        >>> radio = RadioButtons(
        ...     widget_id='view_mode',
        ...     label='View Mode',
        ...     options=['Table', 'Chart', 'Both'],
        ...     default='Chart'
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        options: List[Any] = None,
        default: Optional[Any] = None,
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        if options is None:
            options = []

        if default is None and options:
            default = options[0]

        super().__init__(widget_id, label, default, disabled, help_text, on_change)

        self.options = options

    def to_dash_component(self) -> Any:
        """Convert to Dash RadioItems component."""
        try:
            from dash import dcc
            return dcc.RadioItems(
                id=self.id,
                options=[{'label': str(opt), 'value': opt} for opt in self.options],
                value=self.value,
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class Button(Widget):
    """
    Button widget.

    Streamlit equivalent: st.button()

    Example:
        >>> button = Button(
        ...     widget_id='refresh',
        ...     label='Refresh Data',
        ...     on_click=lambda: print('Refreshing...')
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "Click",
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_click: Optional[Callable] = None
    ):
        super().__init__(widget_id, label, False, disabled, help_text, on_click)

        self.on_click = on_click

    def click(self):
        """Trigger button click."""
        if self.on_click:
            self.on_click()

    def to_dash_component(self) -> Any:
        """Convert to Dash Button component."""
        try:
            from dash import html
            return html.Button(
                id=self.id,
                children=self.label,
                disabled=self.disabled,
                n_clicks=0
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


class ColorPicker(Widget):
    """
    Color picker widget.

    Streamlit equivalent: st.color_picker()

    Example:
        >>> color = ColorPicker(
        ...     widget_id='theme_color',
        ...     label='Theme Color',
        ...     default='#FF0000'
        ... )
    """

    def __init__(
        self,
        widget_id: Optional[str] = None,
        label: str = "",
        default: str = "#000000",
        disabled: bool = False,
        help_text: Optional[str] = None,
        on_change: Optional[Callable] = None
    ):
        super().__init__(widget_id, label, default, disabled, help_text, on_change)

    def to_dash_component(self) -> Any:
        """Convert to Dash Input with color type."""
        try:
            from dash import dcc
            return dcc.Input(
                id=self.id,
                type='color',
                value=self.value,
                disabled=self.disabled
            )
        except ImportError:
            raise ImportError("Dash is required for interactive widgets. Install with: pip install dash")


# ==================== Widget Factory ====================

class WidgetFactory:
    """
    Factory for creating widgets with sensible defaults.

    Provides convenience methods for common widget patterns.
    """

    @staticmethod
    def year_slider(
        widget_id: str = 'year',
        min_year: int = 2020,
        max_year: int = 2024,
        default: Optional[int] = None
    ) -> Slider:
        """Create a year slider widget."""
        if default is None:
            default = max_year

        return Slider(
            widget_id=widget_id,
            label='Year',
            min_value=min_year,
            max_value=max_year,
            default=default,
            step=1
        )

    @staticmethod
    def percentage_slider(
        widget_id: str = 'percentage',
        default: float = 50.0,
        label: str = 'Percentage'
    ) -> Slider:
        """Create a percentage slider (0-100)."""
        return Slider(
            widget_id=widget_id,
            label=label,
            min_value=0.0,
            max_value=100.0,
            default=default,
            step=1.0
        )

    @staticmethod
    def category_select(
        widget_id: str,
        categories: List[str],
        label: str = 'Category',
        default: Optional[str] = None
    ) -> SelectBox:
        """Create a category select box."""
        return SelectBox(
            widget_id=widget_id,
            label=label,
            options=categories,
            default=default
        )

    @staticmethod
    def date_range_this_year(
        widget_id: str = 'date_range',
        label: str = 'Date Range'
    ) -> DateRangePicker:
        """Create a date range picker for current year."""
        today = date.today()
        start_of_year = date(today.year, 1, 1)

        return DateRangePicker(
            widget_id=widget_id,
            label=label,
            default=[start_of_year, today]
        )
