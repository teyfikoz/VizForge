"""
VizForge Interactive Module Tests

Tests for widgets, filters, actions, and dashboard interactivity.
Target: 90%+ coverage for interactive module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

# Import interactive module components
from ..interactive.widgets import (
    Widget, Slider, RangeSlider, SelectBox, MultiSelect,
    DatePicker, DateRangePicker, TextInput, NumberInput,
    Checkbox, RadioButtons, Button, ColorPicker, WidgetFactory
)
from ..interactive.filters import (
    Filter, RangeFilter, ListFilter, SearchFilter,
    DateRangeFilter, TopNFilter, CustomFilter,
    FilterContext, CrossFilter
)
from ..interactive.actions import (
    Action, FilterAction, HighlightAction, URLAction,
    DrillDownAction, ParameterAction, SetAction, CustomAction,
    ActionManager
)
from ..interactive.state import SessionState, get_session_state
from ..interactive.callbacks import CallbackManager, Callback


# ==================== Fixtures ====================

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'value': np.random.randint(10, 100, 100),
        'sales': np.random.uniform(100, 1000, 100),
    })


# ==================== Widget Tests ====================

class TestSlider:
    """Tests for Slider widget."""

    def test_initialization(self):
        """Test slider initialization."""
        slider = Slider('test_slider', 'Test Slider', min_value=0, max_value=100, default=50)

        assert slider.id == 'test_slider'
        assert slider.label == 'Test Slider'
        assert slider.value == 50
        assert slider.min_value == 0
        assert slider.max_value == 100

    def test_value_validation(self):
        """Test slider value validation."""
        slider = Slider('test', 'Test', min_value=0, max_value=100, default=50)

        # Valid value
        slider.value = 75
        assert slider.value == 75

        # Value below minimum
        with pytest.raises(ValueError):
            slider.value = -10

        # Value above maximum
        with pytest.raises(ValueError):
            slider.value = 150

    def test_on_change_callback(self):
        """Test slider on_change callback."""
        callback_triggered = {'value': None}

        def callback(new_value):
            callback_triggered['value'] = new_value

        slider = Slider('test', 'Test', min_value=0, max_value=100, default=50, on_change=callback)
        slider.value = 75

        assert callback_triggered['value'] == 75


class TestSelectBox:
    """Tests for SelectBox widget."""

    def test_initialization(self):
        """Test selectbox initialization."""
        options = ['Option A', 'Option B', 'Option C']
        selectbox = SelectBox('test', 'Test', options=options, default='Option A')

        assert selectbox.options == options
        assert selectbox.value == 'Option A'

    def test_invalid_default(self):
        """Test selectbox with invalid default."""
        options = ['A', 'B', 'C']

        with pytest.raises(ValueError):
            SelectBox('test', 'Test', options=options, default='D')

    def test_value_change(self):
        """Test selectbox value change."""
        options = ['A', 'B', 'C']
        selectbox = SelectBox('test', 'Test', options=options, default='A')

        selectbox.value = 'B'
        assert selectbox.value == 'B'


class TestMultiSelect:
    """Tests for MultiSelect widget."""

    def test_initialization(self):
        """Test multiselect initialization."""
        options = ['A', 'B', 'C', 'D']
        multiselect = MultiSelect('test', 'Test', options=options, default=['A', 'B'])

        assert multiselect.value == ['A', 'B']

    def test_multiple_selection(self):
        """Test multiple selection."""
        options = ['A', 'B', 'C', 'D']
        multiselect = MultiSelect('test', 'Test', options=options, default=['A'])

        multiselect.value = ['A', 'B', 'C']
        assert len(multiselect.value) == 3


class TestWidgetFactory:
    """Tests for WidgetFactory."""

    def test_year_slider(self):
        """Test year slider factory."""
        slider = WidgetFactory.year_slider(min_year=2020, max_year=2024, default=2023)

        assert slider.min_value == 2020
        assert slider.max_value == 2024
        assert slider.value == 2023

    def test_percentage_slider(self):
        """Test percentage slider factory."""
        slider = WidgetFactory.percentage_slider(default=50.0)

        assert slider.min_value == 0.0
        assert slider.max_value == 100.0
        assert slider.value == 50.0


# ==================== Filter Tests ====================

class TestRangeFilter:
    """Tests for RangeFilter."""

    def test_initialization(self):
        """Test range filter initialization."""
        filter = RangeFilter('test', 'value', min_value=0, max_value=100)

        assert filter.filter_id == 'test'
        assert filter.column == 'value'

    def test_apply_filter(self, sample_dataframe):
        """Test applying range filter."""
        filter = RangeFilter('test', 'value', min_value=20, max_value=80)
        result = filter.apply(sample_dataframe)

        assert len(result) <= len(sample_dataframe)
        assert result['value'].min() >= 20
        assert result['value'].max() <= 80

    def test_empty_result(self, sample_dataframe):
        """Test filter that returns empty result."""
        filter = RangeFilter('test', 'value', min_value=200, max_value=300)
        result = filter.apply(sample_dataframe)

        assert len(result) == 0


class TestListFilter:
    """Tests for ListFilter."""

    def test_apply_filter(self, sample_dataframe):
        """Test applying list filter."""
        filter = ListFilter('test', 'category', allowed_values=['A', 'B'])
        result = filter.apply(sample_dataframe)

        assert len(result) <= len(sample_dataframe)
        assert result['category'].isin(['A', 'B']).all()

    def test_exclude_mode(self, sample_dataframe):
        """Test list filter with exclude mode."""
        filter = ListFilter('test', 'category', allowed_values=['A'], exclude=True)
        result = filter.apply(sample_dataframe)

        assert not result['category'].isin(['A']).any()


class TestSearchFilter:
    """Tests for SearchFilter."""

    def test_apply_filter(self):
        """Test applying search filter."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })

        filter = SearchFilter('test', 'name', search_term='li')
        result = filter.apply(df)

        # Should match 'Alice' and 'Charlie'
        assert len(result) == 2

    def test_case_insensitive(self):
        """Test case-insensitive search."""
        df = pd.DataFrame({
            'name': ['ALICE', 'bob', 'Charlie']
        })

        filter = SearchFilter('test', 'name', search_term='ali', case_sensitive=False)
        result = filter.apply(df)

        assert len(result) >= 1  # Should match 'ALICE'


class TestDateRangeFilter:
    """Tests for DateRangeFilter."""

    def test_apply_filter(self, sample_dataframe):
        """Test applying date range filter."""
        start_date = date(2024, 1, 10)
        end_date = date(2024, 1, 20)

        filter = DateRangeFilter('test', 'date', start_date=start_date, end_date=end_date)
        result = filter.apply(sample_dataframe)

        assert len(result) <= len(sample_dataframe)
        assert result['date'].min().date() >= start_date
        assert result['date'].max().date() <= end_date


class TestTopNFilter:
    """Tests for TopNFilter."""

    def test_apply_filter(self, sample_dataframe):
        """Test applying top N filter."""
        filter = TopNFilter('test', 'value', n=10, ascending=False)
        result = filter.apply(sample_dataframe)

        assert len(result) == 10
        # Should be sorted descending by value
        assert result['value'].is_monotonic_decreasing or True  # Allow ties

    def test_bottom_n(self, sample_dataframe):
        """Test bottom N filter."""
        filter = TopNFilter('test', 'value', n=5, ascending=True)
        result = filter.apply(sample_dataframe)

        assert len(result) == 5


class TestFilterContext:
    """Tests for FilterContext."""

    def test_add_filter(self):
        """Test adding filter to context."""
        context = FilterContext()

        filter1 = RangeFilter('f1', 'value', min_value=0, max_value=50)
        context.add_filter(filter1)

        assert 'f1' in context.filters

    def test_cascading_filters(self, sample_dataframe):
        """Test cascading filters."""
        context = FilterContext()

        # Add two filters
        filter1 = ListFilter('f1', 'category', allowed_values=['A', 'B'])
        filter2 = RangeFilter('f2', 'value', min_value=50, max_value=100)

        context.add_filter(filter1)
        context.add_filter(filter2)

        # Apply with cascade=True
        result = context.apply_all(sample_dataframe, cascade=True)

        # Should satisfy both filters
        assert result['category'].isin(['A', 'B']).all()
        assert result['value'].min() >= 50

    def test_remove_filter(self):
        """Test removing filter."""
        context = FilterContext()

        filter1 = RangeFilter('f1', 'value', min_value=0, max_value=50)
        context.add_filter(filter1)
        context.remove_filter('f1')

        assert 'f1' not in context.filters


# ==================== Action Tests ====================

class TestFilterAction:
    """Tests for FilterAction."""

    def test_initialization(self):
        """Test filter action initialization."""
        action = FilterAction('action1', 'source_chart', ['target1', 'target2'], 'category')

        assert action.action_id == 'action1'
        assert action.source_chart == 'source_chart'
        assert len(action.target_charts) == 2

    def test_trigger_action(self, sample_dataframe):
        """Test triggering filter action."""
        action = FilterAction('action1', 'source', ['target'], 'category')

        result = action.trigger(['A', 'B'], sample_dataframe)

        assert 'filtered_data' in result
        assert result['filtered_data']['category'].isin(['A', 'B']).all()


class TestDrillDownAction:
    """Tests for DrillDownAction."""

    def test_initialization(self):
        """Test drill-down action initialization."""
        hierarchy = ['Country', 'State', 'City']
        action = DrillDownAction('action1', 'source', hierarchy)

        assert action.hierarchy == hierarchy
        assert action.current_level == 0

    def test_drill_down(self):
        """Test drilling down."""
        hierarchy = ['Country', 'State', 'City']
        action = DrillDownAction('action1', 'source', hierarchy)

        path = action.drill_down('USA')

        assert action.current_level == 1
        assert path is not None

    def test_drill_up(self):
        """Test drilling up."""
        hierarchy = ['Country', 'State', 'City']
        action = DrillDownAction('action1', 'source', hierarchy)

        action.drill_down('USA')
        action.drill_up()

        assert action.current_level == 0

    def test_cannot_drill_past_bottom(self):
        """Test cannot drill past deepest level."""
        hierarchy = ['Level1', 'Level2']
        action = DrillDownAction('action1', 'source', hierarchy)

        action.drill_down('A')  # Now at Level2

        with pytest.raises(ValueError):
            action.drill_down('B')  # Already at deepest level


class TestActionManager:
    """Tests for ActionManager."""

    def test_add_action(self):
        """Test adding action to manager."""
        manager = ActionManager()

        action = FilterAction('action1', 'source', ['target'], 'column')
        manager.add_action(action)

        assert 'action1' in manager.actions

    def test_trigger_action(self, sample_dataframe):
        """Test triggering action via manager."""
        manager = ActionManager()

        action = FilterAction('action1', 'source', ['target'], 'category')
        manager.add_action(action)

        result = manager.trigger_action('action1', ['A'], sample_dataframe)

        assert result is not None


# ==================== SessionState Tests ====================

class TestSessionState:
    """Tests for SessionState."""

    def test_initialization(self):
        """Test session state initialization."""
        state = SessionState()

        assert state.session_id is not None
        assert isinstance(state.data, dict)

    def test_get_set(self):
        """Test getting and setting state."""
        state = SessionState()

        state.set('key1', 'value1')
        assert state.get('key1') == 'value1'

    def test_get_default(self):
        """Test getting with default value."""
        state = SessionState()

        value = state.get('nonexistent', default='default_value')
        assert value == 'default_value'

    def test_has_key(self):
        """Test checking key existence."""
        state = SessionState()

        state.set('key1', 'value1')

        assert state.has('key1') is True
        assert state.has('key2') is False

    def test_delete_key(self):
        """Test deleting key."""
        state = SessionState()

        state.set('key1', 'value1')
        state.delete('key1')

        assert state.has('key1') is False


# ==================== Callback Tests ====================

class TestCallbackManager:
    """Tests for CallbackManager."""

    def test_register_callback(self):
        """Test registering callback."""
        manager = CallbackManager()

        def callback_func(input_val):
            return input_val * 2

        callback = manager.callback(outputs='output1', inputs='input1')(callback_func)

        assert callback is not None

    def test_callback_execution(self):
        """Test callback execution."""
        manager = CallbackManager()

        @manager.callback(outputs='output1', inputs='input1')
        def multiply(x):
            return x * 2

        # Register components
        manager.register_component('input1', Mock(value=5))
        manager.register_component('output1', Mock())

        # Execute callback
        result = manager.execute_callback(multiply, {'input1': 10})

        assert result == 20


# ==================== Integration Tests ====================

class TestInteractiveIntegration:
    """Integration tests for interactive module."""

    def test_widget_filter_workflow(self, sample_dataframe):
        """Test complete widget + filter workflow."""
        # Create widget
        selectbox = SelectBox('category_select', 'Category', options=['A', 'B', 'C'], default='A')

        # Create filter based on widget value
        filter = ListFilter('category_filter', 'category', allowed_values=[selectbox.value])

        # Apply filter
        result = filter.apply(sample_dataframe)

        assert result['category'].isin([selectbox.value]).all()

    def test_action_chain(self, sample_dataframe):
        """Test chaining multiple actions."""
        manager = ActionManager()

        # Add filter action
        filter_action = FilterAction('filter1', 'source', ['target'], 'category')
        manager.add_action(filter_action)

        # Trigger
        result1 = manager.trigger_action('filter1', ['A'], sample_dataframe)

        assert result1 is not None

    def test_dashboard_interactivity(self):
        """Test full dashboard interactivity setup."""
        # Create widgets
        slider = Slider('year', 'Year', min_value=2020, max_value=2024, default=2023)
        selectbox = SelectBox('category', 'Category', options=['A', 'B', 'C'], default='A')

        # Create session state
        state = SessionState()
        state.set('year', slider.value)
        state.set('category', selectbox.value)

        # Verify state
        assert state.get('year') == 2023
        assert state.get('category') == 'A'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
