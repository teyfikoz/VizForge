"""
VizForge Analytics Module Tests

Tests for calculated fields, hierarchies, aggregations, and parameters.
Target: 90%+ coverage for analytics module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

# Import analytics module components
from ..analytics.calculated_fields import (
    CalculatedField, Expression, ExpressionParser,
    ExpressionType, CalculatedFieldManager
)
from ..analytics.hierarchies import (
    Hierarchy, DrillPath, HierarchyManager
)
from ..analytics.aggregations import (
    Aggregation, AggregationType, WindowFunction,
    WindowType, AggregationEngine
)
from ..analytics.parameters import (
    Parameter, ParameterType, ParameterConstraint,
    ParameterManager, create_numeric_parameter,
    create_list_parameter, create_date_parameter
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_sales_data():
    """Create sample sales data."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'category': ['Electronics', 'Clothing', 'Food'] * 33 + ['Electronics'],
        'product': ['A', 'B', 'C', 'D', 'E'] * 20,
        'revenue': np.random.uniform(100, 1000, 100),
        'cost': np.random.uniform(50, 500, 100),
        'quantity': np.random.randint(1, 50, 100),
    })


@pytest.fixture
def sample_hierarchy_data():
    """Create sample hierarchical data."""
    return pd.DataFrame({
        'Country': ['USA', 'USA', 'USA', 'Canada', 'Canada'] * 20,
        'State': ['CA', 'NY', 'TX', 'ON', 'BC'] * 20,
        'City': ['LA', 'NYC', 'Houston', 'Toronto', 'Vancouver'] * 20,
        'Sales': np.random.randint(1000, 10000, 100),
    })


# ==================== ExpressionParser Tests ====================

class TestExpressionParser:
    """Tests for ExpressionParser."""

    def test_parse_simple_arithmetic(self):
        """Test parsing simple arithmetic expression."""
        parser = ExpressionParser()
        expr = parser.parse('[Revenue] - [Cost]')

        assert expr.raw == '[Revenue] - [Cost]'
        assert 'Revenue' in expr.fields
        assert 'Cost' in expr.fields
        assert expr.type == ExpressionType.ARITHMETIC

    def test_parse_aggregation(self):
        """Test parsing aggregation expression."""
        parser = ExpressionParser()
        expr = parser.parse('SUM([Revenue]) / COUNT([Orders])')

        assert 'SUM' in expr.functions
        assert 'COUNT' in expr.functions
        assert expr.is_aggregation is True
        assert expr.type == ExpressionType.AGGREGATION

    def test_parse_conditional(self):
        """Test parsing conditional expression."""
        parser = ExpressionParser()
        expr = parser.parse('IF([Revenue] > 1000, "High", "Low")')

        assert 'IF' in expr.functions
        assert expr.type == ExpressionType.CONDITIONAL

    def test_validate_valid_expression(self):
        """Test validating valid expression."""
        parser = ExpressionParser()
        is_valid, error = parser.validate('[Revenue] * 2')

        assert is_valid is True
        assert error is None

    def test_validate_unbalanced_brackets(self):
        """Test validating expression with unbalanced brackets."""
        parser = ExpressionParser()
        is_valid, error = parser.validate('[Revenue * 2')

        assert is_valid is False
        assert 'brackets' in error.lower()

    def test_validate_empty_expression(self):
        """Test validating empty expression."""
        parser = ExpressionParser()
        is_valid, error = parser.validate('  ')

        assert is_valid is False


# ==================== CalculatedField Tests ====================

class TestCalculatedField:
    """Tests for CalculatedField."""

    def test_simple_calculation(self, sample_sales_data):
        """Test simple calculated field."""
        field = CalculatedField('Profit', '[revenue] - [cost]')
        result = field.apply(sample_sales_data)

        assert len(result) == len(sample_sales_data)
        expected_profit = sample_sales_data['revenue'] - sample_sales_data['cost']
        assert np.allclose(result, expected_profit)

    def test_percentage_calculation(self, sample_sales_data):
        """Test percentage calculation."""
        field = CalculatedField('Margin', '([revenue] - [cost]) / [revenue] * 100')
        result = field.apply(sample_sales_data)

        assert len(result) == len(sample_sales_data)
        assert result.max() <= 100

    def test_aggregation_field(self, sample_sales_data):
        """Test aggregation calculated field."""
        field = CalculatedField('Total Revenue', 'SUM([revenue])')
        result = field.apply(sample_sales_data)

        # Aggregation should return single value for all rows
        expected_total = sample_sales_data['revenue'].sum()
        assert abs(result.iloc[0] - expected_total) < 0.01

    def test_get_dependencies(self):
        """Test getting field dependencies."""
        field = CalculatedField('Profit', '[revenue] - [cost]')
        deps = field.get_dependencies()

        assert 'revenue' in deps
        assert 'cost' in deps

    def test_invalid_expression(self):
        """Test invalid expression raises error."""
        with pytest.raises(ValueError):
            CalculatedField('Invalid', '[revenue] - ')


class TestCalculatedFieldManager:
    """Tests for CalculatedFieldManager."""

    def test_add_field(self):
        """Test adding calculated field."""
        manager = CalculatedFieldManager()
        field = CalculatedField('Profit', '[revenue] - [cost]')

        manager.add_field(field)

        assert 'Profit' in manager.fields

    def test_apply_all(self, sample_sales_data):
        """Test applying all calculated fields."""
        manager = CalculatedFieldManager()

        # Add two fields
        manager.add_field(CalculatedField('profit', '[revenue] - [cost]'))
        manager.add_field(CalculatedField('margin', '[profit] / [revenue]'))

        result = manager.apply_all(sample_sales_data)

        assert 'profit' in result.columns
        assert 'margin' in result.columns

    def test_dependency_ordering(self, sample_sales_data):
        """Test fields are applied in dependency order."""
        manager = CalculatedFieldManager()

        # Add in wrong order (margin depends on profit)
        manager.add_field(CalculatedField('margin', '[profit] / [revenue]'))
        manager.add_field(CalculatedField('profit', '[revenue] - [cost]'))

        # Should still work due to topological sort
        result = manager.apply_all(sample_sales_data)

        assert 'profit' in result.columns
        assert 'margin' in result.columns


# ==================== Hierarchy Tests ====================

class TestHierarchy:
    """Tests for Hierarchy."""

    def test_initialization(self):
        """Test hierarchy initialization."""
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])

        assert hierarchy.name == 'Geography'
        assert len(hierarchy.levels) == 3
        assert hierarchy.current_level == 0

    def test_get_current_level(self):
        """Test getting current level."""
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])

        assert hierarchy.get_current_level() == 'Country'

    def test_can_drill_down(self):
        """Test checking if can drill down."""
        hierarchy = Hierarchy('Geography', ['Country', 'State'])

        assert hierarchy.can_drill_down() is True

        hierarchy.current_level = 1
        assert hierarchy.can_drill_down() is False

    def test_drill_down(self):
        """Test drilling down."""
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])

        path = hierarchy.drill_down('USA')

        assert hierarchy.current_level == 1
        assert path.current_level == 1
        assert len(path.breadcrumb) == 1

    def test_drill_up(self):
        """Test drilling up."""
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])

        hierarchy.drill_down('USA')
        path = hierarchy.drill_up()

        assert hierarchy.current_level == 0

    def test_reset(self):
        """Test resetting hierarchy."""
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])

        hierarchy.drill_down('USA')
        hierarchy.drill_down('CA')
        hierarchy.reset()

        assert hierarchy.current_level == 0


class TestHierarchyManager:
    """Tests for HierarchyManager."""

    def test_add_hierarchy(self):
        """Test adding hierarchy."""
        manager = HierarchyManager()
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])

        manager.add_hierarchy(hierarchy)

        assert 'Geography' in manager.hierarchies

    def test_drill_down_via_manager(self):
        """Test drilling down via manager."""
        manager = HierarchyManager()
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])
        manager.add_hierarchy(hierarchy)

        path = manager.drill_down('Geography', 'USA')

        assert path.current_level == 1

    def test_apply_filters(self, sample_hierarchy_data):
        """Test applying hierarchy filters to data."""
        manager = HierarchyManager()
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])
        manager.add_hierarchy(hierarchy)

        # Drill down to USA
        manager.drill_down('Geography', 'USA')

        # Apply filters
        filtered = manager.apply_filters(sample_hierarchy_data, 'Geography')

        assert filtered['Country'].eq('USA').all()

    def test_get_breadcrumb(self):
        """Test getting breadcrumb."""
        manager = HierarchyManager()
        hierarchy = Hierarchy('Geography', ['Country', 'State', 'City'])
        manager.add_hierarchy(hierarchy)

        manager.drill_down('Geography', 'USA')
        manager.drill_down('Geography', 'CA')

        breadcrumb = manager.get_breadcrumb('Geography')

        assert len(breadcrumb) == 2


# ==================== Aggregation Tests ====================

class TestAggregation:
    """Tests for Aggregation."""

    def test_sum_aggregation(self, sample_sales_data):
        """Test SUM aggregation."""
        agg = Aggregation(AggregationType.SUM, 'revenue')
        result = agg.apply(sample_sales_data)

        expected = sample_sales_data['revenue'].sum()
        assert abs(result - expected) < 0.01

    def test_avg_aggregation(self, sample_sales_data):
        """Test AVG aggregation."""
        agg = Aggregation(AggregationType.AVG, 'revenue')
        result = agg.apply(sample_sales_data)

        expected = sample_sales_data['revenue'].mean()
        assert abs(result - expected) < 0.01

    def test_grouped_aggregation(self, sample_sales_data):
        """Test grouped aggregation."""
        agg = Aggregation(AggregationType.SUM, 'revenue', group_by=['category'])
        result = agg.apply(sample_sales_data)

        assert len(result) == sample_sales_data['category'].nunique()


class TestWindowFunction:
    """Tests for WindowFunction."""

    def test_running_total(self, sample_sales_data):
        """Test running total window function."""
        window = WindowFunction(WindowType.RUNNING_TOTAL, 'revenue')
        result = window.apply(sample_sales_data)

        # Running total should be monotonic increasing
        assert result.is_monotonic_increasing

    def test_moving_average(self, sample_sales_data):
        """Test moving average window function."""
        window = WindowFunction(WindowType.MOVING_AVG, 'revenue', window_size=5)
        result = window.apply(sample_sales_data)

        assert len(result) == len(sample_sales_data)

    def test_rank(self, sample_sales_data):
        """Test rank window function."""
        window = WindowFunction(WindowType.RANK, 'revenue')
        result = window.apply(sample_sales_data)

        # Ranks should be from 1 to n
        assert result.min() >= 1
        assert result.max() <= len(sample_sales_data)

    def test_partition_by(self, sample_sales_data):
        """Test window function with partition."""
        window = WindowFunction(
            WindowType.RUNNING_TOTAL,
            'revenue',
            partition_by=['category']
        )
        result = window.apply(sample_sales_data)

        assert len(result) == len(sample_sales_data)


# ==================== Parameter Tests ====================

class TestParameter:
    """Tests for Parameter."""

    def test_numeric_parameter(self):
        """Test numeric parameter."""
        param = Parameter('threshold', ParameterType.NUMBER, default_value=100)

        assert param.value == 100

        param.value = 200
        assert param.value == 200

    def test_parameter_validation(self):
        """Test parameter validation."""
        param = Parameter(
            'threshold',
            ParameterType.NUMBER,
            default_value=50,
            min_value=0,
            max_value=100
        )

        # Valid value
        param.value = 75
        assert param.value == 75

        # Invalid value
        with pytest.raises(ValueError):
            param.value = 150

    def test_list_parameter(self):
        """Test list parameter."""
        param = Parameter(
            'region',
            ParameterType.LIST,
            default_value='North',
            allowed_values=['North', 'South', 'East', 'West']
        )

        assert param.value == 'North'

        param.value = 'South'
        assert param.value == 'South'

        # Invalid value
        with pytest.raises(ValueError):
            param.value = 'Invalid'

    def test_on_change_callback(self):
        """Test parameter on_change callback."""
        callback_value = {'value': None}

        def callback(new_val):
            callback_value['value'] = new_val

        param = Parameter('test', ParameterType.NUMBER, default_value=10, on_change=callback)

        param.value = 20

        assert callback_value['value'] == 20


class TestParameterManager:
    """Tests for ParameterManager."""

    def test_add_parameter(self):
        """Test adding parameter."""
        manager = ParameterManager()
        param = Parameter('threshold', ParameterType.NUMBER, default_value=100)

        manager.add_parameter(param)

        assert 'threshold' in manager.parameters

    def test_set_value(self):
        """Test setting parameter value."""
        manager = ParameterManager()
        param = Parameter('threshold', ParameterType.NUMBER, default_value=100)
        manager.add_parameter(param)

        manager.set_value('threshold', 200)

        assert manager.get_value('threshold') == 200

    def test_set_values(self):
        """Test setting multiple parameter values."""
        manager = ParameterManager()
        manager.add_parameter(Parameter('p1', ParameterType.NUMBER, default_value=10))
        manager.add_parameter(Parameter('p2', ParameterType.NUMBER, default_value=20))

        manager.set_values({'p1': 100, 'p2': 200})

        assert manager.get_value('p1') == 100
        assert manager.get_value('p2') == 200

    def test_get_all_values(self):
        """Test getting all parameter values."""
        manager = ParameterManager()
        manager.add_parameter(Parameter('p1', ParameterType.NUMBER, default_value=10))
        manager.add_parameter(Parameter('p2', ParameterType.NUMBER, default_value=20))

        all_values = manager.get_all_values()

        assert all_values['p1'] == 10
        assert all_values['p2'] == 20

    def test_reset_all(self):
        """Test resetting all parameters."""
        manager = ParameterManager()
        param1 = Parameter('p1', ParameterType.NUMBER, default_value=10)
        param2 = Parameter('p2', ParameterType.NUMBER, default_value=20)

        manager.add_parameter(param1)
        manager.add_parameter(param2)

        # Change values
        manager.set_value('p1', 100)
        manager.set_value('p2', 200)

        # Reset
        manager.reset_all()

        assert manager.get_value('p1') == 10
        assert manager.get_value('p2') == 20


# ==================== Helper Function Tests ====================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_numeric_parameter(self):
        """Test creating numeric parameter."""
        param = create_numeric_parameter('threshold', 100, min_value=0, max_value=200)

        assert param.param_type == ParameterType.NUMBER
        assert param.value == 100

    def test_create_list_parameter(self):
        """Test creating list parameter."""
        param = create_list_parameter('region', ['A', 'B', 'C'], 'A')

        assert param.param_type == ParameterType.LIST
        assert param.value == 'A'

    def test_create_date_parameter(self):
        """Test creating date parameter."""
        param = create_date_parameter('start_date', date(2024, 1, 1))

        assert param.param_type == ParameterType.DATE
        assert param.value == date(2024, 1, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
