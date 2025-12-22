"""
VizForge Calculated Fields

Tableau-style calculated fields with expression parsing.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re
from enum import Enum


class ExpressionType(Enum):
    """Types of expressions."""
    ARITHMETIC = "arithmetic"
    LOGICAL = "logical"
    AGGREGATION = "aggregation"
    STRING = "string"
    DATE = "date"
    CONDITIONAL = "conditional"


@dataclass
class Expression:
    """
    Parsed expression object.

    Attributes:
        raw: Raw expression string
        type: Expression type
        fields: Referenced field names
        functions: Function calls in expression
        is_aggregation: Whether expression contains aggregation
    """
    raw: str
    type: ExpressionType
    fields: List[str]
    functions: List[str]
    is_aggregation: bool = False


class ExpressionParser:
    """
    Parse and validate Tableau-style expressions.

    Supports:
    - Arithmetic: +, -, *, /, %, **
    - Logical: AND, OR, NOT, ==, !=, <, >, <=, >=
    - Aggregations: SUM, AVG, COUNT, MIN, MAX, MEDIAN
    - String: CONCAT, UPPER, LOWER, LEFT, RIGHT, LEN
    - Date: YEAR, MONTH, DAY, DATEADD, DATEDIFF
    - Conditional: IF, CASE, IFNULL

    Example:
        >>> parser = ExpressionParser()
        >>> expr = parser.parse("([Profit] / [Revenue]) * 100")
        >>> print(expr.fields)  # ['Profit', 'Revenue']
    """

    # Supported functions
    AGGREGATION_FUNCTIONS = ['SUM', 'AVG', 'COUNT', 'MIN', 'MAX', 'MEDIAN', 'STDEV', 'VAR']
    STRING_FUNCTIONS = ['CONCAT', 'UPPER', 'LOWER', 'LEFT', 'RIGHT', 'LEN', 'TRIM', 'CONTAINS']
    DATE_FUNCTIONS = ['YEAR', 'MONTH', 'DAY', 'DATEADD', 'DATEDIFF', 'TODAY', 'NOW']
    LOGICAL_FUNCTIONS = ['IF', 'CASE', 'IFNULL', 'ISNULL']

    # Operators
    ARITHMETIC_OPS = ['+', '-', '*', '/', '%', '**']
    COMPARISON_OPS = ['==', '!=', '<', '>', '<=', '>=']
    LOGICAL_OPS = ['AND', 'OR', 'NOT']

    def __init__(self):
        """Initialize expression parser."""
        self.all_functions = (
            self.AGGREGATION_FUNCTIONS +
            self.STRING_FUNCTIONS +
            self.DATE_FUNCTIONS +
            self.LOGICAL_FUNCTIONS
        )

    def parse(self, expression: str) -> Expression:
        """
        Parse expression string.

        Args:
            expression: Expression string (e.g., "([Profit] / [Revenue]) * 100")

        Returns:
            Parsed Expression object

        Example:
            >>> expr = parser.parse("SUM([Sales]) / COUNT([Orders])")
            >>> print(expr.is_aggregation)  # True
        """
        # Extract field references [FieldName]
        fields = self._extract_fields(expression)

        # Extract function calls
        functions = self._extract_functions(expression)

        # Determine type
        expr_type = self._determine_type(expression, functions)

        # Check if aggregation
        is_agg = any(func in self.AGGREGATION_FUNCTIONS for func in functions)

        return Expression(
            raw=expression,
            type=expr_type,
            fields=fields,
            functions=functions,
            is_aggregation=is_agg
        )

    def _extract_fields(self, expression: str) -> List[str]:
        """Extract field names from expression."""
        # Pattern: [FieldName]
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, expression)
        return list(set(matches))  # Remove duplicates

    def _extract_functions(self, expression: str) -> List[str]:
        """Extract function names from expression."""
        functions = []

        for func in self.all_functions:
            # Pattern: FUNCTION(...)
            if re.search(rf'\b{func}\s*\(', expression, re.IGNORECASE):
                functions.append(func)

        return functions

    def _determine_type(self, expression: str, functions: List[str]) -> ExpressionType:
        """Determine expression type."""
        # Aggregation if has aggregation function
        if any(func in self.AGGREGATION_FUNCTIONS for func in functions):
            return ExpressionType.AGGREGATION

        # Conditional if has IF/CASE
        if any(func in self.LOGICAL_FUNCTIONS for func in functions):
            return ExpressionType.CONDITIONAL

        # String if has string functions
        if any(func in self.STRING_FUNCTIONS for func in functions):
            return ExpressionType.STRING

        # Date if has date functions
        if any(func in self.DATE_FUNCTIONS for func in functions):
            return ExpressionType.DATE

        # Logical if has logical operators
        if any(op in expression for op in self.LOGICAL_OPS):
            return ExpressionType.LOGICAL

        # Default: arithmetic
        return ExpressionType.ARITHMETIC

    def validate(self, expression: str) -> tuple[bool, Optional[str]]:
        """
        Validate expression syntax.

        Args:
            expression: Expression string

        Returns:
            (is_valid, error_message)

        Example:
            >>> valid, error = parser.validate("SUM([Sales])")
            >>> print(valid)  # True
        """
        try:
            # Check balanced brackets
            if expression.count('[') != expression.count(']'):
                return False, "Unbalanced field brackets []"

            if expression.count('(') != expression.count(')'):
                return False, "Unbalanced parentheses ()"

            # Check for empty fields
            if '[]' in expression:
                return False, "Empty field name []"

            # Basic syntax check
            if not expression.strip():
                return False, "Empty expression"

            return True, None

        except Exception as e:
            return False, str(e)


class CalculatedField:
    """
    Tableau-style calculated field.

    Allows creating derived columns using expressions.

    Example:
        >>> # Simple arithmetic
        >>> profit_margin = CalculatedField(
        ...     'Profit Margin',
        ...     '([Profit] / [Revenue]) * 100'
        ... )
        >>> df['profit_margin'] = profit_margin.apply(df)
        >>>
        >>> # With aggregation
        >>> avg_order_value = CalculatedField(
        ...     'Avg Order Value',
        ...     'SUM([Revenue]) / COUNT([Orders])'
        ... )
    """

    def __init__(
        self,
        name: str,
        expression: str,
        description: str = "",
        data_type: Optional[str] = None
    ):
        """
        Initialize calculated field.

        Args:
            name: Field name
            expression: Tableau-style expression
            description: Field description
            data_type: Expected data type ('number', 'string', 'date', 'boolean')

        Example:
            >>> field = CalculatedField(
            ...     'Growth Rate',
            ...     '([This Year] - [Last Year]) / [Last Year] * 100',
            ...     'Year-over-year growth percentage'
            ... )
        """
        self.name = name
        self.expression_str = expression
        self.description = description
        self.data_type = data_type

        # Parse expression
        self.parser = ExpressionParser()
        self.expression = self.parser.parse(expression)

        # Validate
        is_valid, error = self.parser.validate(expression)
        if not is_valid:
            raise ValueError(f"Invalid expression: {error}")

    def apply(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply calculated field to DataFrame.

        Args:
            data: Input DataFrame

        Returns:
            Series with calculated values

        Example:
            >>> profit_margin = CalculatedField(
            ...     'Margin',
            ...     '([Profit] / [Revenue]) * 100'
            ... )
            >>> df['margin'] = profit_margin.apply(df)
        """
        # Build safe evaluation context
        context = self._build_context(data)

        # Replace field references with actual column names
        eval_expr = self._prepare_expression(data)

        try:
            # Evaluate expression
            result = eval(eval_expr, {"__builtins__": {}}, context)

            # Convert to Series if needed
            if not isinstance(result, pd.Series):
                result = pd.Series([result] * len(data), index=data.index)

            return result

        except Exception as e:
            raise RuntimeError(f"Error evaluating expression '{self.expression_str}': {e}")

    def _build_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Build evaluation context with safe functions."""
        context = {}

        # Add dataframe columns
        for col in data.columns:
            context[col] = data[col]

        # Add aggregation functions
        context.update({
            'SUM': lambda x: x.sum(),
            'AVG': lambda x: x.mean(),
            'COUNT': lambda x: len(x),
            'MIN': lambda x: x.min(),
            'MAX': lambda x: x.max(),
            'MEDIAN': lambda x: x.median(),
            'STDEV': lambda x: x.std(),
            'VAR': lambda x: x.var(),
        })

        # Add string functions
        context.update({
            'CONCAT': lambda *args: ''.join(str(a) for a in args),
            'UPPER': lambda x: x.str.upper() if isinstance(x, pd.Series) else str(x).upper(),
            'LOWER': lambda x: x.str.lower() if isinstance(x, pd.Series) else str(x).lower(),
            'LEN': lambda x: x.str.len() if isinstance(x, pd.Series) else len(str(x)),
            'TRIM': lambda x: x.str.strip() if isinstance(x, pd.Series) else str(x).strip(),
        })

        # Add date functions
        context.update({
            'YEAR': lambda x: x.dt.year if isinstance(x, pd.Series) else x.year,
            'MONTH': lambda x: x.dt.month if isinstance(x, pd.Series) else x.month,
            'DAY': lambda x: x.dt.day if isinstance(x, pd.Series) else x.day,
        })

        # Add logical functions
        context.update({
            'IF': lambda cond, true_val, false_val: np.where(cond, true_val, false_val),
            'IFNULL': lambda x, default: x.fillna(default) if isinstance(x, pd.Series) else (default if pd.isna(x) else x),
            'ISNULL': lambda x: x.isna() if isinstance(x, pd.Series) else pd.isna(x),
        })

        # Add math functions
        context.update({
            'ABS': lambda x: abs(x),
            'SQRT': lambda x: np.sqrt(x),
            'ROUND': lambda x, decimals=0: np.round(x, decimals),
            'FLOOR': lambda x: np.floor(x),
            'CEIL': lambda x: np.ceil(x),
        })

        return context

    def _prepare_expression(self, data: pd.DataFrame) -> str:
        """Prepare expression for evaluation."""
        expr = self.expression_str

        # Replace [FieldName] with FieldName (without brackets)
        # But only if field exists in data
        for field in self.expression.fields:
            if field in data.columns:
                # Replace [Field] with Field
                expr = expr.replace(f'[{field}]', field)
            else:
                raise ValueError(f"Field '{field}' not found in data")

        return expr

    def get_dependencies(self) -> List[str]:
        """
        Get list of fields this calculated field depends on.

        Returns:
            List of field names

        Example:
            >>> field = CalculatedField('Margin', '([Profit] / [Revenue]) * 100')
            >>> print(field.get_dependencies())  # ['Profit', 'Revenue']
        """
        return self.expression.fields

    def is_aggregation(self) -> bool:
        """Check if this is an aggregation field."""
        return self.expression.is_aggregation

    def __repr__(self) -> str:
        """String representation."""
        return f"CalculatedField('{self.name}', '{self.expression_str}')"


class CalculatedFieldManager:
    """
    Manage multiple calculated fields.

    Handles dependencies and evaluation order.

    Example:
        >>> manager = CalculatedFieldManager()
        >>> manager.add_field(CalculatedField('Margin', '[Profit] / [Revenue]'))
        >>> manager.add_field(CalculatedField('Margin %', '[Margin] * 100'))
        >>> df = manager.apply_all(df)
    """

    def __init__(self):
        """Initialize manager."""
        self.fields: Dict[str, CalculatedField] = {}

    def add_field(self, field: CalculatedField) -> 'CalculatedFieldManager':
        """
        Add calculated field.

        Args:
            field: CalculatedField instance

        Returns:
            Self for method chaining
        """
        self.fields[field.name] = field
        return self

    def remove_field(self, name: str):
        """Remove calculated field."""
        if name in self.fields:
            del self.fields[name]

    def get_field(self, name: str) -> Optional[CalculatedField]:
        """Get calculated field by name."""
        return self.fields.get(name)

    def apply_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all calculated fields to DataFrame.

        Respects dependency order.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with calculated fields added

        Example:
            >>> manager.add_field(CalculatedField('Profit', '[Revenue] - [Cost]'))
            >>> manager.add_field(CalculatedField('Margin', '[Profit] / [Revenue]'))
            >>> df = manager.apply_all(df)
        """
        result = data.copy()

        # Sort fields by dependencies
        ordered_fields = self._topological_sort()

        # Apply each field
        for field_name in ordered_fields:
            field = self.fields[field_name]
            result[field_name] = field.apply(result)

        return result

    def _topological_sort(self) -> List[str]:
        """
        Sort fields by dependency order.

        Returns:
            List of field names in evaluation order
        """
        # Simple topological sort
        visited = set()
        order = []

        def visit(field_name: str):
            if field_name in visited:
                return

            field = self.fields[field_name]

            # Visit dependencies first
            for dep in field.get_dependencies():
                if dep in self.fields:  # Dependency is also a calculated field
                    visit(dep)

            visited.add(field_name)
            order.append(field_name)

        # Visit all fields
        for field_name in self.fields:
            visit(field_name)

        return order

    def get_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all calculated fields.

        Returns:
            List of field summaries
        """
        return [
            {
                'name': field.name,
                'expression': field.expression_str,
                'type': field.expression.type.value,
                'dependencies': field.get_dependencies(),
                'is_aggregation': field.is_aggregation()
            }
            for field in self.fields.values()
        ]
