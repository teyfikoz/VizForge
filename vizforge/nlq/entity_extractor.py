"""
VizForge Entity Extractor

Extracts entities (columns, values, time periods) from queries.
Works with DataFrame to find best column matches.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class Entity:
    """Extracted entity."""
    type: str  # 'column', 'value', 'time_period', 'number'
    value: Any
    confidence: float
    matched_text: str


class EntityExtractor:
    """
    Entity extraction from natural language queries.

    Matches query terms to DataFrame columns using fuzzy matching.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize extractor with DataFrame.

        Args:
            dataframe: DataFrame to extract entities from
        """
        self.df = dataframe
        self.columns = list(dataframe.columns)
        self.column_types = self._analyze_column_types()

    def _analyze_column_types(self) -> Dict[str, str]:
        """Analyze column data types."""
        types = {}
        for col in self.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                types[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                types[col] = 'datetime'
            elif pd.api.types.is_categorical_dtype(self.df[col]) or \
                 pd.api.types.is_object_dtype(self.df[col]):
                types[col] = 'categorical'
            else:
                types[col] = 'unknown'
        return types

    def extract_entities(self, query: str) -> List[Entity]:
        """
        Extract all entities from query.

        Args:
            query: Natural language query

        Returns:
            List of extracted entities
        """
        entities = []

        # Extract columns
        entities.extend(self._extract_columns(query))

        # Extract time periods
        entities.extend(self._extract_time_periods(query))

        # Extract numbers
        entities.extend(self._extract_numbers(query))

        # Extract categorical values
        entities.extend(self._extract_values(query))

        return entities

    def _extract_columns(self, query: str) -> List[Entity]:
        """Extract column references from query."""
        entities = []
        query_lower = query.lower()

        for col in self.columns:
            col_lower = col.lower()

            # Exact match
            if col_lower in query_lower:
                entities.append(Entity(
                    type='column',
                    value=col,
                    confidence=1.0,
                    matched_text=col
                ))
                continue

            # Fuzzy match (substring)
            if col_lower in query_lower or query_lower in col_lower:
                confidence = len(col_lower) / max(len(query_lower), len(col_lower))
                if confidence > 0.3:
                    entities.append(Entity(
                        type='column',
                        value=col,
                        confidence=confidence,
                        matched_text=col
                    ))
                    continue

            # Word-by-word match
            col_words = set(re.findall(r'\w+', col_lower))
            query_words = set(re.findall(r'\w+', query_lower))
            common_words = col_words & query_words

            if common_words and len(common_words) / len(col_words) > 0.5:
                confidence = len(common_words) / len(col_words)
                entities.append(Entity(
                    type='column',
                    value=col,
                    confidence=confidence,
                    matched_text=' '.join(common_words)
                ))

        return entities

    def _extract_time_periods(self, query: str) -> List[Entity]:
        """Extract time periods from query."""
        entities = []
        query_lower = query.lower()

        # Relative time periods
        time_patterns = {
            r'\b(last|past)\s+(\d+)\s+(day|week|month|year)s?\b': 'relative',
            r'\b(this|current)\s+(day|week|month|quarter|year)\b': 'current',
            r'\b(next|upcoming)\s+(\d+)\s+(day|week|month|year)s?\b': 'future',
            r'\b(yesterday|today|tomorrow)\b': 'specific',
            r'\b(\d{4})\b': 'year',  # 2024
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b': 'month',
        }

        for pattern, period_type in time_patterns.items():
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    type='time_period',
                    value={'type': period_type, 'text': match.group(0)},
                    confidence=0.9,
                    matched_text=match.group(0)
                ))

        return entities

    def _extract_numbers(self, query: str) -> List[Entity]:
        """Extract numeric values from query."""
        entities = []

        # Extract all numbers
        number_pattern = r'\b(\d+(?:\.\d+)?)\b'
        matches = re.finditer(number_pattern, query)

        for match in matches:
            value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
            entities.append(Entity(
                type='number',
                value=value,
                confidence=1.0,
                matched_text=match.group(1)
            ))

        return entities

    def _extract_values(self, query: str) -> List[Entity]:
        """Extract categorical values from query."""
        entities = []

        # Check for values in categorical columns
        for col in self.columns:
            if self.column_types.get(col) == 'categorical':
                unique_values = self.df[col].unique()

                for val in unique_values:
                    if pd.isna(val):
                        continue

                    val_str = str(val).lower()
                    if val_str in query.lower():
                        entities.append(Entity(
                            type='value',
                            value={'column': col, 'value': val},
                            confidence=0.9,
                            matched_text=val_str
                        ))

        return entities

    def get_numeric_columns(self) -> List[str]:
        """Get all numeric columns."""
        return [col for col, dtype in self.column_types.items() if dtype == 'numeric']

    def get_categorical_columns(self) -> List[str]:
        """Get all categorical columns."""
        return [col for col, dtype in self.column_types.items() if dtype == 'categorical']

    def get_datetime_columns(self) -> List[str]:
        """Get all datetime columns."""
        return [col for col, dtype in self.column_types.items() if dtype == 'datetime']

    def suggest_columns_for_intent(self, intent: str) -> Dict[str, List[str]]:
        """
        Suggest appropriate columns for given intent.

        Args:
            intent: Query intent (trend, comparison, etc.)

        Returns:
            Dictionary with suggested columns for each role
        """
        suggestions = {}

        if intent == 'trend':
            suggestions['x'] = self.get_datetime_columns()
            suggestions['y'] = self.get_numeric_columns()

        elif intent == 'comparison':
            suggestions['x'] = self.get_categorical_columns()
            suggestions['y'] = self.get_numeric_columns()

        elif intent == 'distribution':
            suggestions['values'] = self.get_numeric_columns()

        elif intent == 'correlation':
            numeric_cols = self.get_numeric_columns()
            if len(numeric_cols) >= 2:
                suggestions['x'] = [numeric_cols[0]]
                suggestions['y'] = [numeric_cols[1]]

        elif intent == 'breakdown':
            suggestions['names'] = self.get_categorical_columns()
            suggestions['values'] = self.get_numeric_columns()

        return suggestions
