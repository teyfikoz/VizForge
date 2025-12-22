"""API connectors for REST and GraphQL."""

import pandas as pd
import requests
from typing import Dict, Any, List
from .base import BaseConnector, ConnectionConfig


class RESTConnector(BaseConnector):
    """REST API connector."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.session = None

    def connect(self) -> bool:
        """Initialize REST session."""
        self.session = requests.Session()

        # Add auth if provided
        if self.config.api_key:
            self.session.headers['Authorization'] = f"Bearer {self.config.api_key}"
        elif self.config.username and self.config.password:
            self.session.auth = (self.config.username, self.config.password)

        # Add custom headers
        if 'headers' in self.config.options:
            self.session.headers.update(self.config.options['headers'])

        self._connected = True
        return True

    def disconnect(self) -> bool:
        """Close REST session."""
        if self.session:
            self.session.close()
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            if self.config.url:
                response = self.session.head(self.config.url, timeout=5)
                return response.status_code < 400
            return True
        except:
            return False

    def read(self, endpoint: str = '', params: Dict = None, **kwargs) -> pd.DataFrame:
        """Fetch data from REST API."""
        if not self.is_connected:
            self.connect()

        url = self.config.url + endpoint if self.config.url else endpoint
        response = self.session.get(url, params=params, **kwargs)
        response.raise_for_status()

        data = response.json()

        # Handle different response formats
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check for common data keys
            for key in ['data', 'results', 'items', 'records']:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            # Single record
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported response format: {type(data)}")


class GraphQLConnector(BaseConnector):
    """GraphQL API connector."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.session = None

    def connect(self) -> bool:
        """Initialize GraphQL session."""
        self.session = requests.Session()

        if self.config.api_key:
            self.session.headers['Authorization'] = f"Bearer {self.config.api_key}"

        if 'headers' in self.config.options:
            self.session.headers.update(self.config.options['headers'])

        self._connected = True
        return True

    def disconnect(self) -> bool:
        """Close GraphQL session."""
        if self.session:
            self.session.close()
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test GraphQL connection."""
        try:
            response = self.session.post(
                self.config.url,
                json={'query': '{ __typename }'},
                timeout=5
            )
            return response.status_code < 400
        except:
            return False

    def read(self, query: str, variables: Dict = None, **kwargs) -> pd.DataFrame:
        """Execute GraphQL query."""
        if not self.is_connected:
            self.connect()

        payload = {'query': query}
        if variables:
            payload['variables'] = variables

        response = self.session.post(self.config.url, json=payload, **kwargs)
        response.raise_for_status()

        data = response.json()

        if 'errors' in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        # Extract data
        result_data = data.get('data', {})

        # Find first list in response
        for value in result_data.values():
            if isinstance(value, list):
                return pd.DataFrame(value)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, list):
                        return pd.DataFrame(v)

        return pd.DataFrame([result_data])
