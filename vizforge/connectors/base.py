"""
Base connector interface for all data sources.

All connectors inherit from BaseConnector and implement standard methods.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import pandas as pd


class DataSourceType(Enum):
    """Type of data source."""
    DATABASE = "database"
    CLOUD_STORAGE = "cloud_storage"
    API = "api"
    FILE = "file"
    WEB = "web"
    STREAMING = "streaming"
    DATA_WAREHOUSE = "data_warehouse"


@dataclass
class ConnectionConfig:
    """Configuration for a data source connection."""
    source_type: DataSourceType
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    bucket: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None
    options: Dict[str, Any] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


@dataclass
class DataSource:
    """Represents a connected data source."""
    name: str
    type: DataSourceType
    connector: 'BaseConnector'
    config: ConnectionConfig
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute query on data source."""
        return self.connector.query(query, **kwargs)

    def read(self, **kwargs) -> pd.DataFrame:
        """Read data from source."""
        return self.connector.read(**kwargs)

    def write(self, data: pd.DataFrame, **kwargs) -> bool:
        """Write data to source."""
        return self.connector.write(data, **kwargs)

    def close(self):
        """Close connection."""
        self.connector.close()


class BaseConnector(ABC):
    """Base class for all data connectors."""

    def __init__(self, config: ConnectionConfig):
        """
        Initialize connector.

        Args:
            config: Connection configuration
        """
        self.config = config
        self.connection = None
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to data source.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from data source.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if connection is valid.

        Returns:
            True if connection works, False otherwise
        """
        pass

    @abstractmethod
    def read(self, **kwargs) -> pd.DataFrame:
        """
        Read data from source.

        Returns:
            DataFrame with data
        """
        pass

    def write(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Write data to source (optional, not all sources support writing).

        Args:
            data: DataFrame to write

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support writing")

    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Execute query on data source (optional, only for query-capable sources).

        Args:
            query: Query string

        Returns:
            DataFrame with query results
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support queries")

    def list_tables(self) -> List[str]:
        """
        List available tables/collections (optional, only for structured sources).

        Returns:
            List of table names
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support listing tables")

    def get_schema(self, table: str = None) -> Dict[str, Any]:
        """
        Get schema information (optional).

        Args:
            table: Table name (if applicable)

        Returns:
            Schema information
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support schema retrieval")

    def close(self):
        """Close connection (alias for disconnect)."""
        return self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}(connected={self.is_connected})"


class CachedConnector(BaseConnector):
    """Base class for connectors with caching support."""

    def __init__(self, config: ConnectionConfig, cache_size: int = 100):
        """
        Initialize cached connector.

        Args:
            config: Connection configuration
            cache_size: Maximum cache size (number of queries)
        """
        super().__init__(config)
        self.cache_size = cache_size
        self._cache = {}

    def _get_cache_key(self, query: str, **kwargs) -> str:
        """Generate cache key."""
        return f"{query}:{str(kwargs)}"

    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached result."""
        return self._cache.get(key)

    def _set_cached(self, key: str, data: pd.DataFrame):
        """Set cached result."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = data.copy()

    def _clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
