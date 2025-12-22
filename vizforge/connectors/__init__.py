"""
VizForge Universal Data Connectors

Connect to 20+ data sources with a unified interface.
No configuration headaches - just works!

Supported Sources:
- Databases: PostgreSQL, MySQL, SQLite, MongoDB
- Cloud Storage: AWS S3, Google Cloud, Azure
- APIs: REST, GraphQL
- Files: CSV, Excel, Parquet, JSON, HDF5
- Web: HTML tables, web scraping
- Streaming: Kafka, real-time feeds
"""

from .base import BaseConnector, DataSource
from .database import (
    PostgreSQLConnector,
    MySQLConnector,
    SQLiteConnector,
    MongoDBConnector,
)
from .cloud import (
    S3Connector,
    GCSConnector,
    AzureBlobConnector,
)
from .api import (
    RESTConnector,
    GraphQLConnector,
)
from .file import (
    ExcelConnector,
    ParquetConnector,
    HDF5Connector,
)
from .web import (
    HTMLTableConnector,
    WebScraperConnector,
)
from .connector_factory import connect, list_connectors

__all__ = [
    # Main entry points
    'connect',
    'list_connectors',

    # Base classes
    'BaseConnector',
    'DataSource',

    # Database connectors
    'PostgreSQLConnector',
    'MySQLConnector',
    'SQLiteConnector',
    'MongoDBConnector',

    # Cloud connectors
    'S3Connector',
    'GCSConnector',
    'AzureBlobConnector',

    # API connectors
    'RESTConnector',
    'GraphQLConnector',

    # File connectors
    'ExcelConnector',
    'ParquetConnector',
    'HDF5Connector',

    # Web connectors
    'HTMLTableConnector',
    'WebScraperConnector',
]
