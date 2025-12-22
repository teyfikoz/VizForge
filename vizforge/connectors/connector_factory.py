"""
Connector factory for easy connector creation.

Unified interface for all data sources.
"""

from typing import Union, Dict, Any
from .base import ConnectionConfig, DataSource, DataSourceType
from .database import PostgreSQLConnector, MySQLConnector, SQLiteConnector, MongoDBConnector
from .cloud import S3Connector, GCSConnector, AzureBlobConnector
from .api import RESTConnector, GraphQLConnector
from .file import ExcelConnector, ParquetConnector, HDF5Connector
from .web import HTMLTableConnector, WebScraperConnector


# Connector registry
CONNECTORS = {
    # Databases
    'postgresql': PostgreSQLConnector,
    'postgres': PostgreSQLConnector,
    'mysql': MySQLConnector,
    'sqlite': SQLiteConnector,
    'mongodb': MongoDBConnector,
    'mongo': MongoDBConnector,

    # Cloud Storage
    's3': S3Connector,
    'aws': S3Connector,
    'gcs': GCSConnector,
    'google_cloud': GCSConnector,
    'azure': AzureBlobConnector,
    'azure_blob': AzureBlobConnector,

    # APIs
    'rest': RESTConnector,
    'api': RESTConnector,
    'graphql': GraphQLConnector,

    # Files
    'excel': ExcelConnector,
    'xlsx': ExcelConnector,
    'parquet': ParquetConnector,
    'hdf5': HDF5Connector,
    'h5': HDF5Connector,

    # Web
    'html': HTMLTableConnector,
    'web': WebScraperConnector,
}


def connect(
    source_type: str,
    name: str = None,
    **kwargs
) -> DataSource:
    """
    Connect to a data source.

    Args:
        source_type: Type of data source (e.g., 'postgresql', 's3', 'rest')
        name: Optional name for the data source
        **kwargs: Connection parameters

    Returns:
        Connected DataSource object

    Examples:
        >>> # PostgreSQL
        >>> db = vz.connect('postgresql',
        ...     host='localhost',
        ...     database='mydb',
        ...     username='user',
        ...     password='pass'
        ... )
        >>> df = db.query("SELECT * FROM users")

        >>> # AWS S3
        >>> s3 = vz.connect('s3',
        ...     bucket='my-bucket',
        ...     username='AWS_KEY',
        ...     password='AWS_SECRET'
        ... )
        >>> df = s3.read('data/file.csv', file_type='csv')

        >>> # REST API
        >>> api = vz.connect('rest',
        ...     url='https://api.example.com',
        ...     api_key='YOUR_KEY'
        ... )
        >>> df = api.read('/users')

        >>> # MongoDB
        >>> mongo = vz.connect('mongodb',
        ...     host='localhost',
        ...     database='mydb',
        ...     username='user',
        ...     password='pass'
        ... )
        >>> df = mongo.read('users', {'age': {'$gt': 25}})
    """
    # Get connector class
    connector_class = CONNECTORS.get(source_type.lower())
    if not connector_class:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            f"Supported types: {', '.join(sorted(CONNECTORS.keys()))}"
        )

    # Determine source type enum
    if source_type in ['postgresql', 'postgres', 'mysql', 'sqlite', 'mongodb', 'mongo']:
        ds_type = DataSourceType.DATABASE
    elif source_type in ['s3', 'aws', 'gcs', 'google_cloud', 'azure', 'azure_blob']:
        ds_type = DataSourceType.CLOUD_STORAGE
    elif source_type in ['rest', 'api', 'graphql']:
        ds_type = DataSourceType.API
    elif source_type in ['excel', 'xlsx', 'parquet', 'hdf5', 'h5']:
        ds_type = DataSourceType.FILE
    elif source_type in ['html', 'web']:
        ds_type = DataSourceType.WEB
    else:
        ds_type = DataSourceType.DATABASE

    # Create configuration
    config = ConnectionConfig(
        source_type=ds_type,
        **kwargs
    )

    # Create connector
    connector = connector_class(config)

    # Connect
    connector.connect()

    # Create data source
    data_source = DataSource(
        name=name or f"{source_type}_source",
        type=ds_type,
        connector=connector,
        config=config
    )

    return data_source


def list_connectors() -> Dict[str, str]:
    """
    List all available connectors.

    Returns:
        Dictionary of connector types and their classes
    """
    return {
        'Databases': ['postgresql', 'mysql', 'sqlite', 'mongodb'],
        'Cloud Storage': ['s3', 'gcs', 'azure'],
        'APIs': ['rest', 'graphql'],
        'Files': ['excel', 'parquet', 'hdf5'],
        'Web': ['html', 'web'],
    }


# Convenience functions for common sources

def connect_postgres(**kwargs) -> DataSource:
    """Connect to PostgreSQL database."""
    return connect('postgresql', **kwargs)


def connect_mysql(**kwargs) -> DataSource:
    """Connect to MySQL database."""
    return connect('mysql', **kwargs)


def connect_sqlite(path: str, **kwargs) -> DataSource:
    """Connect to SQLite database."""
    return connect('sqlite', path=path, **kwargs)


def connect_mongodb(**kwargs) -> DataSource:
    """Connect to MongoDB."""
    return connect('mongodb', **kwargs)


def connect_s3(bucket: str, **kwargs) -> DataSource:
    """Connect to AWS S3."""
    return connect('s3', bucket=bucket, **kwargs)


def connect_gcs(bucket: str, **kwargs) -> DataSource:
    """Connect to Google Cloud Storage."""
    return connect('gcs', bucket=bucket, **kwargs)


def connect_azure(container: str, **kwargs) -> DataSource:
    """Connect to Azure Blob Storage."""
    return connect('azure', bucket=container, **kwargs)


def connect_rest(url: str, **kwargs) -> DataSource:
    """Connect to REST API."""
    return connect('rest', url=url, **kwargs)


def connect_graphql(url: str, **kwargs) -> DataSource:
    """Connect to GraphQL API."""
    return connect('graphql', url=url, **kwargs)
