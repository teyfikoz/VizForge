"""
Database connectors for SQL and NoSQL databases.

Supports: PostgreSQL, MySQL, SQLite, MongoDB, Redis
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from .base import BaseConnector, CachedConnector, ConnectionConfig, DataSourceType


class PostgreSQLConnector(CachedConnector):
    """PostgreSQL database connector."""

    def __init__(self, config: ConnectionConfig):
        """Initialize PostgreSQL connector."""
        super().__init__(config)
        self.engine = None

    def connect(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            from sqlalchemy import create_engine

            connection_string = (
                f"postgresql://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port or 5432}/{self.config.database}"
            )

            self.engine = create_engine(connection_string, **self.config.options)
            self.connection = self.engine.connect()
            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "PostgreSQL connector requires 'psycopg2' and 'sqlalchemy'. "
                "Install with: pip install psycopg2-binary sqlalchemy"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def disconnect(self) -> bool:
        """Disconnect from PostgreSQL."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            self.query("SELECT 1")
            return True
        except:
            return False

    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute SQL query."""
        if not self.is_connected:
            self.connect()

        # Check cache
        cache_key = self._get_cache_key(query, **kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Execute query
        df = pd.read_sql(query, self.engine, **kwargs)

        # Cache result
        self._set_cached(cache_key, df)

        return df

    def read(self, table: str = None, query: str = None, **kwargs) -> pd.DataFrame:
        """Read data from table or execute query."""
        if query:
            return self.query(query, **kwargs)
        elif table:
            return self.query(f"SELECT * FROM {table}", **kwargs)
        else:
            raise ValueError("Either 'table' or 'query' must be provided")

    def write(self, data: pd.DataFrame, table: str, if_exists: str = 'append', **kwargs) -> bool:
        """Write DataFrame to PostgreSQL table."""
        if not self.is_connected:
            self.connect()

        try:
            data.to_sql(table, self.engine, if_exists=if_exists, index=False, **kwargs)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to write to PostgreSQL: {e}")

    def list_tables(self) -> List[str]:
        """List all tables in database."""
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """
        df = self.query(query)
        return df['table_name'].tolist()

    def get_schema(self, table: str) -> Dict[str, Any]:
        """Get table schema."""
        query = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table}'
        """
        df = self.query(query)
        return df.to_dict('records')


class MySQLConnector(CachedConnector):
    """MySQL database connector."""

    def __init__(self, config: ConnectionConfig):
        """Initialize MySQL connector."""
        super().__init__(config)
        self.engine = None

    def connect(self) -> bool:
        """Connect to MySQL database."""
        try:
            from sqlalchemy import create_engine

            connection_string = (
                f"mysql+pymysql://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port or 3306}/{self.config.database}"
            )

            self.engine = create_engine(connection_string, **self.config.options)
            self.connection = self.engine.connect()
            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "MySQL connector requires 'pymysql' and 'sqlalchemy'. "
                "Install with: pip install pymysql sqlalchemy"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to MySQL: {e}")

    def disconnect(self) -> bool:
        """Disconnect from MySQL."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test MySQL connection."""
        try:
            self.query("SELECT 1")
            return True
        except:
            return False

    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute SQL query."""
        if not self.is_connected:
            self.connect()

        cache_key = self._get_cache_key(query, **kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        df = pd.read_sql(query, self.engine, **kwargs)
        self._set_cached(cache_key, df)
        return df

    def read(self, table: str = None, query: str = None, **kwargs) -> pd.DataFrame:
        """Read data from table or execute query."""
        if query:
            return self.query(query, **kwargs)
        elif table:
            return self.query(f"SELECT * FROM {table}", **kwargs)
        else:
            raise ValueError("Either 'table' or 'query' must be provided")

    def write(self, data: pd.DataFrame, table: str, if_exists: str = 'append', **kwargs) -> bool:
        """Write DataFrame to MySQL table."""
        if not self.is_connected:
            self.connect()

        try:
            data.to_sql(table, self.engine, if_exists=if_exists, index=False, **kwargs)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to write to MySQL: {e}")

    def list_tables(self) -> List[str]:
        """List all tables in database."""
        query = "SHOW TABLES"
        df = self.query(query)
        return df.iloc[:, 0].tolist()

    def get_schema(self, table: str) -> Dict[str, Any]:
        """Get table schema."""
        query = f"DESCRIBE {table}"
        df = self.query(query)
        return df.to_dict('records')


class SQLiteConnector(CachedConnector):
    """SQLite database connector."""

    def __init__(self, config: ConnectionConfig):
        """Initialize SQLite connector."""
        super().__init__(config)
        self.connection = None

    def connect(self) -> bool:
        """Connect to SQLite database."""
        try:
            import sqlite3

            db_path = self.config.path or self.config.database
            if not db_path:
                raise ValueError("SQLite requires 'path' or 'database' in config")

            self.connection = sqlite3.connect(db_path, **self.config.options)
            self._connected = True
            return True

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to SQLite: {e}")

    def disconnect(self) -> bool:
        """Disconnect from SQLite."""
        if self.connection:
            self.connection.close()
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test SQLite connection."""
        try:
            self.query("SELECT 1")
            return True
        except:
            return False

    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute SQL query."""
        if not self.is_connected:
            self.connect()

        cache_key = self._get_cache_key(query, **kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        df = pd.read_sql(query, self.connection, **kwargs)
        self._set_cached(cache_key, df)
        return df

    def read(self, table: str = None, query: str = None, **kwargs) -> pd.DataFrame:
        """Read data from table or execute query."""
        if query:
            return self.query(query, **kwargs)
        elif table:
            return self.query(f"SELECT * FROM {table}", **kwargs)
        else:
            raise ValueError("Either 'table' or 'query' must be provided")

    def write(self, data: pd.DataFrame, table: str, if_exists: str = 'append', **kwargs) -> bool:
        """Write DataFrame to SQLite table."""
        if not self.is_connected:
            self.connect()

        try:
            data.to_sql(table, self.connection, if_exists=if_exists, index=False, **kwargs)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to write to SQLite: {e}")

    def list_tables(self) -> List[str]:
        """List all tables in database."""
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        df = self.query(query)
        return df['name'].tolist()

    def get_schema(self, table: str) -> Dict[str, Any]:
        """Get table schema."""
        query = f"PRAGMA table_info({table})"
        df = self.query(query)
        return df.to_dict('records')


class MongoDBConnector(BaseConnector):
    """MongoDB database connector."""

    def __init__(self, config: ConnectionConfig):
        """Initialize MongoDB connector."""
        super().__init__(config)
        self.client = None
        self.db = None

    def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient

            if self.config.url:
                connection_string = self.config.url
            else:
                connection_string = f"mongodb://"
                if self.config.username and self.config.password:
                    connection_string += f"{self.config.username}:{self.config.password}@"
                connection_string += f"{self.config.host}:{self.config.port or 27017}"

            self.client = MongoClient(connection_string, **self.config.options)
            self.db = self.client[self.config.database]
            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "MongoDB connector requires 'pymongo'. "
                "Install with: pip install pymongo"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def disconnect(self) -> bool:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            self.client.server_info()
            return True
        except:
            return False

    def read(self, collection: str, query: Dict = None, limit: int = None, **kwargs) -> pd.DataFrame:
        """Read data from MongoDB collection."""
        if not self.is_connected:
            self.connect()

        query = query or {}
        cursor = self.db[collection].find(query, **kwargs)

        if limit:
            cursor = cursor.limit(limit)

        data = list(cursor)
        df = pd.DataFrame(data)

        # Remove MongoDB's _id if present
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)

        return df

    def write(self, data: pd.DataFrame, collection: str, **kwargs) -> bool:
        """Write DataFrame to MongoDB collection."""
        if not self.is_connected:
            self.connect()

        try:
            records = data.to_dict('records')
            self.db[collection].insert_many(records, **kwargs)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to write to MongoDB: {e}")

    def query(self, collection: str, pipeline: List[Dict] = None, **kwargs) -> pd.DataFrame:
        """Execute MongoDB aggregation pipeline."""
        if not self.is_connected:
            self.connect()

        pipeline = pipeline or []
        cursor = self.db[collection].aggregate(pipeline, **kwargs)
        data = list(cursor)
        df = pd.DataFrame(data)

        if '_id' in df.columns:
            df = df.drop('_id', axis=1)

        return df

    def list_tables(self) -> List[str]:
        """List all collections in database."""
        return self.db.list_collection_names()

    def get_schema(self, collection: str) -> Dict[str, Any]:
        """Get collection schema (sample first document)."""
        sample = self.db[collection].find_one()
        if sample:
            return {k: type(v).__name__ for k, v in sample.items()}
        return {}
