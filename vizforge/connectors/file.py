"""File format connectors."""

import pandas as pd
from .base import BaseConnector, ConnectionConfig


class ExcelConnector(BaseConnector):
    """Advanced Excel connector with multiple sheets."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def test_connection(self) -> bool:
        return True

    def read(self, file_path: str = None, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """Read Excel file."""
        path = file_path or self.config.path
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    def write(self, data: pd.DataFrame, file_path: str = None, sheet_name: str = 'Sheet1', **kwargs) -> bool:
        """Write to Excel."""
        path = file_path or self.config.path
        data.to_excel(path, sheet_name=sheet_name, index=False, **kwargs)
        return True


class ParquetConnector(BaseConnector):
    """Parquet file connector."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def test_connection(self) -> bool:
        return True

    def read(self, file_path: str = None, **kwargs) -> pd.DataFrame:
        """Read Parquet file."""
        path = file_path or self.config.path
        return pd.read_parquet(path, **kwargs)

    def write(self, data: pd.DataFrame, file_path: str = None, **kwargs) -> bool:
        """Write to Parquet."""
        path = file_path or self.config.path
        data.to_parquet(path, index=False, **kwargs)
        return True


class HDF5Connector(BaseConnector):
    """HDF5 file connector."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def test_connection(self) -> bool:
        return True

    def read(self, file_path: str = None, key: str = 'data', **kwargs) -> pd.DataFrame:
        """Read HDF5 file."""
        path = file_path or self.config.path
        return pd.read_hdf(path, key=key, **kwargs)

    def write(self, data: pd.DataFrame, file_path: str = None, key: str = 'data', **kwargs) -> bool:
        """Write to HDF5."""
        path = file_path or self.config.path
        data.to_hdf(path, key=key, mode='w', **kwargs)
        return True
