"""
Cloud storage connectors.

Supports: AWS S3, Google Cloud Storage, Azure Blob Storage
"""

import pandas as pd
import io
from typing import List, Dict, Any
from .base import BaseConnector, ConnectionConfig


class S3Connector(BaseConnector):
    """AWS S3 storage connector."""

    def __init__(self, config: ConnectionConfig):
        """Initialize S3 connector."""
        super().__init__(config)
        self.client = None
        self.resource = None

    def connect(self) -> bool:
        """Connect to AWS S3."""
        try:
            import boto3

            self.client = boto3.client(
                's3',
                aws_access_key_id=self.config.username,
                aws_secret_access_key=self.config.password,
                region_name=self.config.options.get('region', 'us-east-1'),
                **{k: v for k, v in self.config.options.items() if k != 'region'}
            )
            self.resource = boto3.resource(
                's3',
                aws_access_key_id=self.config.username,
                aws_secret_access_key=self.config.password,
                region_name=self.config.options.get('region', 'us-east-1')
            )
            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "S3 connector requires 'boto3'. "
                "Install with: pip install boto3"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to S3: {e}")

    def disconnect(self) -> bool:
        """Disconnect from S3."""
        self.client = None
        self.resource = None
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test S3 connection."""
        try:
            self.client.list_buckets()
            return True
        except:
            return False

    def read(self, key: str, bucket: str = None, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """Read file from S3."""
        if not self.is_connected:
            self.connect()

        bucket = bucket or self.config.bucket
        if not bucket:
            raise ValueError("Bucket name required")

        try:
            obj = self.client.get_object(Bucket=bucket, Key=key)
            content = obj['Body'].read()

            if file_type == 'csv':
                return pd.read_csv(io.BytesIO(content), **kwargs)
            elif file_type == 'json':
                return pd.read_json(io.BytesIO(content), **kwargs)
            elif file_type == 'parquet':
                return pd.read_parquet(io.BytesIO(content), **kwargs)
            elif file_type == 'excel':
                return pd.read_excel(io.BytesIO(content), **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to read from S3: {e}")

    def write(self, data: pd.DataFrame, key: str, bucket: str = None, file_type: str = 'csv', **kwargs) -> bool:
        """Write DataFrame to S3."""
        if not self.is_connected:
            self.connect()

        bucket = bucket or self.config.bucket
        if not bucket:
            raise ValueError("Bucket name required")

        try:
            buffer = io.BytesIO()

            if file_type == 'csv':
                data.to_csv(buffer, index=False, **kwargs)
            elif file_type == 'json':
                data.to_json(buffer, **kwargs)
            elif file_type == 'parquet':
                data.to_parquet(buffer, index=False, **kwargs)
            elif file_type == 'excel':
                data.to_excel(buffer, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            buffer.seek(0)
            self.client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to write to S3: {e}")

    def list_tables(self, bucket: str = None, prefix: str = '') -> List[str]:
        """List files in S3 bucket."""
        bucket = bucket or self.config.bucket
        if not bucket:
            raise ValueError("Bucket name required")

        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]


class GCSConnector(BaseConnector):
    """Google Cloud Storage connector."""

    def __init__(self, config: ConnectionConfig):
        """Initialize GCS connector."""
        super().__init__(config)
        self.client = None

    def connect(self) -> bool:
        """Connect to Google Cloud Storage."""
        try:
            from google.cloud import storage

            credentials_path = self.config.options.get('credentials_path')
            if credentials_path:
                import os
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

            self.client = storage.Client(project=self.config.options.get('project'))
            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "GCS connector requires 'google-cloud-storage'. "
                "Install with: pip install google-cloud-storage"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to GCS: {e}")

    def disconnect(self) -> bool:
        """Disconnect from GCS."""
        self.client = None
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test GCS connection."""
        try:
            list(self.client.list_buckets(max_results=1))
            return True
        except:
            return False

    def read(self, blob_name: str, bucket: str = None, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """Read file from GCS."""
        if not self.is_connected:
            self.connect()

        bucket_name = bucket or self.config.bucket
        if not bucket_name:
            raise ValueError("Bucket name required")

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            content = blob.download_as_bytes()

            if file_type == 'csv':
                return pd.read_csv(io.BytesIO(content), **kwargs)
            elif file_type == 'json':
                return pd.read_json(io.BytesIO(content), **kwargs)
            elif file_type == 'parquet':
                return pd.read_parquet(io.BytesIO(content), **kwargs)
            elif file_type == 'excel':
                return pd.read_excel(io.BytesIO(content), **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to read from GCS: {e}")

    def write(self, data: pd.DataFrame, blob_name: str, bucket: str = None, file_type: str = 'csv', **kwargs) -> bool:
        """Write DataFrame to GCS."""
        if not self.is_connected:
            self.connect()

        bucket_name = bucket or self.config.bucket
        if not bucket_name:
            raise ValueError("Bucket name required")

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            buffer = io.BytesIO()
            if file_type == 'csv':
                data.to_csv(buffer, index=False, **kwargs)
            elif file_type == 'json':
                data.to_json(buffer, **kwargs)
            elif file_type == 'parquet':
                data.to_parquet(buffer, index=False, **kwargs)
            elif file_type == 'excel':
                data.to_excel(buffer, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            buffer.seek(0)
            blob.upload_from_file(buffer)
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to write to GCS: {e}")

    def list_tables(self, bucket: str = None, prefix: str = '') -> List[str]:
        """List files in GCS bucket."""
        bucket_name = bucket or self.config.bucket
        if not bucket_name:
            raise ValueError("Bucket name required")

        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]


class AzureBlobConnector(BaseConnector):
    """Azure Blob Storage connector."""

    def __init__(self, config: ConnectionConfig):
        """Initialize Azure Blob connector."""
        super().__init__(config)
        self.service = None

    def connect(self) -> bool:
        """Connect to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient

            connection_string = self.config.options.get('connection_string')
            if connection_string:
                self.service = BlobServiceClient.from_connection_string(connection_string)
            else:
                account_name = self.config.username
                account_key = self.config.password
                if not account_name or not account_key:
                    raise ValueError("Azure requires account_name and account_key or connection_string")

                account_url = f"https://{account_name}.blob.core.windows.net"
                self.service = BlobServiceClient(account_url=account_url, credential=account_key)

            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "Azure connector requires 'azure-storage-blob'. "
                "Install with: pip install azure-storage-blob"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Azure Blob: {e}")

    def disconnect(self) -> bool:
        """Disconnect from Azure Blob."""
        self.service = None
        self._connected = False
        return True

    def test_connection(self) -> bool:
        """Test Azure connection."""
        try:
            list(self.service.list_containers(max_results=1))
            return True
        except:
            return False

    def read(self, blob_name: str, container: str = None, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """Read file from Azure Blob."""
        if not self.is_connected:
            self.connect()

        container_name = container or self.config.bucket
        if not container_name:
            raise ValueError("Container name required")

        try:
            container_client = self.service.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            content = blob_client.download_blob().readall()

            if file_type == 'csv':
                return pd.read_csv(io.BytesIO(content), **kwargs)
            elif file_type == 'json':
                return pd.read_json(io.BytesIO(content), **kwargs)
            elif file_type == 'parquet':
                return pd.read_parquet(io.BytesIO(content), **kwargs)
            elif file_type == 'excel':
                return pd.read_excel(io.BytesIO(content), **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to read from Azure Blob: {e}")

    def write(self, data: pd.DataFrame, blob_name: str, container: str = None, file_type: str = 'csv', **kwargs) -> bool:
        """Write DataFrame to Azure Blob."""
        if not self.is_connected:
            self.connect()

        container_name = container or self.config.bucket
        if not container_name:
            raise ValueError("Container name required")

        try:
            container_client = self.service.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)

            buffer = io.BytesIO()
            if file_type == 'csv':
                data.to_csv(buffer, index=False, **kwargs)
            elif file_type == 'json':
                data.to_json(buffer, **kwargs)
            elif file_type == 'parquet':
                data.to_parquet(buffer, index=False, **kwargs)
            elif file_type == 'excel':
                data.to_excel(buffer, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            buffer.seek(0)
            blob_client.upload_blob(buffer.getvalue(), overwrite=True)
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to write to Azure Blob: {e}")

    def list_tables(self, container: str = None, prefix: str = '') -> List[str]:
        """List files in Azure container."""
        container_name = container or self.config.bucket
        if not container_name:
            raise ValueError("Container name required")

        container_client = self.service.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]
