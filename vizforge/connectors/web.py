"""Web scraping connectors."""

import pandas as pd
import requests
from .base import BaseConnector, ConnectionConfig


class HTMLTableConnector(BaseConnector):
    """HTML table scraper."""

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def test_connection(self) -> bool:
        try:
            response = requests.head(self.config.url, timeout=5)
            return response.status_code < 400
        except:
            return False

    def read(self, url: str = None, table_index: int = 0, **kwargs) -> pd.DataFrame:
        """Read HTML tables from URL."""
        target_url = url or self.config.url
        tables = pd.read_html(target_url, **kwargs)
        return tables[table_index] if tables else pd.DataFrame()


class WebScraperConnector(BaseConnector):
    """Advanced web scraper with BeautifulSoup."""

    def connect(self) -> bool:
        self.session = requests.Session()
        self._connected = True
        return True

    def disconnect(self) -> bool:
        if self.session:
            self.session.close()
        self._connected = False
        return True

    def test_connection(self) -> bool:
        try:
            response = self.session.head(self.config.url, timeout=5)
            return response.status_code < 400
        except:
            return False

    def read(self, url: str = None, selector: str = None, **kwargs) -> pd.DataFrame:
        """Scrape data from website."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("WebScraper requires 'beautifulsoup4'. Install with: pip install beautifulsoup4")

        target_url = url or self.config.url
        response = self.session.get(target_url, **kwargs)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        if selector:
            elements = soup.select(selector)
            data = [{'text': el.get_text(strip=True), 'html': str(el)} for el in elements]
            return pd.DataFrame(data)
        else:
            # Extract all text
            return pd.DataFrame([{'content': soup.get_text(strip=True)}])
