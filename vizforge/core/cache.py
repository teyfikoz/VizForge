"""
VizForge Performance Caching Layer

Intelligent caching for expensive operations to improve performance.
Part of VizForge v1.0.0 - Super AGI features.
"""

from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional
import hashlib
import pickle
import time
import pandas as pd


class ChartCache:
    """
    Smart caching for expensive chart operations.

    Cache strategies:
    - Figure cache (reuse pre-computed figures)
    - Data profiling cache (avoid re-profiling same data)
    - Calculation cache (for calculated fields)
    - LRU eviction for memory management
    """

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of cached items
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}

    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            MD5 hash of serialized arguments
        """
        try:
            # Serialize arguments
            key_data = pickle.dumps((args, sorted(kwargs.items())))
            return hashlib.md5(key_data).hexdigest()
        except Exception:
            # Fallback to string representation
            key_str = str(args) + str(sorted(kwargs.items()))
            return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        if key not in self._cache:
            return None

        # Check TTL
        age = time.time() - self._access_times.get(key, 0)
        if age > self.ttl:
            # Expired - remove from cache
            del self._cache[key]
            del self._access_times[key]
            return None

        # Update access time (LRU)
        self._access_times[key] = time.time()
        return self._cache[key]['value']

    def set(self, key: str, value: Any):
        """
        Store item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]

        self._cache[key] = {'value': value}
        self._access_times[key] = time.time()

    def clear(self):
        """Clear all cached items."""
        self._cache.clear()
        self._access_times.clear()

    @staticmethod
    def cache_data_profile(ttl: int = 3600):
        """
        Decorator to cache data profiling results.

        Args:
            ttl: Time-to-live in seconds

        Returns:
            Decorated function with caching

        Example:
            @ChartCache.cache_data_profile(ttl=1800)
            def profile_dataframe(df):
                # Expensive profiling operation
                return analysis_result
        """
        def decorator(func: Callable):
            cache = {}
            cache_times = {}

            @wraps(func)
            def wrapper(df: pd.DataFrame, *args, **kwargs):
                # Generate cache key from DataFrame shape and columns
                cache_key = hashlib.md5(
                    pickle.dumps((df.shape, tuple(df.columns), tuple(df.dtypes)))
                ).hexdigest()

                # Check cache
                if cache_key in cache:
                    age = time.time() - cache_times[cache_key]
                    if age < ttl:
                        return cache[cache_key]

                # Compute and cache result
                result = func(df, *args, **kwargs)
                cache[cache_key] = result
                cache_times[cache_key] = time.time()

                # Evict old entries (keep last 50)
                if len(cache) > 50:
                    oldest_key = min(cache_times, key=cache_times.get)
                    del cache[oldest_key]
                    del cache_times[oldest_key]

                return result

            wrapper.cache = cache
            wrapper.cache_times = cache_times
            wrapper.clear_cache = lambda: cache.clear() or cache_times.clear()

            return wrapper

        return decorator

    @staticmethod
    def memoize(max_size: int = 128):
        """
        Simple memoization decorator using LRU cache.

        Args:
            max_size: Maximum cache size

        Returns:
            Decorated function with memoization
        """
        def decorator(func: Callable):
            return lru_cache(maxsize=max_size)(func)

        return decorator


# Global cache instance
_global_cache = ChartCache(max_size=200, ttl=3600)


def get_global_cache() -> ChartCache:
    """
    Get the global cache instance.

    Returns:
        Global ChartCache instance
    """
    return _global_cache


def clear_cache():
    """Clear the global cache."""
    _global_cache.clear()


def cache_figure(ttl: int = 1800):
    """
    Decorator to cache Plotly figure generation.

    Args:
        ttl: Time-to-live in seconds (default: 30 minutes)

    Returns:
        Decorated function

    Example:
        @cache_figure(ttl=3600)
        def create_complex_chart(data):
            # Expensive chart creation
            return figure
    """
    def decorator(func: Callable):
        cache = {}
        cache_times = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = hashlib.md5(
                pickle.dumps((args, sorted(kwargs.items())))
            ).hexdigest()

            # Check cache
            if cache_key in cache:
                age = time.time() - cache_times[cache_key]
                if age < ttl:
                    return cache[cache_key]

            # Generate figure
            result = func(*args, **kwargs)
            cache[cache_key] = result
            cache_times[cache_key] = time.time()

            return result

        wrapper.clear_cache = lambda: cache.clear() or cache_times.clear()
        return wrapper

    return decorator
