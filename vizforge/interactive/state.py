"""
VizForge Session State Management

Streamlit-style session state for maintaining state across dashboard interactions.
Part of VizForge v1.0.0 - Super AGI features.
"""

from typing import Any, Dict, Optional
import threading


class SessionState:
    """
    Manage session state for interactive dashboards.

    Provides dictionary-like interface for storing persistent state
    between user interactions (similar to Streamlit's session_state).

    Thread-safe for concurrent dashboard sessions.

    Example:
        >>> state = SessionState()
        >>> state['counter'] = 0
        >>> state['counter'] += 1
        >>> print(state['counter'])  # 1
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session state.

        Args:
            session_id: Unique session identifier (auto-generated if None)
        """
        self._session_id = session_id or self._generate_session_id()
        self._state: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())

    def __getitem__(self, key: str) -> Any:
        """Get item from session state."""
        with self._lock:
            return self._state[key]

    def __setitem__(self, key: str, value: Any):
        """Set item in session state."""
        with self._lock:
            self._state[key] = value

    def __delitem__(self, key: str):
        """Delete item from session state."""
        with self._lock:
            del self._state[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in session state."""
        with self._lock:
            return key in self._state

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item with default value.

        Args:
            key: State key
            default: Default value if key doesn't exist

        Returns:
            Value or default
        """
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set item in session state.

        Args:
            key: State key
            value: Value to store
        """
        with self._lock:
            self._state[key] = value

    def update(self, **kwargs):
        """
        Update multiple state values.

        Args:
            **kwargs: Key-value pairs to update

        Example:
            >>> state.update(counter=5, name='Alice')
        """
        with self._lock:
            self._state.update(kwargs)

    def clear(self):
        """Clear all state."""
        with self._lock:
            self._state.clear()

    def keys(self):
        """Get all state keys."""
        with self._lock:
            return self._state.keys()

    def values(self):
        """Get all state values."""
        with self._lock:
            return self._state.values()

    def items(self):
        """Get all state items."""
        with self._lock:
            return self._state.items()

    def to_dict(self) -> Dict[str, Any]:
        """
        Export state as dictionary.

        Returns:
            Dictionary copy of state
        """
        with self._lock:
            return self._state.copy()

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id


# Global session store (per-session state management)
_session_store: Dict[str, SessionState] = {}
_store_lock = threading.Lock()


def get_session_state(session_id: Optional[str] = None) -> SessionState:
    """
    Get or create session state for a session.

    Args:
        session_id: Session identifier (creates new if None)

    Returns:
        SessionState instance for this session

    Example:
        >>> state = get_session_state()
        >>> state['user'] = 'Alice'
    """
    with _store_lock:
        if session_id is None or session_id not in _session_store:
            session = SessionState(session_id)
            _session_store[session.session_id] = session
            return session

        return _session_store[session_id]


def clear_session(session_id: str):
    """
    Clear a specific session.

    Args:
        session_id: Session to clear
    """
    with _store_lock:
        if session_id in _session_store:
            _session_store[session_id].clear()
            del _session_store[session_id]


def clear_all_sessions():
    """Clear all sessions."""
    with _store_lock:
        _session_store.clear()
