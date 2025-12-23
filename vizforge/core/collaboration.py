"""
Real-Time Collaboration System for VizForge

Multi-user collaboration like Google Docs for data visualization.
Plotly limitation: No collaboration features.
VizForge innovation: Real-time collaborative dashboards.

Features:
- Multi-user editing
- Live cursor tracking
- Change synchronization
- Conflict resolution
- Presence indicators
- Chat and comments
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import time
import json
from enum import Enum
import uuid


class ChangeType(Enum):
    """Types of collaborative changes."""
    CHART_ADD = "chart_add"
    CHART_REMOVE = "chart_remove"
    CHART_UPDATE = "chart_update"
    DATA_UPDATE = "data_update"
    STYLE_UPDATE = "style_update"
    FILTER_UPDATE = "filter_update"
    COMMENT_ADD = "comment_add"


@dataclass
class User:
    """Collaborative user."""
    id: str
    name: str
    color: str  # For cursor/presence
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    cursor_position: Optional[Dict[str, float]] = None
    last_seen: float = field(default_factory=time.time)


@dataclass
class Change:
    """A collaborative change/operation."""
    id: str
    user_id: str
    timestamp: float
    type: ChangeType
    target_id: str  # ID of chart/component being changed
    data: Dict[str, Any]
    applied: bool = False


@dataclass
class Comment:
    """A comment on the dashboard."""
    id: str
    user_id: str
    timestamp: float
    target_id: str  # Chart or component ID
    text: str
    resolved: bool = False
    replies: List['Comment'] = field(default_factory=list)


class CollaborationSession:
    """
    Manages a collaborative editing session.

    Handles:
    - User presence
    - Change tracking
    - Synchronization
    - Conflict resolution
    """

    def __init__(self, session_id: str):
        """
        Initialize collaboration session.

        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.users: Dict[str, User] = {}
        self.changes: List[Change] = []
        self.comments: Dict[str, Comment] = {}
        self._change_callbacks: List[Callable] = []
        self._presence_callbacks: List[Callable] = []

    def join(self, user: User):
        """
        User joins collaboration session.

        Args:
            user: User object
        """
        self.users[user.id] = user
        user.last_seen = time.time()

        # Notify other users
        self._trigger_presence_update({
            'type': 'user_joined',
            'user': user
        })

    def leave(self, user_id: str):
        """
        User leaves collaboration session.

        Args:
            user_id: User ID
        """
        if user_id in self.users:
            user = self.users[user_id]
            del self.users[user_id]

            self._trigger_presence_update({
                'type': 'user_left',
                'user': user
            })

    def add_change(self, change: Change):
        """
        Add a collaborative change.

        Args:
            change: Change object
        """
        # Validate change
        if change.user_id not in self.users:
            raise ValueError(f"Unknown user: {change.user_id}")

        # Check for conflicts
        conflicts = self._detect_conflicts(change)

        if conflicts:
            # Resolve conflicts using operational transformation
            change = self._resolve_conflicts(change, conflicts)

        # Add change
        self.changes.append(change)

        # Notify callbacks
        self._trigger_change_callback(change)

    def _detect_conflicts(self, new_change: Change) -> List[Change]:
        """
        Detect conflicting changes.

        Args:
            new_change: New change to check

        Returns:
            List of conflicting changes
        """
        conflicts = []

        # Find recent changes to same target
        for change in reversed(self.changes[-10:]):  # Last 10 changes
            if (change.target_id == new_change.target_id and
                not change.applied and
                change.user_id != new_change.user_id and
                change.type == new_change.type):
                conflicts.append(change)

        return conflicts

    def _resolve_conflicts(self, new_change: Change, conflicts: List[Change]) -> Change:
        """
        Resolve conflicts using operational transformation.

        Args:
            new_change: New change
            conflicts: Conflicting changes

        Returns:
            Resolved change
        """
        # Simplified operational transformation
        # In production, use full OT algorithm

        resolved = new_change

        for conflict in conflicts:
            # Timestamp-based resolution (last write wins)
            if conflict.timestamp > resolved.timestamp:
                # Merge data
                merged_data = {**resolved.data, **conflict.data}
                resolved.data = merged_data

        return resolved

    def update_cursor(self, user_id: str, position: Dict[str, float]):
        """
        Update user's cursor position.

        Args:
            user_id: User ID
            position: Cursor coordinates {x, y}
        """
        if user_id in self.users:
            self.users[user_id].cursor_position = position
            self.users[user_id].last_seen = time.time()

            self._trigger_presence_update({
                'type': 'cursor_move',
                'user_id': user_id,
                'position': position
            })

    def add_comment(self, comment: Comment):
        """
        Add a comment to the dashboard.

        Args:
            comment: Comment object
        """
        self.comments[comment.id] = comment

        self._trigger_change_callback(Change(
            id=str(uuid.uuid4()),
            user_id=comment.user_id,
            timestamp=time.time(),
            type=ChangeType.COMMENT_ADD,
            target_id=comment.target_id,
            data={'comment_id': comment.id}
        ))

    def resolve_comment(self, comment_id: str, user_id: str):
        """
        Resolve a comment.

        Args:
            comment_id: Comment ID
            user_id: User resolving the comment
        """
        if comment_id in self.comments:
            self.comments[comment_id].resolved = True

    def on_change(self, callback: Callable):
        """
        Register callback for changes.

        Args:
            callback: Function to call on change
        """
        self._change_callbacks.append(callback)

    def on_presence_update(self, callback: Callable):
        """
        Register callback for presence updates.

        Args:
            callback: Function to call on presence change
        """
        self._presence_callbacks.append(callback)

    def _trigger_change_callback(self, change: Change):
        """Trigger all change callbacks."""
        for callback in self._change_callbacks:
            try:
                callback(change)
            except Exception as e:
                print(f"Change callback error: {e}")

    def _trigger_presence_update(self, update: Dict[str, Any]):
        """Trigger all presence update callbacks."""
        for callback in self._presence_callbacks:
            try:
                callback(update)
            except Exception as e:
                print(f"Presence callback error: {e}")

    def get_state(self) -> Dict[str, Any]:
        """
        Get current collaboration state.

        Returns:
            Full session state
        """
        return {
            'session_id': self.session_id,
            'users': [vars(u) for u in self.users.values()],
            'changes': [vars(c) for c in self.changes],
            'comments': [vars(c) for c in self.comments.values()]
        }


class CollaborationServer:
    """
    WebSocket server for real-time collaboration.

    Handles communication between clients.
    """

    def __init__(self, port: int = 8765):
        """
        Initialize collaboration server.

        Args:
            port: WebSocket server port
        """
        self.port = port
        self.sessions: Dict[str, CollaborationSession] = {}

    def create_session(self, dashboard_id: str) -> CollaborationSession:
        """
        Create new collaboration session.

        Args:
            dashboard_id: Dashboard identifier

        Returns:
            Collaboration session
        """
        session = CollaborationSession(dashboard_id)
        self.sessions[dashboard_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """
        Get existing session.

        Args:
            session_id: Session ID

        Returns:
            Session or None
        """
        return self.sessions.get(session_id)

    def broadcast(self, session_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """
        Broadcast message to all users in session.

        Args:
            session_id: Session ID
            message: Message to broadcast
            exclude_user: User ID to exclude (optional)
        """
        session = self.get_session(session_id)
        if not session:
            return

        # In production, this would send via WebSocket
        print(f"Broadcasting to session {session_id}: {message}")


# Global collaboration server
_server = None


def get_collaboration_server(port: int = 8765) -> CollaborationServer:
    """Get global collaboration server."""
    global _server
    if _server is None:
        _server = CollaborationServer(port)
    return _server


# Client-side collaboration helper
def enable_collaboration(dashboard, user: User, session_id: Optional[str] = None) -> CollaborationSession:
    """
    Enable collaboration for a dashboard.

    Args:
        dashboard: Dashboard object
        user: Current user
        session_id: Session ID (creates new if None)

    Returns:
        Collaboration session

    Example:
        user = User(id='user123', name='John Doe', color='#3498db')
        session = enable_collaboration(dashboard, user)
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    server = get_collaboration_server()
    session = server.get_session(session_id) or server.create_session(session_id)

    session.join(user)

    # Add dashboard change tracking
    def track_change(change_type: ChangeType, target_id: str, data: Dict):
        change = Change(
            id=str(uuid.uuid4()),
            user_id=user.id,
            timestamp=time.time(),
            type=change_type,
            target_id=target_id,
            data=data
        )
        session.add_change(change)
        server.broadcast(session_id, {'type': 'change', 'change': vars(change)}, exclude_user=user.id)

    # Hook into dashboard events
    dashboard._collaboration_session = session
    dashboard._collaboration_user = user
    dashboard._track_change = track_change

    return session
